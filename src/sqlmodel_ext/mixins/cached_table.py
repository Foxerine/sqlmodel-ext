"""
CachedTableBaseMixin -- Redis query cache for SQLModel tables.

Add opt-in Redis caching to SQLModel table models. Inherit to enable,
omit if you don't need caching.

Design principles:
- Rich domain model: all cache read/write/invalidation logic lives in the mixin.
- Centralized Redis client: installed once via ``configure_redis()`` at startup.
- Fail fast on misconfiguration: unconfigured Redis -> ``RuntimeError``.
  Transient Redis errors are logged and fall back to the database.
- Full-object caching: always cache every column; field subsets are not supported.
- Serialization: ``model_dump`` (columns only) -> orjson -> ``model_validate``,
  so the loaded instance has a valid ``_sa_instance_state``.
- Explicit failure: the ``no_cache`` parameter only exists on
  ``CachedTableBaseMixin.get()``. Calling a non-cached model with ``no_cache=True``
  raises ``TypeError``, as Python naturally reports.

Two-layer cache architecture:
- ID cache (``id:{ModelName}:{id_value}``) -- single-row equality queries, row-level
  invalidation.
- Query cache (``query:{ModelName}:v{version}:{hash}``) -- condition/list queries,
  invalidated by bumping a namespace version.
- Version key (``ver:{ModelName}``) -- namespace version for the query cache.
  INCR renders all older query keys unreachable.

Invalidation granularity:
- save/update: row-level multi-key DEL of ``id:{cls}:{id}`` +
  O(1) pipeline INCR of ``ver:{cls}``.
- delete(instances): row-level multi-key DEL for each instance + O(1) pipeline
  INCR of ``ver:{cls}``.
- delete(condition): SCAN+DEL of ``id:{cls}:*`` (rare path) + O(1) INCR of
  ``ver:{cls}``.
- STI polymorphism: on subclass changes, pipeline INCR the version key for the
  class itself and every cached ancestor.

Cache skip conditions:
- ``no_cache=True`` (caller explicitly opts out).
- ``load`` contains a non-MANYTOONE or non-cacheable relationship
  (the multi-ID cache optimization cannot handle it).
- ``options is not None`` (an ``ExecutableOption`` may change load behaviour).
- ``with_for_update`` (pessimistic locking must read the latest row).
- ``populate_existing`` (caller explicitly asked to refresh the identity map).
- ``join is not None`` (JOIN target changes don't invalidate the main model
  and would cause phantom reads).
  # TODO: future optimisation -- support JOIN query caching by tracking joined
  # target invalidations.

Dependencies::

    redis.asyncio
    orjson  (optional, falls back to stdlib json)
    sqlmodel_ext.base.SQLModelBase

Usage::

    class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True):
        __cache_ttl__: ClassVar[int] = 1800  # 30 minutes
"""
import ast
import asyncio
import hashlib
import inspect
import textwrap
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self, cast, overload

import json
import logging
from collections.abc import Callable

try:
    import orjson as _json_lib

    def _json_dumps(obj: Any) -> bytes:
        return _json_lib.dumps(obj)

    def _json_loads(data: bytes | str) -> Any:
        return _json_lib.loads(data)
except ImportError:
    _json_lib = None  # type: ignore[assignment]

    def _json_dumps(obj: Any) -> bytes:
        return json.dumps(obj, separators=(",", ":"), default=str).encode("utf-8")

    def _json_loads(data: bytes | str) -> Any:
        return json.loads(data)

logger = logging.getLogger(__name__)

from pydantic import ValidationError
from sqlalchemy import ColumnElement, event, inspect as sa_inspect, select as sa_select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import InstanceState, QueryableAttribute, Session as _SyncSession, make_transient_to_detached
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.orm.relationships import MANYTOONE  # pyright: ignore[reportPrivateImportUsage]
from sqlalchemy.sql import operators
from sqlalchemy.sql.base import ExecutableOption
from sqlalchemy.sql.dml import Delete, Update
from sqlalchemy.sql.elements import BinaryExpression
from sqlmodel import col
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlalchemy.sql._typing import _OnClauseArgument  # pyright: ignore[reportPrivateUsage]

from sqlmodel_ext.base import SQLModelBase
from sqlmodel_ext.mixins.polymorphic import PolymorphicBaseMixin
from sqlmodel_ext.mixins.table import TableBaseMixin
from sqlmodel_ext.pagination import TableViewRequest



class _CacheResultType(StrEnum):
    """Serialization wrapper discriminator -- distinguishes None/single/list results."""
    NONE = 'none'
    LIST = 'list'
    SINGLE = 'single'


# Wrapper JSON field names for cached results
_WRAPPER_TYPE_KEY = '_t'
_WRAPPER_ITEMS_KEY = '_items'
_WRAPPER_DATA_KEY = '_data'
_WRAPPER_CLASS_KEY = '_c'  # Actual class name (restores the correct subclass for polymorphic rows)

# session.info keys -- cache invalidation state tracking
_SESSION_PENDING_CACHE_KEY = '_pending_cache_invalidation_types'
_SESSION_SYNCED_CACHE_KEY = '_synced_cache_invalidation_types'
_SESSION_CASCADE_DELETED_KEY = '_cascade_deleted_for_sync_invalidation'

# Sentinel -- add() scenario: new rows do not need ID-cache invalidation, only
# the query cache must be bumped.
_QUERY_ONLY_INVALIDATION = object()

# Sentinel -- delete(condition): condition deletes cannot extract individual
# IDs, so a model-level full invalidation is required.
_FULL_MODEL_INVALIDATION = object()

# Sentinel returned by _try_load_from_id_caches() for "cache miss"
# (distinct from a cached None result).
_LOAD_CACHE_MISS = object()

# Raw DML statement types (hoisted to a module constant so
# _warn_raw_dml_on_cached does not rebuild the tuple on every execute()).
_DML_TYPES: tuple[type, ...] = (Update, Delete)

# Cache invalidation methods that subclasses must not call directly.
# check_cache_config() walks subclass method bodies with AST to prevent
# MissingGreenlet errors caused by post-commit access to expired attributes.
_FORBIDDEN_DIRECT_CALLS: frozenset[str] = frozenset({
    'invalidate_by_id',
    'invalidate_all',
    '_invalidate_for_model',
    '_invalidate_id_cache',
    '_invalidate_query_caches',
})


class CachedTableBaseMixin(TableBaseMixin):
    """
    Inherit to enable Redis query caching. Omit if caching is unwanted.

    MRO: Model -> CachedTableBaseMixin -> Base -> TableBaseMixin

    ClassVar configuration:
        __cache_ttl__: cache TTL in seconds; subclasses override as needed.
    """

    __cache_ttl__: ClassVar[int] = 3600
    """Cache TTL in seconds. Override via the class-level ``cache_ttl=N`` kwarg (set by the metaclass)."""

    _commit_hook_registered: ClassVar[bool] = False
    """Flag tracking whether the ``after_commit`` event hook has been registered."""

    # ---- internal constants ----
    _CACHE_KEY_PREFIX: ClassVar[str] = 'query'
    """Query cache key prefix. Format: ``query:{ModelName}:v{version}:{hash}``."""

    _ID_CACHE_KEY_PREFIX: ClassVar[str] = 'id'
    """ID cache key prefix. Format: ``id:{ModelName}:{id_value}``."""

    _CACHE_KEY_HASH_LENGTH: ClassVar[int] = 16
    """Length of the MD5 digest suffix used in query cache keys."""

    _VERSION_KEY_PREFIX: ClassVar[str] = 'ver'
    """Version key prefix for query cache. Format: ``ver:{ModelName}``."""

    _SCAN_BATCH_SIZE: ClassVar[int] = 100
    """Count argument for each Redis SCAN iteration (only used in the rare model-level ID cache wipe path)."""

    _subclass_name_cache: ClassVar[dict[str, type]] = {}
    """``class_name -> type`` cache used to avoid repeated recursion in ``_resolve_subclass()``."""

    on_cache_hit: ClassVar[Callable[[str], None] | None] = None
    """Optional callback invoked on cache hit. Receives model class name. Set at startup for metrics."""

    on_cache_miss: ClassVar[Callable[[str], None] | None] = None
    """Optional callback invoked on cache miss. Receives model class name. Set at startup for metrics."""

    # ================================================================
    #  Redis client access (managed via configure_redis())
    # ================================================================

    _redis_client: ClassVar[Any] = None
    """Redis client instance. Must be set via configure_redis() before use."""

    @classmethod
    def configure_redis(cls, client: Any) -> None:
        """Configure the Redis client for caching.

        Must be called once at application startup before any cache operations.

        :param client: A redis.asyncio.Redis instance (decode_responses=False)
        """
        cls._redis_client = client

    @classmethod
    def _get_client(cls) -> Any:
        """Get the Redis client for caching.

        Returns the client configured via configure_redis().
        Raises RuntimeError if not configured.

        Return type is Any because redis.asyncio.Redis generic parameter
        depends on runtime configuration (decode_responses).
        """
        if cls._redis_client is None:
            raise RuntimeError(
                f"{cls.__name__}: Redis client not configured. "
                f"Call CachedTableBaseMixin.configure_redis(redis_client) at startup."
            )
        return cls._redis_client

    # ================================================================
    #  Cache primitives (runtime Redis errors -> logger.error + degrade)
    # ================================================================

    @classmethod
    async def _cache_get(cls, key: str) -> bytes | None:
        """Read from cache. On Redis failure, log and return None (fall back to the DB)."""
        try:
            return await cls._get_client().get(key)
        except RuntimeError:
            raise  # Not configured -- fail fast
        except Exception as e:
            logger.error(f"Redis read error key='{key}': {e}")
            return None

    @classmethod
    async def _cache_set(cls, key: str, value: bytes, ttl: int) -> None:
        """Write to cache. On Redis failure, log and skip (non-critical path)."""
        try:
            await cls._get_client().set(key, value, ex=ttl)
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Redis write error key='{key}': {e}")

    @classmethod
    async def _cache_delete(cls, key: str) -> None:
        """Delete a cache key. Propagates Redis errors (caller decides whether to swallow).

        ``RuntimeError`` (client not configured) propagates. Other exceptions
        also propagate so the sync invalidation path can detect failure,
        skip marking the type as synced, and let the compensation path retry.
        """
        await cls._get_client().delete(key)

    @classmethod
    async def _cache_delete_pattern(cls, pattern: str) -> None:
        """SCAN + DEL pattern deletion (avoids blocking with KEYS).

        Propagates Redis errors (same contract as ``_cache_delete``).
        """
        client = cls._get_client()
        cursor = 0
        while True:
            cursor, keys = await client.scan(
                cursor, match=pattern, count=cls._SCAN_BATCH_SIZE,
            )
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

    # ================================================================
    #  Version management (version bump replaces SCAN+DEL for query cache)
    # ================================================================

    @classmethod
    def _build_version_key(cls) -> str:
        """Build the version key. Format: ``ver:{ModelName}``."""
        return f"{cls._VERSION_KEY_PREFIX}:{cls.__name__}"

    @classmethod
    async def _get_query_version(cls) -> int:
        """Return the current query cache version. Missing key returns 0 (initial).

        On Redis failure, fall back to 0 (query keys degrade to ``v0``).
        """
        try:
            raw = await cls._get_client().get(cls._build_version_key())
            return int(raw) if raw is not None else 0
        except RuntimeError:
            raise  # Not configured -- fail fast
        except Exception as e:
            logger.error(f"Redis version read error ({cls.__name__}): {e}")
            return 0

    @classmethod
    async def _bump_query_version(cls) -> int:
        """Increment the query cache version. O(1) replacement for SCAN+DEL.

        ``INCR`` atomically initialises missing keys to 0 then increments. Old
        query keys at the previous version expire naturally via TTL; no proactive
        cleanup is required.

        :return: the new version (0 on failure).
        """
        try:
            new_ver: int = await cls._get_client().incr(cls._build_version_key())
            return new_ver
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Redis version bump error ({cls.__name__}): {e}")
            return 0

    # ================================================================
    #  Static checks
    # ================================================================

    @classmethod
    def check_cache_config(cls) -> None:
        """Walk all ``CachedTableBaseMixin`` subclasses, validate configuration,
        and register SQLAlchemy session event hooks.

        Call once during application startup, after ``configure_redis()``.

        Checks:
        1. A Redis client is configured.
        2. No subclass (recursive) overrides ``_get_client`` (would break cache access).
        3. ``__cache_ttl__`` is a positive int (``None`` is rejected).
        4. No subclass method calls cache-invalidation methods directly (AST check).

        Side effects:
        - Registers SQLAlchemy Session ``after_commit``/``after_rollback``/
          ``persistent_to_deleted`` event hooks.
        """
        # Verify the Redis client is available.
        _ = cls._get_client()

        violations: list[str] = []

        def _check_forbidden_calls(sub: type) -> None:
            """AST-check subclass method bodies for direct invalidation calls.

            When a subclass method bypasses CRUD helpers and still needs to
            invalidate the cache, it should use::

                _register_pending_invalidation()     # record pending IDs
                await session.commit()               # enhanced AsyncSession flushes them

            (the sqlmodel-ext enhanced ``AsyncSession`` synchronously
            invalidates every registered pending entry after commit).

            Direct calls to ``invalidate_by_id`` and friends can touch expired
            attributes after commit, triggering ``MissingGreenlet``.
            """
            for attr_name, attr in sub.__dict__.items():
                # Unwrap descriptors and collect function objects to scan.
                funcs: list[Any] = []
                if isinstance(attr, (classmethod, staticmethod)):
                    funcs.append(attr.__func__)
                elif isinstance(attr, property):
                    for accessor in (attr.fget, attr.fset, attr.fdel):
                        if accessor is not None:
                            funcs.append(accessor)
                elif inspect.isfunction(attr):
                    funcs.append(attr)
                seen: set[str] = set()
                for func_obj in funcs:
                    try:
                        source = textwrap.dedent(inspect.getsource(func_obj))
                        tree = ast.parse(source)
                    except (OSError, TypeError, SyntaxError):
                        continue
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call):
                            continue
                        func = node.func
                        call_name: str | None = None
                        if isinstance(func, ast.Attribute):
                            call_name = func.attr
                        elif isinstance(func, ast.Name):
                            call_name = func.id
                        if call_name and call_name in _FORBIDDEN_DIRECT_CALLS and call_name not in seen:
                            seen.add(call_name)
                            violations.append(f"  - {sub.__name__}.{attr_name}() -> {call_name}()")

        def _check_subclasses(parent: type) -> None:
            for sub in parent.__subclasses__():
                if '_get_client' in sub.__dict__:
                    raise TypeError(f"{sub.__name__} must not override _get_client")
                ttl = getattr(sub, '__cache_ttl__', None)
                if not isinstance(ttl, int) or ttl <= 0:
                    raise ValueError(
                        f"{sub.__name__}.__cache_ttl__ must be a positive int, got: {ttl!r}"
                    )
                _check_forbidden_calls(sub)
                _check_subclasses(sub)

        _check_subclasses(cls)

        if violations:
            nl = '\n'
            raise TypeError(
                "The following subclass methods call cache-invalidation methods "
                "directly (risking MissingGreenlet after commit):\n"
                f"{nl.join(violations)}\n"
                "Use _register_pending_invalidation() followed by "
                "`await session.commit()` (the enhanced AsyncSession "
                "synchronously invalidates the registered pendings) instead."
            )

        # Install session event hooks (idempotent).
        cls._register_session_commit_hook()

        # Warmup: every model subclass is loaded by now, so pre-build the
        # {table name: [cached classes]} index. This keeps the raw-DML warning
        # hot path of the enhanced AsyncSession.execute() free of lazy
        # construction (first call already hits the index).
        cls._build_cached_tablename_index()

    @classmethod
    def _register_session_commit_hook(cls) -> None:
        """Install SQLAlchemy Session ``after_commit``/``after_rollback`` hooks.

        ``after_commit``: flush the invalidation queue accumulated in
        ``session.info``. Covers every commit path: CRUD methods with
        ``commit=True`` and bare ``session.commit()`` calls.

        ``after_rollback``: drop the queued invalidations (rows were rolled
        back, nothing to invalidate).

        Idempotent: repeated calls install the hooks only once.

        Limitation (fire-and-forget): ``after_commit`` handlers are synchronous
        by SQLAlchemy contract and cannot ``await`` async invalidation.
        ``loop.create_task()`` schedules a compensation task, leaving a tiny
        window (typically < 1 ms) between commit return and cache clearing.
        ``commit=True`` CRUD methods already invalidate synchronously during
        ``await`` (no window), so this path only matters for
        ``commit=False`` -> ``session.commit()``. TTL provides the eventual
        consistency backstop.
        """
        if cls._commit_hook_registered:
            return

        def _after_commit_handler(session: _SyncSession) -> None:
            pending: dict[type, set[Any]] | None = session.info.pop(_SESSION_PENDING_CACHE_KEY, None)
            if not pending:
                return
            loop = asyncio.get_running_loop()

            # Create the sync-invalidation tracking dict for this commit cycle.
            # After super() returns (i.e. after this handler runs), CRUD methods
            # push their successfully-invalidated (type, instance_ids) pairs
            # into this dict. The _compensate task then dedupes by instance_id
            # and only compensates IDs that were not sync-invalidated.
            synced: dict[type, set[Any]] = {}
            session.info[_SESSION_SYNCED_CACHE_KEY] = synced

            async def _compensate(
                    to_invalidate: dict[type, set[Any]],
                    already_synced: dict[type, set[Any]],
            ) -> None:
                sentinels = {_QUERY_ONLY_INVALIDATION, _FULL_MODEL_INVALIDATION}
                for model_type, pending_ids in to_invalidate.items():
                    if not issubclass(model_type, CachedTableBaseMixin):
                        continue

                    needs_full = _FULL_MODEL_INVALIDATION in pending_ids
                    synced_ids = already_synced.get(model_type)

                    if synced_ids is not None:
                        # Sync path already invalidated this type (query cache covered).
                        if needs_full and _FULL_MODEL_INVALIDATION not in synced_ids:
                            # Pending includes the full-invalidation sentinel but
                            # the sync path did not perform a full invalidation —
                            # compensate at the model level now.
                            try:
                                await model_type._invalidate_for_model()
                            except Exception as e:
                                logger.error(f"post-commit model-level invalidation failed ({model_type.__name__}): {e}")
                            continue
                        # Compensate only the ID caches that the sync path missed.
                        remaining = pending_ids - synced_ids - sentinels
                        if remaining:
                            try:
                                for _id in remaining:
                                    await model_type._invalidate_id_cache(_id)
                            except Exception as e:
                                logger.error(f"post-commit ID-cache invalidation failed ({model_type.__name__}): {e}")
                        continue

                    # This type was not touched by the sync path at all — full
                    # compensation (non-migrated callers).
                    logger.warning(
                        f"fallback compensation triggered: {model_type.__name__} "
                        f"did not go through the sync invalidation path "
                        f"(pending_ids={len(pending_ids)}; this commit bypassed the "
                        f"enhanced AsyncSession.commit() -- typically a "
                        f"begin()/savepoint path, covered by this fallback)"
                    )
                    try:
                        if needs_full:
                            await model_type._invalidate_for_model()
                        else:
                            real_ids = pending_ids - sentinels
                            has_query_only = _QUERY_ONLY_INVALIDATION in pending_ids
                            if real_ids:
                                for _id in real_ids:
                                    await model_type._invalidate_id_cache(_id)
                                await model_type._invalidate_query_caches()
                            elif has_query_only:
                                await model_type._invalidate_query_caches()
                            else:
                                await model_type._invalidate_for_model()
                    except Exception as e:
                        logger.error(f"post-commit compensation failed ({model_type.__name__}): {e}")

            # fire-and-forget: instance_ids handled by the sync path get deduped
            # via `synced`; only commit=False accumulations that were not
            # sync-invalidated end up being compensated here.
            _ = loop.create_task(_compensate(pending, synced))

        def _after_rollback_handler(session: _SyncSession) -> None:
            session.info.pop(_SESSION_PENDING_CACHE_KEY, None)
            session.info.pop(_SESSION_SYNCED_CACHE_KEY, None)
            session.info.pop(_SESSION_CASCADE_DELETED_KEY, None)

        def _on_persistent_to_deleted(session: _SyncSession, instance: object) -> None:
            """Cascade delete cache invalidation -- listen for the
            persistent -> deleted state transition during flush.

            In SA async mode, ``cascade_delete`` relationships with
            ``passive_deletes=False`` are resolved by SA during flush: it
            ``SELECT``s the children, then issues one ``DELETE`` per child.
            Each of those ``DELETE``s fires this event.

            Double-write strategy:
            1. ``_SESSION_PENDING_CACHE_KEY`` -- after_commit compensation
               path (fire-and-forget).
            2. ``_SESSION_CASCADE_DELETED_KEY`` -- ``delete()`` sync path
               (awaits invalidation with no window).
            """
            if isinstance(instance, CachedTableBaseMixin):
                _id = getattr(instance, 'id', None)
                if _id is not None:
                    _cls = type(instance)
                    CachedTableBaseMixin._register_pending_invalidation(
                        session, _cls, _id,  # pyright: ignore[reportArgumentType]  # SA event delivers _SyncSession, .info is compatible
                    )
                    cascade_info: dict[type, set[Any]] = session.info.setdefault(
                        _SESSION_CASCADE_DELETED_KEY, {}
                    )
                    cascade_info.setdefault(_cls, set()).add(_id)

        event.listen(_SyncSession, "after_commit", _after_commit_handler)
        event.listen(_SyncSession, "after_rollback", _after_rollback_handler)
        event.listen(_SyncSession, "persistent_to_deleted", _on_persistent_to_deleted)

        cls._commit_hook_registered = True
        logger.debug("CachedTableBaseMixin: session commit/rollback/cascade event hooks registered")

    # ================================================================
    #  ID query detection
    # ================================================================

    @classmethod
    def _extract_id_from_condition(cls, condition: Any) -> Any | None:
        """Detect whether ``condition`` is an ``id == value`` equality query.

        :return: the literal ID value, or ``None`` if the condition is not an ID equality query.
        """
        if not isinstance(condition, BinaryExpression):
            return None
        if condition.operator is not operators.eq:
            return None
        left = condition.left
        if hasattr(left, 'key') and left.key == 'id':
            right = condition.right
            if hasattr(right, 'value'):
                return right.value
        return None

    @classmethod
    def _build_id_cache_key(cls, id_value: Any) -> str:
        """Build an ID cache key. Format: ``id:{ModelName}:{id_value}``."""
        return f"{cls._ID_CACHE_KEY_PREFIX}:{cls.__name__}:{id_value}"

    # ================================================================
    #  load relationship cache (multi-ID union lookup)
    # ================================================================

    @classmethod
    def _has_pending_invalidation(cls, session: AsyncSession) -> bool:
        """Return True when the session has pending invalidations related to ``cls``."""
        pending: dict[type, set[Any]] | None = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if not pending:
            return False
        return any(
            issubclass(pending_type, cls) or issubclass(cls, pending_type)
            for pending_type in pending
        )

    @classmethod
    def _analyze_load_relations(
            cls,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]],
    ) -> 'list[tuple[str, type[CachedTableBaseMixin], str]] | None':
        """Check whether every entry of ``load`` is a MANYTOONE relationship
        targeting a cacheable model.

        Only direct relationships owned by ``cls`` (including inherited ones)
        are considered. Chained loads such as
        ``Character.tool_set -> ToolSet.tools`` are not supported.

        :return: ``[(rel_name, target_cls, fk_attr_name), ...]`` on success,
                 or ``None`` if the optimization is not applicable.
        """
        load_list = load if isinstance(load, list) else [load]
        mapper = sa_inspect(cls)
        relationships = mapper.relationships  # pyright: ignore[reportOptionalMemberAccess]
        result: list[tuple[str, type[CachedTableBaseMixin], str]] = []

        for attr in load_list:
            # Require the relationship to be owned by cls (including inheritance).
            attr_owner: type[Any] = attr.class_  # pyright: ignore[reportAssignmentType]
            if not issubclass(cls, attr_owner):
                return None  # chained load; cannot be served from the cache

            rel_name: str = attr.key
            if rel_name not in relationships:
                return None  # not a relationship attribute

            rel_prop = relationships[rel_name]
            if rel_prop.direction is not MANYTOONE:
                return None  # only MANYTOONE (FK on the main model) is supported

            target_cls = rel_prop.mapper.class_
            if not issubclass(target_cls, CachedTableBaseMixin):
                return None  # target model is not cacheable

            # Extract the FK attribute name (e.g. ``permission_id``).
            local_col = rel_prop.local_remote_pairs[0][0]
            fk_attr_name: str = local_col.key
            result.append((rel_name, target_cls, fk_attr_name))

        return result

    @classmethod
    async def _try_load_from_id_caches(
            cls,
            session: AsyncSession,
            id_value: Any,
            rel_info: 'list[tuple[str, type[CachedTableBaseMixin], str]]',
    ) -> Any:
        """Attempt to assemble the main model + its MANYTOONE relationships from
        multiple ID caches in a single pass.

        Steps:
        1. Read the main model's ID cache and deserialize.
        2. For each entry in ``rel_info``, look up the related model's ID cache
           by the main row's FK value.
        3. If everything hits, merge into the session via
           ``session.merge(load=False)`` plus ``set_committed_value`` and return.
        4. If any key misses, return ``_LOAD_CACHE_MISS`` so the caller can fall
           back to the database.

        :return: the main model instance | None (cached empty result)
                 | ``_LOAD_CACHE_MISS`` (caller must fall back to the DB).
        """
        # 1. Main model ID cache lookup.
        main_cache_key = cls._build_id_cache_key(id_value)
        main_raw = await cls._cache_get(main_cache_key)
        if main_raw is None:
            return _LOAD_CACHE_MISS

        try:
            main_obj = cls._deserialize_result(main_raw, 'first')
        except Exception:
            try:
                await cls._cache_delete(main_cache_key)
            except Exception:
                pass
            return _LOAD_CACHE_MISS

        if main_obj is None:
            return None  # cached empty result

        # 2. Collect the related FKs, build the cache key list, and fetch them
        #    in a single pipeline mget call. (N relationships go from N+1 Redis
        #    RTTs down to 2: one GET for the main model + one MGET for all relations.)
        rel_entries: list[tuple[str, 'type[CachedTableBaseMixin]', str | None]] = []
        """(rel_name, target_cls, cache_key | None)"""
        pipeline_keys: list[str] = []

        for rel_name, target_cls, fk_attr_name in rel_info:
            fk_value = getattr(main_obj, fk_attr_name, None)
            if fk_value is None:
                rel_entries.append((rel_name, target_cls, None))
                continue
            if target_cls._has_pending_invalidation(session):
                return _LOAD_CACHE_MISS
            rel_cache_key = target_cls._build_id_cache_key(fk_value)
            rel_entries.append((rel_name, target_cls, rel_cache_key))
            pipeline_keys.append(rel_cache_key)

        # Batch-read every relationship cache in a single RTT.
        pipeline_results: list[bytes | None] = []
        if pipeline_keys:
            try:
                pipeline_results = await cls._get_client().mget(pipeline_keys)
            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"Redis mget error: {e}")
                return _LOAD_CACHE_MISS

        # 3. Deserialize the related objects.
        pipeline_idx = 0
        rel_objects: list[tuple[str, Any]] = []
        for rel_name, target_cls, cache_key in rel_entries:
            if cache_key is None:
                rel_objects.append((rel_name, None))
                continue

            rel_raw = pipeline_results[pipeline_idx]
            pipeline_idx += 1
            if rel_raw is None:
                return _LOAD_CACHE_MISS

            try:
                rel_obj = target_cls._deserialize_result(rel_raw, 'first')
            except Exception:
                try:
                    await target_cls._cache_delete(cache_key)
                except Exception:
                    pass
                return _LOAD_CACHE_MISS

            rel_objects.append((rel_name, rel_obj))

        # 4. All caches hit -- merge into the session identity map.
        # Merge related objects first.
        merged_rels: list[tuple[str, Any]] = []
        for rel_name, rel_obj in rel_objects:
            if rel_obj is not None:
                make_transient_to_detached(rel_obj)
                rel_obj = await session.merge(rel_obj, load=False)
            merged_rels.append((rel_name, rel_obj))

        # Then merge the main object.
        make_transient_to_detached(main_obj)
        main_obj = await session.merge(main_obj, load=False)

        # Install the relationship attributes without tracking them as changes.
        for rel_name, rel_obj in merged_rels:
            set_committed_value(main_obj, rel_name, rel_obj)

        return main_obj

    @classmethod
    async def _write_load_result_to_id_caches(
            cls,
            result: Any,
            rel_info: 'list[tuple[str, type[CachedTableBaseMixin], str]]',
    ) -> None:
        """After a DB query with ``load``, write the main model and each
        MANYTOONE relationship model into their own ID caches.

        Main models and related models are cached independently (each has
        its own ID cache key and TTL), and are invalidated independently
        (each ``save``/``update``/``delete`` naturally invalidates its own cache).
        """
        if result is None:
            return

        items = [result] if not isinstance(result, list) else result

        for item in items:
            if item is None:
                continue

            # Write the main model to cls's ID cache.
            item_id = getattr(item, 'id', None)
            if item_id is not None:
                try:
                    cache_key = cls._build_id_cache_key(item_id)
                    serialized = cls._serialize_result(item)
                    await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
                except Exception as e:
                    logger.error(f"main-model cache write failed ({cls.__name__}:{item_id}): {e}")

            # Write each relationship model to its target class's ID cache.
            for rel_name, target_cls, _fk_attr in rel_info:
                rel_obj = getattr(item, rel_name, None)
                if rel_obj is None:
                    continue
                rel_id = getattr(rel_obj, 'id', None)
                if rel_id is None:
                    continue
                actual_rel_cls = type(rel_obj)
                if not issubclass(actual_rel_cls, CachedTableBaseMixin):
                    continue
                try:
                    cache_key = target_cls._build_id_cache_key(rel_id)
                    serialized = target_cls._serialize_result(rel_obj)
                    await target_cls._cache_set(cache_key, serialized, target_cls.__cache_ttl__)
                except Exception as e:
                    logger.error(f"relationship-model cache write failed ({target_cls.__name__}:{rel_id}): {e}")

    # ================================================================
    #  Cache key construction
    # ================================================================

    @classmethod
    def _build_cache_key(
            cls,
            condition: Any,
            fetch_mode: str,
            offset: int | None,
            limit: int | None,
            order_by: Any,
            load: Any,
            filter_expr: Any,
            table_view: Any,
            *time_args: Any,
            version: int = 0,
    ) -> str:
        """Build a deterministic cache key from the query parameters (versioned).

        ``table_view`` is first merged into the explicit parameters (mirroring
        the merge logic in ``table.py``) so that semantically equivalent queries
        produce the same key regardless of whether they were passed via
        ``table_view`` or as explicit kwargs.

        ``time_args`` order: ``created_before``, ``created_after``,
        ``updated_before``, ``updated_after``.

        ``version``: query cache version, bumped by each call to
        ``_invalidate_query_caches()``.

        Format: ``{_CACHE_KEY_PREFIX}:{ModelName}:v{version}:{md5_hash[:_CACHE_KEY_HASH_LENGTH]}``.
        """
        try:
            from sqlalchemy.dialects import postgresql
            _dialect = postgresql.dialect()
        except Exception:
            _dialect = None

        def _compile_sql(expr: Any) -> str:
            """Compile SQL expression to string, with dialect fallback."""
            if _dialect is not None:
                try:
                    return str(expr.compile(dialect=_dialect, compile_kwargs={"literal_binds": True}))
                except Exception:
                    pass
            try:
                return str(expr.compile(compile_kwargs={"literal_binds": True}))
            except Exception:
                return repr(expr)

        # ---- Normalization: merge table_view into explicit kwargs (mirrors
        #      the merge logic in table.py). ----
        # time_args: [created_before, created_after, updated_before, updated_after]
        merged_times = list(time_args)
        if table_view is not None:
            # Time fields: explicit args win; pull from table_view only when None.
            tv_times = [
                table_view.created_before_datetime,
                table_view.created_after_datetime,
                table_view.updated_before_datetime,
                table_view.updated_after_datetime,
            ]
            for i in range(min(len(merged_times), 4)):
                if merged_times[i] is None:
                    merged_times[i] = tv_times[i]
            # Pagination: explicit args win.
            if offset is None:
                offset = table_view.offset
            if limit is None:
                limit = table_view.limit
            # Ordering: explicit order_by wins; otherwise use the table_view description.
            if order_by is None:
                parts_order = f"ob={table_view.order},{'d' if table_view.desc else 'a'}"
            else:
                parts_order = None
        else:
            parts_order = None

        parts: list[str] = [fetch_mode]

        # condition → SQL string
        if condition is None:
            parts.append("none")
        elif isinstance(condition, bool):
            parts.append(str(condition))
        else:
            parts.append(_compile_sql(condition))

        # pagination (normalized)
        if offset is not None:
            parts.append(f"o={offset}")
        if limit is not None:
            parts.append(f"l={limit}")

        # ordering (normalized)
        if order_by:
            for ob in order_by:
                parts.append(_compile_sql(ob))
        elif parts_order is not None:
            parts.append(parts_order)

        # load args (affect returned data content)
        if load is not None:
            load_list = load if isinstance(load, list) else [load]
            parts.append("load=" + ",".join(str(item.key) for item in load_list))

        # filter (bool values also need inclusion: False = WHERE FALSE, True = no condition)
        if filter_expr is not None:
            if isinstance(filter_expr, bool):
                parts.append(f"f={filter_expr}")
            else:
                try:
                    parts.append("f=" + _compile_sql(filter_expr))
                except Exception:
                    parts.append("f=" + repr(filter_expr))

        # Time filters (normalized: table_view time fields have been merged).
        for i, ta in enumerate(merged_times):
            if ta is not None:
                parts.append(f"t{i}={ta.isoformat()}")

        key_hash = hashlib.md5("|".join(parts).encode()).hexdigest()[:cls._CACHE_KEY_HASH_LENGTH]
        return f"{cls._CACHE_KEY_PREFIX}:{cls.__name__}:v{version}:{key_hash}"

    # ================================================================
    #  Serialization / deserialization
    # ================================================================

    @classmethod
    def _serialize_item(cls, item: Any) -> bytes:
        """Serialize a single ``SQLModelBase`` instance to bytes, including its
        concrete class name.

        In polymorphic scenarios (STI) a query against the base class may
        return subclass instances. Recording the concrete class name ensures
        that deserialization restores the correct subclass.

        Only column attributes are serialized, excluding:
        - Relationship attributes (``lazy='raise_on_sql'`` would throw, and
          the ID cache only stores column data).
        - ``computed_field`` results (they may depend on relationships, e.g.
          ``ToolSet.tool_count -> self.tools``).

        With a ``load`` query the main model and each relationship model are
        serialized independently into their own ID caches.
        """
        if isinstance(item, SQLModelBase):
            mapper = sa_inspect(type(item))
            column_fields: set[str] = {prop.key for prop in mapper.column_attrs}
            item_dict: dict[str, Any] = item.model_dump(mode='json', include=column_fields)
            item_dict[_WRAPPER_CLASS_KEY] = type(item).__name__
            return _json_dumps(item_dict)
        return _json_dumps(item)

    @classmethod
    def _serialize_result(cls, result: Any) -> bytes:
        """Serialize a ``get()`` query result to bytes for Redis storage.

        Uses the configured JSON serializer plus a wrapper format that
        distinguishes None / single / list. Each ``SQLModelBase`` entry embeds
        a ``_c`` field with the concrete class name (polymorphic safe).
        """
        if result is None:
            return _json_dumps({_WRAPPER_TYPE_KEY: _CacheResultType.NONE})
        elif isinstance(result, list):
            serialized_items = [cls._serialize_item(item).decode('utf-8') for item in result]
            items_json = "[" + ",".join(serialized_items) + "]"
            wrapper = (
                f'{{"{_WRAPPER_TYPE_KEY}":"{_CacheResultType.LIST}"'
                f',"{_WRAPPER_ITEMS_KEY}":{items_json}}}'
            )
            return wrapper.encode('utf-8')
        else:
            data_json = cls._serialize_item(result).decode('utf-8')
            wrapper = (
                f'{{"{_WRAPPER_TYPE_KEY}":"{_CacheResultType.SINGLE}"'
                f',"{_WRAPPER_DATA_KEY}":{data_json}}}'
            )
            return wrapper.encode('utf-8')

    @classmethod
    def _resolve_subclass(cls, class_name: str | None) -> type:
        """Resolve a concrete subclass by its class name (for polymorphic deserialization).

        Lookups are cached in ``_subclass_name_cache`` so that after the
        initial recursive walk the resolution is O(1). The cache key is
        prefixed with the lookup-origin class name so different inheritance
        trees do not collide.
        """
        if class_name is None or cls.__name__ == class_name:
            return cls

        lookup_key = f"{cls.__name__}.{class_name}"
        cached = CachedTableBaseMixin._subclass_name_cache.get(lookup_key)
        if cached is not None:
            return cached

        def _walk(klass: type) -> type | None:
            for sub in klass.__subclasses__():
                if sub.__name__ == class_name:
                    return sub
                found = _walk(sub)
                if found is not None:
                    return found
            return None

        resolved = _walk(cls) or cls
        CachedTableBaseMixin._subclass_name_cache[lookup_key] = resolved
        return resolved

    @classmethod
    def _deserialize_item(cls, item_data: dict[str, Any]) -> Any:
        """Rebuild a single model instance from a cached dict.

        Reads the ``_c`` field to determine the concrete subclass (polymorphic
        safe), pops it, and then calls ``model_validate`` on the remaining data.
        """
        class_name = item_data.pop(_WRAPPER_CLASS_KEY, None)
        actual_cls = cls._resolve_subclass(class_name)
        return actual_cls.model_validate(item_data)

    @classmethod
    def _deserialize_result(cls, raw: bytes, _fetch_mode: str) -> Any:
        """Rebuild a ``get()`` query result from cached bytes.

        Uses ``json.loads`` followed by ``model_validate`` (not
        ``model_validate_json``) because ``model_validate_json`` returns
        ``str`` for UUID fields on ``table=True`` models.

        :raises ValidationError: when the cached schema no longer matches.
        :raises JSONDecodeError: invalid JSON payload.
        """
        cached = _json_loads(raw)
        result_type = cached.get(_WRAPPER_TYPE_KEY)
        if result_type == _CacheResultType.NONE:
            return None
        elif result_type == _CacheResultType.LIST:
            return [cls._deserialize_item(item) for item in cached[_WRAPPER_ITEMS_KEY]]
        elif result_type == _CacheResultType.SINGLE:
            return cls._deserialize_item(cached[_WRAPPER_DATA_KEY])
        raise ValueError(
            f"unknown cached result type: {result_type!r} (expected "
            f"{_CacheResultType.NONE!r}/{_CacheResultType.LIST!r}/{_CacheResultType.SINGLE!r})"
        )

    # ================================================================
    #  Invalidation
    # ================================================================

    @classmethod
    def _cached_ancestors(cls) -> list[type['CachedTableBaseMixin']]:
        """Collect the cached ancestors (``CachedTableBaseMixin`` subclasses)
        in the MRO, excluding ``cls`` itself (STI inheritance chain)."""
        return [
            ancestor
            for ancestor in cls.__mro__
            if (
                ancestor is not cls
                and ancestor is not object
                and ancestor is not CachedTableBaseMixin
                and isinstance(ancestor, type)
                and issubclass(ancestor, CachedTableBaseMixin)
            )
        ]

    @classmethod
    async def _invalidate_id_cache(cls, instance_id: Any) -> None:
        """Row-level DEL: a single multi-key DELETE that drops the ID cache
        for ``cls`` and each cached ancestor.

        Redis ``DELETE`` natively accepts multiple keys (one command, one RTT,
        atomic execution).
        """
        prefix = cls._ID_CACHE_KEY_PREFIX
        keys = [f"{prefix}:{cls.__name__}:{instance_id}"]
        for ancestor in cls._cached_ancestors():
            keys.append(f"{prefix}:{ancestor.__name__}:{instance_id}")
        await cls._get_client().delete(*keys)

    @classmethod
    async def _invalidate_query_caches(cls) -> None:
        """O(1) version bump: pipeline ``INCR`` for the class and every
        cached ancestor in a single round-trip.

        ``pipeline(transaction=False)`` batches the ``INCR`` commands into one
        RTT. Query keys at older versions expire naturally via TTL; no
        ``SCAN+DEL`` is required.
        """
        ancestors = cls._cached_ancestors()
        if not ancestors:
            # No ancestors: issue a single INCR to avoid pipeline overhead.
            await cls._bump_query_version()
            return
        client = cls._get_client()
        async with client.pipeline(transaction=False) as pipe:
            pipe.incr(cls._build_version_key())
            for ancestor in ancestors:
                pipe.incr(ancestor._build_version_key())
            await pipe.execute()

    @classmethod
    async def _invalidate_for_model(cls, _instance_id: Any | None = None) -> None:
        """Invalidate both the ID cache and the query cache version.

        Called by ``save``/``update``/``delete``.

        - With ``_instance_id``: row-level multi-key DEL of that single ID cache.
        - Without ``_instance_id``: model-level SCAN+DEL of every ID cache (rare path).
        - Query cache: always bumped via O(1) pipeline INCR.
        """
        if _instance_id is not None:
            await cls._invalidate_id_cache(_instance_id)
        else:
            # No instance_id -> clear every ID cache at the model level
            # (only triggered by condition deletes; rare path).
            id_prefix = cls._ID_CACHE_KEY_PREFIX
            await cls._cache_delete_pattern(f"{id_prefix}:{cls.__name__}:*")
            for ancestor in cls._cached_ancestors():
                await cls._cache_delete_pattern(f"{id_prefix}:{ancestor.__name__}:*")
        # Query cache is always bumped at O(1).
        await cls._invalidate_query_caches()

    @classmethod
    async def invalidate_by_id(cls, *_ids: Any) -> None:
        """Public API -- manually invalidate the cache for the given IDs.

        Intended for external callers only (admin scripts, tests, non-model code).
        Raw SQL methods inside a model should use
        ``_register_pending_invalidation()`` followed by
        ``await session.commit()`` (the enhanced AsyncSession synchronously
        invalidates registered pendings after commit)
        to avoid ``MissingGreenlet`` from post-commit attribute access.

        Each ID performs a row-level multi-key DEL on its ``id:`` cache; the
        query cache is bumped at O(1) via pipeline INCR.

        Redis errors are logged and swallowed (fire-and-forget) to prevent a
        "DB committed but API returned 500" mismatch.
        """
        try:
            for _id in _ids:
                await cls._invalidate_id_cache(_id)
            await cls._invalidate_query_caches()
        except Exception as e:
            logger.error(f"invalidate_by_id() Redis failure ({cls.__name__}, ids={_ids}): {e}")

    @classmethod
    async def invalidate_all(cls) -> None:
        """Public API -- invalidate every cache entry for this model (id: + query:).

        Redis errors are logged and swallowed (same contract as ``invalidate_by_id``).
        """
        try:
            await cls._invalidate_for_model()
        except Exception as e:
            logger.error(f"invalidate_all() Redis failure ({cls.__name__}): {e}")

    # ================================================================
    #  get() override -- cache read path
    # ================================================================

    @overload
    @classmethod
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["all"],
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> list[Self]: ...

    @overload
    @classmethod
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["one"],
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["first"] = ...,
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> Self | None: ...

    @classmethod  # @override -- runtime MRO override of TableBaseMixin.get(); pyright cannot see this statically
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["one", "first", "all"] = "first",
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> list[Self] | Self | None:
        """Cached ``get()`` -- intercepts ``TableBaseMixin.get()`` and returns
        directly on cache hit.

        - ``no_cache`` only exists on this mixin; passing it to a non-cached
          model raises ``TypeError`` (explicit failure).
        - If the transaction has uncommitted changes (``commit=False`` CRUD),
          the cache is skipped for both reads and writes.
        - Cache hits are merged into the identity map via
          ``session.merge(load=False)``.
        - When ``load`` specifies a MANYTOONE cacheable relationship, a
          multi-ID union lookup is attempted (zero SQL).
        - Full skip conditions are documented at the top of the module.
        """
        skip_cache = (
            no_cache
            or options is not None
            or with_for_update
            or populate_existing
            # TODO: future optimization -- support JOIN query caching by
            # tracking the join target's invalidations.
            or join is not None
        )

        # Uncommitted changes in the transaction -> skip cache (both read and write).
        if not skip_cache and cls._has_pending_invalidation(session):
            skip_cache = True

        # ---- load query: multi-ID cache union optimization ----
        # When the relationship is MANYTOONE and the target is cacheable,
        # assemble the result from the main-model and relationship ID caches.
        if load is not None:
            rel_info: list[tuple[str, type[CachedTableBaseMixin], str]] | None = None
            if not skip_cache and jti_subclasses is None:
                rel_info = cls._analyze_load_relations(load)
                if rel_info is not None:
                    # Detect a simple ID query; multi-ID caching only supports
                    # single-row ID equality queries.
                    id_value = cls._extract_id_from_condition(condition)
                    is_simple = (
                        id_value is not None
                        and fetch_mode in ('first', 'one')
                        and offset is None and limit is None
                        and order_by is None and filter is None
                        and table_view is None
                        and created_before_datetime is None and created_after_datetime is None
                        and updated_before_datetime is None and updated_after_datetime is None
                    )
                    if is_simple:
                        cached = await cls._try_load_from_id_caches(session, id_value, rel_info)
                        if cached is not _LOAD_CACHE_MISS:
                            if cls.on_cache_hit is not None:
                                cls.on_cache_hit(cls.__name__)
                            if cached is None and fetch_mode == 'one':
                                # ``fetch_mode='one'`` type contract guarantees raise (not None);
                                # when the cache hit is None we must raise, matching the SQL
                                # path's ``result.one()`` behavior. Otherwise downstream
                                # ``try/except NoResultFound`` callers (e.g. dep injection that
                                # maps "missing" to 503) would silently see None and fail later.
                                raise NoResultFound(
                                    f"No row was found when one was required: {cls.__name__}(id={id_value})"
                                )
                            return cached
                        # Redis was actually queried and missed.

            # Cache miss / not a simple query / not optimizable -> DB query.
            result = await super().get(
                session, condition,
                offset=offset, limit=limit, fetch_mode=fetch_mode,
                join=join, options=options, load=load,
                order_by=order_by, filter=filter,
                with_for_update=with_for_update, table_view=table_view,
                jti_subclasses=jti_subclasses,
                populate_existing=populate_existing,
                created_before_datetime=created_before_datetime,
                created_after_datetime=created_after_datetime,
                updated_before_datetime=updated_before_datetime,
                updated_after_datetime=updated_after_datetime,
            )

            # Write the main-model + relationship-model ID caches
            # (optimizable scenarios only).
            if rel_info is not None and not skip_cache:
                await cls._write_load_result_to_id_caches(result, rel_info)

            return result

        # ---- Non-load query ----
        cache_key: str | None = None
        if not skip_cache:
            # Detect a pure ID equality query (no pagination/ordering/time filters).
            id_value = cls._extract_id_from_condition(condition)
            is_simple_id_query = (
                id_value is not None
                and fetch_mode in ('first', 'one')
                and offset is None and limit is None
                and order_by is None and filter is None
                and table_view is None
                and created_before_datetime is None and created_after_datetime is None
                and updated_before_datetime is None and updated_after_datetime is None
            )

            if is_simple_id_query:
                cache_key = cls._build_id_cache_key(id_value)
            else:
                # Fetch the current version and embed it in the query key
                # (bumping the version makes older keys naturally unreachable).
                _query_version = await cls._get_query_version()
                cache_key = cls._build_cache_key(
                    condition, fetch_mode, offset, limit,
                    order_by, load, filter, table_view,
                    created_before_datetime, created_after_datetime,
                    updated_before_datetime, updated_after_datetime,
                    version=_query_version,
                )

            raw = await cls._cache_get(cache_key)
            if raw is not None:
                # Phase 1: deserialize. Failure here means the cached payload
                # is corrupt (schema change, etc.).
                try:
                    result = cls._deserialize_result(raw, fetch_mode)
                except Exception as e:
                    # Bad cache: try to delete the key and fall back to the DB
                    # (cleanup failure does not affect the normal query path).
                    logger.warning(
                        f"cache deserialization failed, dropping bad key and querying DB: "
                        f"{cache_key} ({type(e).__name__}: {e})"
                    )
                    try:
                        await cls._cache_delete(cache_key)
                    except Exception as del_err:
                        logger.error(f"bad-cache cleanup failed key='{cache_key}': {del_err}")
                    # fall through to DB query below
                else:
                    # Phase 2: merge into the session identity map to preserve
                    # the same semantics as a DB query.
                    if cls.on_cache_hit is not None:
                        cls.on_cache_hit(cls.__name__)
                    if result is None:
                        if fetch_mode == 'one':
                            # ``fetch_mode='one'`` type contract guarantees raise (not None);
                            # raise on cache hit to mirror the SQL path's ``result.one()`` behavior.
                            raise NoResultFound(
                                f"No row was found when one was required: {cls.__name__}"
                            )
                        return result
                    if isinstance(result, list):
                        merged_list: list[Self] = []
                        for item in result:
                            make_transient_to_detached(item)
                            merged_list.append(await session.merge(item, load=False))
                        return merged_list
                    else:
                        make_transient_to_detached(result)
                        result = await session.merge(result, load=False)
                    return result

        # Cache miss callback (only when Redis was actually queried, not skip_cache path)
        if cache_key is not None and cls.on_cache_miss is not None:
            cls.on_cache_miss(cls.__name__)

        # DB query via MRO super() -> TableBaseMixin.get().
        # Note: do NOT forward no_cache -- TableBaseMixin.get() does not accept it.
        result = await super().get(
            session, condition,
            offset=offset, limit=limit, fetch_mode=fetch_mode,
            join=join, options=options, load=load,
            order_by=order_by, filter=filter,
            with_for_update=with_for_update, table_view=table_view,
            jti_subclasses=jti_subclasses,
            populate_existing=populate_existing,
            created_before_datetime=created_before_datetime,
            created_after_datetime=created_after_datetime,
            updated_before_datetime=updated_before_datetime,
            updated_after_datetime=updated_after_datetime,
        )

        # Write through to the cache (unless we were told to skip).
        if not skip_cache and cache_key is not None:
            try:
                serialized = cls._serialize_result(result)
                await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
            except Exception as e:
                logger.error(f"cache serialize/write failed: {type(e).__name__}: {e}")

        return result

    # ================================================================
    #  Deferred-commit compensation (session.info tracking + after_commit event)
    # ================================================================

    @staticmethod
    def _register_pending_invalidation(
            session: AsyncSession,
            model_type: type,
            instance_id: Any | None = None,
    ) -> None:
        """Register a pending invalidation entry (``model_type`` plus an
        optional ``instance_id``) into ``session.info`` so the ``after_commit``
        compensation path can eventually run it.

        ``pending`` structure: ``dict[type, set[Any]]``

        - key: model type
        - value: set of pending ``instance_id`` values for that type

        Sentinel semantics:
        - ``_FULL_MODEL_INVALIDATION``: condition delete; model-level full
          invalidation required (highest priority, never downgraded).
        - ``_QUERY_ONLY_INVALIDATION``: ``add()`` path; only the query cache
          needs to be bumped.
        - Plain UUID/int: row-level ID invalidation.

        Recording the actual ``instance_id`` lets the ``after_commit``
        compensation path perform row-level invalidation instead of a
        model-level SCAN+DEL that would wipe unrelated rows.
        """
        pending: dict[type, set[Any]] = session.info.setdefault(_SESSION_PENDING_CACHE_KEY, {})
        ids = pending.setdefault(model_type, set())
        if instance_id is not None:
            ids.add(instance_id)

    # ================================================================
    #  save() override -- invalidate on write
    # ================================================================

    async def save(
            self,
            session: AsyncSession,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            refresh: bool = True,
            commit: bool = True,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            optimistic_retry_count: int = 0,
    ) -> Self:  # MRO override TableBaseMixin.save()
        """``save()`` invalidates the cache first, then refreshes via ``get()``
        so that ``get()`` cannot hit stale data.

        Flow (``commit=True``, ``refresh=True``):
        1. ``_register_pending_invalidation()`` -- record what must be invalidated.
        2. ``super().save(refresh=False)`` -- commit; the enhanced
           ``AsyncSession.commit()`` synchronously flushes the registered
           invalidations, ensuring the old row leaves Redis.
        3. ``get()`` refresh -- the cache is now empty, so ``get()`` hits the
           DB and repopulates the cache.

        Newly created objects (``id is None``) register
        ``_QUERY_ONLY_INVALIDATION`` (no ID cache can possibly exist) and the
        DB-generated id is fetched from ``result`` after save for sync-path marking.
        """
        model_type = type(self)
        # Newly created rows may have id=None until flush; only the query
        # cache needs invalidation in that case.
        instance_id = getattr(self, 'id', None)
        if instance_id is not None:
            self._register_pending_invalidation(session, model_type, instance_id)
        else:
            self._register_pending_invalidation(session, model_type, _QUERY_ONLY_INVALIDATION)

        # refresh=False: skip super()'s internal get(); we refresh here
        # AFTER invalidation.
        # Cache invalidation has moved up into session.commit() (the
        # sqlmodel-ext enhanced AsyncSession): after commit it synchronously
        # invalidates everything registered via _register_pending_invalidation
        # above, so this method no longer invalidates on its own.
        # With commit=False nothing is invalidated (the data is not committed
        # yet); it is deferred to the caller's session.commit().
        result = await super().save(
            session,
            refresh=False,
            commit=commit,
            optimistic_retry_count=optimistic_retry_count,
        )

        # After super().save(refresh=False) + commit the object is expired;
        # a direct getattr(result, 'id') would trigger synchronous lazy
        # loading and raise MissingGreenlet. Use sa_inspect to pull the id
        # from the identity map without issuing a DB query.
        _insp = cast(InstanceState[Any], sa_inspect(result))
        if _insp.identity:
            instance_id = _insp.identity[0]

        # Write-through refresh: bypass the cache read, hit the DB, then
        # repopulate the ID cache proactively.
        # Bypass read: avoid partial-invalidation races where an old row
        # could still be served.
        # Proactive repopulate: keep the hit rate high so the next external
        # get() lands directly on fresh data.
        # Only backfill when commit=True; commit=False rows may still be
        # rolled back and must not reach the cache.
        if refresh:
            assert instance_id is not None, f"{model_type.__name__} has id=None after save"
            result = await model_type.get(
                session, model_type.id == instance_id,
                load=load, jti_subclasses=jti_subclasses,
                no_cache=True,
            )
            assert result is not None, f"{model_type.__name__} record not found (id={instance_id})"

            # Actively repopulate the ID cache (commit=True, no load).
            # Also refill every ancestor cache key -- same rationale as the
            # identical fix in update() below.
            if commit and load is None:
                try:
                    serialized = model_type._serialize_result(result)
                    classes_to_refill = [model_type, *model_type._cached_ancestors()]
                    for refill_cls in classes_to_refill:
                        cache_key = refill_cls._build_id_cache_key(instance_id)
                        await refill_cls._cache_set(cache_key, serialized, refill_cls.__cache_ttl__)
                except Exception as e:
                    logger.error(f"save() post-commit cache backfill failed ({model_type.__name__}): {e}")

        return result  # noqa: RLC007  callers with refresh=False explicitly accept the expired object

    # ================================================================
    #  update() override -- invalidate on write
    # ================================================================

    async def update(
            self,
            session: AsyncSession,
            other: SQLModelBase,
            extra_data: dict[str, Any] | None = None,
            exclude_unset: bool = True,
            exclude: set[str] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            refresh: bool = True,
            commit: bool = True,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            optimistic_retry_count: int = 0,
    ) -> Self:  # MRO override
        """``update()`` invalidates the cache first, then refreshes via ``get()``.
        Mirrors the ``save()`` flow."""
        model_type = type(self)
        instance_id = getattr(self, 'id', None)
        self._register_pending_invalidation(session, model_type, instance_id)

        # refresh=False: skip super()'s internal get(); refresh here after
        # invalidation.
        result = await super().update(
            session, other,
            extra_data=extra_data,
            exclude_unset=exclude_unset,
            exclude=exclude,
            refresh=False,
            commit=commit,
            optimistic_retry_count=optimistic_retry_count,
        )

        # Cache invalidation has moved up into session.commit() (the
        # sqlmodel-ext enhanced AsyncSession): after commit it synchronously
        # invalidates everything registered via _register_pending_invalidation
        # above, so this method no longer invalidates on its own.

        # Write-through refresh: bypass cache read, hit the DB, then backfill.
        if refresh:
            assert instance_id is not None, f"{model_type.__name__} has id=None after update"
            result = await model_type.get(
                session, model_type.id == instance_id,
                load=load, jti_subclasses=jti_subclasses,
                no_cache=True,
            )
            assert result is not None, f"{model_type.__name__} record not found (id={instance_id})"

            # Actively repopulate the ID cache (commit=True, no load).
            # **Refill ancestor caches too**: under STI / inheritance,
            # invalidation clears the `id:<Class>:<id>` key for self *and*
            # every cached ancestor (see _invalidate_id_cache +
            # _cached_ancestors), but the historical implementation only
            # refilled self's key. Callers querying through an ancestor class
            # (e.g. a polymorphic root) would hit a MISS -> DB query ->
            # repopulate -- theoretically correct, but within that window the
            # same-session identity_map can intercept the merge and hand back
            # a stale Python instance.
            #
            # Fix: write the same fresh result to self + every ancestor cache
            # key, symmetric with _invalidate_id_cache -- refill exactly the
            # keys that were invalidated. Deserialization routes through
            # _polymorphic_name to the concrete subclass, so reading a parent
            # key still yields the correct subclass instance.
            if commit and load is None:
                try:
                    serialized = model_type._serialize_result(result)
                    classes_to_refill = [model_type, *model_type._cached_ancestors()]
                    for refill_cls in classes_to_refill:
                        cache_key = refill_cls._build_id_cache_key(instance_id)
                        await refill_cls._cache_set(cache_key, serialized, refill_cls.__cache_ttl__)
                except Exception as e:
                    logger.error(f"update() post-commit cache backfill failed ({model_type.__name__}): {e}")

        return result  # noqa: RLC007  callers with refresh=False explicitly accept the expired object

    # ================================================================
    #  delete() override -- invalidate on delete
    # ================================================================

    @classmethod  # MRO override TableBaseMixin.delete()
    async def delete(
            cls,
            session: AsyncSession,
            instances: Self | list[Self] | None = None,
            *,
            condition: ColumnElement[bool] | bool | None = None,
            commit: bool = True,
    ) -> int:
        """Invalidate the cache after ``delete()``.

        - With ``instances``: row-level DEL of each instance's ``id:`` cache
          plus a query-cache bump.
        - With ``condition`` (or no args): model-level SCAN+DEL (``id:`` + ``query:``).
        - Cascade delete (``passive_deletes=False``): SA handles children
          during flush; the ``persistent_to_deleted`` event registers
          child-model invalidations, flushed after commit by the enhanced
          ``AsyncSession`` via ``_flush_invalidations``.
        - Cascade delete (``passive_deletes=True``): SA skips loading
          children. This method pre-queries child IDs before DELETE and
          registers them as pending so the post-commit flush invalidates them.
        """
        # Extract instance IDs before super().delete() because the object
        # may become inaccessible after deletion.
        instance_ids: list[Any] = []
        if instances is not None:
            _instances = instances if isinstance(instances, list) else [instances]
            for inst in _instances:
                _id = getattr(inst, 'id', None)
                if _id is not None:
                    instance_ids.append(_id)

        # Pre-query child IDs for passive_deletes relationships (they cannot
        # be queried after the DB CASCADE delete). When passive_deletes=True,
        # SA does not load children and persistent_to_deleted does not fire,
        # so we must query child IDs ourselves before DELETE and invalidate
        # their caches afterwards.
        #
        # The walk is BFS over the entire passive_deletes chain — a single
        # level is not enough. Example: A -[passive=True]-> B -[passive=True]-> C
        # all use DB CASCADE, SA never loads B or C, persistent_to_deleted
        # only fires for A. Stale rows of both B and C linger in Redis until
        # TTL. The BFS prefetches every cached level before issuing DELETE.
        passive_targets: dict[type[CachedTableBaseMixin], list[Any]] = {}
        # Queue items: (current_cls, IDs from the previous level, or None for
        # full-model propagation when the upstream delete used a condition).
        passive_queue: list[tuple[type, list[Any] | None]] = [
            (cls, list(instance_ids) if instance_ids else None)
        ]
        while passive_queue:
            current_cls, current_ids = passive_queue.pop(0)
            try:
                current_mapper = sa_inspect(current_cls)
                if current_mapper is None:
                    continue
                for rel in current_mapper.relationships:
                    if not rel.cascade.delete or not rel.passive_deletes:
                        continue
                    target = rel.mapper.class_
                    if not (isinstance(target, type) and issubclass(target, CachedTableBaseMixin)):
                        continue
                    if current_ids is None:
                        # Condition delete or already-full propagation.
                        if _FULL_MODEL_INVALIDATION not in passive_targets.get(target, []):
                            passive_targets.setdefault(target, []).append(_FULL_MODEL_INVALIDATION)
                            passive_queue.append((target, None))
                        continue
                    remote_cols = list(rel.remote_side)
                    if len(remote_cols) != 1:
                        # Composite FK is not supported (matches the original
                        # single-level behaviour).
                        continue
                    target_mapper = sa_inspect(target)
                    if target_mapper is None:
                        continue
                    target_pk = target_mapper.primary_key[0]
                    stmt = sa_select(target_pk).where(remote_cols[0].in_(current_ids))
                    rows = await session.execute(stmt)
                    child_ids = [row[0] for row in rows]
                    if child_ids:
                        passive_targets.setdefault(target, []).extend(child_ids)
                        passive_queue.append((target, child_ids))
            except Exception as e:
                logger.warning(f"passive_deletes pre-query failed ({current_cls.__name__}): {e}")

        # Clear any leftover cascade data from a previous run (defensive;
        # normal flows never leave residue).
        session.info.pop(_SESSION_CASCADE_DELETED_KEY, None)

        # Register pending with instance_ids so the compensation path can
        # also invalidate at the row level.
        for _id in instance_ids:
            cls._register_pending_invalidation(session, cls, _id)
        if not instance_ids:
            # Condition deletes cannot extract individual IDs; use the
            # sentinel to request a model-level full invalidation.
            cls._register_pending_invalidation(session, cls, _FULL_MODEL_INVALIDATION)
        # Register passive_deletes targets in pending as a compensation safety net.
        for target_cls, child_ids in passive_targets.items():
            for child_id in child_ids:
                cls._register_pending_invalidation(session, target_cls, child_id)

        result = await super().delete(session, instances, condition=condition, commit=commit)
        # Cache invalidation has moved up into session.commit() (the
        # sqlmodel-ext enhanced AsyncSession):
        # - cls itself + the pre-queried passive_deletes=True targets were
        #   registered via _register_pending_invalidation before
        #   super().delete(), so they land in the pre-commit pending snapshot;
        # - passive_deletes=False cascade children are pushed into
        #   _SESSION_CASCADE_DELETED_KEY by the persistent_to_deleted event
        #   during super().delete()'s flush and are handled by
        #   _flush_invalidations after commit (overlap between the two
        #   sources is deduplicated by the set union).
        # This method no longer invalidates on its own.
        return result

    # ================================================================
    #  add() override -- invalidate on write
    # ================================================================

    @classmethod  # MRO override TableBaseMixin.add()
    async def add(
            cls,
            session: AsyncSession,
            instances: Self | list[Self],
            refresh: bool = True,
            commit: bool = True,
    ) -> Self | list[Self]:
        """``add()`` invalidates the cache first, then refreshes via ``get()``.

        The query cache is always bumped (list queries may need to include
        the new rows). Instances with an explicit ID also get their ``id:``
        cache invalidated to prevent stale data on ID reuse.
        """
        # Collect explicitly-provided IDs (caller-passed, not from
        # default_factory). ``model_fields_set`` only contains fields that
        # were explicitly passed to the constructor.
        items = instances if isinstance(instances, list) else [instances]
        explicit_ids: list[Any] = [
            _id for item in items
            if isinstance(item, SQLModelBase) and 'id' in item.model_fields_set
            and (_id := getattr(item, 'id', None)) is not None
        ]

        cls._register_pending_invalidation(session, cls, _QUERY_ONLY_INVALIDATION)
        for _id in explicit_ids:
            cls._register_pending_invalidation(session, cls, _id)

        # refresh=False: skip super()'s internal get(); this method handles
        # the refresh after invalidation.
        result = await super().add(session, instances, refresh=False, commit=commit)

        # Cache invalidation has moved up into session.commit() (the
        # sqlmodel-ext enhanced AsyncSession): after commit it synchronously
        # invalidates everything registered via _register_pending_invalidation
        # above, so this method no longer invalidates on its own.

        # Write-through refresh: bypass cache read, hit the DB, then
        # actively repopulate the ID cache. Only backfill when commit=True;
        # commit=False rows may still be rolled back and must not reach
        # the cache.
        if refresh:
            if isinstance(result, list):
                refreshed: list[Self] = []
                for inst in result:
                    # After commit the object is expired; use sa_inspect to
                    # read the id safely without triggering lazy loading.
                    _insp = cast(InstanceState[Any], sa_inspect(inst))
                    _inst_id = _insp.identity[0] if _insp.identity else None
                    assert _inst_id is not None, f"{cls.__name__} has id=None after add"
                    r = await cls.get(session, cls.id == _inst_id, no_cache=True)
                    assert r is not None, f"{cls.__name__} record not found (id={_inst_id})"
                    if commit:
                        try:
                            cache_key = cls._build_id_cache_key(_inst_id)
                            serialized = cls._serialize_result(r)
                            await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
                        except Exception as e:
                            logger.error(f"add() post-commit cache backfill failed ({cls.__name__}): {e}")
                    refreshed.append(r)
                return refreshed
            else:
                _insp = cast(InstanceState[Any], sa_inspect(result))
                _result_id = _insp.identity[0] if _insp.identity else None
                assert _result_id is not None, f"{cls.__name__} has id=None after add"
                r = await cls.get(session, cls.id == _result_id, no_cache=True)
                assert r is not None, f"{cls.__name__} record not found (id={_result_id})"
                if commit:
                    try:
                        cache_key = cls._build_id_cache_key(_result_id)
                        serialized = cls._serialize_result(r)
                        await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
                    except Exception as e:
                        logger.error(f"add() post-commit cache backfill failed ({cls.__name__}): {e}")
                return r

        return result

    # ================================================================
    #  Post-commit sync invalidation (invoked by _flush_invalidations)
    # ================================================================

    @classmethod
    async def _do_sync_invalidation(
            cls,
            session: AsyncSession,
            captured_ids: set[Any],
    ) -> None:
        """Internal implementation of post-commit sync invalidation.

        Checks whether ``pending`` has already been consumed by ``after_commit``
        and, if so, runs sync invalidation and marks the type as ``synced``.

        ``synced`` is marked BEFORE issuing Redis calls to eliminate the race
        with the fire-and-forget ``_compensate`` task: ``after_commit``
        schedules ``_compensate`` via ``create_task``, which may be picked up
        by the event loop while this method awaits Redis. If ``synced`` were
        marked after the Redis call, ``_compensate`` would misclassify this
        type as "did not go through the sync path" and emit a fallback
        WARNING. Marking first closes the window.

        Redis-failure correctness: marking ``synced`` first means
        ``_compensate`` will not fallback, but its fallback also calls Redis
        and would fail anyway when Redis is down. TTL provides eventual
        consistency in that case.

        :param session: the async session
        :param captured_ids: the set of pending IDs snapshotted before commit
            (may contain sentinels).
        """
        current_pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if current_pending and cls in current_pending:
            return  # pending not yet consumed (commit hasn't happened) -- skip

        sentinels = {_QUERY_ONLY_INVALIDATION, _FULL_MODEL_INVALIDATION}
        real_ids = captured_ids - sentinels
        needs_full = _FULL_MODEL_INVALIDATION in captured_ids

        # Mark synced before issuing Redis calls to prevent _compensate from
        # racing us on the next await point.
        synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
        if isinstance(synced, dict):
            synced.setdefault(cls, set()).update(captured_ids)

        try:
            if needs_full:
                await cls._invalidate_for_model()
            elif real_ids:
                for _id in real_ids:
                    await cls._invalidate_id_cache(_id)
                await cls._invalidate_query_caches()
            elif _QUERY_ONLY_INVALIDATION in captured_ids:
                await cls._invalidate_query_caches()
        except Exception as e:
            logger.error(f"sync cache invalidation failed ({cls.__name__}): {e}")

    # ================================================================
    #  Session-level invalidation helpers
    #  (delegated to by sqlmodel_ext.session.AsyncSession)
    # ================================================================

    @staticmethod
    def _capture_session_pending(session: AsyncSession) -> dict[type, set[Any]]:
        """Snapshot the pending invalidations accumulated in ``session.info`` before commit.

        The ``after_commit`` event pops the whole pending dict during commit,
        so it must be snapshotted beforehand for the post-commit sync
        invalidation. Returns an empty dict when nothing is pending.
        """
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if not pending:
            return {}
        return {k: set(v) for k, v in pending.items()}

    @staticmethod
    async def _flush_invalidations(
            session: AsyncSession,
            captured: dict[type, set[Any]],
    ) -> None:
        """Post-commit sync invalidation: ``captured`` (pre-commit pendings) + cascade children.

        Covers two invalidation sources:

        1. ``captured`` -- entries registered by CRUD methods via
           ``_register_pending_invalidation`` before commit (explicit IDs, the
           ``_QUERY_ONLY_INVALIDATION`` sentinel for new rows, the
           ``_FULL_MODEL_INVALIDATION`` sentinel for condition deletes, and
           ``delete()``'s pre-queried ``passive_deletes=True`` targets).
        2. ``_SESSION_CASCADE_DELETED_KEY`` -- ``passive_deletes=False``
           cascade children, pushed by the ``persistent_to_deleted`` event
           during ``super().commit()``'s flush (i.e. after the ``captured``
           snapshot point). ``after_commit`` does not pop this key, so it is
           drained here to keep cascades on the sync path (otherwise they
           would fall back to fire-and-forget compensation, reopening the
           stale window).

        Race with the fire-and-forget ``_compensate`` task: first mark ALL
        types as synced in one awaitless pass, then await
        ``_do_sync_invalidation`` per type -- otherwise an await point could
        let ``_compensate`` misclassify a not-yet-marked type as "did not go
        through the sync path" and emit a fallback WARNING (same rationale as
        ``_do_sync_invalidation``).

        Zero-cost return when nothing is pending and no cascade occurred
        (equivalent to a plain commit).
        """
        cascade: dict[type, set[Any]] | None = session.info.pop(_SESSION_CASCADE_DELETED_KEY, None)
        if not captured and not cascade:
            return

        # Merge both sources. captured's passive=True targets and cascade's
        # passive=False children never overlap by construction, but the set
        # union deduplicates harmlessly anyway.
        merged: dict[type, set[Any]] = {k: set(v) for k, v in captured.items()}
        if cascade:
            for mt, ids in cascade.items():
                merged.setdefault(mt, set()).update(ids)

        # Mark synced first in one awaitless pass to close the _compensate
        # race. Pending dict keys are always model types, so issubclass is
        # enough to decide whether a type is cached.
        synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
        if isinstance(synced, dict):
            for mt, ids in merged.items():
                if issubclass(mt, CachedTableBaseMixin):
                    synced.setdefault(mt, set()).update(ids)

        # Sync-invalidate every type that inherits CachedTableBaseMixin.
        for mt, ids in merged.items():
            if issubclass(mt, CachedTableBaseMixin):
                await mt._do_sync_invalidation(session, ids)

    @staticmethod
    def _clear_session_cache_state(session: AsyncSession) -> None:
        """Clear cache-invalidation tracking state from ``session.info`` (called by ``reset()``).

        The ``after_rollback`` triggered by ``reset()``'s internal rollback
        already clears these three keys; this is an idempotent second pass
        covering edges where ``after_rollback`` never fires (reset without an
        active transaction, the startup window before event registration).
        """
        session.info.pop(_SESSION_PENDING_CACHE_KEY, None)
        session.info.pop(_SESSION_SYNCED_CACHE_KEY, None)
        session.info.pop(_SESSION_CASCADE_DELETED_KEY, None)

    # ================================================================
    #  Misuse hardening
    #  (invoked by sqlmodel_ext.session.AsyncSession commit/refresh/execute)
    # ================================================================

    @staticmethod
    def _autoregister_session_mutations(session: AsyncSession) -> None:
        """Auto-register pending invalidations for every ``CachedTableBaseMixin`` instance in the session before commit.

        Covers the bare paths that bypass CRUD methods -- ``session.add(x)`` /
        direct attribute mutation / ``session.delete(x)`` followed by a plain
        ``session.commit()`` (without ``Model.save()/delete()``). Forms a set
        union with the CRUD methods' explicit registrations, so there is no
        double invalidation; deletes are additionally registered by the
        ``persistent_to_deleted`` event -- this is belt-and-suspenders.

        - new (pending INSERT; int PKs may still have id=None) -> query-cache
          invalidation only (``_QUERY_ONLY_INVALIDATION``)
        - dirty (pending UPDATE) / deleted -> row-level ID + query cache
        """
        # Iterate directly (no list() copy): _register_pending_invalidation
        # only writes session.info and never mutates the new/dirty/deleted
        # sets, so there is no mutation-during-iteration risk. dirty is
        # computed from SA's incrementally-maintained _modified set
        # (O(modified objects), not a full-table scan), so the cost scales
        # with the commit workload.
        sync = session.sync_session
        for inst in sync.new:
            if isinstance(inst, CachedTableBaseMixin):
                CachedTableBaseMixin._register_pending_invalidation(
                    session, type(inst), _QUERY_ONLY_INVALIDATION,
                )
        for inst in sync.dirty:
            if isinstance(inst, CachedTableBaseMixin):
                _id = getattr(inst, 'id', None)
                CachedTableBaseMixin._register_pending_invalidation(
                    session, type(inst), _id if _id is not None else _QUERY_ONLY_INVALIDATION,
                )
        for inst in sync.deleted:
            if isinstance(inst, CachedTableBaseMixin):
                _id = getattr(inst, 'id', None)
                if _id is not None:
                    CachedTableBaseMixin._register_pending_invalidation(session, type(inst), _id)

    @staticmethod
    async def _refresh_via_cache(session: AsyncSession, instance: Any) -> bool:
        """Cache-aware refresh: reload ``instance`` through ``Model.get()`` (Redis cache + STI polymorphic loading).

        When the instance is in the identity map, ``get()``'s internal
        ``session.merge(load=False)`` updates its column values in place --
        avoiding a bare ``session.refresh()`` that bypasses the Redis cache
        (always hits the DB) and skips STI subclass columns.

        :return: True = refreshed via the cache path; False = not possible
            (transient / no PK), caller falls back to the native refresh.
        """
        insp = cast(InstanceState[Any], sa_inspect(instance))
        if insp.identity is None:
            return False
        cls = type(instance)
        # get() with a Redis hit is O(0) DB work; a miss queries the DB and
        # backfills the cache. The return value is discarded -- merge already
        # updated the identity-map instance in place.
        await cls.get(session, col(cls.id) == insp.identity[0])
        return True

    _cached_tablename_index: ClassVar[dict[str, list[type['CachedTableBaseMixin']]] | None] = None
    """Lazily-built {SQL table name: [cached model classes mapped to it]} index for _warn_raw_dml_on_cached."""

    @classmethod
    def _build_cached_tablename_index(cls) -> dict[str, list[type['CachedTableBaseMixin']]]:
        """Build {SQL table name: [cached model classes]} (STI maps several classes to one table). Cached in a ClassVar after the first call."""
        if CachedTableBaseMixin._cached_tablename_index is None:
            index: dict[str, list[type[CachedTableBaseMixin]]] = {}
            stack: list[type] = list(CachedTableBaseMixin.__subclasses__())
            seen: set[type] = set()
            while stack:
                sub = stack.pop()
                if sub in seen:
                    continue
                seen.add(sub)
                stack.extend(sub.__subclasses__())
                tablename = getattr(sub, '__tablename__', None)
                if isinstance(tablename, str):
                    index.setdefault(tablename, []).append(sub)
            CachedTableBaseMixin._cached_tablename_index = index
        return CachedTableBaseMixin._cached_tablename_index

    @staticmethod
    def _warn_raw_dml_on_cached(session: AsyncSession, statement: Any) -> None:
        """Warn when a bare ``execute(UPDATE/DELETE)`` hits a cached table without registered invalidation.

        CRUD's ``delete(condition=...)`` also goes through ``execute(Delete)``,
        but it calls ``_register_pending_invalidation`` before executing -- so
        the warning only fires when **none** of the cached classes mapped to
        the table has a pending registration, avoiding false positives on
        legitimate CRUD paths. Non-DML statements (SELECT etc.) early-return
        with negligible overhead.
        """
        if not isinstance(statement, _DML_TYPES):
            return
        tablename = getattr(getattr(statement, 'table', None), 'name', None)
        if not isinstance(tablename, str):
            return
        classes = CachedTableBaseMixin._build_cached_tablename_index().get(tablename)
        if not classes:
            return  # not a cached table
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if pending and any(c in pending for c in classes):
            return  # registered by a CRUD path, not a bare statement
        logger.warning(
            f"raw execute({statement.__class__.__name__}) hits cached table {tablename!r} "
            f"without registered invalidation -- this bypasses cache invalidation. "
            f"Use {classes[0].__name__}.save()/delete() (auto-invalidates), or call "
            f"await {classes[0].__name__}.invalidate_by_id(...) explicitly after raw DML."
        )
