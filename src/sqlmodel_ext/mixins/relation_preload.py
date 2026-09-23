"""
Relation Preloading Mixin and transaction-contract decorators.

Provides method-level relationship declaration and on-demand incremental loading,
preventing MissingGreenlet errors while maintaining optimal SQL query count.

Design principles:
- On-demand: Only loads relationships needed by the called method
- Incremental: Already-loaded relationships are not re-loaded
- Optimal: Same relationship queried only once, different relationships loaded incrementally
- Zero-invasive: Callers need no changes
- Commit-safe: Uses SQLAlchemy inspect to detect real loading state

Usage::

    from sqlmodel_ext.mixins import RelationPreloadMixin
    from sqlmodel_ext.mixins.relation_preload import requires_relations

    class MyFunction(RelationPreloadMixin, Function, table=True):
        generator: Generator = Relationship(...)

        @requires_relations('generator', Generator.config)
        async def cost(self, params, context, session) -> int:
            return self.generator.config.price  # auto-loaded

Supports AsyncGenerator::

    @requires_relations('twitter_api')
    async def _call(self, ...) -> AsyncGenerator[ToolResponse, None]:
        yield ToolResponse(...)  # decorator handles async generators correctly

Transaction-contract decorators (all **fail-closed** -- a guard that cannot
locate the session raises instead of silently passing):

- :func:`requires_for_update` -- ``self`` must have been loaded with
  ``get(with_for_update=True)``.
- :func:`requires_locked_param` / :func:`validate_locked_instances` -- the
  instances in a parameter must be FOR UPDATE locked.
- :func:`requires_repeatable_read` -- the session must be verified to run in
  REPEATABLE READ.
- :func:`requires_read_committed` -- the transaction must run in READ
  COMMITTED (live ``SHOW`` check, PostgreSQL only).
"""
import inspect as python_inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Mapping, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast as typing_cast

from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.orm import QueryableAttribute, RelationshipProperty
from sqlalchemy.orm.attributes import set_committed_value
from sqlmodel import col
from sqlmodel.ext.asyncio.session import AsyncSession

from .table import SESSION_FOR_UPDATE_KEY, SESSION_REPEATABLE_READ_KEY

logger = logging.getLogger(__name__)


def _extract_session(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AsyncSession | None:
    """
    Extract AsyncSession from method parameters.

    Search order:
    1. kwargs named 'session'
    2. Positional arg at 'session' parameter position
    3. kwargs with AsyncSession type

    ``args`` must exclude ``self`` (instance-method heuristic).
    """
    if 'session' in kwargs:
        return kwargs['session']

    try:
        sig = python_inspect.signature(func)
        param_names = list(sig.parameters.keys())

        if 'session' in param_names:
            idx = param_names.index('session') - 1  # subtract self
            if 0 <= idx < len(args):
                return args[idx]
    except (ValueError, TypeError):
        pass

    for value in kwargs.values():
        if isinstance(value, AsyncSession):
            return value

    return None


def _bind_session(
    signature: python_inspect.Signature,
    context: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AsyncSession:
    """Locate the ``session`` argument by binding the **full** call arguments to the signature (fail-closed).

    Unlike :func:`_extract_session` (which assumes ``args[0]`` is ``self``
    and was stripped by the caller), ``Signature.bind`` lets Python do the
    binding, so instance methods, classmethods, plain functions, keyword
    arguments and defaults are all handled correctly.

    :raises TypeError: the arguments do not bind to the signature
    :raises RuntimeError: no ``AsyncSession`` found in the ``session`` parameter
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as e:
        raise TypeError(f"{context}: argument binding failed -- {e}") from e
    bound.apply_defaults()
    session = bound.arguments.get('session')
    if not isinstance(session, AsyncSession):
        raise RuntimeError(
            f"{context}: cannot locate an AsyncSession in the call arguments "
            f"(got {type(session).__name__}); the isolation-level guard cannot verify. "
            "Make sure the method has a `session` parameter and that every inner "
            "decorator uses @wraps (inspect.signature follows __wrapped__)."
        )
    return session


def _is_obj_relation_loaded(obj: Any, rel_name: str) -> bool:
    """
    Check if an object's relationship is loaded.

    :param obj: The object to check
    :param rel_name: Relationship attribute name
    :returns: True if loaded, False if unloaded or expired
    """
    try:
        state = sa_inspect(obj)
        return rel_name not in state.unloaded
    except Exception:
        return False


def _find_relation_to_class(from_class: type, to_class: type) -> str | None:
    """
    Find a relationship attribute name pointing from one class to another.

    :param from_class: Source class
    :param to_class: Target class
    :returns: Relationship attribute name, or None
    """
    for attr_name in dir(from_class):
        try:
            attr = getattr(from_class, attr_name, None)
            if attr is None:
                continue
            if hasattr(attr, 'property') and hasattr(attr.property, 'mapper'):
                target_class = attr.property.mapper.class_
                if target_class == to_class:
                    return attr_name
        except AttributeError:
            continue
    return None


def requires_relations(
    *relations: str | QueryableAttribute[Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator declaring method's required relationships with auto-loading.

    Parameter formats:
    - String: attribute name on this class (e.g. ``'generator'``)
    - QueryableAttribute: external class attribute (e.g. ``Generator.config``)

    Behavior:
    - Checks if relationships are loaded before method execution
    - Unloaded relationships are incrementally loaded (single query)
    - Already-loaded relationships are skipped
    - If no session can be found in the call arguments, loading is skipped;
      callers invoking such methods must call
      :meth:`RelationPreloadMixin.ensure_relations_loaded` themselves.

    Supports both regular async methods and async generators.

    Example::

        @requires_relations('generator', Generator.config)
        async def cost(self, params, context, session) -> int:
            return self.generator.config.price
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        is_async_gen = python_inspect.isasyncgenfunction(func)

        if is_async_gen:
            @wraps(func)
            async def gen_wrapper(self: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
                session = _extract_session(func, args, kwargs)
                if session is not None:
                    await self.ensure_relations_loaded(session, relations)
                async for item in func(self, *args, **kwargs):
                    yield item
            setattr(gen_wrapper, '_required_relations', relations)
            return gen_wrapper  # type: ignore[return-value]
        else:
            @wraps(func)
            async def func_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                session = _extract_session(func, args, kwargs)
                if session is not None:
                    await self.ensure_relations_loaded(session, relations)
                return await func(self, *args, **kwargs)
            setattr(func_wrapper, '_required_relations', relations)
            return func_wrapper

    return decorator


_P = ParamSpec('_P')
_R = TypeVar('_R')


def requires_for_update(
        func: Callable[_P, Awaitable[_R]],
) -> Callable[_P, Coroutine[Any, Any, _R]]:
    """
    Decorator declaring that self must be obtained via FOR UPDATE.

    At runtime, checks ``session.info`` for lock records before method execution.
    If self was not obtained via ``Model.get(with_for_update=True)``,
    raises RuntimeError immediately.

    **fail-closed**: if the session cannot be extracted from the call
    arguments the decorator raises instead of skipping the check. A failed
    extraction is exactly the signal that the guard stopped working (e.g. an
    inner decorator without ``@wraps`` hides the ``session`` parameter from
    ``inspect.signature``) and must not look like "guard passed".

    **Signature preserving**: typed with ``ParamSpec``, so type checkers keep
    checking the decorated method's parameters (a ``Callable[..., Any]``
    signature would erase them).

    Static analysis: sets ``_requires_for_update = True`` metadata on the wrapper,
    enabling lint tools (e.g. relation_load_checker) to verify locking at call sites.

    Example::

        @requires_for_update
        async def adjust_balance(self, session: AsyncSession, *, amount: int) -> None:
            ...

        # Caller must lock first
        user = await User.get(session, User.id == uid, with_for_update=True)
        await user.adjust_balance(session, amount=-100)
    """
    @wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # With ParamSpec, self is args[0] (this decorator is for instance methods).
        self = args[0]
        session = _extract_session(func, args[1:], kwargs)
        cls_name = type(self).__name__
        if session is None:
            raise RuntimeError(
                f"{cls_name}.{func.__name__}() cannot extract an AsyncSession from its arguments; "
                "the row-lock guard cannot verify. Make sure the method has a `session` parameter "
                "and that every inner decorator uses @wraps (inspect.signature follows __wrapped__)."
            )
        locked: set[int] = session.info.get(SESSION_FOR_UPDATE_KEY, set())
        if id(self) not in locked:
            raise RuntimeError(
                f"{cls_name}.{func.__name__}() requires a FOR UPDATE locked instance. "
                f"Call {cls_name}.get(session, ..., with_for_update=True) first."
            )
        return await func(*args, **kwargs)

    setattr(wrapper, '_requires_for_update', True)
    return wrapper


def validate_locked_instances(
        session: AsyncSession,
        instances: Sequence[Any],
        *,
        context: str,
) -> None:
    """Check that every instance is in the session's FOR UPDATE lock tracking set; raise otherwise.

    The validation body of :func:`requires_locked_param`, exported for
    mid-flight re-validation (a decorator can only check arguments at method
    entry, not e.g. instances returned by a callback).

    An empty sequence also raises: claiming "these instances are locked"
    while passing none is a contract violation (fail-closed).

    :param context: prefix for the error message (who requires the lock)
    :raises RuntimeError: empty sequence, or an instance was not locked
    """
    if not instances:
        raise RuntimeError(f"{context}: requires at least one FOR UPDATE locked instance (got none)")
    locked: set[int] = session.info.get(SESSION_FOR_UPDATE_KEY, set())
    for inst in instances:
        if id(inst) not in locked:
            cls_name = type(inst).__name__
            raise RuntimeError(
                f"{context}: {cls_name} instance is not FOR UPDATE locked -- call "
                f"{cls_name}.get(session, ..., with_for_update=True) first and hold the "
                "lock until the transaction commits"
            )


def requires_locked_param(
        param_name: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator factory: the instance(s) in parameter ``param_name`` must be FOR UPDATE locked.

    The parametrized sibling of :func:`requires_for_update` (which can only
    express "self is locked"): for classmethods, ``@asynccontextmanager``
    factories and other callables that operate on a *batch* of instances.

    Semantics (fail-closed): a ``None`` argument skips the check (e.g. a
    ``condition``-based code path); a single instance is normalized to a
    one-element sequence; an empty sequence raises. The session is always
    taken from the decorated callable's ``session`` parameter -- a callable
    without one raises.

    Two decorated shapes are supported: coroutine functions (``async def``)
    and synchronous factories (e.g. ``@asynccontextmanager`` products -- the
    check then runs when the context manager is created, i.e. before entry).
    """
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = python_inspect.signature(func)
        context = f"{func.__qualname__} contract on {param_name!r}"

        def _validate(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError as e:
                raise TypeError(f"{context}: argument binding failed -- {e}") from e
            bound.apply_defaults()
            if 'session' not in bound.arguments:
                raise RuntimeError(
                    f"{context}: the decorated callable has no `session` parameter, "
                    "the row-lock guard cannot verify (fail-closed)"
                )
            target = bound.arguments.get(param_name)
            if target is None:
                return
            instances = list(target) if isinstance(target, (list, tuple)) else [target]
            validate_locked_instances(
                bound.arguments['session'], instances, context=context,
            )

        if python_inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
                _validate(args, kwargs)
                return await typing_cast('Callable[_P, Awaitable[Any]]', func)(*args, **kwargs)
            setattr(async_wrapper, '_requires_locked_param', param_name)
            return typing_cast(Callable[_P, _R], async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            _validate(args, kwargs)
            return func(*args, **kwargs)
        setattr(sync_wrapper, '_requires_locked_param', param_name)
        return sync_wrapper

    return decorator


def requires_repeatable_read(
        func: Callable[_P, Awaitable[_R]],
) -> Callable[_P, Coroutine[Any, Any, _R]]:
    """
    Decorator declaring that the method must run in a **REPEATABLE READ** transaction.

    For orchestration that reads the same source data across several
    statements (e.g. a deep copy reading nodes, then files, then edges):
    under READ COMMITTED every statement takes a new snapshot, and a
    concurrent committed write can make the reads inconsistent ("torn").

    Same contract family as :func:`requires_for_update` and equally
    fail-closed: no locatable session, or no REPEATABLE READ marker in
    ``session.info``, raises ``RuntimeError``. The marker is written by
    :meth:`sqlmodel_ext.AsyncSession.enter_repeatable_read` only after the
    level was **read back from the database**, and cleared by ``reset()`` /
    ``close()`` (the connection returns to the pool and is reset), so its
    presence means the session really is in REPEATABLE READ.

    No built-in retry: a serialization failure (SQLSTATE ``40001``) requires
    discarding the whole session, which the decorated method cannot do from
    inside it. Retry ownership belongs to the outermost entry point
    (:meth:`sqlmodel_ext.SessionFactory.run_in_repeatable_read`).

    The session is located via ``Signature.bind``, so instance methods,
    classmethods and plain functions are all supported.

    Example::

        @requires_repeatable_read
        async def clone(self, session: AsyncSession, ...) -> ...:
            ...  # every read sees the same transaction snapshot

        # outermost entry point:
        return await session_factory.run_in_repeatable_read(_clone, description="deep copy")
    """
    signature = python_inspect.signature(func)
    context = f"{func.__qualname__}()"

    @wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        session = _bind_session(signature, context, args, kwargs)
        if not session.info.get(SESSION_REPEATABLE_READ_KEY, False):
            raise RuntimeError(
                f"{context} requires a REPEATABLE READ transaction (reads across statements "
                "must see one snapshot). Orchestrate this call from the outermost entry point "
                "via session_factory.run_in_repeatable_read(...)."
            )
        return await func(*args, **kwargs)

    setattr(wrapper, '_requires_repeatable_read', True)
    return wrapper


def requires_read_committed(
        func: Callable[_P, Awaitable[_R]],
) -> Callable[_P, Coroutine[Any, Any, _R]]:
    """
    Decorator declaring that the method must run in a **READ COMMITTED** transaction (fail-closed, live check).

    For cross-process coordination that relies on "every statement takes a
    new snapshot" to see rows committed concurrently (e.g. re-checking a
    hand-off window). REPEATABLE READ / SERIALIZABLE use a transaction-wide
    snapshot and would not see those commits.

    Unlike :func:`requires_repeatable_read` this deliberately does **not**
    consult the ``session.info`` marker: "no RR marker" also covers
    SERIALIZABLE and an isolation level set directly on the connection or
    engine -- trusting the marker would fail open. It reads the level back
    with ``SHOW transaction_isolation`` on every call (one cheap round trip).
    **PostgreSQL only** (``SHOW`` is PostgreSQL syntax).

    The session is located via ``Signature.bind``, so instance methods,
    classmethods and plain functions are all supported.

    Example::

        @requires_read_committed
        async def claim(self, session: AsyncSession, ...) -> ...:
            ...  # each statement sees the latest committed state

        @classmethod
        @requires_read_committed          # classmethods work too (decorator innermost)
        async def load_ready(cls, session: AsyncSession, ...) -> ...:
            ...
    """
    signature = python_inspect.signature(func)
    context = f"{func.__qualname__}()"

    @wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        session = _bind_session(signature, context, args, kwargs)
        connection = await session.connection()
        isolation = await connection.scalar(text('SHOW transaction_isolation'))
        if isolation != 'read committed':
            raise RuntimeError(
                f"{context} requires a READ COMMITTED transaction (a per-statement snapshot is "
                f"needed to see concurrent commits); actual isolation level = {isolation!r}. "
                "Do not call it inside run_in_repeatable_read or any transaction whose isolation "
                "level was changed (session / connection / engine)."
            )
        return await func(*args, **kwargs)

    setattr(wrapper, '_requires_read_committed', True)
    return wrapper


class RelationPreloadMixin:
    """
    Relation Preloading Mixin.

    Provides on-demand incremental loading to ensure optimal SQL query count.

    Features:
    - On-demand: Only loads relationships needed by the called method
    - Incremental: Already-loaded relationships are not re-loaded
    - In-place update: Modifies self directly, no instance replacement
    - Import-time validation: String relationship names verified at class creation
    - Commit-safe: Uses SQLAlchemy inspect for real state detection
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate all @requires_relations declarations at class creation time.

        Only SQLModel classes are validated: relationships can only be
        declared on SQLModel classes, so a plain Python mixin cannot own the
        declared relations itself -- its ``@requires_relations`` methods are
        validated when a concrete SQLModel class mixes it in (``dir(cls)``
        includes inherited methods).
        """
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '__sqlmodel_relationships__'):
            return

        all_annotations: set[str] = set()
        for klass in cls.__mro__:
            if hasattr(klass, '__annotations__'):
                all_annotations.update(klass.__annotations__.keys())

        sqlmodel_relationships: set[str] = set()
        for klass in cls.__mro__:
            if hasattr(klass, '__sqlmodel_relationships__'):
                sqlmodel_relationships.update(klass.__sqlmodel_relationships__.keys())

        all_available_names = all_annotations | sqlmodel_relationships

        for method_name in dir(cls):
            if method_name.startswith('__'):
                continue

            try:
                method = getattr(cls, method_name, None)
            except AttributeError:
                continue

            if method is None or not hasattr(method, '_required_relations'):
                continue

            for spec in method._required_relations:
                if isinstance(spec, str):
                    if spec not in all_available_names and not hasattr(cls, spec):
                        raise AttributeError(
                            f"{cls.__name__}.{method_name} declares relation '{spec}', "
                            f"but {cls.__name__} has no such attribute"
                        )

    def _is_relation_loaded(self, rel_name: str) -> bool:
        """
        Check if a relationship is truly loaded (via SQLAlchemy inspect).

        Handles commit-induced expiration automatically.

        :param rel_name: Relationship attribute name
        :returns: True if loaded, False if unloaded or expired
        """
        try:
            state = sa_inspect(self)
            if state is None:
                return False
            return rel_name not in state.unloaded
        except Exception:
            return False

    async def ensure_relations_loaded(
        self,
        session: AsyncSession,
        relations: tuple[str | QueryableAttribute[Any], ...],
    ) -> None:
        """
        Ensure specified relationships are loaded, incrementally loading missing ones.

        Public entry point: besides the automatic call from the
        ``@requires_relations`` wrapper, orchestration code must call it
        explicitly before invoking methods whose signature has no session
        (the wrapper then cannot find one and skips preloading -- correctness
        must not depend on that branch).

        :param session: Database session
        :param relations: Required relationship specs
        """
        to_load: list[str | QueryableAttribute[Any]] = []
        direct_keys: set[str] = set()
        nested_parent_keys: set[str] = set()

        for rel in relations:
            if isinstance(rel, str):
                if not self._is_relation_loaded(rel):
                    to_load.append(rel)
                    direct_keys.add(rel)
            else:
                parent_class: type = rel.property.parent.class_
                parent_attr = _find_relation_to_class(self.__class__, parent_class)

                if parent_attr is None:
                    logger.warning(
                        f"Cannot find relationship path from {self.__class__.__name__} "
                        f"to {parent_class.__name__}, cannot check if {rel.key} is loaded"
                    )
                    to_load.append(rel)
                    continue

                if not self._is_relation_loaded(parent_attr):
                    if parent_attr not in direct_keys and parent_attr not in nested_parent_keys:
                        to_load.append(parent_attr)
                        nested_parent_keys.add(parent_attr)
                    to_load.append(rel)
                else:
                    parent_obj = getattr(self, parent_attr)
                    if not _is_obj_relation_loaded(parent_obj, rel.key):
                        if parent_attr not in direct_keys and parent_attr not in nested_parent_keys:
                            to_load.append(parent_attr)
                            nested_parent_keys.add(parent_attr)
                        to_load.append(rel)

        if not to_load:
            return

        load_options = self._specs_to_load_options(to_load)
        if not load_options:
            return

        state = sa_inspect(self)
        if state is None or state.key is None:
            logger.warning(f"Cannot get primary key for {self.__class__.__name__}")
            return
        pk_tuple = state.key[1]
        pk_value = pk_tuple[0]

        # RelationPreloadMixin must be combined with TableBaseMixin (id / get);
        # the type system cannot express that constraint.
        cls: Any = self.__class__
        fresh = await cls.get(
            session,
            cls.id == pk_value,
            load=load_options,
        )

        if fresh is None:
            logger.warning(f"Cannot load relations: {self.__class__.__name__} id={pk_value} not found")
            return

        all_direct_keys = direct_keys | nested_parent_keys
        for key in all_direct_keys:
            value = getattr(fresh, key, None)
            object.__setattr__(self, key, value)

    @classmethod
    def _specs_to_load_options(
        cls,
        specs: list[str | QueryableAttribute[Any]],
    ) -> list[QueryableAttribute[Any]]:
        """
        Convert relationship specs to load parameters.

        - String -> ``cls.{name}``
        - QueryableAttribute -> used directly
        """
        result: list[QueryableAttribute[Any]] = []

        for spec in specs:
            if isinstance(spec, str):
                rel = getattr(cls, spec, None)
                if rel is not None:
                    result.append(rel)
                else:
                    logger.warning(f"Relation '{spec}' not found on {cls.__name__}")
            else:
                result.append(spec)

        return result

    @classmethod
    def bulk_preload_unsupported_reason(
        cls,
        spec: 'str | QueryableAttribute[Any]',
    ) -> str | None:
        """Decide whether a relationship spec is outside the batch capability of :meth:`ensure_relations_loaded_bulk`.

        Returns ``None`` when supported (a direct string spec naming a
        many-to-one relationship on a single-column FK to the target's ``id``
        with a plain FK-equality ``primaryjoin``); otherwise a human-readable
        reason. This predicate is the **single source of truth** for the
        batch boundary: ``ensure_relations_loaded_bulk`` uses it to filter
        (unsupported specs are left to the per-instance fallback), and
        startup checks can use it to fail loudly on declarations that cannot
        be batched.
        """
        if not isinstance(spec, str):
            return "nested QueryableAttribute spec (batching only supports direct relationship names)"
        if not hasattr(cls, spec):
            return f"relationship '{spec}' does not exist on {cls.__name__}"
        mapped_property = getattr(cls, spec).property
        # A mapped attribute that is not a relationship (plain column, ...):
        # ColumnProperty has no uselist/secondary -- check the type first so
        # the predicate never raises.
        if not isinstance(mapped_property, RelationshipProperty):
            return (
                f"attribute '{spec}' is not a relationship "
                f"({type(mapped_property).__name__}; batching only supports relationships)"
            )
        relationship_property = mapped_property
        if relationship_property.uselist:
            return f"collection relationship '{spec}' (one-to-many)"
        if relationship_property.secondary is not None:
            return f"many-to-many relationship '{spec}'"
        pairs = relationship_property.local_remote_pairs
        if pairs is None or len(pairs) != 1:
            return f"composite-FK or unresolved relationship '{spec}'"
        local_col, remote_col = pairs[0]
        if remote_col.key != 'id':
            return f"relationship '{spec}' does not reference the primary key"
        # The primaryjoin must be a plain FK equality: extra predicates (e.g.
        # ``AND target.is_enabled``) filter rows in the ORM loader, while the
        # manual batch assembly queries by id only and would bypass them --
        # and once assembled the relationship counts as loaded, so the
        # per-instance fallback would not correct it. compare() is structural
        # and asymmetric, so both directions are tried.
        primaryjoin = relationship_property.primaryjoin
        if not (
            primaryjoin.compare(local_col == remote_col)
            or primaryjoin.compare(remote_col == local_col)
        ):
            return f"relationship '{spec}' has a primaryjoin with predicates beyond the FK equality"
        return None

    @classmethod
    async def ensure_relations_loaded_bulk(
        cls,
        session: AsyncSession,
        instances: Sequence['RelationPreloadMixin'],
        specs_by_class: Mapping[type, tuple[str | QueryableAttribute[Any], ...]],
    ) -> None:
        """Batch-preload relationships for a (possibly heterogeneous) collection of instances.

        Query count = **0-1 owner refresh query + 1 query per distinct target
        inheritance-tree root**. The query that produced the owner instances
        is not repeated; only owners whose required FK columns are unloaded /
        expired get one ``id IN (...)`` refresh. For fresh instances the whole
        chain is therefore ``1 + R`` where R is the number of distinct target
        roots (relationships whose targets belong to the same STI/JTI tree are
        merged into one polymorphic ``IN`` on the root).

        1. **Owner refresh (on demand)**: one ``id IN`` for instances whose FK
           columns cannot be read safely.
        2. **Per target root**: collect target ids from the owners' FK
           columns, query the root once, and attach each target with
           ``set_committed_value`` (SQLAlchemy's public manual-preload API --
           afterwards ``inspect().unloaded`` no longer contains the
           relationship, same as a loader).

        Only many-to-one single-column-FK relationships are batched (see
        :meth:`bulk_preload_unsupported_reason`); everything else is left to
        the per-instance fallback. **A non-NULL FK whose target row is missing
        is deliberately not assembled** -- the relationship stays unloaded so
        a ``lazy='raise_on_sql'`` access fails loudly instead of pretending
        "no relationship".

        Callers should still call ``ensure_relations_loaded`` per instance
        afterwards (a no-op fast path when loaded) -- **this method is a
        query-count optimization, not a source of correctness**.

        :param cls: any class of the tree (only a call anchor; the owner
            refresh queries the mapper's base class)
        :param session: Database session (instances should belong to its identity map)
        :param instances: Instances to preload (may be heterogeneous;
            pending / transient / already loaded ones are skipped)
        :param specs_by_class: Relationship specs per concrete class (same
            format as ``ensure_relations_loaded``)
        """
        # Only persistent instances with at least one unloaded batchable
        # relationship. Primary keys come from the identity key -- never via
        # ``.id``: after a commit expired the object, attribute access would
        # trigger a synchronous lazy refresh (MissingGreenlet).
        pending: list[Any] = []
        for instance in instances:
            specs = specs_by_class.get(type(instance))
            if specs is None:
                continue
            supported_specs = [
                spec for spec in specs
                if isinstance(spec, str)
                and type(instance).bulk_preload_unsupported_reason(spec) is None
            ]
            if not supported_specs:
                continue
            if all(instance._is_relation_loaded(spec) for spec in supported_specs):
                continue
            state = sa_inspect(instance)
            if state is None or state.key is None:
                continue  # pending / transient: no persistent identity, left to the fallback
            pending.append(instance)
        if not pending:
            return

        # Collect the batchable distinct relationship properties.
        properties: dict[Any, list[Any]] = {}  # property -> owner instances
        for instance in pending:
            for spec in specs_by_class[type(instance)]:
                if not isinstance(spec, str):
                    continue
                if type(instance).bulk_preload_unsupported_reason(spec) is not None:
                    continue
                prop = getattr(type(instance), spec).property
                properties.setdefault(prop, []).append(instance)
        if not properties:
            return

        # Owner refresh -- only for owners whose needed FK columns are
        # unloaded / expired (fresh instances cost zero extra queries).
        needed_fk_keys_by_class: dict[type, set[str]] = {}
        for prop, owners in properties.items():
            fk_col_key = prop.local_remote_pairs[0][0].key
            for owner in owners:
                needed_fk_keys_by_class.setdefault(type(owner), set()).add(fk_col_key)
        stale_ids: list[Any] = []
        for instance in pending:
            state = sa_inspect(instance)
            if state is None:
                continue
            needed_keys = needed_fk_keys_by_class.get(type(instance))
            if needed_keys and needed_keys & set(state.unloaded):
                stale_ids.append(state.key[1][0])
        if stale_ids:
            root_cls: Any = typing_cast(Any, sa_inspect(cls)).mapper.base_mapper.class_
            _ = await root_cls.get(
                session,
                col(root_cls.id).in_(stale_ids),
                fetch_mode='all',
            )

        # Group by target inheritance-tree root: targets of different
        # properties that share one STI/JTI tree are loaded by a single
        # polymorphic IN on the root.
        root_batches: dict[Any, tuple[set[Any], list[tuple[Any, str, Any]]]] = {}
        for prop, owners in properties.items():
            local_col, _remote_col = prop.local_remote_pairs[0]  # remote is the PK (guaranteed by the predicate)
            target_root = prop.mapper.base_mapper.class_
            ids, assignments = root_batches.setdefault(target_root, (set(), []))
            for owner in owners:
                fk = getattr(owner, local_col.key)
                assignments.append((owner, prop.key, fk))
                if fk is not None:
                    ids.add(fk)

        for target_root, (target_ids, assignments) in root_batches.items():
            targets_by_id: dict[Any, Any] = {}
            if target_ids:
                any_root: Any = target_root
                targets = await any_root.get(
                    session,
                    col(any_root.id).in_(target_ids),
                    fetch_mode='all',
                )
                targets_by_id = {target.id: target for target in targets}
            for owner, relation_key, fk in assignments:
                if fk is None:
                    set_committed_value(owner, relation_key, None)
                elif fk in targets_by_id:
                    set_committed_value(owner, relation_key, targets_by_id[fk])
                # Non-NULL FK but missing target: deliberately not assembled.

    # ==================== Optional manual preload API ====================

    @classmethod
    def get_relations_for_method(cls, method_name: str) -> list[QueryableAttribute[Any]]:
        """
        Get relationships declared by a specific method.

        :param method_name: Method name
        :returns: List of QueryableAttribute
        """
        method = getattr(cls, method_name, None)
        if method is None or not hasattr(method, '_required_relations'):
            return []

        result: list[QueryableAttribute[Any]] = []
        for spec in method._required_relations:
            if isinstance(spec, str):
                rel = getattr(cls, spec, None)
                if rel:
                    result.append(rel)
            else:
                result.append(spec)

        return result

    @classmethod
    def get_relations_for_methods(cls, *method_names: str) -> list[QueryableAttribute[Any]]:
        """
        Get deduplicated relationships for multiple methods.

        :param method_names: Method names
        :returns: Deduplicated list of QueryableAttribute
        """
        seen: set[str] = set()
        result: list[QueryableAttribute[Any]] = []

        for method_name in method_names:
            for rel in cls.get_relations_for_method(method_name):
                key = rel.key
                if key not in seen:
                    seen.add(key)
                    result.append(rel)

        return result

    async def preload_for(self, session: AsyncSession, *method_names: str) -> 'RelationPreloadMixin':
        """
        Manually preload relationships for specified methods.

        Usually not needed -- the decorator handles this automatically.

        :param session: Database session
        :param method_names: Method names whose relationships to preload
        :returns: self (supports chaining)
        """
        all_relations: list[str | QueryableAttribute[Any]] = []

        for method_name in method_names:
            method = getattr(self.__class__, method_name, None)
            if method and hasattr(method, '_required_relations'):
                all_relations.extend(method._required_relations)

        if all_relations:
            await self.ensure_relations_loaded(session, tuple(all_relations))

        return self
