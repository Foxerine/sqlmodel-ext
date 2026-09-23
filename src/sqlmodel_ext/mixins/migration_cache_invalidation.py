"""
Migration-driven precise cache invalidation.

Schema migrations do not clear stale values in the Redis cache. After a
migration an old cache entry may:

- **fail to deserialize** -> ``CachedTableBaseMixin.get()`` self-heals (drops
  the key and reads the database);
- **deserialize fine but carry wrong values** (e.g. a numeric column was
  rescaled; the old cached value still validates) -> self-healing cannot
  help and wrong data is served.

This module handles the second case: a migration explicitly declares "I
changed the meaning of existing data", and at startup the affected
``CachedTableBaseMixin`` namespaces are invalidated -- without a blunt
``FLUSHDB`` that would also wipe unrelated keys (sessions, rate limits, ...).

Declarations (decentralized)
============================

Each migration that needs cache invalidation declares a module-level constant
in **its own migration file**; this module only discovers, deduplicates and
executes them::

    # at the top of alembic/versions/<revision>.py
    CACHE_INVALIDATIONS: dict[str, list[str]] = {
        '<topic>-v<N>': ['User', 'Transaction', ...],
    }

Without Alembic (or for tests), pass the tasks directly via ``tasks=``.

Mechanism: discover -> sentinel -> idempotent execution
=======================================================

Call ``run_pending_migration_cache_invalidations()`` once at startup, after
migrations ran and ``CachedTableBaseMixin.configure_redis()`` was called:

1. Collect every migration's ``CACHE_INVALIDATIONS`` declaration (walks the
   Alembic script directory; requires the optional ``alembic`` package).
2. For each task check the Redis sentinel ``cache_invalidation:<task>``.
3. **Missing** -> first run: ``Model.invalidate_all(strict=True)`` for each
   listed model, then write the sentinel.
4. **Present** -> already done, skip (idempotent).

The ``cache_invalidation:*`` namespace is never touched by
``invalidate_all()`` (its prefix is not ``id:`` / ``query:``), so sentinels
persist; after a manual ``FLUSHDB`` they are gone together with the stale
cache and re-running the invalidations is harmless.

Concurrent startup of several workers: the first one runs the invalidation
and writes the sentinel, later ones skip. In the worst case two workers both
run ``invalidate_all()`` -- an idempotent version bump + delete.

When a migration must declare an invalidation
=============================================

| Change | Declare? | Why |
|--------|----------|-----|
| ``ADD COLUMN`` | no | old cache entries validate with the field default |
| ``DROP COLUMN`` | no* | old entries carry an extra key; ``extra='forbid'`` makes their deserialization fail, and a failed cache read falls back to the DB and self-heals |
| change column type, same meaning | no | coerced by Pydantic, or deserialization fails and self-heals |
| change column type **and rescale / re-encode values** | **yes** | old values validate but are wrong |
| drop a column + add a column **backfilled with non-default values** | **yes** | old entries get the default, not the backfill |
| new STI subtype + data migration | maybe | depends on whether old discriminator values are still valid |
| add / rename index, add / drop trigger | no | row data in the cache is unaffected |
| ``UPDATE`` existing rows | **yes** | existing values changed |

\\* except the drop + backfilled-add case above.

Rule of thumb: "an old cache entry deserializes successfully but its meaning
changed" must be declared; "fails to deserialize" or "meaning unchanged" is
left to self-healing.

Declaration rules
=================

1. Sentinel names ``<topic>-v<N>``: ``v<N>`` keeps a future invalidation of the
   same topic from colliding with the old sentinel.
2. List the **concrete subclasses that actually write cache entries**:
   ``invalidate_all()`` only walks **up** the MRO (itself + cached
   ancestors), so a parent cannot clear its subclasses' namespaces (cache
   keys look like ``id:<queried class __name__>:<pk>``).
3. Use Python class names (``__name__``), not SQL table names.
"""
import logging

from sqlmodel_ext.mixins.cached_table import CachedTableBaseMixin

try:
    from alembic.config import Config as _AlembicConfig
    from alembic.script import ScriptDirectory as _AlembicScriptDirectory
except ImportError:
    _AlembicConfig = None
    _AlembicScriptDirectory = None

logger = logging.getLogger(__name__)

_SENTINEL_PREFIX = 'cache_invalidation'
"""Sentinel key prefix. The ``cache_invalidation:*`` namespace is disjoint from ``id:*`` / ``query:*``."""

DEFAULT_ALEMBIC_INI = 'alembic.ini'
"""Default Alembic configuration path (relative to the working directory)."""


def collect_migration_invalidation_tasks(alembic_ini: str = DEFAULT_ALEMBIC_INI) -> dict[str, list[str]]:
    """Walk the Alembic script directory and merge every migration's ``CACHE_INVALIDATIONS`` declaration.

    Best-effort: a failure to load the Alembic configuration or to import a
    migration module is logged and skipped -- it must not block startup.

    :param alembic_ini: path to ``alembic.ini``
    :returns: sentinel -> list of model class names (merged over all migrations)
    :raises RuntimeError: the optional ``alembic`` package is not installed
    """
    if _AlembicConfig is None or _AlembicScriptDirectory is None:
        raise RuntimeError(
            "collect_migration_invalidation_tasks() requires the 'alembic' package; "
            + "install it or pass the tasks to run_pending_migration_cache_invalidations(tasks=...)"
        )
    tasks: dict[str, list[str]] = {}
    try:
        script_dir = _AlembicScriptDirectory.from_config(_AlembicConfig(alembic_ini))
    except Exception as e:
        logger.warning(f"loading the Alembic script directory failed, skipping migration cache invalidation discovery: {e}")
        return tasks

    for script in script_dir.walk_revisions():
        try:
            module = script.module
        except Exception as e:
            logger.warning(f"importing migration {script.revision} failed, skipping its cache invalidation declaration: {e}")
            continue
        decl = getattr(module, 'CACHE_INVALIDATIONS', None)
        if not decl:
            continue
        if not isinstance(decl, dict):
            logger.warning(f"CACHE_INVALIDATIONS of migration {script.revision} is not a dict, ignored")
            continue
        for sentinel, model_names in decl.items():
            if sentinel in tasks and tasks[sentinel] != list(model_names):
                logger.warning(
                    f"cache invalidation sentinel {sentinel} declared twice with different content "
                    + f"(migration {script.revision} overrides the earlier declaration)"
                )
            tasks[sentinel] = list(model_names)
    return tasks


async def run_pending_migration_cache_invalidations(
        base_class: type,
        *,
        tasks: dict[str, list[str]] | None = None,
        alembic_ini: str = DEFAULT_ALEMBIC_INI,
) -> None:
    """Run every migration cache invalidation task that has not run yet (idempotent, sentinel-guarded).

    Call once at startup after migrations and after
    ``CachedTableBaseMixin.configure_redis()``. Already processed tasks
    (sentinel present) are skipped.

    Redis being unavailable is logged and startup continues (best-effort; the
    cache layer cannot serve anything without Redis either). A task whose
    invalidation fails for any model does **not** get its sentinel written,
    so it is retried on the next startup (``invalidate_all(strict=True)``
    surfaces the failure). One failed task never blocks the others. A model
    name that no longer exists counts as "nothing to invalidate", not as a
    failure (otherwise the task would retry forever).

    :param base_class: root class whose ``__subclasses__()`` tree contains the
        cached models (typically ``SQLModelBase``); used to resolve class names
    :param tasks: explicit ``sentinel -> [model class name, ...]`` tasks;
        ``None`` (default) collects them from Alembic migrations
        (:func:`collect_migration_invalidation_tasks`, requires ``alembic``)
    :param alembic_ini: path to ``alembic.ini`` when collecting from Alembic
    """
    invalidation_tasks = tasks if tasks is not None else collect_migration_invalidation_tasks(alembic_ini)
    if not invalidation_tasks:
        logger.debug("no migration declared cache invalidation tasks, skipping")
        return

    name_to_cls: dict[str, type[CachedTableBaseMixin]] = {}

    def visit(cls: type) -> None:
        if issubclass(cls, CachedTableBaseMixin):
            name_to_cls[cls.__name__] = cls
        for sub in cls.__subclasses__():
            visit(sub)

    visit(base_class)

    try:
        cache_client = CachedTableBaseMixin._get_client()  # pyright: ignore[reportPrivateUsage]
    except Exception as e:
        logger.warning(f"Redis client is not configured, skipping migration cache invalidation: {e}")
        return

    for sentinel, table_names in invalidation_tasks.items():
        sentinel_key = f"{_SENTINEL_PREFIX}:{sentinel}"
        try:
            already_done = await cache_client.exists(sentinel_key)
        except Exception as e:
            logger.warning(f"checking sentinel {sentinel_key} failed, skipping: {e}")
            continue
        if already_done:
            logger.debug(f"migration cache invalidation task {sentinel} already done, skipping")
            continue

        logger.warning(
            f"running migration cache invalidation task {sentinel} for the first time "
            + f"({len(table_names)} model(s))..."
        )
        cleared = 0
        failed = 0
        for table_name in table_names:
            cls = name_to_cls.get(table_name)
            if cls is None:
                logger.warning(f"  [{sentinel}] model {table_name} not found (possibly removed), skipping")
                continue
            try:
                # strict=True: the default swallows Redis errors, which would
                # make this handler dead code and write the sentinel anyway.
                await cls.invalidate_all(strict=True)
                cleared += 1
            except Exception:
                failed += 1
                logger.exception(f"  [{sentinel}] invalidating {table_name} failed")

        if failed > 0:
            logger.error(
                f"migration cache invalidation task {sentinel}: {failed} model(s) failed; the sentinel "
                + f"is NOT written and the task will be retried on the next startup "
                + f"(cleared={cleared}/{len(table_names)})"
            )
            continue

        try:
            # The sentinel never expires: a new invalidation needs a new sentinel name.
            await cache_client.set(sentinel_key, "1")
            logger.info(
                f"migration cache invalidation task {sentinel} done ({cleared}/{len(table_names)} model(s))"
            )
        except Exception:
            logger.exception(f"writing sentinel {sentinel_key} failed, the task will be retried on the next startup")
