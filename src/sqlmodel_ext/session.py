"""sqlmodel-ext enhanced ``AsyncSession`` -- the library's canonical session type.

A subclass of sqlmodel's ``AsyncSession`` that upgrades cache correctness from
"documented convention enforced by review" to "adapted automatically at
runtime", closing four footguns:

- ``commit()``: auto-registers every ``CachedTableBaseMixin`` mutation in the
  session before commit (new/dirty/deleted -- including bare ``session.add()``
  / attribute mutation / ``session.delete()`` paths that never went through the
  CRUD methods), then synchronously invalidates after commit -- eliminating
  the "forgot the cache-aware commit -> stale cache / fire-and-forget window"
  class of bugs.
- ``reset()``: releases the connection + clears FOR UPDATE lock tracking +
  cache-invalidation tracking state.
- ``refresh()``: whole-object refresh of a cached model automatically goes
  through ``Model.get()`` (Redis cache hit + STI polymorphic subclass column
  loading), avoiding a bare ``session.refresh()`` that bypasses the cache and
  skips subclass columns.
- ``execute()``: WARNs when a bare ``UPDATE``/``DELETE`` hits a cached table
  without a registered invalidation (a known gap that bypasses cache
  invalidation).

Invalidation/refresh logic belongs to ``CachedTableBaseMixin`` (delegated via
its protected helpers); this class stays thin and only orchestrates ordering,
leaving room for more framework capabilities to grow on the session type.

Wiring: point every session factory at this class --
``async_sessionmaker(class_=AsyncSession)``. Business code then simply
``await session.commit()`` / ``await session.reset()`` /
``await session.refresh(obj)`` and gets cache-aware behavior.

Models that do not inherit ``CachedTableBaseMixin`` are unaffected: every hook
degrades to the upstream behavior when no cached model is involved.
"""
from typing import Any
from collections.abc import Iterable

from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSessionBase

from sqlmodel_ext.mixins.cached_table import CachedTableBaseMixin
from sqlmodel_ext.mixins.table import SESSION_FOR_UPDATE_KEY


class AsyncSession(_AsyncSessionBase):
    """Cache-aware ``AsyncSession``: commit/reset/refresh/execute auto-adapt to the Redis cache.

    All cache logic is delegated to ``CachedTableBaseMixin``'s protected
    helpers (``_xxx``) -- they are the framework-internal contract between
    this wrapper and cached_table; application code must not call them
    directly, hence the explicit reportPrivateUsage waivers below.
    """

    async def commit(self) -> None:
        """Commit + synchronously invalidate the Redis cache of every ``CachedTableBaseMixin`` model involved.

        Ordering: auto-register new/dirty/deleted cached models (covers bare
        add/mutate/delete + commit paths) -> snapshot the pendings (the
        ``after_commit`` event pops them during commit) -> ``super().commit()``
        actually commits (flush may append cascade children via the
        ``persistent_to_deleted`` event) -> synchronously invalidate the
        snapshot + cascade children after commit. Degrades to a plain commit
        when nothing is pending.
        """
        CachedTableBaseMixin._autoregister_session_mutations(self)  # pyright: ignore[reportPrivateUsage]
        captured = CachedTableBaseMixin._capture_session_pending(self)  # pyright: ignore[reportPrivateUsage]
        await super().commit()
        await CachedTableBaseMixin._flush_invalidations(self, captured)  # pyright: ignore[reportPrivateUsage]

    async def reset(self) -> None:
        """Reset the session (release transaction/connection) + clear tracking state in ``session.info``.

        Clears ``SESSION_FOR_UPDATE_KEY`` (no event clears it, must be popped
        manually) + the three cache-invalidation tracking keys (the
        ``after_rollback`` fired by ``reset()``'s internal rollback already
        cleared them; this is an idempotent second pass).
        """
        await super().reset()
        self.info.pop(SESSION_FOR_UPDATE_KEY, None)
        CachedTableBaseMixin._clear_session_cache_state(self)  # pyright: ignore[reportPrivateUsage]

    async def refresh(
            self,
            instance: object,
            attribute_names: Iterable[str] | None = None,
            with_for_update: Any = None,
    ) -> None:
        """Refresh an ORM instance. Whole-object refresh of cached models goes through ``Model.get()`` (cache + STI polymorphism).

        ``attribute_names`` (partial refresh) / ``with_for_update`` (row lock)
        / non-cached models -> native ``session.refresh()`` (semantics the
        cache layer cannot express equivalently).
        """
        if (
            attribute_names is None
            and with_for_update is None
            and isinstance(instance, CachedTableBaseMixin)
            and await CachedTableBaseMixin._refresh_via_cache(self, instance)  # pyright: ignore[reportPrivateUsage]
        ):
            return
        await super().refresh(instance, attribute_names=attribute_names, with_for_update=with_for_update)

    async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]  # wide passthrough signature, only to insert the warning before DML
        """Pass ``execute`` through, warning on bare ``UPDATE``/``DELETE`` that bypass cache invalidation."""
        CachedTableBaseMixin._warn_raw_dml_on_cached(self, statement)  # pyright: ignore[reportPrivateUsage]
        return await super().execute(statement, *args, **kwargs)
