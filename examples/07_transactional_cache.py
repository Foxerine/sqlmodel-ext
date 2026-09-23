"""
07 -- Transaction-aware two-level Redis cache (``CachedTableBaseMixin``), on fakeredis.

Run (needs ``fakeredis``, installed by the ``dev`` extra)::

    python examples/07_transactional_cache.py

* Inherit ``CachedTableBaseMixin`` and every ``get()`` is cached: equality on
  the primary key uses the ID layer (``id:{Model}:{id}``), anything else the
  query layer (``query:{Model}:v{version}:{hash}``).
* Writes through the model (``save`` / ``update`` / ``delete``) *and* bare
  ``session.add()`` / attribute changes are invalidated automatically by the
  enhanced ``sqlmodel_ext.AsyncSession`` right after ``commit()``; a rollback
  drops the pending invalidations.
* A query that depends on a table with uncommitted writes in the current
  transaction is neither read from nor written to the shared cache.
* Raw DML bypasses ORM tracking: pair it with
  ``Model.invalidate_on_commit(session, *ids)`` (or ``invalidate_all()``),
  otherwise the session logs a warning.
"""
import asyncio

import fakeredis.aioredis
from sqlalchemy import event, update
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, col

from sqlmodel_ext import (
    AsyncSession,
    CachedTableBaseMixin,
    NonNegativeInt,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
)


class Setting(SQLModelBase, CachedTableBaseMixin, UUIDTableBaseMixin, table=True, cache_ttl=600):
    key: Str64
    """Setting name."""

    value: NonNegativeInt
    """Setting value."""


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    selects: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _count_selects(_conn: object, _cursor: object, statement: str, *_args: object) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    redis = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(redis)
    CachedTableBaseMixin.check_cache_config()  # startup check; also installs session hooks
    assert Setting.__cache_ttl__ == 600

    async with AsyncSession(engine) as session:
        setting = await Setting(key="max_upload_mb", value=10).save(session)
        setting_id = setting.id

        # --- ID layer: second read is served from Redis ----------------------
        _ = await Setting.get(session, col(Setting.id) == setting_id, fetch_mode='one')
        selects.clear()
        cached = await Setting.get(session, col(Setting.id) == setting_id, fetch_mode='one')
        assert cached.value == 10 and selects == [], "pure cache hit: zero SQL"

        # --- writes invalidate after commit ----------------------------------
        cached.value = 20
        cached = await cached.save(session)
        fresh = await Setting.get(session, col(Setting.id) == setting_id, fetch_mode='one')
        assert fresh.value == 20

        # --- rollback keeps serving the committed state ----------------------
        fresh.value = 999
        session.add(fresh)
        await session.rollback()
        after_rollback = await Setting.get(session, col(Setting.id) == setting_id, fetch_mode='one')
        assert after_rollback.value == 20

        # --- raw DML: register the invalidation explicitly -------------------
        _ = await session.exec(
            update(Setting).where(col(Setting.id) == setting_id).values(value=30),
        )
        Setting.invalidate_on_commit(session, setting_id)
        await session.commit()
        after_raw = await Setting.get(session, col(Setting.id) == setting_id, fetch_mode='one')
        assert after_raw.value == 30, "no stale value after raw DML"

        # --- authorization reads bypass every cache layer --------------------
        selects.clear()
        _ = await Setting.get(session, col(Setting.id) == setting_id, fetch_mode='one', authoritative=True)
        assert selects, "authoritative=True always reads the database"

    await Setting.invalidate_all(strict=True)
    CachedTableBaseMixin.configure_redis(None)
    await redis.aclose()
    await engine.dispose()
    print("[OK] 07_transactional_cache: hits, invalidation, rollback and raw DML behave")


if __name__ == "__main__":
    asyncio.run(main())
