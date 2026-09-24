"""Independent final-review probes for the 0.5.1 release."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, PaginationRequest, SQLModelBase, UUIDTableBaseMixin


class Final051CachedRow(SQLModelBase, CachedTableBaseMixin, UUIDTableBaseMixin, table=True):
    """Cached row used to exercise savepoint hand-over."""

    value: int


@pytest_asyncio.fixture
async def final_051_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(client)
    CachedTableBaseMixin.check_cache_config()
    try:
        yield client
    finally:
        CachedTableBaseMixin._redis_client = None
        await client.aclose()


@pytest.mark.asyncio
async def test_savepoint_release_defers_invalidation_until_outer_commit(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with AsyncSession(engine) as session:
        row = await Final051CachedRow(value=1).save(session)
        row_id = row.id
        assert await Final051CachedRow.get(session, Final051CachedRow.id == row_id) is not None

    version_key = "ver:Final051CachedRow"
    before_raw = await final_051_redis.get(version_key)
    before = 0 if before_raw is None else int(before_raw)
    caplog.set_level(logging.WARNING)

    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id, no_cache=True)
        assert row is not None
        async with session.begin_nested():
            row.value = 2
            await session.flush()
        assert await final_051_redis.exists(f"id:Final051CachedRow:{row_id}") == 1
        await session.commit()

    for _ in range(20):
        await asyncio.sleep(0)
    after_raw = await final_051_redis.get(version_key)
    after = 0 if after_raw is None else int(after_raw)
    assert await final_051_redis.exists(f"id:Final051CachedRow:{row_id}") == 0
    assert after == before + 1
    assert not [record for record in caplog.records if "fallback compensation triggered" in record.getMessage()]


def test_invalid_preceding_fields_do_not_hide_their_own_errors() -> None:
    with pytest.raises(Exception) as exc_info:
        PaginationRequest(offset=-1, order="not-a-sort", after_id=uuid.uuid4())
    errors = exc_info.value.errors()
    assert {error["loc"] for error in errors} == {("offset",), ("order",)}


# ---------------------------------------------------------------------------
# B1: a bare mutation flushed BEFORE the enhanced commit (savepoint flush,
# manual flush, autoflush) must still be invalidated by the outer commit.
# ---------------------------------------------------------------------------


async def _seed_cached_row(engine: AsyncEngine, value: int = 1) -> uuid.UUID:
    """Insert a row and warm its ID cache; return the id."""
    async with AsyncSession(engine) as session:
        row = await Final051CachedRow(value=value).save(session)
        row_id = row.id
        assert await Final051CachedRow.get(session, Final051CachedRow.id == row_id) is not None
    return row_id


async def _version(redis: fakeredis.aioredis.FakeRedis) -> int:
    raw = await redis.get("ver:Final051CachedRow")
    return 0 if raw is None else int(raw)


async def _id_cached(redis: fakeredis.aioredis.FakeRedis, row_id: uuid.UUID) -> bool:
    return await redis.exists(f"id:Final051CachedRow:{row_id}") == 1


async def _cached_value(engine: AsyncEngine, row_id: uuid.UUID) -> int:
    """What a fresh reader gets through the cache."""
    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id)
        assert row is not None
        return row.value


@pytest.mark.asyncio
async def test_manual_flush_then_commit_invalidates(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    row_id = await _seed_cached_row(engine)
    before = await _version(final_051_redis)
    caplog.set_level(logging.WARNING)

    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id, no_cache=True)
        assert row is not None
        row.value = 2
        await session.flush()
        assert not session.dirty
        await session.commit()

    assert not await _id_cached(final_051_redis, row_id)
    assert await _version(final_051_redis) == before + 1
    assert not [r for r in caplog.records if "fallback compensation triggered" in r.getMessage()]


@pytest.mark.asyncio
async def test_autoflush_then_commit_invalidates(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    row_id = await _seed_cached_row(engine)
    before = await _version(final_051_redis)

    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id, no_cache=True)
        assert row is not None
        row.value = 2
        # A DB-backed query autoflushes the pending UPDATE.
        _ = (await session.exec(select(Final051CachedRow))).all()
        assert not session.dirty
        await session.commit()

    assert not await _id_cached(final_051_redis, row_id)
    assert await _version(final_051_redis) == before + 1


@pytest.mark.asyncio
async def test_flushed_bare_add_invalidates_query_cache(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    _ = await _seed_cached_row(engine)
    async with AsyncSession(engine) as session:
        assert len(await Final051CachedRow.get(session, fetch_mode="all")) == 1
    before = await _version(final_051_redis)

    async with AsyncSession(engine) as session:
        session.add(Final051CachedRow(value=9))
        await session.flush()
        assert not session.new
        await session.commit()

    assert await _version(final_051_redis) == before + 1
    async with AsyncSession(engine) as session:
        assert len(await Final051CachedRow.get(session, fetch_mode="all")) == 2


@pytest.mark.asyncio
async def test_flushed_bare_delete_invalidates(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    row_id = await _seed_cached_row(engine)
    before = await _version(final_051_redis)

    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id, no_cache=True)
        assert row is not None
        await session.delete(row)
        await session.flush()
        assert not session.deleted
        await session.commit()

    assert not await _id_cached(final_051_redis, row_id)
    assert await _version(final_051_redis) == before + 1


@pytest.mark.asyncio
async def test_savepoint_rollback_does_not_drop_outer_invalidation(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """An outer flushed change survives a later savepoint rollback and is invalidated."""
    row_id = await _seed_cached_row(engine)
    other_id = await _seed_cached_row(engine, value=10)
    before = await _version(final_051_redis)

    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id, no_cache=True)
        other = await Final051CachedRow.get(session, Final051CachedRow.id == other_id, no_cache=True)
        assert row is not None and other is not None
        row.value = 2
        await session.flush()
        nested = await session.begin_nested()
        other.value = 20
        await session.flush()
        await nested.rollback()
        await session.commit()

    assert not await _id_cached(final_051_redis, row_id)
    assert await _version(final_051_redis) == before + 1
    # The rolled-back row may be invalidated once more (safe direction); what
    # matters is that readers see the committed state.
    assert await _cached_value(engine, row_id) == 2
    assert await _cached_value(engine, other_id) == 10


@pytest.mark.asyncio
async def test_savepoint_only_rollback_keeps_cache_correct(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,  # noqa: ARG001  -- configures the cache
) -> None:
    row_id = await _seed_cached_row(engine)

    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id, no_cache=True)
        assert row is not None
        nested = await session.begin_nested()
        row.value = 2
        await session.flush()
        await nested.rollback()
        await session.commit()

    assert await _cached_value(engine, row_id) == 1


@pytest.mark.asyncio
async def test_outermost_rollback_drops_flushed_pendings(
    engine: AsyncEngine,
    final_051_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    row_id = await _seed_cached_row(engine)
    before = await _version(final_051_redis)

    async with AsyncSession(engine) as session:
        row = await Final051CachedRow.get(session, Final051CachedRow.id == row_id, no_cache=True)
        assert row is not None
        row.value = 2
        await session.flush()
        await session.rollback()
        assert "_pending_cache_invalidation_types" not in session.info
        await session.commit()

    assert await _id_cached(final_051_redis, row_id)
    assert await _version(final_051_redis) == before
    assert await _cached_value(engine, row_id) == 1
