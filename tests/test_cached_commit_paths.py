"""
Post-commit cache invalidation: enhanced ``commit()`` vs. the fallback compensation.

The enhanced ``sqlmodel_ext.AsyncSession.commit()`` invalidates the committed
pendings synchronously; the ``after_commit`` event hands them over instead of
scheduling the fire-and-forget fallback. Regression for the spurious
``"fallback compensation triggered"`` WARNING (and the second invalidation
that came with it) on every commit through the enhanced session.

Observable used for "invalidated exactly once": every query-cache
invalidation ``INCR``s ``ver:{Model}`` by one.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy import event as sa_event
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession as PlainAsyncSession

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin

FALLBACK_MESSAGE = "fallback compensation triggered"


class CommitPathGadget(SQLModelBase, CachedTableBaseMixin, UUIDTableBaseMixin, table=True):
    """Cached model for the commit-path tests."""
    name: str
    quantity: int = 0


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(client)
    CachedTableBaseMixin.check_cache_config()
    try:
        yield client
    finally:
        CachedTableBaseMixin._redis_client = None
        await client.aclose()


async def _drain() -> None:
    """Let fire-and-forget compensation tasks finish."""
    for _ in range(50):
        await asyncio.sleep(0)


async def _version(client: fakeredis.aioredis.FakeRedis) -> int:
    raw = await client.get("ver:CommitPathGadget")
    return 0 if raw is None else int(raw)


def _fallback_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if FALLBACK_MESSAGE in r.getMessage()]


async def _load(s: AsyncSession, gid: Any) -> CommitPathGadget:
    gadget = await CommitPathGadget.get(s, CommitPathGadget.id == gid, no_cache=True)
    assert gadget is not None
    return gadget


async def _seed(engine: AsyncEngine, client: fakeredis.aioredis.FakeRedis) -> Any:
    """Insert one row and cache it by id; returns the id."""
    async with AsyncSession(engine) as s:
        gadget = await CommitPathGadget(name="seed", quantity=1).save(s)
        gid = gadget.id
        cached = await CommitPathGadget.get(s, CommitPathGadget.id == gid)
        assert cached is not None
    await _drain()
    assert await client.exists(f"id:CommitPathGadget:{gid}") == 1
    return gid


@pytest.mark.asyncio
async def test_enhanced_commit_invalidates_once_without_fallback_warning(
    engine: AsyncEngine,
    fake_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gid = await _seed(engine, fake_redis)
    caplog.set_level(logging.WARNING)
    before = await _version(fake_redis)

    async with AsyncSession(engine) as s:
        gadget = await _load(s, gid)
        gadget.quantity = 2                     # bare mutation
        s.add(CommitPathGadget(name="new"))     # bare insert
        await s.commit()
        await _drain()

    assert _fallback_records(caplog) == []
    assert await fake_redis.exists(f"id:CommitPathGadget:{gid}") == 0
    assert await _version(fake_redis) == before + 1, "one commit must invalidate the query cache exactly once"


@pytest.mark.asyncio
async def test_crud_commit_and_begin_block_have_no_fallback_warning(
    engine: AsyncEngine,
    fake_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gid = await _seed(engine, fake_redis)
    caplog.set_level(logging.WARNING)

    async with AsyncSession(engine) as s:
        before = await _version(fake_redis)
        _ = await CommitPathGadget(name="crud").save(s)          # commit=True CRUD
        await _drain()
        assert await _version(fake_redis) == before + 1

    async with AsyncSession(engine) as s:
        before = await _version(fake_redis)
        async with s.begin():
            gadget = await _load(s, gid)
            gadget.name = "in-begin"
        await _drain()
        assert await _version(fake_redis) == before + 1

    assert _fallback_records(caplog) == []


@pytest.mark.asyncio
async def test_pending_registered_during_the_commit_flush_is_invalidated(
    engine: AsyncEngine,
    fake_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pending registered by the commit's own flush (after any pre-commit snapshot) is not lost."""
    gid = await _seed(engine, fake_redis)
    caplog.set_level(logging.WARNING)

    async with AsyncSession(engine) as s:
        s.add(CommitPathGadget(name="triggers a flush"))

        def _register_during_flush(sync_session: Any, _ctx: Any, _instances: Any) -> None:
            # Stands in for cascade children (``persistent_to_deleted``) and
            # any other registration that happens inside super().commit().
            CachedTableBaseMixin._register_pending_invalidation(sync_session, CommitPathGadget, gid)

        sa_event.listen(s.sync_session, "before_flush", _register_during_flush)
        await s.commit()
        await _drain()

    assert await fake_redis.exists(f"id:CommitPathGadget:{gid}") == 0
    assert _fallback_records(caplog) == []


@pytest.mark.asyncio
async def test_plain_session_commit_still_falls_back_with_warning(
    engine: AsyncEngine,
    fake_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A commit that bypasses the enhanced commit() keeps the compensation + WARNING."""
    gid = await _seed(engine, fake_redis)
    caplog.set_level(logging.WARNING)
    before = await _version(fake_redis)

    async with PlainAsyncSession(engine) as s:
        gadget = await s.get(CommitPathGadget, gid)
        assert gadget is not None
        gadget.quantity = 5
        # A plain session does not autoregister; register as a CRUD method would.
        CachedTableBaseMixin._register_pending_invalidation(s, CommitPathGadget, gid)  # pyright: ignore[reportArgumentType]
        await s.commit()
        await _drain()

    records = _fallback_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "bypassed the enhanced AsyncSession.commit()" in records[0].getMessage()
    assert await fake_redis.exists(f"id:CommitPathGadget:{gid}") == 0
    assert await _version(fake_redis) == before + 1


@pytest.mark.asyncio
async def test_run_sync_commit_on_enhanced_session_falls_back(
    engine: AsyncEngine,
    fake_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Committing the sync session directly bypasses the enhanced commit() even on the enhanced class."""
    gid = await _seed(engine, fake_redis)
    caplog.set_level(logging.WARNING)

    async with AsyncSession(engine) as s:
        gadget = await _load(s, gid)
        gadget.quantity = 6
        CachedTableBaseMixin._register_pending_invalidation(s, CommitPathGadget, gid)
        await s.run_sync(lambda sync_session: sync_session.commit())
        await _drain()

    assert len(_fallback_records(caplog)) == 1
    assert await fake_redis.exists(f"id:CommitPathGadget:{gid}") == 0


@pytest.mark.asyncio
async def test_commit_error_after_database_commit_falls_back(
    engine: AsyncEngine,
    fake_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """super().commit() raising after after_commit fired: the hand-over goes to the fallback."""
    gid = await _seed(engine, fake_redis)
    caplog.set_level(logging.WARNING)

    async with AsyncSession(engine) as s:
        gadget = await _load(s, gid)
        gadget.quantity = 7

        def _fail_after_commit(_sync_session: Any) -> None:
            raise RuntimeError("boom after the database committed")

        # Registered after the library's after_commit handler, so it runs second.
        sa_event.listen(s.sync_session, "after_commit", _fail_after_commit)
        with pytest.raises(RuntimeError, match="boom after the database committed"):
            await s.commit()
        await _drain()

    records = _fallback_records(caplog)
    assert len(records) == 1
    assert "super().commit() raised after the database committed" in records[0].getMessage()
    assert await fake_redis.exists(f"id:CommitPathGadget:{gid}") == 0


@pytest.mark.asyncio
async def test_interrupted_sync_invalidation_hands_remaining_to_fallback(
    engine: AsyncEngine,
    fake_redis: fakeredis.aioredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gid = await _seed(engine, fake_redis)
    caplog.set_level(logging.WARNING)
    original = CommitPathGadget._do_sync_invalidation.__func__  # pyright: ignore[reportFunctionMemberAccess]
    calls: list[int] = []

    async def _cancel_first_call(cls: type[CommitPathGadget], captured_ids: set[Any]) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise asyncio.CancelledError
        await original(cls, captured_ids)

    monkeypatch.setattr(CommitPathGadget, "_do_sync_invalidation", classmethod(_cancel_first_call))

    async with AsyncSession(engine) as s:
        gadget = await _load(s, gid)
        gadget.quantity = 8
        with pytest.raises(asyncio.CancelledError):
            await s.commit()
        await _drain()

    records = _fallback_records(caplog)
    assert len(records) == 1
    assert "interrupted after the database committed" in records[0].getMessage()
    assert await fake_redis.exists(f"id:CommitPathGadget:{gid}") == 0
