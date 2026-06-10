"""
CachedTableBaseMixin Redis cache behavior, exercised against fakeredis.

Covered behaviors:
- ID-layer cache: first get-by-id hits the DB, second is served from Redis
  (asserted via SQL statement counting + ``id:{Model}:{id}`` key presence).
- Query-layer cache: identical condition queries reuse the cached result;
  writes bump ``ver:{Model}`` so stale query keys become unreachable.
- Write invalidation: save/update/delete drop the ID cache and bump the
  query-cache version; save/update backfill the ID cache with fresh data.
- commit/rollback consistency with the enhanced (cache-aware)
  ``sqlmodel_ext.AsyncSession``: bare mutations auto-invalidate on commit,
  rollback drops pending invalidations and keeps the cache serving the last
  committed state; ``commit=False`` writes skip the cache entirely.
- Serialization roundtrip (orjson): datetime / UUID fields survive
  cache write + read in a brand-new session with zero SQL.
- Degradation: unconfigured Redis fails fast with RuntimeError on cached
  reads; transient Redis errors fall back to the database.

Redis injection point: ``CachedTableBaseMixin.configure_redis(client)``
(class-level ``_redis_client`` ClassVar shared by all cached models).
The ``after_commit``/``after_rollback`` session hooks required by the sync
invalidation path are installed once via ``check_cache_config()``.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import datetime

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy import event as sa_event
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin
from sqlmodel_ext.mixins.cached_table import _SESSION_PENDING_CACHE_KEY


# ---------------------------------------------------------------------------
# Models (module scope, globally-unique names per repo convention)
# ---------------------------------------------------------------------------

class CacheGadget(SQLModelBase, CachedTableBaseMixin, UUIDTableBaseMixin, table=True):
    """Plain cached model used by most cache behavior tests."""
    name: str
    quantity: int = 0


class CacheEvent(SQLModelBase, CachedTableBaseMixin, UUIDTableBaseMixin, table=True):
    """Cached model with datetime + UUID payload for serialization tests."""
    title: str
    happened_at: datetime
    ref_id: uuid.UUID


class CacheGadgetPatch(SQLModelBase):
    """Non-table DTO used to drive ``update()``."""
    quantity: int


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """Fresh FakeRedis per test, injected via configure_redis().

    Also runs check_cache_config() once so the after_commit/after_rollback
    session hooks (required by the sync invalidation path) are installed.
    """
    client = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(client)
    CachedTableBaseMixin.check_cache_config()
    try:
        yield client
    finally:
        CachedTableBaseMixin._redis_client = None
        await client.aclose()


@pytest_asyncio.fixture
async def cache_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Enhanced (cache-aware) AsyncSession bound to the per-test engine."""
    async with AsyncSession(engine) as s:
        yield s


@pytest_asyncio.fixture
async def sql_log(engine: AsyncEngine) -> AsyncIterator[list[str]]:
    """Record every SQL statement sent to the (per-test) engine."""
    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        statements.append(statement)

    sa_event.listen(engine.sync_engine, "before_cursor_execute", _record)
    try:
        yield statements
    finally:
        sa_event.remove(engine.sync_engine, "before_cursor_execute", _record)


def _select_count(statements: list[str]) -> int:
    return sum(1 for s in statements if s.lstrip().upper().startswith("SELECT"))


async def _drain() -> None:
    """Let fire-and-forget post-commit compensation tasks finish."""
    for _ in range(20):
        await asyncio.sleep(0)


def _id_key(model: type, id_value: object) -> str:
    return f"id:{model.__name__}:{id_value}"


# ---------------------------------------------------------------------------
# 1. ID-layer cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_by_id_first_db_then_cache(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
    sql_log: list[str],
) -> None:
    gadget = await CacheGadget(name="hammer", quantity=3).save(cache_session)
    gid = gadget.id
    await _drain()

    # Start from an empty cache so the first get must hit the DB.
    await fake_redis.flushdb()
    sql_log.clear()

    first = await CacheGadget.get(cache_session, CacheGadget.id == gid)
    assert first is not None and first.quantity == 3
    selects_after_first = _select_count(sql_log)
    assert selects_after_first >= 1, "first get must query the database"

    # The ID cache key must now exist.
    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 1

    second = await CacheGadget.get(cache_session, CacheGadget.id == gid)
    assert second is not None
    assert second.id == gid and second.name == "hammer" and second.quantity == 3
    assert _select_count(sql_log) == selects_after_first, "second get must be served from Redis"


@pytest.mark.asyncio
async def test_save_backfills_id_cache(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
) -> None:
    """save() proactively repopulates the ID cache (write-through refresh)."""
    gadget = await CacheGadget(name="prefilled", quantity=7).save(cache_session)
    await _drain()
    assert await fake_redis.exists(_id_key(CacheGadget, gadget.id)) == 1


@pytest.mark.asyncio
async def test_no_cache_param_skips_cache_entirely(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
) -> None:
    gadget = await CacheGadget(name="nocache", quantity=1).save(cache_session)
    await _drain()
    await fake_redis.flushdb()

    got = await CacheGadget.get(cache_session, CacheGadget.id == gadget.id, no_cache=True)
    assert got is not None
    assert await fake_redis.keys("*") == [], "no_cache=True must not read or write Redis"


# ---------------------------------------------------------------------------
# 2. Query-layer cache: hit + version-bump invalidation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_cache_hit_and_invalidation_on_write(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
    sql_log: list[str],
) -> None:
    await CacheGadget(name="qa", quantity=10).save(cache_session)
    await CacheGadget(name="qb", quantity=1).save(cache_session)
    await _drain()
    await fake_redis.flushdb()
    sql_log.clear()

    # First condition query -> DB + query-cache write at v0.
    r1 = await CacheGadget.get(cache_session, CacheGadget.quantity >= 5, fetch_mode="all")
    assert {g.name for g in r1} == {"qa"}
    selects_after_first = _select_count(sql_log)
    assert selects_after_first >= 1
    query_keys = await fake_redis.keys("query:CacheGadget:*")
    assert len(query_keys) == 1
    assert query_keys[0].decode().startswith("query:CacheGadget:v0:")

    # Identical query -> cache hit, zero extra SQL.
    r2 = await CacheGadget.get(cache_session, CacheGadget.quantity >= 5, fetch_mode="all")
    assert {g.name for g in r2} == {"qa"}
    assert _select_count(sql_log) == selects_after_first

    # A write bumps ver:CacheGadget -> the old query key becomes unreachable.
    await CacheGadget(name="qc", quantity=7).save(cache_session)
    await _drain()
    raw_ver = await fake_redis.get("ver:CacheGadget")
    assert raw_ver is not None and int(raw_ver) >= 1

    sql_log.clear()
    r3 = await CacheGadget.get(cache_session, CacheGadget.quantity >= 5, fetch_mode="all")
    assert {g.name for g in r3} == {"qa", "qc"}, "post-write query must see the new row"
    assert _select_count(sql_log) >= 1, "version bump must force a DB re-query"

    # The re-query was cached under the new version namespace.
    new_keys = [k.decode() for k in await fake_redis.keys("query:CacheGadget:*")]
    assert any(k.startswith(f"query:CacheGadget:v{int(raw_ver)}:") for k in new_keys)


# ---------------------------------------------------------------------------
# 3. Write invalidation: update / delete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_invalidates_and_backfills_id_cache(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
    sql_log: list[str],
) -> None:
    gadget = await CacheGadget(name="upd", quantity=1).save(cache_session)
    gid = gadget.id
    await _drain()
    ver_before = await fake_redis.get("ver:CacheGadget")

    updated = await gadget.update(cache_session, CacheGadgetPatch(quantity=42))
    await _drain()
    assert updated.quantity == 42

    # Query-cache version bumped by the update.
    ver_after = await fake_redis.get("ver:CacheGadget")
    assert ver_after is not None
    assert int(ver_after) > int(ver_before or 0)

    # ID cache was backfilled with the *new* value: cache-served get, no SQL.
    sql_log.clear()
    got = await CacheGadget.get(cache_session, CacheGadget.id == gid)
    assert got is not None and got.quantity == 42
    assert _select_count(sql_log) == 0, "refreshed ID cache must serve the updated row"


@pytest.mark.asyncio
async def test_delete_instance_removes_id_cache(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
) -> None:
    gadget = await CacheGadget(name="gone", quantity=1).save(cache_session)
    gid = gadget.id
    await _drain()
    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 1

    deleted = await CacheGadget.delete(cache_session, gadget)
    assert deleted == 1
    await _drain()

    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 0, (
        "delete() must drop the row's ID cache key"
    )
    got = await CacheGadget.get(cache_session, CacheGadget.id == gid)
    assert got is None


@pytest.mark.asyncio
async def test_delete_by_condition_wipes_model_id_caches(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
) -> None:
    a = await CacheGadget(name="ca", quantity=5).save(cache_session)
    a_id = a.id  # capture before the next commit expires `a`
    b = await CacheGadget(name="cb", quantity=6).save(cache_session)
    b_id = b.id
    await _drain()
    assert await fake_redis.exists(_id_key(CacheGadget, a_id)) == 1
    assert await fake_redis.exists(_id_key(CacheGadget, b_id)) == 1
    ver_before = int(await fake_redis.get("ver:CacheGadget") or 0)

    deleted = await CacheGadget.delete(cache_session, condition=CacheGadget.quantity >= 5)
    assert deleted == 2
    await _drain()

    # Condition delete -> model-level SCAN+DEL of every id: key + version bump.
    assert await fake_redis.keys("id:CacheGadget:*") == []
    assert int(await fake_redis.get("ver:CacheGadget") or 0) > ver_before

    rows = await CacheGadget.get(cache_session, fetch_mode="all")
    assert rows == []


# ---------------------------------------------------------------------------
# 4. commit / rollback consistency (enhanced cache-aware AsyncSession)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_commit_false_skips_cache_then_rollback_clears_pending(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
    sql_log: list[str],
) -> None:
    gadget = CacheGadget(name="pending", quantity=5)
    await gadget.save(cache_session, commit=False, refresh=False)

    # Uncommitted write -> pending invalidation recorded for the type.
    pending = cache_session.info.get(_SESSION_PENDING_CACHE_KEY)
    assert pending is not None and CacheGadget in pending

    # get() must skip the cache (read AND write) while the txn is dirty.
    sql_log.clear()
    got = await CacheGadget.get(cache_session, CacheGadget.id == gadget.id)
    assert got is not None and got.name == "pending"
    assert _select_count(sql_log) >= 1
    assert await fake_redis.keys("id:CacheGadget:*") == [], (
        "uncommitted rows must never be written to Redis"
    )

    await cache_session.rollback()
    # after_rollback hook drops the queued invalidations.
    assert _SESSION_PENDING_CACHE_KEY not in cache_session.info

    # Row was rolled back; cache path is active again and sees no row.
    got2 = await CacheGadget.get(cache_session, CacheGadget.id == gadget.id)
    assert got2 is None
    assert await fake_redis.keys("ver:CacheGadget") == [], (
        "rollback must not bump the query-cache version"
    )


@pytest.mark.asyncio
async def test_bare_mutation_enhanced_commit_invalidates(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
) -> None:
    """Bare attribute mutation + session.commit() (no CRUD helper) must still
    invalidate via the enhanced AsyncSession's auto-registration."""
    gadget = await CacheGadget(name="bare", quantity=1).save(cache_session)
    gid = gadget.id
    await _drain()
    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 1
    ver_before = int(await fake_redis.get("ver:CacheGadget") or 0)

    gadget.quantity = 99  # dirty, never goes through save()
    await cache_session.commit()
    await _drain()

    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 0, (
        "cache-aware commit must drop the stale ID cache"
    )
    assert int(await fake_redis.get("ver:CacheGadget") or 0) > ver_before

    got = await CacheGadget.get(cache_session, CacheGadget.id == gid)
    assert got is not None and got.quantity == 99
    # The miss re-populated the ID cache with fresh data.
    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 1


@pytest.mark.asyncio
async def test_rollback_keeps_cache_serving_committed_state(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
    sql_log: list[str],
) -> None:
    gadget = await CacheGadget(name="rb", quantity=1).save(cache_session)
    gid = gadget.id
    await _drain()
    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 1

    gadget.quantity = 77  # never committed
    await cache_session.rollback()

    # Cache untouched by the rollback and still serves the committed value.
    assert await fake_redis.exists(_id_key(CacheGadget, gid)) == 1
    sql_log.clear()
    got = await CacheGadget.get(cache_session, CacheGadget.id == gid)
    assert got is not None and got.quantity == 1
    assert _select_count(sql_log) == 0, "post-rollback get must be a pure cache hit"


# ---------------------------------------------------------------------------
# 5. Serialization roundtrip (orjson): datetime / UUID fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_roundtrip_preserves_datetime_and_uuid(
    fake_redis: fakeredis.aioredis.FakeRedis,
    cache_session: AsyncSession,
    engine: AsyncEngine,
    sql_log: list[str],
) -> None:
    moment = datetime(2024, 5, 17, 12, 30, 45, 123456)
    ref = uuid.uuid4()
    event_row = await CacheEvent(title="launch", happened_at=moment, ref_id=ref).save(cache_session)
    eid = event_row.id
    await _drain()
    await fake_redis.flushdb()

    # Populate the cache from the DB in session #1.
    from_db = await CacheEvent.get(cache_session, CacheEvent.id == eid)
    assert from_db is not None
    assert await fake_redis.exists(_id_key(CacheEvent, eid)) == 1

    # Read back in a brand-new session: must come from Redis with zero SQL.
    async with AsyncSession(engine) as fresh_session:
        sql_log.clear()
        from_cache = await CacheEvent.get(fresh_session, CacheEvent.id == eid)
        assert _select_count(sql_log) == 0, "fresh-session read must be served from Redis"

    assert from_cache is not None
    assert isinstance(from_cache.id, uuid.UUID) and from_cache.id == eid
    assert isinstance(from_cache.ref_id, uuid.UUID) and from_cache.ref_id == ref
    assert isinstance(from_cache.happened_at, datetime) and from_cache.happened_at == moment
    assert from_cache.title == "launch"
    assert from_cache.created_at == from_db.created_at
    assert from_cache.updated_at == from_db.updated_at


@pytest.mark.asyncio
async def test_serialize_deserialize_wrappers_roundtrip(
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """Direct unit roundtrip of the None / single / list wrapper formats."""
    moment = datetime(2030, 1, 2, 3, 4, 5, 678901)
    ref = uuid.uuid4()
    ev = CacheEvent(title="unit", happened_at=moment, ref_id=ref)

    # single
    single = CacheEvent._deserialize_result(CacheEvent._serialize_result(ev), "first")
    assert isinstance(single, CacheEvent)
    assert single.id == ev.id and single.ref_id == ref and single.happened_at == moment

    # list
    listed = CacheEvent._deserialize_result(CacheEvent._serialize_result([ev]), "all")
    assert isinstance(listed, list) and len(listed) == 1
    assert listed[0].id == ev.id and listed[0].happened_at == moment

    # None (cached empty result)
    none_back = CacheEvent._deserialize_result(CacheEvent._serialize_result(None), "first")
    assert none_back is None


# ---------------------------------------------------------------------------
# 6. Degradation: unconfigured / failing Redis
# ---------------------------------------------------------------------------

class _BrokenRedis:
    """Stand-in client whose every command fails like a downed Redis."""

    def __getattr__(self, name: str):
        async def _fail(*args: object, **kwargs: object) -> object:
            raise ConnectionError("redis is down")
        return _fail


@pytest.mark.asyncio
async def test_unconfigured_redis_fails_fast_on_cached_read(
    cache_session: AsyncSession,
) -> None:
    """Documented contract: unconfigured Redis -> RuntimeError (fail fast)."""
    previous = CachedTableBaseMixin._redis_client
    CachedTableBaseMixin._redis_client = None
    try:
        # Writes survive: every redis touch on the write path is wrapped in
        # best-effort error handling, and the refresh uses no_cache=True.
        gadget = await CacheGadget(name="noredis", quantity=2).save(cache_session)
        await _drain()
        assert gadget.quantity == 2

        with pytest.raises(RuntimeError, match="Redis client not configured"):
            await CacheGadget.get(cache_session, CacheGadget.id == gadget.id)

        # Explicit opt-out still works without Redis.
        got = await CacheGadget.get(cache_session, CacheGadget.id == gadget.id, no_cache=True)
        assert got is not None and got.name == "noredis"
    finally:
        CachedTableBaseMixin._redis_client = previous


@pytest.mark.asyncio
async def test_transient_redis_errors_fall_back_to_db(
    cache_session: AsyncSession,
) -> None:
    """Runtime Redis failures are logged and degrade gracefully to the DB."""
    previous = CachedTableBaseMixin._redis_client
    CachedTableBaseMixin.configure_redis(_BrokenRedis())
    try:
        gadget = await CacheGadget(name="degraded", quantity=11).save(cache_session)
        await _drain()

        # ID query: cache read fails -> DB fallback.
        got = await CacheGadget.get(cache_session, CacheGadget.id == gadget.id)
        assert got is not None and got.quantity == 11

        # Condition query: version read fails (-> v0), cache read fails -> DB.
        rows = await CacheGadget.get(cache_session, CacheGadget.quantity >= 10, fetch_mode="all")
        assert {g.name for g in rows} == {"degraded"}

        # Update path: invalidation + backfill failures must not break the write.
        updated = await gadget.update(cache_session, CacheGadgetPatch(quantity=12))
        await _drain()
        assert updated.quantity == 12
    finally:
        CachedTableBaseMixin._redis_client = previous
