"""
Tests for the auxiliary mixins: ResourceQuotaMixin, MixinTableScanMixin,
TrgmSearchableMixin / TrgmSearchRequest, migration cache invalidation and
the cache TTL constants.

``TrgmSearchableMixin`` is PostgreSQL-only; it is tested at the SQL
compilation level with the PostgreSQL dialect (same approach as
``test_pg_dialect_types.py``).
"""
import uuid
from typing import Any, ClassVar

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy import ColumnElement
from sqlalchemy.dialects import postgresql
from sqlmodel import Field, col
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    AsyncSession as ExtAsyncSession,
    CachedTableBaseMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
)
from sqlmodel_ext.mixins import (
    BIGRAM_FUNCTION_SQL,
    CACHE_TTL_COLD,
    CACHE_TTL_HOT,
    CACHE_TTL_WARM,
    CallerDidNotCommitError,
    MixinTableScanMixin,
    QuotaExceededError,
    QuotaOwnerNotFoundError,
    ResourceQuotaMixin,
    TrgmSearchableMixin,
    TrgmSearchRequest,
    run_pending_migration_cache_invalidations,
)


# ---------------------------------------------------------------------------
# ResourceQuotaMixin
# ---------------------------------------------------------------------------

class QuotaOwner(SQLModelBase, UUIDTableBaseMixin, table=True):
    max_items: int = 2


class QuotaItem(ResourceQuotaMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    owner_id: uuid.UUID = Field(foreign_key='quotaowner.id')
    label: str = "x"

    @classmethod
    async def _lock_owner(cls, session: AsyncSession, owner_id: Any, *, with_for_update: bool = True) -> Any:
        return await QuotaOwner.get(session, col(QuotaOwner.id) == owner_id, with_for_update=with_for_update)

    @classmethod
    def _quota_condition(cls, owner_id: Any) -> ColumnElement[bool]:
        return col(cls.owner_id) == owner_id

    @classmethod
    def _quota_max(cls, owner: Any) -> int:
        return owner.max_items


async def _make_owner(session: AsyncSession, max_items: int = 2) -> uuid.UUID:
    return (await QuotaOwner(max_items=max_items).save(session)).id


@pytest.mark.asyncio
class TestResourceQuota:
    async def test_acquire_and_insert_within_quota(self, session: AsyncSession) -> None:
        owner_id = await _make_owner(session)
        for _ in range(2):
            async with QuotaItem.acquire_quota_lock(session, owner_id):
                await QuotaItem(owner_id=owner_id).save(session)
        assert await QuotaItem.count(session) == 2

    async def test_quota_exceeded(self, session: AsyncSession) -> None:
        owner_id = await _make_owner(session, max_items=1)
        async with QuotaItem.acquire_quota_lock(session, owner_id):
            await QuotaItem(owner_id=owner_id).save(session)
        with pytest.raises(QuotaExceededError) as exc_info:
            async with QuotaItem.acquire_quota_lock(session, owner_id):
                pass
        assert exc_info.value.max_allowed == 1
        assert exc_info.value.current_count == 1
        await session.rollback()

    async def test_count_checks_total_request(self, session: AsyncSession) -> None:
        owner_id = await _make_owner(session, max_items=2)
        with pytest.raises(QuotaExceededError):
            async with QuotaItem.acquire_quota_lock(session, owner_id, count=3):
                pass
        await session.rollback()
        with pytest.raises(ValueError):
            async with QuotaItem.acquire_quota_lock(session, owner_id, count=0):
                pass

    async def test_missing_owner(self, session: AsyncSession) -> None:
        with pytest.raises(QuotaOwnerNotFoundError):
            async with QuotaItem.acquire_quota_lock(session, uuid.uuid4()):
                pass

    async def test_caller_must_commit(self, session: AsyncSession) -> None:
        owner_id = await _make_owner(session)
        with pytest.raises(CallerDidNotCommitError):
            async with QuotaItem.acquire_quota_lock(session, owner_id):
                await QuotaItem(owner_id=owner_id).save(session, commit=False)
        await session.rollback()

    async def test_defer_commit_disables_guard(self, session: AsyncSession) -> None:
        owner_id = await _make_owner(session)
        async with QuotaItem.acquire_quota_lock(session, owner_id, defer_commit=True):
            await QuotaItem(owner_id=owner_id).save(session, commit=False)
        await session.commit()
        assert await QuotaItem.count(session) == 1

    async def test_idempotent_check_short_circuits(self, session: AsyncSession) -> None:
        owner_id = await _make_owner(session, max_items=1)
        async with QuotaItem.acquire_quota_lock(session, owner_id):
            await QuotaItem(owner_id=owner_id, label="only").save(session)

        async def existing() -> QuotaItem | None:
            return await QuotaItem.get(session, col(QuotaItem.label) == "only")

        # The quota is full, but the idempotent re-check finds the row first.
        async with QuotaItem.acquire_quota_lock(session, owner_id, idempotent_check=existing) as found:
            assert found is not None and found.label == "only"
        await session.rollback()

    async def test_preflight(self, session: AsyncSession) -> None:
        owner_id = await _make_owner(session, max_items=1)
        await QuotaItem.preflight_quota(session, owner_id)
        await QuotaItem(owner_id=owner_id).save(session)
        with pytest.raises(QuotaExceededError):
            await QuotaItem.preflight_quota(session, owner_id)
        with pytest.raises(QuotaOwnerNotFoundError):
            await QuotaItem.preflight_quota(session, uuid.uuid4())
        with pytest.raises(ValueError):
            await QuotaItem.preflight_quota(session, owner_id, count=0)

    async def test_unimplemented_contract_raises(self) -> None:
        class Incomplete(ResourceQuotaMixin):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete._quota_max(object())
        with pytest.raises(NotImplementedError):
            Incomplete._quota_condition(1)


# ---------------------------------------------------------------------------
# MixinTableScanMixin
# ---------------------------------------------------------------------------

class RemoteKeyMixin(MixinTableScanMixin):
    """A field-level mixin mounted on several tables."""
    remote_key: str | None = None


class ScanVendorA(RemoteKeyMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "a"


class ScanVendorB(RemoteKeyMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "b"


@pytest.mark.asyncio
class TestMixinTableScan:
    async def test_discovers_concrete_tables(self) -> None:
        tables = RemoteKeyMixin._concrete_mixin_subclasses()
        assert set(tables) == {ScanVendorA, ScanVendorB}

    async def test_scan_rows_where(self, session: AsyncSession) -> None:
        await ScanVendorA(remote_key="k1").save(session)
        await ScanVendorA(remote_key="other").save(session)
        await ScanVendorB(remote_key="k1").save(session)

        hits = [
            (table_cls, row.remote_key)
            async for table_cls, row in RemoteKeyMixin._scan_rows_where(
                session, lambda table_cls: col(table_cls.remote_key) == "k1",
            )
        ]
        assert sorted((t.__name__, k) for t, k in hits) == [("ScanVendorA", "k1"), ("ScanVendorB", "k1")]


# ---------------------------------------------------------------------------
# TrgmSearchableMixin (PostgreSQL dialect compilation)
# ---------------------------------------------------------------------------

class TrgmDoc(TrgmSearchableMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    __trgm_text_columns__: ClassVar[tuple[str, ...]] = ('description',)
    name: str
    description: str = ""


def _pg(clause: Any) -> str:
    # The pyformat paramstyle doubles literal '%' signs; undo that for readability.
    compiled = str(clause.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    return compiled.replace('%%', '%')


class TestTrgmSearchable:
    def test_condition_shape(self) -> None:
        sql = _pg(TrgmDoc.trgm_search_condition("abc"))
        assert "public.bigrams(trgmdoc.name)" in sql
        assert "public.bigrams(trgmdoc.description)" in sql
        assert "trgmdoc.name % 'abc'" in sql
        assert "ILIKE" in sql.upper()
        # the text column never uses the similarity operator
        assert "trgmdoc.description % " not in sql

    def test_short_query_skips_bigram_guard(self) -> None:
        sql = _pg(TrgmDoc.trgm_search_condition("a"))
        assert "bigrams" not in sql
        assert "trgmdoc.name % 'a'" in sql

    def test_like_wildcards_are_escaped(self) -> None:
        sql = _pg(TrgmDoc.trgm_search_condition("50%_off"))
        assert "50/%/_off" in sql

    def test_unknown_column_raises(self) -> None:
        class BadTrgm(TrgmSearchableMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
            __trgm_name_column__: ClassVar[str] = 'missing'
            title: str

        with pytest.raises(RuntimeError, match="not on the mapper"):
            BadTrgm.trgm_search_condition("abc")

    def test_request_apply_condition(self) -> None:
        assert TrgmSearchRequest(query=None).apply_condition(TrgmDoc, None) is None
        assert TrgmSearchRequest(query="   ").normalized_query is None
        base = col(TrgmDoc.name) == "x"
        assert TrgmSearchRequest(query="  ").apply_condition(TrgmDoc, base) is base
        combined = _pg(TrgmSearchRequest(query=" abc ").apply_condition(TrgmDoc, base))
        assert combined.startswith("trgmdoc.name = 'x' AND")
        assert "'abc'" in combined

    def test_request_rejects_nul_and_overlong(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrgmSearchRequest(query="a\x00b")
        with pytest.raises(ValidationError):
            TrgmSearchRequest(query="x" * 65)

    def test_bigram_function_sql(self) -> None:
        assert "CREATE OR REPLACE FUNCTION public.bigrams(t text)" in BIGRAM_FUNCTION_SQL
        assert "IMMUTABLE" in BIGRAM_FUNCTION_SQL


# ---------------------------------------------------------------------------
# Migration cache invalidation
# ---------------------------------------------------------------------------

class MigCachedThing(SQLModelBase, CachedTableBaseMixin, UUIDTableBaseMixin, table=True):
    name: str


class _FailingInvalidation(Exception):
    pass


@pytest_asyncio.fixture
async def mig_redis():
    client = fakeredis.aioredis.FakeRedis()
    previous = CachedTableBaseMixin._redis_client
    CachedTableBaseMixin.configure_redis(client)
    try:
        yield client
    finally:
        CachedTableBaseMixin._redis_client = previous
        await client.aclose()


@pytest.mark.asyncio
class TestMigrationCacheInvalidation:
    async def test_runs_once_and_writes_sentinel(self, mig_redis, engine) -> None:
        async with ExtAsyncSession(engine) as s:
            thing = await MigCachedThing(name="t").save(s)
            tid = thing.id
        assert await mig_redis.exists(f"id:MigCachedThing:{tid}") == 1

        tasks = {'rescale-v1': ['MigCachedThing', 'RemovedModel']}
        await run_pending_migration_cache_invalidations(SQLModelBase, tasks=tasks)
        assert await mig_redis.exists(f"id:MigCachedThing:{tid}") == 0
        assert await mig_redis.get('cache_invalidation:rescale-v1') == b"1"

        # Second run: sentinel present -> skipped (the re-filled key survives).
        await mig_redis.set(f"id:MigCachedThing:{tid}", b"x")
        await run_pending_migration_cache_invalidations(SQLModelBase, tasks=tasks)
        assert await mig_redis.exists(f"id:MigCachedThing:{tid}") == 1

    async def test_failure_leaves_sentinel_unwritten(self, mig_redis, monkeypatch: pytest.MonkeyPatch) -> None:
        async def failing(cls: Any, *, strict: bool = False) -> None:
            assert strict is True
            raise _FailingInvalidation("redis down")

        monkeypatch.setattr(MigCachedThing, "invalidate_all", classmethod(failing))
        await run_pending_migration_cache_invalidations(
            SQLModelBase, tasks={'broken-v1': ['MigCachedThing'], 'fine-v1': ['RemovedModel']},
        )
        assert await mig_redis.exists('cache_invalidation:broken-v1') == 0
        # One failing task does not block the others.
        assert await mig_redis.exists('cache_invalidation:fine-v1') == 1

    async def test_no_tasks_is_noop(self, mig_redis) -> None:
        await run_pending_migration_cache_invalidations(SQLModelBase, tasks={})
        assert await mig_redis.keys('*') == []


# ---------------------------------------------------------------------------
# Cache TTL constants
# ---------------------------------------------------------------------------

def test_cache_ttl_constants() -> None:
    assert CACHE_TTL_HOT < CACHE_TTL_WARM < CACHE_TTL_COLD
    assert CACHE_TTL_COLD == CachedTableBaseMixin.__cache_ttl__
