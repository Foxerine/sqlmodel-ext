"""
Enhanced ``AsyncSession``: orchestration-level behavior without Redis.

Full cache invalidation requires a Redis client; these tests cover the parts
that must work standalone:

- ``commit()`` degrades to a plain commit when no cached model is involved.
- ``reset()`` clears the FOR UPDATE tracking key and the three
  cache-invalidation tracking keys from ``session.info``.
- ``refresh()`` falls back to the native refresh for non-cached models.
- ``execute()`` passes statements through unchanged (SELECT path).
- ``commit_count``, post-commit callbacks, the ``begin()`` wrapper,
  best-effort rollback, raw-DML write registration, ``set_local_timeouts``
  and the ``SessionFactory.run_in_repeatable_read`` retry orchestration.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text, update
from sqlmodel import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSessionBase

from sqlmodel_ext import AsyncSession, SESSION_FOR_UPDATE_KEY, SQLModelBase, UUIDTableBaseMixin
from sqlmodel_ext.mixins.cached_table import (
    _SESSION_CASCADE_DELETED_KEY,
    _SESSION_COMMITTED_PENDING_KEY,
    _SESSION_ENHANCED_COMMIT_KEY,
    _SESSION_FLUSHED_TABLES,
    _SESSION_PENDING_CACHE_KEY,
)
from sqlmodel_ext.mixins.table import SESSION_REPEATABLE_READ_KEY
from sqlmodel_ext.session import (
    POST_COMMIT_CALLBACKS_KEY,
    SERIALIZATION_FAILURE_SQLSTATE,
    RepeatableReadSnapshotConflictError,
    SerializationRetryExhaustedError,
    SessionFactory,
)



class SessEntry(SQLModelBase, UUIDTableBaseMixin, table=True):
    """Plain (non-cached, non-polymorphic) model for session tests."""
    name: str


@pytest_asyncio.fixture
async def enhanced_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Enhanced session bound to the fresh per-test engine."""
    async with AsyncSession(engine) as s:
        yield s


@pytest.mark.asyncio
async def test_commit_plain_model_degrades_to_plain_commit(
    enhanced_session: AsyncSession,
) -> None:
    """Non-cached models commit normally -- no Redis required."""
    func = SessEntry(name="plain")
    enhanced_session.add(func)
    await enhanced_session.commit()

    result = await enhanced_session.execute(select(SessEntry))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].name == "plain"


@pytest.mark.asyncio
async def test_reset_clears_tracking_state(enhanced_session: AsyncSession) -> None:
    enhanced_session.info[SESSION_FOR_UPDATE_KEY] = {123}
    enhanced_session.info[_SESSION_PENDING_CACHE_KEY] = {}
    enhanced_session.info[_SESSION_CASCADE_DELETED_KEY] = {}
    enhanced_session.info[_SESSION_ENHANCED_COMMIT_KEY] = True
    enhanced_session.info[_SESSION_COMMITTED_PENDING_KEY] = {}

    await enhanced_session.reset()

    assert SESSION_FOR_UPDATE_KEY not in enhanced_session.info
    assert _SESSION_PENDING_CACHE_KEY not in enhanced_session.info
    assert _SESSION_CASCADE_DELETED_KEY not in enhanced_session.info
    assert _SESSION_ENHANCED_COMMIT_KEY not in enhanced_session.info
    assert _SESSION_COMMITTED_PENDING_KEY not in enhanced_session.info


@pytest.mark.asyncio
async def test_refresh_falls_back_for_non_cached_model(
    enhanced_session: AsyncSession,
) -> None:
    """Non-cached models take the native session.refresh() path."""
    func = SessEntry(name="before")
    enhanced_session.add(func)
    await enhanced_session.commit()

    await enhanced_session.refresh(func)
    assert func.name == "before"


@pytest.mark.asyncio
async def test_crud_save_via_enhanced_session(enhanced_session: AsyncSession) -> None:
    """TableBaseMixin.save() works unchanged on the enhanced session."""
    func = SessEntry(name="crud")
    func = await func.save(enhanced_session)
    assert func.name == "crud"


# ---------------------------------------------------------------------------
# commit_count / post-commit callbacks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_commit_count_increments_only_on_successful_commit(
    enhanced_session: AsyncSession,
) -> None:
    assert enhanced_session.commit_count == 0
    await SessEntry(name="a").save(enhanced_session)
    assert enhanced_session.commit_count == 1
    await SessEntry(name="b").save(enhanced_session, commit=False)
    assert enhanced_session.commit_count == 1
    await enhanced_session.rollback()
    assert enhanced_session.commit_count == 1


@pytest.mark.asyncio
async def test_post_commit_callback_runs_once_after_commit(
    enhanced_session: AsyncSession,
) -> None:
    calls: list[str] = []

    async def cb() -> None:
        calls.append("ran")

    enhanced_session.add_post_commit_callback(cb)
    assert calls == []
    await enhanced_session.commit()
    assert calls == ["ran"]
    await enhanced_session.commit()  # queue cleared -- not run again
    assert calls == ["ran"]


@pytest.mark.asyncio
@pytest.mark.parametrize("discard", ["rollback", "reset", "close"])
async def test_post_commit_callback_discarded(enhanced_session: AsyncSession, discard: str) -> None:
    calls: list[str] = []

    async def cb() -> None:
        calls.append("ran")

    enhanced_session.add_post_commit_callback(cb)
    await getattr(enhanced_session, discard)()
    await enhanced_session.commit()
    assert calls == []


@pytest.mark.asyncio
async def test_post_commit_callback_failure_does_not_block_others(
    enhanced_session: AsyncSession,
) -> None:
    calls: list[str] = []

    async def bad() -> None:
        raise ValueError("boom")

    async def good() -> None:
        calls.append("good")

    enhanced_session.add_post_commit_callback(bad)
    enhanced_session.add_post_commit_callback(good)
    await enhanced_session.commit()  # does not raise
    assert calls == ["good"]


@pytest.mark.asyncio
async def test_post_commit_callback_rejected_inside_savepoint(
    enhanced_session: AsyncSession,
) -> None:
    async def cb() -> None:
        pass

    await SessEntry(name="x").save(enhanced_session, commit=False)
    async with enhanced_session.begin_nested():
        with pytest.raises(RuntimeError, match="savepoint"):
            enhanced_session.add_post_commit_callback(cb)
    await enhanced_session.rollback()


@pytest.mark.asyncio
async def test_fail_soft_when_observed_runs_all_callbacks(
    enhanced_session: AsyncSession,
) -> None:
    calls: list[str] = []

    async def cancelled() -> None:
        raise asyncio.CancelledError()

    async def good() -> None:
        calls.append("good")

    enhanced_session.add_post_commit_callback(cancelled)
    enhanced_session.add_post_commit_callback(good)
    await enhanced_session.commit(fail_soft_when_observed=True)
    assert calls == ["good"]
    assert enhanced_session.commit_count == 1


@pytest.mark.asyncio
async def test_default_commit_propagates_callback_cancellation(
    enhanced_session: AsyncSession,
) -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError()

    enhanced_session.add_post_commit_callback(cancelled)
    with pytest.raises(asyncio.CancelledError):
        await enhanced_session.commit()
    # The commit itself happened before the callbacks ran.
    assert enhanced_session.commit_count == 1


# ---------------------------------------------------------------------------
# begin() wrapper
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_begin_context_routes_exit_through_enhanced_commit(engine: AsyncEngine) -> None:
    calls: list[str] = []

    async def cb() -> None:
        calls.append("ran")

    async with AsyncSession(engine) as s:
        async with s.begin():
            s.add(SessEntry(name="in-begin"))
            s.add_post_commit_callback(cb)
        assert calls == ["ran"]
        assert s.commit_count == 1
        # the session is reusable after the block
        rows = (await s.exec(select(SessEntry))).all()
        assert [r.name for r in rows] == ["in-begin"]


@pytest.mark.asyncio
async def test_begin_context_exception_rolls_back_and_discards(engine: AsyncEngine) -> None:
    calls: list[str] = []

    async def cb() -> None:
        calls.append("ran")

    async with AsyncSession(engine) as s:
        with pytest.raises(ValueError):
            async with s.begin():
                s.add(SessEntry(name="rolled-back"))
                s.add_post_commit_callback(cb)
                raise ValueError("abort")
        await s.commit()
        assert calls == []
        assert (await s.exec(select(SessEntry))).all() == []


@pytest.mark.asyncio
async def test_await_begin_returns_enhanced_session(engine: AsyncEngine) -> None:
    calls: list[str] = []

    async def cb() -> None:
        calls.append("ran")

    async with AsyncSession(engine) as s:
        tx = await s.begin()
        assert tx is s
        assert s.in_transaction()
        s.add_post_commit_callback(cb)
        await tx.commit()
        assert calls == ["ran"]


@pytest.mark.asyncio
async def test_explicit_commit_inside_begin_block_not_committed_twice(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as s:
        async with s.begin():
            s.add(SessEntry(name="early"))
            await s.commit()
        assert s.commit_count == 1
        _ = (await s.exec(select(SessEntry))).all()


# ---------------------------------------------------------------------------
# rollback(best_effort_budget_seconds=)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_best_effort_rollback_swallows_failure_and_invalidates(
    enhanced_session: AsyncSession, monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalidated: list[bool] = []

    async def failing_rollback(self: object) -> None:
        raise RuntimeError("connection lost")

    async def fake_invalidate(self: object) -> None:
        invalidated.append(True)

    monkeypatch.setattr(_AsyncSessionBase, "rollback", failing_rollback)
    monkeypatch.setattr(_AsyncSessionBase, "invalidate", fake_invalidate)
    await enhanced_session.rollback(best_effort_budget_seconds=1.0)  # must not raise
    assert invalidated == [True]


@pytest.mark.asyncio
async def test_plain_rollback_failure_still_discards_callbacks(
    enhanced_session: AsyncSession, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cb() -> None:
        pass

    async def failing_rollback(self: object) -> None:
        raise RuntimeError("connection lost")

    enhanced_session.add_post_commit_callback(cb)
    monkeypatch.setattr(_AsyncSessionBase, "rollback", failing_rollback)
    with pytest.raises(RuntimeError):
        await enhanced_session.rollback()
    assert POST_COMMIT_CALLBACKS_KEY not in enhanced_session.info


# ---------------------------------------------------------------------------
# reset() / close() clear every tracking key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["reset", "close"])
async def test_reset_and_close_clear_repeatable_read_marker(
    enhanced_session: AsyncSession, method: str,
) -> None:
    enhanced_session.info[SESSION_REPEATABLE_READ_KEY] = True
    enhanced_session.info[SESSION_FOR_UPDATE_KEY] = {1}
    await getattr(enhanced_session, method)()
    assert SESSION_REPEATABLE_READ_KEY not in enhanced_session.info
    assert SESSION_FOR_UPDATE_KEY not in enhanced_session.info


# ---------------------------------------------------------------------------
# raw DML observation points
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["execute", "exec", "scalar"])
async def test_raw_dml_registers_uncommitted_table(
    enhanced_session: AsyncSession, entry: str,
) -> None:
    await SessEntry(name="row").save(enhanced_session)
    table = SessEntry.__table__  # type: ignore[attr-defined]
    stmt = update(table).values(name="changed").returning(table.c.id)
    await getattr(enhanced_session, entry)(stmt)
    assert table in enhanced_session.info[_SESSION_FLUSHED_TABLES]
    await enhanced_session.commit()
    # cleared at the end of the outermost transaction
    assert _SESSION_FLUSHED_TABLES not in enhanced_session.info


@pytest.mark.asyncio
async def test_raw_text_write_registers_all_tables_but_select_does_not(
    enhanced_session: AsyncSession,
) -> None:
    await enhanced_session.exec(text("SELECT 1"))
    assert _SESSION_FLUSHED_TABLES not in enhanced_session.info
    await enhanced_session.exec(text("DELETE FROM sessentry"))
    registered = enhanced_session.info[_SESSION_FLUSHED_TABLES]
    assert SessEntry.__table__ in registered  # type: ignore[attr-defined]
    assert len(registered) > 1
    await enhanced_session.rollback()


# ---------------------------------------------------------------------------
# set_local_timeouts (PostgreSQL only -- statement shape checked via a stub)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_local_timeouts_issues_transaction_local_set_config(
    enhanced_session: AsyncSession, monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[str] = []

    class _Result:
        def one(self) -> tuple[str]:
            return ("",)

    async def fake_exec(self: object, statement: object, *args: object, **kwargs: object) -> _Result:
        compiled.append(str(statement.compile(  # type: ignore[attr-defined]
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True},
        )))
        return _Result()

    monkeypatch.setattr(AsyncSession, "exec", fake_exec)
    await enhanced_session.set_local_timeouts(lock_timeout_ms=1500, statement_timeout_ms=30000)
    assert "set_config('lock_timeout', '1500ms', true)" in compiled[0]
    assert "set_config('statement_timeout', '30000ms', true)" in compiled[1]


# ---------------------------------------------------------------------------
# SessionFactory.run_in_repeatable_read (retry orchestration; the PostgreSQL
# isolation switch is stubbed because SQLite has no REPEATABLE READ)
# ---------------------------------------------------------------------------

class _SerializationFailure(Exception):
    sqlstate = SERIALIZATION_FAILURE_SQLSTATE


@pytest_asyncio.fixture
async def rr_factory(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> SessionFactory:
    async def fake_enter(self: AsyncSession) -> None:
        self.info[SESSION_REPEATABLE_READ_KEY] = True

    monkeypatch.setattr(AsyncSession, "enter_repeatable_read", fake_enter)
    return SessionFactory(bind=engine, class_=AsyncSession)


@pytest.mark.asyncio
async def test_run_in_repeatable_read_retries_serialization_failure(rr_factory: SessionFactory) -> None:
    attempts = 0

    async def op(s: AsyncSession) -> str:
        nonlocal attempts
        attempts += 1
        assert s.info[SESSION_REPEATABLE_READ_KEY] is True
        if attempts == 1:
            raise DBAPIError(None, None, _SerializationFailure("40001"), False)
        await SessEntry(name="rr").save(s)
        return "done"

    assert await rr_factory.run_in_repeatable_read(op, description="probe") == "done"
    assert attempts == 2


@pytest.mark.asyncio
async def test_run_in_repeatable_read_retries_snapshot_conflict_then_exhausts(rr_factory: SessionFactory) -> None:
    attempts = 0

    async def op(s: AsyncSession) -> None:
        nonlocal attempts
        attempts += 1
        raise RepeatableReadSnapshotConflictError("winner invisible")

    with pytest.raises(SerializationRetryExhaustedError):
        await rr_factory.run_in_repeatable_read(op, description="probe", max_attempts=3)
    assert attempts == 3


@pytest.mark.asyncio
async def test_run_in_repeatable_read_does_not_retry_other_db_errors(rr_factory: SessionFactory) -> None:
    class _Deadlock(Exception):
        sqlstate = '40P01'

    attempts = 0

    async def op(s: AsyncSession) -> None:
        nonlocal attempts
        attempts += 1
        raise DBAPIError(None, None, _Deadlock("deadlock"), False)

    with pytest.raises(DBAPIError):
        await rr_factory.run_in_repeatable_read(op, description="probe")
    assert attempts == 1


@pytest.mark.asyncio
async def test_run_in_repeatable_read_never_reruns_after_commit(
    rr_factory: SessionFactory, engine: AsyncEngine,
) -> None:
    attempts = 0

    async def op(s: AsyncSession) -> None:
        nonlocal attempts
        attempts += 1
        await SessEntry(name="once").save(s, refresh=False)
        raise RepeatableReadSnapshotConflictError("surfaced after commit")

    with pytest.raises(RepeatableReadSnapshotConflictError):
        await rr_factory.run_in_repeatable_read(op, description="probe", max_attempts=3)
    assert attempts == 1
    async with AsyncSession(engine) as check:
        rows = (await check.exec(select(SessEntry))).all()
        assert [r.name for r in rows] == ["once"]


@pytest.mark.asyncio
async def test_run_in_repeatable_read_requires_commit(rr_factory: SessionFactory) -> None:
    async def op(s: AsyncSession) -> str:
        return "no commit"

    with pytest.raises(RuntimeError, match="without committing"):
        await rr_factory.run_in_repeatable_read(op, description="probe")


@pytest.mark.asyncio
async def test_run_in_repeatable_read_rejects_zero_attempts(rr_factory: SessionFactory) -> None:
    async def op(s: AsyncSession) -> None:
        pass

    with pytest.raises(ValueError):
        await rr_factory.run_in_repeatable_read(op, description="probe", max_attempts=0)


@pytest.mark.asyncio
async def test_enter_repeatable_read_fails_loud_when_level_not_verified(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the read-back does not report REPEATABLE READ, raise and leave no marker."""
    class _FakeConnection:
        async def scalar(self, statement: object) -> str:
            return 'read committed'

    async def fake_connection(self: object, **kwargs: object) -> _FakeConnection:
        return _FakeConnection()

    monkeypatch.setattr(AsyncSession, "connection", fake_connection)
    async with AsyncSession(engine) as s:
        with pytest.raises(RuntimeError, match="REPEATABLE READ"):
            await s.enter_repeatable_read()
        assert SESSION_REPEATABLE_READ_KEY not in s.info
