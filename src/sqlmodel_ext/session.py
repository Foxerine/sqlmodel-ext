"""sqlmodel-ext enhanced ``AsyncSession`` -- the library's canonical session type.

A subclass of sqlmodel's ``AsyncSession`` that upgrades cache correctness from
"documented convention enforced by review" to "adapted automatically at
runtime", and adds transaction-level helpers:

- ``commit()``: auto-registers every ``CachedTableBaseMixin`` mutation in the
  session before commit (new/dirty/deleted -- including bare ``session.add()``
  / attribute mutation / ``session.delete()`` paths that never went through the
  CRUD methods), then synchronously invalidates after commit, then runs the
  registered post-commit callbacks (see ``add_post_commit_callback``).
  ``commit_count`` records how many commits actually succeeded.
- ``rollback()``: discards pending post-commit callbacks; optional bounded
  best-effort mode (``best_effort_budget_seconds``).
- ``begin()``: ``async with session.begin():`` routes its exit through the
  enhanced ``commit()`` / ``rollback()`` (the native context manager commits
  the underlying transaction directly and would skip invalidation and
  callbacks).
- ``reset()`` / ``close()``: release the connection + clear FOR UPDATE lock
  tracking, the REPEATABLE READ marker, pending callbacks and cache
  tracking state (a closed session can be reused, stale state would fail open).
- ``refresh()``: delegates to the native ``session.refresh()`` -- a refresh
  must read the database; it never goes through the cache.
- ``execute()`` / ``exec()`` / ``scalar()`` / ``stream()`` /
  ``stream_scalars()``: register the tables written by raw DML as "written
  but uncommitted" (so dependent queries in the same transaction are never
  published to the shared cache) and warn when a raw ``UPDATE``/``DELETE``
  hits a cached table without a registered invalidation.
- ``enter_repeatable_read()`` / ``set_local_timeouts()``: PostgreSQL
  transaction helpers.

:class:`SessionFactory` (an ``async_sessionmaker`` subclass) adds
``run_in_repeatable_read()``: run an idempotent operation in its own
REPEATABLE READ session and retry the whole session on serialization
conflicts.

Invalidation logic belongs to ``CachedTableBaseMixin`` (delegated via its
helpers); this class only orchestrates ordering.

Wiring: point every session factory at this class --
``SessionFactory(engine, class_=AsyncSession)`` (or
``async_sessionmaker(engine, class_=AsyncSession)`` if you do not need
``run_in_repeatable_read``). Business code then simply
``await session.commit()`` / ``await session.reset()`` and gets cache-aware
behavior.

Models that do not inherit ``CachedTableBaseMixin`` are unaffected: every hook
degrades to the upstream behavior when no cached model is involved.
"""
import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar
from collections.abc import Awaitable, Callable, Generator, Iterable

from sqlalchemy import func, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSessionBase

from sqlmodel_ext.mixins.cached_table import CachedTableBaseMixin
from sqlmodel_ext.mixins.table import SESSION_FOR_UPDATE_KEY, SESSION_REPEATABLE_READ_KEY

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

REPEATABLE_READ_ISOLATION_LEVEL = 'REPEATABLE READ'
"""Isolation level name passed to ``execution_options(isolation_level=...)`` (SQLAlchemy spelling)."""

REPEATABLE_READ_REPORTED_LEVEL = 'repeatable read'
"""What PostgreSQL's ``SHOW transaction_isolation`` reports for that level."""

SERIALIZATION_FAILURE_SQLSTATE = '40001'
"""PostgreSQL ``serialization_failure``: under a REPEATABLE READ snapshot the target row was changed and committed concurrently; the transaction must be redone."""

MAX_REPEATABLE_READ_ATTEMPTS = 3
"""Default attempt limit (including the first) for REPEATABLE READ serialization conflicts.

Absorbs occasional contention without masking persistent contention (which
would multiply the load of a heavy transaction); on exhaustion
``SerializationRetryExhaustedError`` lets the caller report a conflict."""

SESSION_COMMIT_COUNT_KEY = '_commit_count'
"""Key in session.info: number of successful ``commit()`` calls on this session.

Read through :attr:`AsyncSession.commit_count`. Answers "did the operation
commit?": ``in_transaction()`` is useless for that (a re-query after commit
opens a new read transaction), and ``new/dirty/deleted`` are emptied by a
plain ``flush()``; only the commit count neither over- nor under-reports."""

POST_COMMIT_CALLBACKS_KEY = '_post_commit_callbacks'
"""Key in session.info: async side-effect callbacks to run only **after a real commit**.

For irreversible external side effects that may only happen once the DB
change is durable (e.g. deleting external objects after the row is deleted)
-- awaiting them inline would produce side effects for changes that are later
rolled back.

Contract: ``add_post_commit_callback`` registers; the next successful
``commit()`` of this session runs them in order **after** the commit and
clears the queue; ``rollback()`` / ``reset()`` / ``close()`` discard them
(the DB change they belong to was not persisted). Registration inside a
savepoint is rejected (a nested rollback cannot undo just that savepoint's
callbacks). A failing callback never blocks later callbacks or the commit
result (the commit already happened -- failures are only logged)."""


class RepeatableReadSnapshotConflictError(Exception):
    """A unique-constraint conflict under a REPEATABLE READ snapshot whose winner is invisible to this snapshot.

    Unique constraints are checked against the *committed current* state,
    while a REPEATABLE READ snapshot only sees commits made before the
    transaction began. When a concurrent transaction commits the same key
    after our snapshot, our INSERT hits the constraint (proving the winner
    exists), yet no SELECT in this transaction can see it -- this session
    cannot converge; only a new snapshot can. (Under READ COMMITTED this
    shape does not exist.)

    This is a **control-flow signal**, not a failure: domain code raises it
    after confirming "session is in REPEATABLE READ + unique conflict + the
    winner is not visible in the snapshot", and
    :meth:`SessionFactory.run_in_repeatable_read` treats it exactly like
    SQLSTATE ``40001`` -- roll back, discard the session, run again.
    """


class SerializationRetryExhaustedError(Exception):
    """A REPEATABLE READ transaction hit serialization conflicts on every attempt.

    Raised by :meth:`SessionFactory.run_in_repeatable_read` after
    ``max_attempts`` consecutive ``SQLSTATE 40001`` /
    :class:`RepeatableReadSnapshotConflictError` conflicts. Occasional
    conflicts are absorbed by the retries; exhausting them indicates
    persistent contention. Suggested HTTP mapping: **409** (a state conflict
    that may succeed later, not a server fault).

    Deadlocks (``40P01``) are deliberately **not** retried: a consistent lock
    order should make them impossible, and retrying would disguise a lock
    ordering bug as "occasionally slow".
    """
    status_code: int = 409


class AsyncSession(_AsyncSessionBase):
    """Cache-aware ``AsyncSession`` with post-commit callbacks and transaction helpers.

    All cache logic is delegated to ``CachedTableBaseMixin``'s protected
    helpers (``_xxx``) -- they are the framework-internal contract between
    this wrapper and cached_table; application code must not call them
    directly, hence the explicit reportPrivateUsage waivers below.
    """

    @property
    def commit_count(self) -> int:
        """Number of successful ``commit()`` calls on this session.

        The single source for "a commit was observed": record a baseline
        before a critical write and compare afterwards
        (``session.commit_count > baseline`` => the database committed; a
        sufficient, not necessary, signal).
        """
        return self.info.get(SESSION_COMMIT_COUNT_KEY, 0)

    def add_post_commit_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register an async side effect to run after the next real commit (contract: :data:`POST_COMMIT_CALLBACKS_KEY`).

        :raises RuntimeError: called inside a savepoint (a nested rollback
            could not precisely undo the registration).
        """
        if self.in_nested_transaction():
            raise RuntimeError(
                "post-commit callbacks cannot be registered inside a savepoint "
                "(a nested rollback could not undo them precisely)"
            )
        self.info.setdefault(POST_COMMIT_CALLBACKS_KEY, []).append(callback)

    async def set_local_timeouts(self, *, lock_timeout_ms: int, statement_timeout_ms: int) -> None:
        """Pin ``lock_timeout`` / ``statement_timeout`` to the **current transaction** (PostgreSQL only).

        Uses ``set_config(..., is_local=True)``, equivalent to ``SET LOCAL``:
        the values revert automatically when the transaction ends. Call it
        once at the start of a transaction (or after a rollback started a new
        one) to bound every following statement and lock wait.
        """
        _ = (await self.exec(
            select(func.set_config('lock_timeout', f"{lock_timeout_ms}ms", True)),
        )).one()
        _ = (await self.exec(
            select(func.set_config('statement_timeout', f"{statement_timeout_ms}ms", True)),
        )).one()

    async def enter_repeatable_read(self) -> None:
        """Raise this session's transaction to **REPEATABLE READ**, verify it, and record it in ``session.info`` (PostgreSQL only).

        For orchestration that reads the same source data across several
        statements and needs a transaction-wide snapshot.

        **Must be the first database action of the session**: when the
        connection is already established, SQLAlchemy only emits a
        ``SAWarning`` and silently ignores ``execution_options`` (the level
        stays READ COMMITTED). Therefore the level is read back with
        ``SHOW transaction_isolation`` and a mismatch raises -- turning a
        silent correctness loss into a loud failure. The read-back uses the
        raw connection (not a session statement), bypassing the raw-DML
        observation hooks.

        No built-in retry: a serialization failure requires discarding the
        whole session (see :meth:`SessionFactory.run_in_repeatable_read`);
        calling this directly means the caller owns retries.

        :raises RuntimeError: the isolation level could not be verified.
        """
        connection = await self.connection(
            execution_options={'isolation_level': REPEATABLE_READ_ISOLATION_LEVEL}
        )
        actual_level = await connection.scalar(text('SHOW transaction_isolation'))
        if actual_level != REPEATABLE_READ_REPORTED_LEVEL:
            raise RuntimeError(
                "could not raise the session to REPEATABLE READ (the isolation level setting was "
                f"silently ignored): expected {REPEATABLE_READ_REPORTED_LEVEL!r}, got {actual_level!r}. "
                "Most likely the session already executed SQL (the connection was established, so "
                "execution_options were ignored) -- enter_repeatable_read() must be the first "
                "database action of a new session."
            )
        self.info[SESSION_REPEATABLE_READ_KEY] = True

    async def commit(self, *, fail_soft_when_observed: bool = False) -> None:
        """Commit + synchronously invalidate every involved ``CachedTableBaseMixin`` model + run post-commit callbacks.

        Ordering: auto-register new/dirty/deleted cached models (covers bare
        add/mutate/delete + commit paths) -> pop the callback queue -> mark
        the session as inside the enhanced commit -> ``super().commit()``
        actually commits (its flush may register more pendings, e.g. cascade
        children via the ``persistent_to_deleted`` event); the ``after_commit``
        event pops the complete pending set and, because of the mark, hands
        it over to this method instead of scheduling the fire-and-forget
        fallback -> clear the mark, take the hand-over -> bump
        ``commit_count`` -> synchronously invalidate the hand-over (each
        pending exactly once) -> run the callbacks in order. Degrades to a
        plain commit when nothing is pending. If ``super().commit()`` raises
        after the database committed, the hand-over goes to the fallback
        compensation before the error propagates.

        :param fail_soft_when_observed: "observed-commit" completion mode.
            When ``True``: if ``super().commit()`` itself raises
            (``commit_count`` unchanged) the error propagates as-is -- the
            database may or may not have committed and the caller must treat
            it as uncertain. Every step **after** the commit (cache
            invalidation, each callback) is individually fail-soft (catches
            ``BaseException`` including cancellation, logs, continues), so no
            failure can drop the remaining already-popped callbacks.
            Default ``False``: errors raised by the invalidation step itself
            and cancellation propagate; callback ``Exception``\\s are logged
            and skipped.

        Redis failures are **not** among the propagated errors, in either
        mode: each cached model's invalidation catches and logs its own Redis
        error (the database has committed; failing the request would not undo
        it), and the affected cache entries converge when their TTL expires.
        Until then those entries can serve pre-commit data. Callers that need
        a hard guarantee must not rely on the cache for that read
        (``no_cache=True``).
        """
        CachedTableBaseMixin._autoregister_session_mutations(self)  # pyright: ignore[reportPrivateUsage]
        callbacks: list[Callable[[], Awaitable[None]]] = self.info.pop(POST_COMMIT_CALLBACKS_KEY, [])
        # While marked, the after_commit event hands its popped pendings to
        # this method instead of scheduling the fire-and-forget fallback.
        CachedTableBaseMixin._begin_enhanced_commit(self)  # pyright: ignore[reportPrivateUsage]
        try:
            await super().commit()
        except BaseException:
            # A non-empty hand-over means after_commit fired, i.e. the database
            # committed before the failure; the synchronous path below will not
            # run, so fall back to the fire-and-forget compensation.
            CachedTableBaseMixin._schedule_fallback_compensation(  # pyright: ignore[reportPrivateUsage]
                CachedTableBaseMixin._end_enhanced_commit(self),  # pyright: ignore[reportPrivateUsage]
                "super().commit() raised after the database committed",
            )
            raise
        committed = CachedTableBaseMixin._end_enhanced_commit(self)  # pyright: ignore[reportPrivateUsage]
        # Count right after the real commit and before invalidation/callbacks:
        # their failure does not change the fact that the database committed.
        self.info[SESSION_COMMIT_COUNT_KEY] = self.commit_count + 1
        if fail_soft_when_observed:
            try:
                await CachedTableBaseMixin._flush_invalidations(self, committed)  # pyright: ignore[reportPrivateUsage]
            except BaseException:
                logger.exception(
                    "cache invalidation after commit failed (including cancellation) -- the database "
                    "committed; suppressed (fail-soft), continuing with post-commit callbacks"
                )
            for callback in callbacks:
                try:
                    await callback()
                except BaseException:
                    logger.exception(
                        "post-commit callback failed (including cancellation) -- the database "
                        "committed; suppressed (fail-soft), continuing with the next callback"
                    )
            return
        await CachedTableBaseMixin._flush_invalidations(self, committed)  # pyright: ignore[reportPrivateUsage]
        for callback in callbacks:
            try:
                await callback()
            except Exception:
                # The commit already happened: log only, never roll back, and
                # do not block later callbacks.
                logger.exception("post-commit callback failed (the database change is committed)")

    async def rollback(self, *, best_effort_budget_seconds: float | None = None) -> None:
        """Roll back + discard pending post-commit callbacks (their DB change was not persisted).

        The callbacks are discarded in ``finally``: even if the native
        rollback raises, a reused session must not run stale callbacks on an
        unrelated later commit.

        :param best_effort_budget_seconds: bounded abort mode. The rollback
            runs under ``asyncio.timeout``; on failure or timeout the
            connection is ``invalidate()``d on a best-effort basis and **no
            exception is raised**. This is not a hard upper bound
            (``invalidate`` may still wait inside the driver); the residual
            risk is row locks held until the connection finally dies.
            ``CancelledError`` still propagates. Default ``None``: a normal
            rollback whose errors propagate.
        """
        if best_effort_budget_seconds is not None:
            try:
                async with asyncio.timeout(best_effort_budget_seconds):
                    await self.rollback()
            except Exception:
                logger.exception(
                    f"rollback failed / timed out (budget {best_effort_budget_seconds:.1f}s) -- "
                    "invalidating the connection on a best-effort basis"
                )
                try:
                    await self.invalidate()
                except Exception:
                    logger.exception("invalidate also failed -- the connection is left to server-side timeouts")
            return
        try:
            await super().rollback()
        finally:
            _ = self.info.pop(POST_COMMIT_CALLBACKS_KEY, None)

    def begin(self) -> '_PostCommitAwareSessionBegin':  # pyright: ignore[reportIncompatibleMethodOverride]  # returns an enhanced handle with the same protocols
        """Enhanced transaction context for ``async with session.begin():`` and ``await session.begin()``.

        The native ``AsyncSessionTransaction.__aexit__`` commits the
        underlying synchronous transaction directly, bypassing this class's
        ``commit()`` -- cache invalidation and post-commit callbacks would be
        silently skipped. This wrapper keeps the native entry semantics
        (including "a transaction is already begun" errors) and routes only
        the **exit** through the enhanced ``commit()`` / ``rollback()``.
        ``begin_nested()`` is unaffected (a savepoint release must not
        trigger outermost side effects).
        """
        return _PostCommitAwareSessionBegin(self)

    async def reset(self) -> None:
        """Reset the session (release transaction/connection) + clear tracking state in ``session.info``.

        Clears ``SESSION_FOR_UPDATE_KEY``, pending post-commit callbacks, the
        REPEATABLE READ marker and the cache-invalidation tracking keys. Done
        in ``finally`` so a failing native reset cannot leave stale state.
        The REPEATABLE READ marker **must** be cleared: the connection goes
        back to the pool, its isolation level is reset, and a stale marker
        would let ``@requires_repeatable_read`` pass on a READ COMMITTED
        transaction.
        """
        try:
            await super().reset()
        finally:
            self._clear_tracking_state()

    async def close(self) -> None:
        """Close the session + clear tracking state (symmetric with ``reset()``).

        A closed SQLAlchemy session is reusable, but ``close()`` fires only
        ``after_transaction_end`` -- not ``after_commit`` / ``after_rollback``
        -- so lock ids, callbacks or the REPEATABLE READ marker left behind
        would make the next transaction fail open.
        """
        try:
            await super().close()
        finally:
            self._clear_tracking_state()

    def _clear_tracking_state(self) -> None:
        """Drop every per-transaction tracking key from ``session.info`` (shared by ``reset()`` / ``close()``)."""
        self.info.pop(SESSION_FOR_UPDATE_KEY, None)
        _ = self.info.pop(POST_COMMIT_CALLBACKS_KEY, None)
        _ = self.info.pop(SESSION_REPEATABLE_READ_KEY, None)
        CachedTableBaseMixin._clear_session_cache_state(self)  # pyright: ignore[reportPrivateUsage]

    async def refresh(
            self,
            instance: object,
            attribute_names: Iterable[str] | None = None,
            with_for_update: Any = None,
    ) -> None:
        """Refresh an ORM instance -- delegates to the native ``session.refresh()``.

        ``refresh()`` means "discard in-memory changes and re-read from the
        database"; the native implementation is correct by construction
        (expire -> SELECT -> ``ObjectDeletedError`` if the row is gone). It
        never goes through the Redis cache.
        """
        await super().refresh(instance, attribute_names=attribute_names, with_for_update=with_for_update)

    # Raw-DML observation points. In SQLAlchemy's AsyncSession, ``exec()``
    # calls ``sync_session.exec``, ``scalar()`` calls ``sync_session.scalar``
    # and ``stream()`` / ``stream_scalars()`` call ``sync_session.execute``
    # -- none of them route through ``execute()`` (only ``scalars()`` does),
    # so each needs its own hook. Writes issued on a raw connection remain
    # invisible.
    #
    # The hooks are defined only at runtime. Type checkers keep seeing the
    # parent's overloaded signatures, so ``await session.exec(select(User))``
    # still infers ``ScalarResult[User]`` instead of collapsing to ``Any``;
    # the wide passthrough signatures below exist purely to forward arguments.

    if not TYPE_CHECKING:
        async def execute(self, statement, *args, **kwargs):
            """Pass ``execute`` through after registering raw-DML write tables (see ``CachedTableBaseMixin.register_raw_dml_write``)."""
            CachedTableBaseMixin.register_raw_dml_write(self, statement)
            return await super().execute(statement, *args, **kwargs)

        async def exec(self, statement, *args, **kwargs):
            """Pass ``exec`` (SQLModel's preferred entry point) through after registering raw-DML write tables."""
            CachedTableBaseMixin.register_raw_dml_write(self, statement)
            return await super().exec(statement, *args, **kwargs)

        async def scalar(self, statement, *args, **kwargs):
            """Pass ``scalar`` through after registering raw-DML write tables."""
            CachedTableBaseMixin.register_raw_dml_write(self, statement)
            return await super().scalar(statement, *args, **kwargs)

        async def stream(self, statement, *args, **kwargs):
            """Pass ``stream`` through after registering raw-DML write tables."""
            CachedTableBaseMixin.register_raw_dml_write(self, statement)
            return await super().stream(statement, *args, **kwargs)

        async def stream_scalars(self, statement, *args, **kwargs):
            """Pass ``stream_scalars`` through after registering raw-DML write tables."""
            CachedTableBaseMixin.register_raw_dml_write(self, statement)
            return await super().stream_scalars(statement, *args, **kwargs)


class _PostCommitAwareSessionBegin:
    """Enhanced transaction handle returned by :meth:`AsyncSession.begin`.

    Faithfully supports both native ``AsyncSessionTransaction`` protocols:

    - ``await session.begin()``: delegates to the native start (begins the
      transaction) and returns **the session itself**, whose ``commit`` /
      ``rollback`` are already the enhanced ones. (Returning the native
      handle would let ``tx.commit()`` bypass invalidation and callbacks.)
    - ``async with session.begin():``: delegates entry to the native handle
      and routes the **exit** through the enhanced ``commit()`` /
      ``rollback()``.

    Explicit commit/rollback inside the block: the exit first checks
    ``in_transaction()``; if the transaction was already closed it does not
    commit again (that would raise) and only runs the native exit to restore
    internal references.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session: AsyncSession = session
        self._inner: Any = _AsyncSessionBase.begin(session)

    def __await__(self) -> Generator[Any, Any, AsyncSession]:
        async def _start() -> AsyncSession:
            _ = await self._inner
            return self._session
        return _start().__await__()

    async def __aenter__(self) -> AsyncSession:
        _ = await self._inner.__aenter__()
        return self._session

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._session.in_transaction():
            # Already committed/rolled back inside the block (the enhanced
            # path ran there): only restore native internals.
            await self._inner.__aexit__(exc_type, exc, tb)
            return
        primary: BaseException | None = None
        try:
            if exc_type is not None:
                await self._session.rollback()
            else:
                await self._session.commit()
        except BaseException as e:
            primary = e
        if primary is None:
            # Transaction closed by the enhanced commit/rollback; the native
            # exit only restores references (and does not commit twice).
            await self._inner.__aexit__(exc_type, exc, tb)
            return
        # The enhanced commit/rollback failed: pass the *primary* exception to
        # the native exit so it takes its error branch (with the block's
        # original None args it would try to commit again). A cleanup failure
        # is only logged; the primary exception is re-raised.
        try:
            await self._inner.__aexit__(type(primary), primary, primary.__traceback__)
        except BaseException:
            logger.exception("native cleanup of the begin() context failed (the primary exception is re-raised)")
        raise primary


class SessionFactory(async_sessionmaker[AsyncSession]):
    """Session factory -- an ``async_sessionmaker`` subclass carrying cross-session orchestration.

    Calling it still yields an :class:`AsyncSession` (pass
    ``class_=AsyncSession``). It additionally provides
    :meth:`run_in_repeatable_read`: retrying on serialization conflicts
    requires **discarding the whole session**, which a session cannot do for
    itself. Being a subclass, it can be passed wherever an
    ``async_sessionmaker[AsyncSession]`` is expected.
    """

    async def run_in_repeatable_read(
            self,
            operation: Callable[[AsyncSession], Awaitable[_T]],
            *,
            description: str,
            max_attempts: int = MAX_REPEATABLE_READ_ATTEMPTS,
    ) -> _T:
        """Run ``operation`` in its **own REPEATABLE READ session**; redo it entirely on serialization conflicts (PostgreSQL only).

        Each attempt: new session -> :meth:`AsyncSession.enter_repeatable_read`
        (set + verify) -> ``operation(session)`` -> check it committed ->
        return. On ``SQLSTATE 40001`` (serialization_failure) or
        :class:`RepeatableReadSnapshotConflictError`, roll back, **discard the
        session** and run ``operation`` again from scratch; after
        ``max_attempts`` raise :class:`SerializationRetryExhaustedError`.

        Contract for ``operation`` (each item is a correctness requirement):

        1. **It must commit itself.** A session that exits without a commit
           rolls back; if ``commit_count`` did not grow, ``RuntimeError`` is
           raised instead of returning a "successful" result for data that
           was never persisted. (Re-querying after the commit is fine -- the
           check counts commits, not open transactions.)
        2. **It must be re-runnable**: capture only immutable inputs (ids,
           DTOs, scalars). Never capture ORM instances, the session,
           generated ids or mutable containers from a previous attempt --
           they belong to a rolled-back transaction.
        3. **Authorization belongs inside it** too: a retry re-runs the whole
           business action on a new snapshot.

        The return value must not be an ORM instance bound to the session
        (it is closed when this method returns); build a DTO inside
        ``operation``.

        Only attempts that have **not committed** are re-run: if a conflict
        surfaces *after* this attempt's commit (a re-query / post-processing
        is also a database action), a re-run would repeat an already
        persisted business action, so the error propagates unchanged.
        Deadlocks (``40P01``) are never retried.

        :param operation: re-runnable orchestration taking this attempt's session
        :param description: action name used in retry / exhaustion log messages
        :param max_attempts: total attempts including the first, must be >= 1
        :raises SerializationRetryExhaustedError: ``max_attempts`` consecutive conflicts
        :raises RuntimeError: ``operation`` returned without committing
        :raises ValueError: ``max_attempts < 1``
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        last_error: DBAPIError | RepeatableReadSnapshotConflictError | None = None
        for attempt in range(1, max_attempts + 1):
            async with self() as session:
                await session.enter_repeatable_read()
                commits_before: int = session.commit_count
                try:
                    result = await operation(session)
                except RepeatableReadSnapshotConflictError as e:
                    if session.commit_count > commits_before:
                        # Committed already: a rollback cannot undo it and a
                        # re-run would repeat the business action.
                        raise
                    last_error = e
                    await session.rollback()
                    logger.warning(
                        f"{description}: REPEATABLE READ snapshot conflict (a unique conflict proves a "
                        f"concurrent winner committed but it is invisible to this snapshot); discarding "
                        f"the session and retrying: attempt={attempt}/{max_attempts}"
                    )
                    continue
                except DBAPIError as e:
                    orig = e.orig
                    # asyncpg's SerializationError reaches us wrapped as a
                    # generic DBAPIError (no distinguishable subclass), so the
                    # only reliable signal is the ``sqlstate`` the dialect
                    # adapter sets on every PostgreSQL error. No attribute =
                    # not a PostgreSQL error = not retryable.
                    sqlstate = getattr(orig, 'sqlstate', None) if orig is not None else None
                    if sqlstate != SERIALIZATION_FAILURE_SQLSTATE:
                        raise
                    if session.commit_count > commits_before:
                        raise
                    last_error = e
                    await session.rollback()
                    logger.warning(
                        f"{description}: serialization failure (SQLSTATE {SERIALIZATION_FAILURE_SQLSTATE}); "
                        f"discarding the session and retrying: attempt={attempt}/{max_attempts}"
                    )
                    continue
                if session.commit_count == commits_before:
                    raise RuntimeError(
                        f"{description}: the operation returned without committing. "
                        "run_in_repeatable_read requires the operation to commit itself; "
                        "otherwise the session rolls back on exit and the writes are lost"
                    )
                return result

        raise SerializationRetryExhaustedError(
            f"{description}: serialization conflict on {max_attempts} consecutive attempts "
            f"(SQLSTATE {SERIALIZATION_FAILURE_SQLSTATE} or REPEATABLE READ snapshot conflict); giving up"
        ) from last_error
