"""
04 -- Pessimistic locking contracts, post-commit callbacks and isolation helpers.

Run::

    python examples/04_locking_and_isolation.py

What SQLite can demonstrate (and this script asserts):

* ``get(..., with_for_update=True)`` records the returned instances as locked
  in ``session.info``; SQLite simply omits the ``FOR UPDATE`` clause, but the
  bookkeeping is identical to PostgreSQL.
* ``@requires_for_update`` turns "the caller forgot to lock the row" into a
  loud ``RuntimeError`` instead of a lost update under concurrency.
* ``@requires_locked_param('accounts')`` does the same for a batch argument.
* The lock record is cleared on commit: a method cannot reuse a lock from a
  previous transaction.
* ``session.add_post_commit_callback()`` runs an irreversible side effect only
  after a real commit; a rollback discards it. ``session.commit_count`` tells
  whether a commit actually happened.

PostgreSQL-only helpers (not executed here -- they issue ``SHOW`` /
``set_config``):

* ``await session.set_local_timeouts(lock_timeout_ms=..., statement_timeout_ms=...)``
  bounds lock waits for the current transaction only.
* ``SessionFactory(engine, class_=AsyncSession).run_in_repeatable_read(op, description=...)``
  runs ``op(session)`` in its own REPEATABLE READ session and re-runs it from
  scratch on SQLSTATE 40001 (up to ``max_attempts``, default 3), raising
  ``SerializationRetryExhaustedError`` (409) when contention persists.
  ``op`` must commit itself and must capture only immutable inputs.
* ``@requires_repeatable_read`` / ``@requires_read_committed`` guard methods
  that depend on a specific isolation level (fail-closed).
* ``get(..., with_for_update=True, skip_locked=True)`` lets N workers each
  claim a different queue row.
"""
import asyncio

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, col

from sqlmodel_ext import (
    AsyncSession,
    NonNegativeInt,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
    requires_for_update,
    requires_locked_param,
)


class Account(SQLModelBase, UUIDTableBaseMixin, table=True):
    owner: Str64
    """Account owner."""

    balance: NonNegativeInt
    """Balance in cents."""

    @requires_for_update
    async def withdraw(self, session: AsyncSession, *, amount: int) -> None:
        """Read-modify-write: only safe on a row locked by this transaction."""
        if amount > self.balance:
            raise ValueError("insufficient balance")
        self.balance -= amount
        session.add(self)

    @classmethod
    @requires_locked_param('accounts')
    async def pool(cls, session: AsyncSession, accounts: list['Account']) -> int:
        """Move every balance into the first account; all rows must be locked by the caller."""
        total = sum(account.balance for account in accounts)
        for account in accounts:
            account.balance = 0
        accounts[0].balance = total
        session.add_all(accounts)
        return total


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        alice = await Account(owner="alice", balance=1_000).save(session)
        alice_id = alice.id

        # --- forgetting the lock fails loudly -------------------------------
        unlocked = await Account.get(session, col(Account.id) == alice_id, fetch_mode='one')
        try:
            await unlocked.withdraw(session, amount=100)
        except RuntimeError as exc:
            assert "FOR UPDATE" in str(exc)
        else:
            raise AssertionError("withdraw() on an unlocked row must raise")

        # --- the correct shape: lock, mutate, commit ------------------------
        locked = await Account.get(
            session, col(Account.id) == alice_id, fetch_mode='one', with_for_update=True,
        )
        await locked.withdraw(session, amount=100)
        commits_before = session.commit_count
        await session.commit()
        assert session.commit_count == commits_before + 1

        # The lock ended with the transaction -- the old instance is no longer "locked".
        try:
            await locked.withdraw(session, amount=1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a lock must not outlive its transaction")

        # --- batch contract ---------------------------------------------------
        bob = await Account(owner="bob", balance=0).save(session)
        unlocked_rows = await Account.get(session, fetch_mode='all')
        try:
            _ = await Account.pool(session, unlocked_rows)
        except RuntimeError:
            pass
        else:
            raise AssertionError("pool() on unlocked rows must raise")
        rows = await Account.get(session, fetch_mode='all', with_for_update=True)
        assert await Account.pool(session, rows) == 900
        await session.rollback()  # discard the demo mutation
        _ = await Account.delete(session, bob)

        # --- post-commit side effects ----------------------------------------
        sent: list[str] = []

        async def notify() -> None:
            sent.append("balance changed")

        session.add_post_commit_callback(notify)
        await session.rollback()
        assert sent == [], "a rolled-back change must not trigger its side effect"

        locked = await Account.get(
            session, col(Account.id) == alice_id, fetch_mode='one', with_for_update=True,
        )
        await locked.withdraw(session, amount=50)
        session.add_post_commit_callback(notify)
        await session.commit()
        assert sent == ["balance changed"], "callbacks run only after a real commit"

        final = await Account.get(session, col(Account.id) == alice_id, fetch_mode='one')
        assert final.balance == 850

    await engine.dispose()
    print("[OK] 04_locking_and_isolation: lock contracts and post-commit callbacks enforced")


if __name__ == "__main__":
    asyncio.run(main())
