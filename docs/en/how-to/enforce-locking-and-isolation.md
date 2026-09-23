# Enforce row locks and isolation levels

**Goal**: turn "this method must be called while holding a row lock / under a specific isolation level" from a convention in a comment into a contract **enforced at runtime**; attach "side effects that may only happen after a successful commit" to after the commit; put upper bounds on lock waits and statements within a transaction.

**Prerequisites**:

- The session is constructed from the enhanced `sqlmodel_ext.AsyncSession`: `SessionFactory(engine, class_=AsyncSession)`
- For `run_in_repeatable_read` / `requires_read_committed` / `set_local_timeouts`: PostgreSQL

All contract decorators are **fail-closed**: when a guard cannot find the session it raises instead of silently letting the call through — "the guard is broken" must look different from "the guard passed".

## 1. Read-modify-write: `with_for_update`

```python
account = await Account.get(session, Account.id == account_id, with_for_update=True)
account.balance += amount
account = await account.save(session)
```

- Generates `SELECT ... FOR UPDATE`; the row lock is held until the end of the transaction.
- **Forces `populate_existing`** (cannot be turned off): the database returns the latest row, but if the object is already in the identity map, SQLAlchemy by default hands you the **stale object** — the following read-modify-write is then a lost update. A locking read always overwrites it with the database values.
- On cached models, a locking read automatically bypasses Redis.
- `get_one(..., with_for_update=True)` and `get_exist_one(..., with_for_update=True)` work the same way; the latter is often used to close the TOCTOU window between "existence check → delete".

### Work queues: `skip_locked`

N workers each claim **different** candidate rows instead of queueing on the same row:

```python
job = await Job.get(
    session,
    Job.status == 'queued',
    order_by=[col(Job.created_at)],
    with_for_update=True,
    skip_locked=True,     # FOR UPDATE SKIP LOCKED
)
if job is None:
    return   # either there are no jobs, or all jobs are locked by other workers
```

::: warning Don't use `skip_locked` for existence checks
Returning 0 rows means both "there are none" and "they are all locked by someone else". `skip_locked` only takes effect with `with_for_update=True`.
:::

### Authorization reads: `authoritative`

When the result is used to decide "whether something is allowed", you must read the authoritative value — not a possibly stale object in the identity map, and not the Redis cache:

```python
member = await Membership.get_one(session, membership_id, authoritative=True)
if member.role != 'admin':
    raise HTTPException(403)
```

At this layer `authoritative=True` is equivalent to `populate_existing=True`; cached models additionally bypass Redis. The caller passes only this one parameter, and each model knows how many layers it has to bypass. It does not lock — use `with_for_update` when you need a lock.

## 2. Require `self` to be locked: `@requires_for_update`

```python
from decimal import Decimal
from sqlmodel_ext import RelationPreloadMixin, requires_for_update, SignedDecimal38_18
from sqlmodel_ext.session import AsyncSession

class Account(SQLModelBase, UUIDTableBaseMixin, RelationPreloadMixin, table=True):
    balance: SignedDecimal38_18 = Decimal('0')

    @requires_for_update
    async def adjust_balance(self, session: AsyncSession, *, amount: Decimal) -> None:
        self.balance += amount
        await self.save(session, commit=False, refresh=False)
```

```python
account = await Account.get_one(session, account_id)
await account.adjust_balance(session, amount=Decimal('5'))   # RuntimeError: requires a FOR UPDATE locked instance

account = await Account.get_one(session, account_id, with_for_update=True)
await account.adjust_balance(session, amount=Decimal('5'))   # OK
await session.commit()
```

Lock tracking follows the transaction: it is cleared after the outermost commit / rollback (the locks have been released); a savepoint rollback restores the set as it was on entering the savepoint (PostgreSQL releases locks acquired inside the savepoint); a savepoint release keeps it; `reset()` / `close()` clear it. So calling `adjust_balance` again after the commit fails again — which is exactly what you want.

The decorator is declared with `ParamSpec`, so the decorated method's signature stays visible to type checkers.

## 3. Require instances in parameters to be locked: `@requires_locked_param`

`@requires_for_update` can only express "`self` is locked". For classmethods / plain functions / `@asynccontextmanager` factories that operate on a **batch** of instances, use the parameterized version:

```python
from sqlmodel_ext.mixins import requires_locked_param

class Account(...):
    @classmethod
    @requires_locked_param('accounts')
    async def settle(cls, session: AsyncSession, accounts: list['Account']) -> None:
        for account in accounts:
            account.balance = Decimal('0')
        await session.flush()
```

```python
accounts = await Account.get(session, Account.owner_id == owner_id, fetch_mode='all', with_for_update=True)
await Account.settle(session, accounts)      # OK
```

- Parameter is `None` → the check is skipped; a single instance → treated as a one-element sequence; **empty sequence → `RuntimeError`**
- The session is always taken from the decorated function's `session` parameter; no such parameter → `RuntimeError`
- When decorating an `@asynccontextmanager` factory, the check runs when the context manager is **created** (before entering)

Instances obtained mid-execution (e.g. returned by a callback) are checked again with the same validation core:

```python
from sqlmodel_ext.mixins import validate_locked_instances

validate_locked_instances(session, returned_accounts, context="batch settle")
```

## 4. Isolation-level contracts

### Need a consistent snapshot: REPEATABLE READ

When reading the same set of source data across multiple statements (e.g. a deep copy that reads nodes, files and edges one after another), each statement under READ COMMITTED gets a new snapshot, and concurrent commits can make those reads inconsistent with each other.

```python
from sqlmodel_ext.mixins import requires_repeatable_read

class Board(...):
    @requires_repeatable_read
    async def clone(self, session: AsyncSession) -> 'BoardCloneResult':
        ...  # all reads see the same transaction snapshot
```

The decorator only checks the RR marker in `session.info`, and that marker is written **only** by `enter_repeatable_read()` after it has **read back** the level from the database to confirm it (it is cleared on `reset()` / `close()`) — "I requested it" is never recorded as "it is in effect". The correct way to enter RR is at the outermost entry point:

```python
from sqlmodel_ext.session import SessionFactory

session_factory = SessionFactory(engine, class_=AsyncSession)

async def _clone(session: AsyncSession) -> BoardCloneResult:
    board = await Board.get_one(session, board_id)          # capture only immutable inputs (board_id)
    result = await board.clone(session)
    await session.commit()                                   # you must commit yourself
    return result                                            # return a DTO, not an ORM instance

result = await session_factory.run_in_repeatable_read(_clone, description="clone board")
```

`run_in_repeatable_read` opens a **new session** for each attempt, calls `enter_repeatable_read()`, and runs `operation`; on SQLSTATE `40001` or `RepeatableReadSnapshotConflictError` it rolls back, discards the session and reruns from scratch, and after `max_attempts` (default 3) consecutive conflicts it raises `SerializationRetryExhaustedError` (`status_code = 409`). `operation` returning without committing → `RuntimeError` (otherwise you'd get a "successful" result that was never persisted). Conflicts that only appear after the commit are not rerun. Deadlocks `40P01` are deliberately not retried — consistent lock ordering should make them impossible, and retrying would only disguise a lock-ordering bug as "occasionally slow".

::: tip The "invisible winner" in unique-constraint conflicts
Under REPEATABLE READ, unique constraints are checked against the **currently committed** state, while the snapshot only sees commits from before the transaction started. When a concurrent transaction commits the same key after your snapshot, your INSERT hits the constraint (proving the winner exists), yet no SELECT in this transaction can see it. In that case raise `RepeatableReadSnapshotConflictError` in your domain code, and `run_in_repeatable_read` will rerun with a fresh snapshot just as it does for `40001`.
:::

When calling `enter_repeatable_read()` directly, it must be the **first** database action on a new session; otherwise SQLAlchemy only emits a `SAWarning` and silently keeps the original level — here it reads the level back and raises `RuntimeError`, turning a silent correctness loss into a loud failure.

### Need to see concurrent commits: READ COMMITTED

When you rely on "a new snapshot per statement" to see rows other processes just committed (e.g. re-checking a handoff window):

```python
from sqlmodel_ext.mixins import requires_read_committed

class Handoff(...):
    @classmethod
    @requires_read_committed          # place it below @classmethod (innermost)
    async def load_ready(cls, session: AsyncSession) -> list['Handoff']:
        ...
```

It runs `SHOW transaction_isolation` **live on every call** instead of looking at the `session.info` marker — "no RR marker" could also mean SERIALIZABLE, or an isolation level changed directly on the connection / engine, and trusting the marker would fail open. PostgreSQL only.

## 5. Side effects that run only after commit: post-commit callbacks

After deleting a database row you need to delete the file in object storage — if you `await` the file deletion inline and the transaction later rolls back, the file is gone but the row is still there. Attach irreversible external side effects to after the commit:

```python
async def delete_blob() -> None:
    await storage.delete(blob_key)          # capture immutable values, not ORM instances

session.add_post_commit_callback(delete_blob)
await Attachment.delete(session, attachment)    # delete_blob runs only after the commit succeeds
```

- Run in registration order after the next **real** commit, then the queue is cleared
- `rollback()` / `reset()` / `close()` discard callbacks that haven't run
- Registering inside a savepoint → `RuntimeError` (a savepoint rollback cannot precisely undo it)
- A failing callback is only logged; it does not block later callbacks or change the commit result
- Exiting `async with session.begin():` also goes through the enhanced `commit()`, so callbacks run as usual

When you need a signal that "the database has definitely committed", look at `commit_count`, not `in_transaction()` (re-querying after a commit opens a new read transaction):

```python
baseline = session.commit_count
await do_critical_write(session)
if session.commit_count > baseline:
    ...   # the database has committed
```

When the final stretch of side effects must be "best effort, and none may be lost", use `await session.commit(fail_soft_when_observed=True)`: a failure of the commit itself is raised as usual (the outcome is uncertain; the caller treats it as "possibly committed"); the cache invalidation and each callback **after** the commit are individually fault-tolerant, and not even cancellation makes the remaining already-popped callbacks get lost.

## 6. Put bounds on transactions

### Lock wait and statement timeouts

```python
await session.set_local_timeouts(lock_timeout_ms=2_000, statement_timeout_ms=10_000)
```

Uses `set_config(..., is_local=True)` (equivalent to `SET LOCAL`) to pin them to the **current transaction**; they are restored automatically when the transaction ends. Call it once at the start of the transaction (call it again after a rollback has started a new transaction). PostgreSQL only.

### Bounded rollback

A rollback in a cleanup path (e.g. cancellation handling, shutdown) can itself hang. Give it a budget:

```python
await session.rollback(best_effort_budget_seconds=5.0)
```

Rolls back under `asyncio.timeout`; on failure or timeout it makes a best-effort `invalidate()` of the connection and **does not raise** (`CancelledError` still propagates). This is not a hard upper bound — `invalidate()` can still wait inside the driver; the residual risk is row locks being held until the connection finally drops. The default (not passed) is a normal rollback, with errors propagating as usual.

## Related reference

- [Decorators and enhanced session reference](/en/reference/decorators)
- [`with_for_update` / `skip_locked` / `authoritative` of `get()`](/en/reference/crud-methods#get)
- [Handle concurrent updates (optimistic locking)](./handle-concurrent-updates)
