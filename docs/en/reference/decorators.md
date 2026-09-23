# Decorators & helpers

::: tip
This is reference documentation. To learn how to use `@requires_relations` to fix MissingGreenlet, see [Prevent MissingGreenlet errors](/en/how-to/prevent-missing-greenlet); to learn how to combine the lock and isolation-level contracts, see [Enforce row locks and isolation levels](/en/how-to/enforce-locking-and-isolation).
:::

## `@requires_relations`

```python
from sqlmodel_ext import requires_relations
```

```python
def requires_relations(
    *relations: str | QueryableAttribute[Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]
```

| Parameter | Type | Meaning |
|------|------|------|
| `*relations` | `str` | Name of a direct relationship attribute on this class (e.g. `'profile'`) |
| `*relations` | `QueryableAttribute` | Nested relationship (e.g. `Generator.config`) |

**Preconditions**:

- The decorated class must inherit `RelationPreloadMixin`
- The decorated method must be an `async def` (plain coroutine) or `async def ... yield` (async generator)
- One of the method's parameters is named `session`, or one of its kwargs is of type `AsyncSession`

**Runtime behavior**:

1. Extract the `AsyncSession` from the arguments
2. Call `self.ensure_relations_loaded(session, relations)` to incrementally load missing relationships
3. Execute the original method

When no session is found, preloading is **skipped** (no error) — callers of such a method must call `ensure_relations_loaded()` themselves first.

**Import-time validation**: `RelationPreloadMixin.__init_subclass__` checks string relationship names at class definition time; a nonexistent one raises `AttributeError`.

**Attached attribute**: the `_required_relations` tuple.

## Transaction-contract decorators

All four decorators below are **fail-closed**: whenever the guard can't find a session (for example an inner decorator didn't use `@wraps`, so `inspect.signature` can't see the `session` parameter), it raises instead of silently letting the call through.

| Decorator | Contract | How it checks |
|------|------|------|
| `@requires_for_update` | `self` must have been obtained via `get(with_for_update=True)` | Whether `id(self)` is in `session.info[SESSION_FOR_UPDATE_KEY]` |
| `@requires_locked_param(name)` | Every instance in parameter `name` must hold a FOR UPDATE lock | Same as above, checked one by one |
| `@requires_repeatable_read` | Must run inside a **verified** REPEATABLE READ transaction | The RR marker in `session.info` (written by `enter_repeatable_read()` after reading the level back to confirm it) |
| `@requires_read_committed` | Must run inside a READ COMMITTED transaction | A live `SHOW transaction_isolation` on every call (PostgreSQL only) |

### `@requires_for_update`

```python
from sqlmodel_ext import requires_for_update
```

```python
def requires_for_update(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[Any, Any, R]]
```

Declared with `ParamSpec`, so it **preserves the decorated method's signature** — the type checker keeps validating call arguments.

**Runtime behavior**: extracts the session from the arguments (not found → `RuntimeError`); `id(self)` not in the lock set → `RuntimeError`; otherwise executes the original method.

**Attached attribute**: `_requires_for_update = True` (for static analysis).

Lifecycle of the lock set: cleared on the outermost commit / rollback (row locks are released with the transaction); a savepoint rollback restores the snapshot taken when entering the savepoint (PostgreSQL releases locks acquired inside the savepoint); a savepoint release keeps it (the locks transfer to the outer transaction); the enhanced session's `reset()` / `close()` also clear it.

### `@requires_locked_param`

```python
from sqlmodel_ext.mixins import requires_locked_param
```

```python
def requires_locked_param(
    param_name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]
```

The parameterized version of `@requires_for_update`, for classmethods, `@asynccontextmanager` factories and other callables that operate on a **batch** of instances.

- Parameter value is `None` → the check is skipped (e.g. a condition-driven branch)
- A single instance → normalized to a one-element sequence
- An empty sequence → `RuntimeError` (claiming "these instances are locked" while passing none violates the contract)
- The session is always taken from the decorated function's `session` parameter; no such parameter → `RuntimeError`
- Supports two shapes: coroutine functions; synchronous factories (e.g. the product of `@asynccontextmanager` — the check runs when the context manager is **created**, i.e. before entering it)

**Attached attribute**: `_requires_locked_param = param_name`.

### `validate_locked_instances()`

```python
from sqlmodel_ext.mixins import validate_locked_instances
```

```python
def validate_locked_instances(
    session: AsyncSession,
    instances: Sequence[Any],
    *,
    context: str,
) -> None
```

The validation core of `@requires_locked_param`, exported separately for re-validation **mid-execution** (the decorator can only check parameters at entry and can't see, for example, instances returned by a callback). An empty sequence or any unlocked instance → `RuntimeError`, with a message starting with `context`.

### `@requires_repeatable_read`

```python
from sqlmodel_ext.mixins import requires_repeatable_read
```

```python
def requires_repeatable_read(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[Any, Any, R]]
```

For orchestration that reads the same set of source data across multiple statements and needs a transaction-level snapshot (e.g. a deep copy that reads nodes, files and edges in turn). The session is located via `Signature.bind`; instance methods, classmethods and plain functions are all supported.

- No RR marker → `RuntimeError`, advising you to orchestrate from the outermost entry point with `SessionFactory.run_in_repeatable_read(...)`
- **No built-in retry**: a serialization failure (SQLSTATE `40001`) requires discarding the whole session, which the decorated method can't do; retrying belongs to the outermost entry point

**Attached attribute**: `_requires_repeatable_read = True`.

### `@requires_read_committed`

```python
from sqlmodel_ext.mixins import requires_read_committed
```

```python
def requires_read_committed(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[Any, Any, R]]
```

For cross-process coordination that relies on "a new snapshot per statement" to see concurrent commits (e.g. re-checking a handoff window).

It **deliberately does not read** the `session.info` marker: the absence of an RR marker could also mean SERIALIZABLE, or an isolation level set directly on the connection / engine — trusting the marker would fail open. Every call reads the level back with `SHOW transaction_isolation` (one cheap round trip); anything other than `'read committed'` → `RuntimeError`. **PostgreSQL only**.

When decorating a classmethod, place it below `@classmethod` (innermost).

**Attached attribute**: `_requires_read_committed = True`.

## `rel()`

```python
from sqlmodel_ext import rel
```

```python
def rel(relationship: object) -> QueryableAttribute[Any]
```

Asserts the type of a SQLModel `Relationship` field as `QueryableAttribute` so basedpyright doesn't report a type error. Input is a `QueryableAttribute` → returns the original object; otherwise → `AttributeError` (e.g. an instance attribute was passed by mistake).

**Typical usage**: `load=rel(User.profile)`, `load=[rel(User.profile), rel(Profile.avatar)]`.

## `cond()`

```python
from sqlmodel_ext import cond
```

```python
def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]
```

Narrows a column comparison expression (inferred as `bool` by basedpyright) to `ColumnElement[bool]`, so the `&` / `|` operators pass type checking. At runtime it is equivalent to `cast(ColumnElement[bool], expr)`.

```python
scope = cond(UserFile.user_id == current_user.id)
condition = scope & cond(UserFile.status == FileStatusEnum.uploaded)
```

## Enhanced `AsyncSession` {#session-reset}

```python
from sqlmodel_ext import AsyncSession
from sqlmodel_ext.session import SessionFactory
```

`sqlmodel_ext.AsyncSession` is a subclass of sqlmodel's `AsyncSession` and the canonical session type of this library. Construct it with `SessionFactory(engine, class_=AsyncSession)` (when you need `run_in_repeatable_read`) or `async_sessionmaker(engine, class_=AsyncSession)`. When no cached models are involved, every hook degrades to upstream behavior.

| Member | Description |
|------|------|
| `commit(*, fail_soft_when_observed=False)` | Before commit, automatically registers changes to all cached models (including bare `session.add()` / attribute changes / `session.delete()`); after commit, invalidates **synchronously**, then runs the post-commit callbacks in order. `fail_soft_when_observed=True`: a failure of the commit itself is raised as usual; every step **after** the commit (invalidation, each callback) is individually fault-tolerant (catches `BaseException` including cancellation, logs, continues). Default: invalidation errors propagate; a callback's `Exception` is logged and skipped, cancellation propagates |
| `commit_count` (property) | Number of successful commits on this session. The single source of truth for "was it committed": record a baseline before a critical write, and `commit_count > baseline` afterwards means the database has committed (a sufficient but not necessary signal) |
| `add_post_commit_callback(callback)` | Registers an `async` zero-argument callback, run **only after the next real commit**; discarded by `rollback()` / `reset()` / `close()`; registering inside a savepoint → `RuntimeError`; a failing callback doesn't block later callbacks nor change the commit result |
| `rollback(*, best_effort_budget_seconds=None)` | Rolls back + discards pending callbacks. Passing a budget enters **bounded abandon** mode: rolls back under `asyncio.timeout`, and on failure or timeout does a best-effort `invalidate()` of the connection, **without raising** (`CancelledError` still propagates). Not a hard cap (`invalidate` may still wait inside the driver) |
| `begin()` | Exiting `async with session.begin():` goes through the enhanced `commit()` / `rollback()` (the native context manager would commit the underlying transaction directly, skipping invalidation and callbacks); `await session.begin()` returns the session itself. `begin_nested()` is unaffected |
| `reset()` / `close()` | Release the transaction and connection, and in a `finally` clear the FOR UPDATE lock tracking, pending callbacks, the REPEATABLE READ marker and cache tracking state |
| `refresh(instance, attribute_names=None, with_for_update=None)` | Delegates to the native `refresh()`: must read the database, **never uses the cache** |
| `execute` / `exec` / `scalar` / `stream` / `stream_scalars` | Before passing through, call `CachedTableBaseMixin.register_raw_dml_write()` to register tables written by raw DML |
| `set_local_timeouts(*, lock_timeout_ms, statement_timeout_ms)` | Uses `set_config(..., is_local=True)` (equivalent to `SET LOCAL`) to pin both timeouts to the **current transaction**; they revert automatically when the transaction ends. PostgreSQL only |
| `enter_repeatable_read()` | Raises this session's transaction to REPEATABLE READ and, **after reading it back to verify**, records a marker in `session.info`. Must be the first database action of a fresh session (otherwise SQLAlchemy only emits a `SAWarning` and silently keeps the original level — this reads it back and raises `RuntimeError`). No built-in retry. PostgreSQL only |

### Object state after `session.reset()`

Typical scenario: an endpoint / task needs long external I/O midway, so it first calls `session.reset()` to release the DB connection. See [Release the database connection during long I/O](/en/how-to/release-connection-during-long-io).

- All ORM objects become **detached**
- **Already-loaded scalar fields are not expired**, so accessing them is safe (no SQL triggered); later commits can no longer expire them either
- Accessing relationship fields that weren't preloaded raises
- Writes require re-fetching an attached instance with `Model.get()` first
- Any subsequent `await Model.get/save` automatically checks out a new connection from the pool

## `SessionFactory`

```python
from sqlmodel_ext.session import (
    SessionFactory,
    RepeatableReadSnapshotConflictError,
    SerializationRetryExhaustedError,
    MAX_REPEATABLE_READ_ATTEMPTS,
)
```

A subclass of `async_sessionmaker[AsyncSession]`; can be passed anywhere an `async_sessionmaker[AsyncSession]` is expected.

```python
async def run_in_repeatable_read(
    self,
    operation: Callable[[AsyncSession], Awaitable[T]],
    *,
    description: str,
    max_attempts: int = MAX_REPEATABLE_READ_ATTEMPTS,   # 3
) -> T
```

Each attempt: new session → `enter_repeatable_read()` → `operation(session)` → check it committed → return. On SQLSTATE `40001` or `RepeatableReadSnapshotConflictError` it rolls back, **discards the session**, and reruns from scratch; `max_attempts` consecutive conflicts → `SerializationRetryExhaustedError` (`status_code = 409`). PostgreSQL only.

The contract for `operation`:

1. **It must commit itself** — returning without a commit raises `RuntimeError` rather than returning a "successful" result that was never persisted
2. **It must be re-runnable** — capture only immutable inputs (ids, DTOs, scalars); don't capture ORM instances / sessions / generated ids from a previous attempt
3. **Authorization goes inside it too** — a retry reruns the whole business action on a new snapshot
4. The return value must not be an ORM instance bound to the session (the session is closed when the method returns); build a DTO inside `operation`

A conflict that only appears after the commit is not rerun (rerunning would repeat an already-persisted business action); it is raised as-is. Deadlocks (`40P01`) are deliberately **not** retried — consistent lock ordering should make them impossible.

`RepeatableReadSnapshotConflictError`: a control-flow signal raised by domain code after confirming "the session is in REPEATABLE READ + a unique-constraint conflict + the winner is invisible in the snapshot"; `run_in_repeatable_read` treats it exactly like `40001`.

## Constants

```python
from sqlmodel_ext import SESSION_FOR_UPDATE_KEY
from sqlmodel_ext.mixins import SESSION_REPEATABLE_READ_KEY
```

| Constant | Value | Description |
|------|------|------|
| `SESSION_FOR_UPDATE_KEY` | `'_for_update_locked'` | Key in `session.info` for the set of `id()`s of FOR UPDATE-locked instances |
| `SESSION_REPEATABLE_READ_KEY` | `'_repeatable_read_verified'` | Key in `session.info` for the marker "isolation level **verified** as REPEATABLE READ" |
