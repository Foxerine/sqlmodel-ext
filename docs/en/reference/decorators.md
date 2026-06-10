# Decorators & helpers

::: tip
This is reference documentation. To learn how to use `@requires_relations` to fix MissingGreenlet, see [Prevent MissingGreenlet errors](/en/how-to/prevent-missing-greenlet).
:::

## `@requires_relations`

```python
from sqlmodel_ext import requires_relations
```

**Signature**:

```python
def requires_relations(
    *relations: str | QueryableAttribute[Any],
) -> Callable[[F], F]
```

**Parameters**:

| Parameter | Type | Meaning |
|-----------|------|---------|
| `*relations` | `str` | Direct relation attribute name on this class (e.g. `'profile'`) |
| `*relations` | `QueryableAttribute` | Nested relation (e.g. `Generator.config`) |

**Prerequisites**:

- The decorated class must inherit `RelationPreloadMixin`
- The decorated method must be `async def` (regular coroutine) or `async def ... yield` (async generator)
- One of the method's parameters must be named `session`, or a kwarg must be of type `AsyncSession`

**Runtime behavior**:

1. Auto-extracts `AsyncSession` from arguments
2. Calls `self._ensure_relations_loaded(session, relations)` to load missing relations
3. Already-loaded relations are not re-queried (incremental loading)
4. Nested relations automatically resolve their intermediate paths
5. Executes the original method

**Import-time validation**: `RelationPreloadMixin.__init_subclass__` checks at class definition time that the string names in `relations` exist as class attributes or SQLModel relationships; otherwise raises `AttributeError`.

**Attached attribute**: the decorated function gains a `_required_relations` tuple storing the declaration.

## `@requires_for_update`

```python
from sqlmodel_ext import requires_for_update
```

**Signature**:

```python
def requires_for_update(func: F) -> F
```

**Prerequisites**:

- The decorated class must inherit `RelationPreloadMixin`
- Callers must first acquire the instance via `cls.get(session, ..., with_for_update=True)`

**Runtime behavior**:

1. Extracts `AsyncSession` from arguments
2. Checks whether `session.info[SESSION_FOR_UPDATE_KEY]` contains `id(self)`
3. Not present → `RuntimeError`
4. Present → executes the original method

**Attached attribute**: the decorated function gains `_requires_for_update = True`.

## `rel()`

```python
from sqlmodel_ext import rel
```

**Signature**:

```python
def rel(relationship: object) -> QueryableAttribute[Any]
```

**Purpose**: type-cast a SQLModel `Relationship` field to `QueryableAttribute`, so basedpyright stops complaining.

**Runtime behavior**:

- Input is a `QueryableAttribute` → return as-is
- Otherwise → `AttributeError`

**Typical usage**: `load=rel(User.profile)`, `load=[rel(User.profile), rel(Profile.avatar)]`.

## `cond()`

```python
from sqlmodel_ext import cond
```

**Signature**:

```python
def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]
```

**Purpose**: narrow a column comparison (which basedpyright infers as `bool`) into `ColumnElement[bool]`, so `&` / `|` operators don't trip type errors.

**Runtime behavior**: equivalent to `cast(ColumnElement[bool], expr)` — no runtime check.

**Typical usage**:

```python
scope = cond(UserFile.user_id == current_user.id)
condition = scope & cond(UserFile.status == FileStatusEnum.uploaded)
```

## `session.reset()` (enhanced AsyncSession) {#session-reset}

```python
from sqlmodel_ext import AsyncSession
```

**Signature** (enhanced method on `sqlmodel_ext.session.AsyncSession`):

```python
async def reset(self) -> None
```

> Replaces the removed `safe_reset()` helper since 0.4.0. Construct sessions with `async_sessionmaker(class_=AsyncSession)` and simply `await session.reset()`.

**Purpose**: after the upstream `reset()` (releasing the transaction and connection), automatically clears the FOR UPDATE lock-tracking set in `session.info[SESSION_FOR_UPDATE_KEY]` and the cache-invalidation tracking state — preventing tracking state from leaking into the next session reuse cycle.

**Typical use case**: an HTTP endpoint / Taskiq task needs to do long external I/O mid-flight (S3, ffprobe, third-party HTTP polling, etc.). Call `session.reset()` first to release the DB connection, so the connection isn't blocked on network I/O and the pool isn't exhausted. See [Release the DB connection during long I/O](/en/how-to/release-connection-during-long-io).

**Object state after the call**:

- All ORM objects enter the **detached** state (`sa_inspect(obj).detached == True`)
- But **already-loaded scalar fields are NOT expired** — they remain in `obj.__dict__`, so reading them is safe (no SQL, no `MissingGreenlet`)
- Accessing un-preloaded relation fields raises `InvalidRequestError` (lazy load fails on detached objects)
- Writes (save / update / delete) will fail — re-query a fresh attached instance via `Model.get()` first
- Any subsequent `await Model.get/save` that issues SQL will transparently check out a new connection from the pool

## `sanitize_integrity_error()`

```python
TableBaseMixin.sanitize_integrity_error(
    e: IntegrityError,
    default_message: str = "Data integrity constraint violation",
) -> str
```

Static method on `TableBaseMixin`. Extracts a user-safe error message from an `IntegrityError`.

**Behavior**:

- SQLSTATE `23514` (`check_violation`): take the first line of the error, strip the `ERROR:` prefix, return it (PostgreSQL trigger messages are business-meaningful and safe to surface)
- Other constraint errors (FK, unique, etc.): return `default_message` (avoid leaking table structure)

## Constants

```python
from sqlmodel_ext import SESSION_FOR_UPDATE_KEY
```

**`SESSION_FOR_UPDATE_KEY`**: the string `'_for_update_locked'`. `get(with_for_update=True)` uses it to track locked instance `id()`s in `session.info`. `@requires_for_update` reads this key for its runtime check.
