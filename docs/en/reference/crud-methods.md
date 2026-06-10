# CRUD methods

::: tip
This is reference documentation. For typical patterns and common tasks, see the [how-to guides](/en/how-to/) or [Getting started](/en/tutorials/01-getting-started).
:::

All methods are defined on `TableBaseMixin` and exposed via MRO to every class that inherits it. `UUIDTableBaseMixin` overloads `get_one()` / `get_exist_one()` to accept `uuid.UUID` IDs.

Common type variable: `T = TypeVar('T', bound='TableBaseMixin')`.

## `add()`

```python
@classmethod
async def add(
    cls: type[T],
    session: AsyncSession,
    instances: T | list[T],
    refresh: bool = True,
    commit: bool = True,
) -> T | list[T]
```

Bulk insert new records.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instances` | — | A single instance or a list of instances |
| `refresh` | `True` | After commit, re-fetch via `cls.get()` to pick up DB-generated fields |
| `commit` | `True` | When `False`, only `flush()` (no `commit()`) |

**Return type**: matches the input shape — single instance in, single instance out; list in, list out.

## `save()`

```python
async def save(
    self: T,
    session: AsyncSession,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    refresh: bool = True,
    commit: bool = True,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    optimistic_retry_count: int = 0,
) -> T
```

INSERT or UPDATE the current instance. SQLAlchemy decides based on whether the instance is already in the session.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `load` | `None` | Relations to eagerly load after save (single or list) |
| `refresh` | `True` | After commit, re-fetch via `cls.get()` (avoids MissingGreenlet) |
| `commit` | `True` | When `False`, only flush — useful for batched operations |
| `jti_subclasses` | `None` | JTI relation eager-loading option (requires `load`); `'all'` loads every subclass |
| `optimistic_retry_count` | `0` | Number of automatic retries on optimistic-lock conflict |

**Raises**: `OptimisticLockError` (after retries are exhausted).

::: danger Always use the return value
`session.commit()` expires every object in the session. Always write `user = await user.save(session)`; never discard the return value.
:::

## `update()`

```python
async def update(
    self: T,
    session: AsyncSession,
    other: SQLModelBase,
    extra_data: dict[str, Any] | None = None,
    exclude_unset: bool = True,
    exclude: set[str] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    refresh: bool = True,
    commit: bool = True,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    optimistic_retry_count: int = 0,
) -> T
```

Partial-update the current instance using fields from `other` (PATCH semantics).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `other` | — | Model instance carrying new data (typically `XxxUpdateRequest`) |
| `extra_data` | `None` | Extra dict layered on top of `other` |
| `exclude_unset` | `True` | Only update fields that were **explicitly set** on `other` |
| `exclude` | `None` | Exclude these fields from the update |
| `load`, `refresh`, `commit`, `jti_subclasses`, `optimistic_retry_count` | — | Same as `save()` |

**Raises**: `OptimisticLockError`.

## `delete()`

```python
@classmethod
async def delete(
    cls: type[T],
    session: AsyncSession,
    instances: T | list[T] | None = None,
    *,
    condition: ColumnElement[bool] | bool | None = None,
    commit: bool = True,
) -> int
```

Delete by instance or by condition. **The two modes are mutually exclusive** — exactly one of `instances` and `condition` must be provided.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instances` | `None` | Single instance or list (instance mode) |
| `condition` | `None` | WHERE condition (condition mode, bulk delete) |
| `commit` | `True` | Whether to commit |

**Returns**: number of deleted rows (`int`).

**Raises**: `ValueError` (when both or neither of `instances` / `condition` are provided).

## `get()`

The most powerful query method, with `@overload` declarations giving precise return types per `fetch_mode` literal.

```python
@classmethod
async def get(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    offset: int | None = None,
    limit: int | None = None,
    fetch_mode: Literal["one", "first", "all"] = "first",
    join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
    options: list[ExecutableOption] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    order_by: list[ColumnElement[Any]] | None = None,
    filter: ColumnElement[bool] | bool | None = None,
    with_for_update: bool = False,
    table_view: TableViewRequest | None = None,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    populate_existing: bool = False,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> T | list[T] | None
```

### `fetch_mode` and return types

| `fetch_mode` | Return type | 0 rows | Multiple rows |
|--------------|-------------|--------|---------------|
| `"first"` (default) | `T \| None` | `None` | Returns the first |
| `"one"` | `T` | `NoResultFound` | `MultipleResultsFound` |
| `"all"` | `list[T]` | `[]` | All rows |

### Parameters

| Parameter | Type | Meaning |
|-----------|------|---------|
| `condition` | `ColumnElement[bool]` | Main WHERE condition |
| `offset` / `limit` | `int` | Pagination (explicit args take precedence over `table_view`) |
| `join` | `type` or `(type, on)` tuple | JOIN another table |
| `options` | `list[ExecutableOption]` | Custom SQLAlchemy options (e.g. `selectinload`) |
| `load` | `QueryableAttribute` or `list` | Eager-load relations (auto-builds nested chains) |
| `order_by` | `list[ColumnElement]` | Sort expressions |
| `filter` | `ColumnElement[bool]` | Additional WHERE condition |
| `with_for_update` | `bool` | `SELECT ... FOR UPDATE` (row lock); locked instance ID is recorded in `session.info[SESSION_FOR_UPDATE_KEY]` |
| `table_view` | `TableViewRequest` | DTO bundle for pagination + sorting + time filters |
| `jti_subclasses` | `list[type] \| 'all'` | JTI polymorphic subclass loading (requires `load`) |
| `populate_existing` | `bool` | Force-overwrite identity-map objects |
| `created_before/after_datetime` | `datetime` | Half-open time filter |
| `updated_before/after_datetime` | `datetime` | Half-open time filter |

**Raises**:

- `ValueError` — `jti_subclasses` provided without `load`
- `ValueError` — `jti_subclasses` used on a nested relation chain
- `ValueError` — `jti_subclasses` target class is not a `PolymorphicBaseMixin`

### Polymorphic query behavior

| Scenario | Behavior |
|----------|----------|
| JTI model (`is_jti=True`) | Auto-uses `with_polymorphic(cls, '*')` to JOIN every sub-table |
| STI model (`is_sti=True`) | Auto-adds `WHERE _polymorphic_name IN (...)` filter |
| `with_for_update` + JTI | Uses `FOR UPDATE OF <main_table>` (avoids LEFT JOIN nullable-side restrictions) |

## `get_one()`

```python
@classmethod
async def get_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin overrides to uuid.UUID
    *,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    with_for_update: bool = False,
) -> T
```

Shortcut for `get(cls.id == id, fetch_mode='one')`.

**Raises**: `NoResultFound` (record not found), `MultipleResultsFound` (multiple records — should never happen for unique IDs).

## `get_exist_one()`

```python
@classmethod
async def get_exist_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin overrides to uuid.UUID
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    *,
    detail: str = "Not found",
) -> T
```

Like `get_one()`, but the not-found exception is friendlier:

| Environment | Exception |
|-------------|-----------|
| FastAPI installed | `HTTPException(status_code=404, detail=detail)` |
| FastAPI not installed | `RecordNotFoundError` |

`detail` (keyword-only) customizes the 404 message — replacing the
"get, then hand-write the null check + raise" boilerplate, e.g.
`detail="Character not found"`.

The decision is made at module import time and cached as `_HAS_FASTAPI`.

## `count()`

```python
@classmethod
async def count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> int
```

Returns the number of records matching the condition. Backed by `SELECT COUNT(*)`.

## `get_with_count()`

```python
@classmethod
async def get_with_count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    table_view: TableViewRequest | None = None,
    # ... all the same parameters as get()
) -> ListResponse[T]
```

Combination of `count()` + `get(fetch_mode="all")`, returning `ListResponse[T]`. Typically used by LIST endpoints.

## Method cheat sheet

| Method | Type | Equivalent SQL | Returns |
|--------|------|----------------|---------|
| `add()` | `@classmethod` | `INSERT` | `T` or `list[T]` |
| `save()` | instance method | `INSERT` or `UPDATE` | refreshed `T` |
| `update()` | instance method | `UPDATE` (PATCH) | refreshed `T` |
| `delete()` | `@classmethod` | `DELETE` | `int` (rows deleted) |
| `get()` | `@classmethod` | `SELECT ... WHERE ...` | `T \| list[T] \| None` |
| `get_one()` | `@classmethod` | `SELECT WHERE id = ?` | `T` |
| `get_exist_one()` | `@classmethod` | `SELECT WHERE id = ?` + 404 | `T` |
| `count()` | `@classmethod` | `SELECT COUNT(*)` | `int` |
| `get_with_count()` | `@classmethod` | `COUNT + SELECT` | `ListResponse[T]` |
