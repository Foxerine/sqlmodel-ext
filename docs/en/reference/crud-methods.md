# CRUD methods

::: tip
This is reference documentation. For typical patterns and common tasks, see the [how-to guides](/en/how-to/) or [Getting started](/en/tutorials/01-getting-started).
:::

All methods are defined on `TableBaseMixin` and exposed via MRO to every class that inherits it. `UUIDTableBaseMixin` swaps `id` for a **UUIDv7** primary key and overloads `get_one()` / `get_exist_one()` to accept only `uuid.UUID`.

Common type variable: `T = TypeVar('T', bound='TableBaseMixin')`.

::: tip Use with basedpyright
The return type of `get()` is determined precisely by the `fetch_mode` literal, `delete()` uses `@overload` to enforce "either `instances` or `condition`", and `get_one()` is overloaded by primary-key type — all of these constraints live at the **type level**, so basedpyright flags misuse before you run anything.
:::

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

Inserts one or more new records.

| Parameter | Default | Description |
|------|--------|------|
| `instances` | — | A single instance or a list of instances |
| `refresh` | `True` | After commit, re-fetch via `cls.get()` to bind database-generated fields |
| `commit` | `True` | When `False`, only `flush()` without `commit()` |

**Return type**: matches the input type of `instances` — a single instance in, a single instance out; a list in, a list out.

## `save()`

```python
async def save(
    self: T,
    session: AsyncSession,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    refresh: bool = True,
    commit: bool = True,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    optimistic_retry_count: int | None = None,
) -> T
```

INSERTs or UPDATEs the current instance. SQLAlchemy decides based on the instance state.

| Parameter | Default | Description |
|------|--------|------|
| `load` | `None` | Relationships to preload after saving (single or list) |
| `refresh` | `True` | After commit, re-fetch with `cls.get()` (avoids MissingGreenlet) |
| `commit` | `True` | When `False`, only flush; suitable for batch operations |
| `jti_subclasses` | `None` | JTI relationship preload option (requires `load`); `'all'` loads every subclass |
| `optimistic_retry_count` | `None` | Number of automatic retries on optimistic-lock conflicts. `None` = use the model policy `__optimistic_retry_default__` (`3` for `OptimisticLockMixin` models, `0` otherwise); explicit `0` = no retry |

**Behavior details**:

- When a persistent instance has column changes, `save()` **explicitly** assigns `updated_at` (rather than relying only on the column-level `onupdate`) — under JTI, an UPDATE that only changes child-table columns doesn't touch the parent table, so `onupdate` wouldn't fire.
- On retry it re-reads the latest row and re-applies only **the columns this instance actually modified** (per SQLAlchemy attribute history), so it never overwrites other columns someone else has already committed with your stale values.

**Raises**: `OptimisticLockError` (retries exhausted, or the record was found deleted during a retry).

::: danger Always use the return value
`session.commit()` expires every object in the session. Always write `user = await user.save(session)` — never discard the return value.
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
    optimistic_retry_count: int | None = None,
) -> T
```

Partially updates the current instance with fields from `other` (PATCH semantics).

| Parameter | Default | Description |
|------|--------|------|
| `other` | — | Model instance carrying the new data, typically an `XxxUpdate` DTO derived with `partial=True` |
| `extra_data` | `None` | Extra field dict, layered on top of `other` |
| `exclude_unset` | `True` | Only apply fields in `other.model_fields_set`; an explicitly passed `None` also counts as "set" and is written as NULL |
| `exclude` | `None` | Exclude certain fields from the update |
| `load`, `refresh`, `commit`, `jti_subclasses` | — | Same as `save()` |
| `optimistic_retry_count` | `None` | Same as `save()`; on retry it re-reads the latest row and then re-applies the changes from `other` |

::: tip Pairs with `partial=True`
Fields of a DTO derived with `partial=True` default to `Unset`, and `Unset` fields **never appear in `model_dump()`** — so fields that weren't sent are naturally not written, independent of the `exclude_unset` switch. See [Integrate with FastAPI](/en/how-to/integrate-with-fastapi).
:::

A non-empty update (with data or with `extra_data`) always explicitly assigns `updated_at`; an empty update leaves it alone.

**Raises**: `OptimisticLockError`.

## `delete()`

```python
@overload
@classmethod
async def delete(cls: type[T], session: AsyncSession, instances: T | list[T], *, commit: bool = ...) -> int: ...
@overload
@classmethod
async def delete(cls: type[T], session: AsyncSession, *, condition: ColumnElement[bool] | bool, commit: bool = ...) -> int: ...
```

Deletes by instance or by condition. **The two modes are mutually exclusive** — the two `@overload`s make "passing neither" fail type checking with "no matching overload"; the same is validated at runtime.

| Parameter | Default | Description |
|------|--------|------|
| `instances` | `None` | A single instance or a list (instance mode) |
| `condition` | `None` | WHERE condition (condition mode, bulk delete; STI subclasses automatically get the discriminator filter appended, so sibling subclasses' rows are never deleted by mistake) |
| `commit` | `True` | Whether to commit |

**Return value**: number of deleted records (`int`).

**Raises**:

| Exception | Condition |
|------|------|
| `ValueError` | Both or neither of `instances` and `condition` are provided |
| `ResourceReferencedError` | The row being deleted is still referenced by a foreign key (`RESTRICT` / `NO ACTION`) — the row **exists** and was not deleted. Only translated when the `IntegrityError` is caught inside this method **and** its `statement` is a `DELETE`; PostgreSQL only (relies on SQLSTATE `23503`). See [Handle deletes of still-referenced rows](/en/how-to/handle-referenced-deletes) |
| `OptimisticLockError` | An optimistic-lock conflict occurred within this flush (typically: a versioned `DELETE` on an `OptimisticLockMixin` model hit 0 rows). `record_id` and `expected_version` are always `None` (at the flush level the conflict can't be attributed to a specific row); **never retried**; condition mode never raises it (a bulk DELETE has no per-row version check) |

::: warning The `commit=False` boundary
Both translations above only cover SQL issued **inside this method**. In instance mode with `commit=False`, the actual `DELETE` is issued by your later flush/commit, outside this method, and is not translated.
:::

## `get()`

The most complex query method; `@overload` provides a precise mapping from the `fetch_mode` literal to the return type.

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
    join: type[TableBaseMixin] | tuple[type[TableBaseMixin], OnClauseArgument] | None = None,
    options: list[ExecutableOption] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    order_by: list[ColumnElement[Any]] | None = None,
    filter: ColumnElement[bool] | bool | None = None,
    with_for_update: bool = False,
    skip_locked: bool = False,
    table_view: TableViewArgument | None = None,  # TimeFilterRequest | PageWindowRequest
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    populate_existing: bool = False,
    authoritative: bool = False,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> T | list[T] | None
```

On `CachedTableBaseMixin` models, `get()` has one extra parameter, `no_cache: bool = False` (see [Mixins](./mixins#cachedtablebasemixin)).

### `fetch_mode` and return types

| `fetch_mode` | Return type | 0 rows | Multiple rows |
|---|---|---|---|
| `"first"` (default) | `T \| None` | `None` | Returns the first |
| `"one"` | `T` | `NoResultFound` | `MultipleResultsFound` |
| `"all"` | `list[T]` | `[]` | Returns all |

### Parameters

| Parameter | Type | Meaning |
|------|------|------|
| `condition` | `ColumnElement[bool]` | Main WHERE condition |
| `offset` / `limit` | `int` | Pagination (explicit arguments take precedence over `table_view`) |
| `join` | `type` or `(type, on)` tuple | JOIN another table |
| `options` | `list[ExecutableOption]` | Custom SQLAlchemy options (e.g. `selectinload`) |
| `load` | `QueryableAttribute` or `list` | Preload relationships (nested chains built automatically; bidirectional relationship pairs are cycle-broken by list order) |
| `order_by` | `list[ColumnElement]` | Ordering expressions |
| `filter` | `ColumnElement[bool]` | Additional WHERE condition |
| `with_for_update` | `bool` | `SELECT ... FOR UPDATE` row lock. Instance `id()`s are written to `session.info[SESSION_FOR_UPDATE_KEY]`; **forces** `populate_existing` (cannot be turned off), guaranteeing you get the latest database values rather than a stale object from the identity map |
| `skip_locked` | `bool` | `FOR UPDATE SKIP LOCKED`: skip rows locked by other transactions instead of waiting. Only effective when `with_for_update=True`. "0 rows" may also mean "every candidate is locked by someone else", so **do not** use it for existence checks |
| `table_view` | `TableViewRequest` (or any `TimeFilterRequest` / `PageWindowRequest`; each part is applied when present) | Bundle of pagination + ordering + time filtering + keyset cursor parameters. `order` always appends `id` in the same direction as a tiebreaker; `after_id` applies the keyset cursor |
| `jti_subclasses` | `list[type] \| 'all'` | Subclass loading for JTI polymorphic relationships (requires `load`) |
| `populate_existing` | `bool` | Lock-free forced overwrite of identity-map objects with database data |
| `authoritative` | `bool` | The single switch for **authorization reads** (the result decides whether something is allowed, so it must be authoritative): at this layer equivalent to `populate_existing=True`; cached models additionally bypass Redis. Monotonically merged with `populate_existing` (`or`) |
| `created_before/after_datetime` | `datetime` | Time filter (left-closed, right-open) |
| `updated_before/after_datetime` | `datetime` | Time filter (left-closed, right-open) |

**Raises**:

- `ValueError` — `jti_subclasses` without a matching `load`; used on a nested relationship chain; target class is not a `PolymorphicBaseMixin`
- `KeysetCursorUnsupportedError` — `after_id` used together with an explicit `order_by` or `join`
- `KeysetCursorInvalidError` — the `after_id` anchor doesn't exist or isn't visible within this query (the two are deliberately indistinguishable)
- `ValueError` — `after_id` used on a table without a UUID primary key (programming error; use offset pagination instead)

For the full keyset cursor rules, see [Keyset cursor pagination](/en/how-to/keyset-pagination).

### Polymorphic query behavior

| Scenario | Behavior |
|------|------|
| JTI model | Automatically uses `with_polymorphic(cls, '*')` to JOIN all subtables |
| STI model | Automatically adds `WHERE _polymorphic_name IN (...)` (`get()` / `count()` / `delete(condition=)` / keyset anchor / aggregate methods share the same filter) |
| `with_for_update` + polymorphic | Uses `FOR UPDATE OF <main table>` (avoids the restriction on the nullable side of a LEFT JOIN) |

## `get_one()`

```python
@classmethod
async def get_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin overloads this as uuid.UUID
    *,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    with_for_update: bool = False,
    authoritative: bool = False,
) -> T
```

Shortcut for `get(col(cls.id) == id, fetch_mode='one')`. `authoritative` has the same semantics as in `get()`.

**Raises**: `NoResultFound` (not found), `MultipleResultsFound`.

## `get_exist_one()`

```python
@classmethod
async def get_exist_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin overloads this as uuid.UUID
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    *,
    detail: str = "Not found",
    with_for_update: bool = False,
) -> T
```

Like `get_one()`, but with a friendlier exception when not found:

| Environment | Exception |
|------|------|
| FastAPI installed | `HTTPException(status_code=404, detail=detail)` |
| FastAPI not installed | `RecordNotFoundError` |

- `detail` (keyword-only) customizes the 404 message, e.g. `detail="Character not found"`.
- `with_for_update` (keyword-only) is forwarded to `get()`: reads the row with `SELECT ... FOR UPDATE` (which by itself bypasses the Redis cache and the identity map). The typical use is closing the TOCTOU window between the "existence check" and the "delete" — a concurrent second request blocks, and after the first commits it finds no row and gets the same 404 as a serial second delete.

The decision is made at module import time: `sqlmodel_ext.mixins.table` tries `from fastapi import HTTPException` on import and records `None` if that fails.

## `count()`

```python
@classmethod
async def count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    distinct_column: Mapped[Any] | ColumnElement[Any] | None = None,
    time_filter: TimeFilterRequest | None = None,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> int
```

Returns the number of matching records via `SELECT COUNT(*)`. Passing `distinct_column` switches to `COUNT(DISTINCT col)` (e.g. "distinct active users"). Non-null fields in `time_filter` take precedence over the individually passed time parameters.

## `distinct_column()`

```python
@classmethod
async def distinct_column(
    cls: type[T],
    session: AsyncSession,
    column: Mapped[V] | ColumnElement[V],
    condition: ColumnElement[bool] | None = None,
    *,
    limit: int | None = None,
) -> list[V]
```

Returns the distinct values of a column (database-level `SELECT DISTINCT`); the return type is inferred from the column type (`col(Model.owner_id)` → `list[UUID]`). STI filtering matches `get()` / `count()`.

## `group_sum()`

```python
@classmethod
async def group_sum(
    cls: type[T],
    session: AsyncSession,
    sum_columns: Sequence[Mapped[Any] | ColumnElement[Any]],
    *,
    group_by: Mapped[GK] | ColumnElement[GK] | None = None,
    condition: ColumnElement[bool] | None = None,
    order_by: ColumnElement[Any] | None = None,
) -> list[GroupSumRow[GK]]
```

Computes `COUNT(*)` and each `COALESCE(SUM(col), 0)` in a single query.

- Without `group_by` → whole-table aggregate, returns **exactly one** element (`key=None`)
- With `group_by` (a column or expression, e.g. a `date_trunc` time bucket) → one row per group; ordered by `group_by` ascending by default, overridable with `order_by`

**Raises**: `ValueError` (`sum_columns` is empty — for a plain count use `count()`).

### `GroupSumRow[GK]`

```python
from sqlmodel_ext.mixins import GroupSumRow
```

| Field | Type | Description |
|------|------|------|
| `key` | `GK` | Group key (the value of `group_by`); `None` for a whole-table aggregate |
| `count` | `NonNegativeBigInt` | Number of rows in the group |
| `totals` | `list[Decimal]` | Result of each summed column, aligned **by position** with `sum_columns` |

For conditional sums (`SUM ... FILTER (WHERE ...)`), call once per condition and merge by `key` in Python. For usage see [Aggregate queries](/en/how-to/aggregate-queries).

## `get_with_count()`

```python
@classmethod
async def get_with_count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    join: type[TableBaseMixin] | tuple[type[TableBaseMixin], OnClauseArgument] | None = None,
    options: list[ExecutableOption] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    order_by: list[ColumnElement[Any]] | None = None,
    filter: ColumnElement[bool] | bool | None = None,
    table_view: TableViewRequest | None = None,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
) -> ListResponse[T]
```

First `get(fetch_mode="all")` (which performs all keyset cursor validation), then `count()`, assembled into a `ListResponse[T]`. `count` is the size of the **whole filtered set** and is not affected by `after_id`. Typically used for LIST endpoints.

## IntegrityError friendly-message registry

Static methods on `TableBaseMixin`. Register "constraint name → user-visible message" at the **same place** the constraint is declared, so the query path never leaks table names / column names / SQL. The first registration wins (`setdefault`, guarding against repeated imports).

| Method | Direction / purpose |
|------|------|
| `register_unique_violation_message(constraint_name, friendly_message)` | UNIQUE violation (23505) |
| `register_foreign_key_violation_message(constraint_name, friendly_message)` | INSERT/UPDATE of a child row pointing at a nonexistent parent (23503, "referenced resource does not exist", 404 semantics) |
| `register_check_violation_message(constraint_name, friendly_message)` | ORM-declared `CheckConstraint` (23514 with a `constraint_name`) |
| `register_fk_delete_restrict_message(constraint_name, friendly_message)` | DELETE of a parent row that is still referenced (409 semantics); only `delete()` looks it up |
| `lookup_unique_violation_message` / `lookup_foreign_key_violation_message` / `lookup_check_violation_message` / `lookup_fk_delete_restrict_message(constraint_name)` | Look up by name; returns `None` on a miss or when the name is `None` |
| `lookup_integrity_violation_message(e)` | Pure lookup: returns the registered business message on a hit, otherwise `None`; a trigger `RAISE EXCEPTION` (23514 without a `constraint_name`) counts as a hit and returns its first-line message. **Does not** consult the `fk_delete_restrict` registry |
| `sanitize_integrity_error(e, default_message=...)` | Adds a fallback on top of `lookup_integrity_violation_message`: on a miss, logs and returns `default_message` |
| `extract_trigger_message(orig)` | Extracts the first-line business message from a trigger exception (strips the `ERROR:` prefix and the `DETAIL:` / `CONTEXT:` lines) |

When you need to distinguish "hit / miss" (e.g. counting user errors separately from platform errors), use `lookup_integrity_violation_message` — `sanitize_*` folds both cases into one string. SQLSTATE detection is PostgreSQL-specific; other databases only ever get `default_message`.

## Method quick reference

| Method | Kind | SQL | Return value |
|------|------|---------|--------|
| `add()` | `@classmethod` | `INSERT` | `T` or `list[T]` |
| `save()` | instance method | `INSERT` or `UPDATE` | refreshed `T` |
| `update()` | instance method | `UPDATE` (PATCH) | refreshed `T` |
| `delete()` | `@classmethod` | `DELETE` | `int` (deleted count) |
| `get()` | `@classmethod` | `SELECT ... WHERE ...` | `T \| list[T] \| None` |
| `get_one()` | `@classmethod` | `SELECT WHERE id = ?` | `T` |
| `get_exist_one()` | `@classmethod` | `SELECT WHERE id = ?` + 404 | `T` |
| `count()` | `@classmethod` | `SELECT COUNT(*)` / `COUNT(DISTINCT col)` | `int` |
| `distinct_column()` | `@classmethod` | `SELECT DISTINCT col` | `list[V]` |
| `group_sum()` | `@classmethod` | `SELECT [key,] COUNT(*), SUM(...) [GROUP BY]` | `list[GroupSumRow[GK]]` |
| `get_with_count()` | `@classmethod` | `SELECT` + `COUNT` | `ListResponse[T]` |
