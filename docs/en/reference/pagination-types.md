# Pagination types

::: tip
This is reference documentation. To learn how to wire pagination into an endpoint, see [Paginate a list endpoint](/en/how-to/paginate-a-list-endpoint) and [Keyset cursor pagination](/en/how-to/keyset-pagination).
:::

Class hierarchy:

```
PageWindowRequest            offset / limit / desc
  └── PaginationRequest      + order / after_id (keyset cursor)
TimeFilterRequest            created_* / updated_* time bounds
TableViewRequest             TimeFilterRequest + PaginationRequest
```

They are all pure data classes that only carry parameters; the SQL clauses are built by `TableBaseMixin.get()`.

## `PageWindowRequest`

```python
from sqlmodel_ext.pagination import PageWindowRequest
```

The lowest layer: just the window and direction. For consumers whose "ordering column is fixed by their own domain semantics and only need a window + direction" — it deliberately **does not carry** `order` / `after_id`, to avoid exposing fields in the public schema that nobody consumes and that would be silently dropped.

| Field | Type | Default | Constraint |
|------|------|--------|------|
| `offset` | `int \| None` | `0` | `ge=0, le=MAX_TABLE_VIEW_OFFSET` |
| `limit` | `int \| None` | `DEFAULT_PAGE_SIZE` (`50`) | `ge=1, le=MAX_PAGE_SIZE` (`100`) |
| `desc` | `bool \| None` | `True` | — |

`get(table_view=PageWindowRequest(...))` only applies `offset` / `limit`; ordering is determined by your own `order_by`.

## `PaginationRequest`

```python
from sqlmodel_ext import PaginationRequest
```

Inherits `PageWindowRequest`, adding the ordering column and the keyset cursor.

| Field | Type | Default | Description |
|------|------|--------|------|
| `order` | `Literal["created_at", "updated_at", "id"] \| None` | `"created_at"` | Ordering column. Subclasses can override the `Literal` to add domain ordering columns (`get()` resolves the column by name, so every value must be a real column) |
| `after_id` | `uuid.UUID \| None` | `None` | Keyset cursor: returns only records ordered **after** this record (pass the id of the last item on the previous page) |

`get()` always appends `id` in the same direction after `order` as a tiebreaker (the composite `(ordering column, id)`), so rows sharing the same timestamp are neither duplicated nor skipped at page boundaries, for both offset and keyset pagination.

**Construction-time validation** (field validators on `after_id`; a violation raises `ValidationError` located at `after_id`):

| Rule | Reason |
|------|------|
| `after_id` can only be paired with `order` of `created_at` / `id` (or `None`) | A mutable ordering column (`updated_at` or a domain column) would let the anchor move after an update, breaking the "no duplicates, no gaps" guarantee |
| `after_id` and a non-zero `offset` are **mutually exclusive** | The two would stack rather than being alternatives: keyset already means "after the anchor", and adding `offset=N` would skip N more rows that could then never be reached. `offset` defaults to 0, and "forgot to reset it" is the most natural misuse, so it is made unrepresentable at construction time |

Runtime rules (in `get()`): the anchor must be visible to this query (`condition` + `filter` + STI filter), otherwise `KeysetCursorInvalidError`; used together with an explicit `order_by` or `join` → `KeysetCursorUnsupportedError`; only tables with a UUID primary key are supported. See [Keyset cursor pagination](/en/how-to/keyset-pagination) for details.

## `TimeFilterRequest`

```python
from sqlmodel_ext import TimeFilterRequest
```

| Field | Type | Default | Semantics |
|------|------|--------|------|
| `created_after_datetime` | `AwareDatetime \| None` | `None` | `created_at >= value` |
| `created_before_datetime` | `AwareDatetime \| None` | `None` | `created_at < value` |
| `updated_after_datetime` | `AwareDatetime \| None` | `None` | `updated_at >= value` |
| `updated_before_datetime` | `AwareDatetime \| None` | `None` | `updated_at < value` |

Time ranges are left-closed, right-open `[after, before)`. All bounds are **`AwareDatetime`**: a datetime without a timezone is rejected at validation time (it can't be compared with timezone-aware database values and would be silently interpreted in the database's timezone).

**Construction-time validation** (field validators; a violation raises `ValidationError` located at the `*_before_datetime` field named below):

- `created_after_datetime >= created_before_datetime` → at `created_before_datetime`
- `updated_after_datetime >= updated_before_datetime` → at `updated_before_datetime`
- `created_after_datetime >= updated_before_datetime` → at `updated_before_datetime` (a record's update time can't be earlier than its creation time)

## `TableViewRequest`

```python
from sqlmodel_ext import TableViewRequest
```

```python
class TableViewRequest(TimeFilterRequest, PaginationRequest):
    pass
```

Carries pagination + ordering + keyset cursor + time filter parameters together. `get()` / `get_with_count()` accept a `table_view` parameter; explicitly passed `offset` / `limit` / `order_by` / time parameters take precedence, falling back to `table_view` when not provided.

## `query_dependency()`

```python
from sqlmodel_ext import query_dependency

TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]
```

Turns a query-parameter DTO (`TableViewRequest`, `PaginationRequest`, `PageWindowRequest`, `TimeFilterRequest`, `TrgmSearchRequest`, or a subclass of your own) into a FastAPI dependency. Requires the `fastapi` extra (`ImportError` when called without FastAPI; importing `sqlmodel_ext` never needs it).

| Behavior | Detail |
|------|------|
| Query parameters | One per model field, named after the field's alias (the field name when it has none), with the field's type, constraints, default and docstring description — the OpenAPI schema matches the model |
| Validation | The dependency constructs the model, so every validator runs, cross-field ones included |
| Errors | A `ValidationError` is re-raised as `fastapi.exceptions.RequestValidationError` with every location prefixed by `'query'` (`["query", "after_id"]`; a model-level error without a field becomes `["query"]`). FastAPI's default handler answers 422 |
| Other query parameters | Ignored; the endpoint can declare its own next to the dependency |
| Caching | One callable per model class, so FastAPI's per-request dependency cache treats repeated uses as one dependency |
| Rejected models (`TypeError`) | `table=True` models (they skip validation), fields with a `default_factory`, fields whose `validation_alias` is not a single string |

Why not `Depends()` on the class: FastAPI validates each parameter, then calls the class; a `ValidationError` from that call is not a `RequestValidationError`, so cross-field errors become 500s. See [Paginate a list endpoint](/en/how-to/paginate-a-list-endpoint#why-query-dependency).

## Constants

```python
from sqlmodel_ext.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_SHARED_PAGE_WINDOW,
    MAX_TABLE_VIEW_OFFSET,
)
```

| Constant | Value | Description |
|------|------|------|
| `DEFAULT_PAGE_SIZE` | `50` | Default `limit` |
| `MAX_PAGE_SIZE` | `100` | Upper bound of `limit` |
| `MAX_SHARED_PAGE_WINDOW` | `1000` | The largest window any `PageWindowRequest` subclass may allow; a subclass that relaxes `limit` above this value must also tighten its own `offset` upper bound |
| `MAX_TABLE_VIEW_OFFSET` | `JS_MAX_SAFE_INTEGER - MAX_SHARED_PAGE_WINDOW` | Upper bound of `offset` |

Why `offset` needs an upper bound: with only `ge=0`, `offset=10**100` passes validation and fails only in the database driver (`bigint out of range`). Why `2**53 - 1` rather than the int8 maximum: this bound becomes part of the API contract as the OpenAPI `maximum`, and most JS/TS clients map `integer` to an IEEE-754 double. Why subtract one more window: callers often compute the next page with `offset + limit`, and reserving a window keeps that addition in range.

## `ListResponse[T]`

```python
from sqlmodel_ext import ListResponse
```

Inherits from `pydantic.BaseModel` (**not** `SQLModelBase`); a generic class.

::: info Why it doesn't inherit SQLModelBase
As a generic container used as a FastAPI **response_model**, SQLModel + Generic still generates a wrong JSON schema for parameterized fields (`{"items": {}}` instead of a `$ref`); see sqlmodel#1002. Generic containers that are only method return values and never enter OpenAPI (such as `GroupSumRow`) are unaffected and inherit `SQLModelBase` as usual.
:::

| Field | Type | Description |
|------|------|------|
| `count` | `int` | Total number of records matching the conditions (not affected by `after_id`) |
| `items` | `list[T]` | Data of the current page |

```python
model_config = ConfigDict(use_attribute_docstrings=True)
```

**Typical return type**: `get_with_count()` returns `ListResponse[T]`.

## Info response mixins (DTO)

```python
from sqlmodel_ext import (
    IntIdInfoMixin,
    UUIDIdInfoMixin,
    DatetimeInfoMixin,
    IntIdDatetimeInfoMixin,
    UUIDIdDatetimeInfoMixin,
)
```

Mixins for response DTOs. These fields **always have a value** in API responses, so they are declared required (no `| None`) — unlike `id: int | None` in `TableBaseMixin` (None before INSERT).

| Mixin | Fields |
|-------|------|
| `IntIdInfoMixin` | `id: int` |
| `UUIDIdInfoMixin` | `id: UUID` |
| `DatetimeInfoMixin` | `created_at: datetime`, `updated_at: datetime` |
| `IntIdDatetimeInfoMixin` | Combination of the two above (int id) |
| `UUIDIdDatetimeInfoMixin` | Combination of the two above (UUID id) |

All mixins inherit `SQLModelBase`.
