# Pagination types

::: tip
This is reference documentation. To learn how to wire pagination into an endpoint, see [Paginate a list endpoint](/en/how-to/paginate-a-list-endpoint).
:::

## `PaginationRequest`

```python
from sqlmodel_ext import PaginationRequest
```

Inherits from `SQLModelBase`. DTO carrying pagination and sorting parameters.

**Fields**:

| Field | Type | Default | Constraint |
|-------|------|---------|------------|
| `offset` | `int \| None` | `0` | `ge=0` |
| `limit` | `int \| None` | `50` | `ge=1, le=100` |
| `desc` | `bool \| None` | `True` | — |
| `order` | `Literal["created_at", "updated_at"] \| None` | `"created_at"` | — |

## `TimeFilterRequest`

```python
from sqlmodel_ext import TimeFilterRequest
```

Inherits from `SQLModelBase`. DTO carrying time filter parameters.

**Fields**:

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `created_after_datetime` | `datetime \| None` | `None` | `created_at >= value` |
| `created_before_datetime` | `datetime \| None` | `None` | `created_at < value` |
| `updated_after_datetime` | `datetime \| None` | `None` | `updated_at >= value` |
| `updated_before_datetime` | `datetime \| None` | `None` | `updated_at < value` |

Time intervals are half-open: `[after, before)`.

**`model_post_init` validation**:

- `created_after_datetime >= created_before_datetime` → `ValueError`
- `updated_after_datetime >= updated_before_datetime` → `ValueError`
- `created_after_datetime >= updated_before_datetime` → `ValueError` (creation time cannot be later than update time)

## `TableViewRequest`

```python
from sqlmodel_ext import TableViewRequest
```

```python
class TableViewRequest(TimeFilterRequest, PaginationRequest):
    pass
```

Combination of `TimeFilterRequest` and `PaginationRequest`. Carries pagination + sorting + time filtering parameters in one DTO.

`TableBaseMixin.get()` / `get_with_count()` accept `table_view: TableViewRequest | None`. When `offset` / `limit` / `order_by` / time filter parameters are also provided as explicit arguments, **explicit arguments take precedence**; missing ones fall back to `table_view`.

## `ListResponse[T]`

```python
from sqlmodel_ext import ListResponse
```

Inherits from `pydantic.BaseModel` (**not** `SQLModelBase`). Generic class.

::: info Why not SQLModelBase
SQLModel's metaclass conflicts with `Generic[T]` schema generation; see sqlmodel#1002. `ListResponse` deliberately uses `BaseModel` instead.
:::

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `count` | `int` | Total number of records matching the query |
| `items` | `list[T]` | The current page's items |

**`model_config`**:

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

Mixins for response DTOs. These fields **always have values** in API responses, so they are declared as required (no `| None`) — distinct from `TableBaseMixin`'s `id: int | None` (which is None before INSERT).

| Mixin | Fields |
|-------|--------|
| `IntIdInfoMixin` | `id: int` |
| `UUIDIdInfoMixin` | `id: UUID` |
| `DatetimeInfoMixin` | `created_at: datetime`, `updated_at: datetime` |
| `IntIdDatetimeInfoMixin` | Combines the above two (int id) |
| `UUIDIdDatetimeInfoMixin` | Combines the above two (UUID id) |

All mixins inherit from `SQLModelBase`.
