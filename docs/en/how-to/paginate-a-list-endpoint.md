# Paginate a list endpoint

**Goal**: make a list endpoint accept query parameters like `?offset=`, `?limit=`, `?desc=`, `?order=`, `?after_id=`, `?created_after_datetime=`, and return a `{count, items}` response.

**Prerequisites**:

- You already have a FastAPI endpoint
- Your model inherits `TableBaseMixin` or `UUIDTableBaseMixin`
- You have an `XxxResponse` DTO

## 1. Declare the request parameters as a FastAPI dependency

```python
from typing import Annotated
from fastapi import Depends
from sqlmodel_ext import TableViewRequest, query_dependency

TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]
```

`TableViewRequest` carries pagination (`offset` / `limit` / `desc` / `order`), a keyset cursor (`after_id`) and time filtering (`created_after_datetime` / `created_before_datetime` / `updated_after_datetime` / `updated_before_datetime`). `query_dependency()` declares one query parameter per model field (same type, constraints, default, title, description, examples, deprecation and `json_schema_extra` as the model field, so each parameter's OpenAPI schema is the field's JSON schema) and constructs the `TableViewRequest` itself.

### Why `query_dependency()`

Single-field errors (`limit=0`, a time without a timezone) are validated per parameter by FastAPI and are a 422 directly. But **cross-field** rules — `after_id` and `offset` are mutually exclusive, `after_id` can't be combined with `order=updated_at`, time ranges must be in order — only fire when the `TableViewRequest` is constructed. With a bare `Depends()`, FastAPI calls `TableViewRequest(...)` itself; the Pydantic `ValidationError` raised there is not a `RequestValidationError`, FastAPI does not catch it, and the request fails with a **500**.

`query_dependency()` re-raises that `ValidationError` as a `RequestValidationError` whose error locations start with `query`, so FastAPI's default handler answers **422** in the same shape as any other query-parameter error — no exception handler to register:

```json
{"detail": [{"type": "value_error", "loc": ["query", "after_id"],
             "msg": "Value error, after_id and offset cannot be combined: ...", "input": "..."}]}
```

A cross-field error is located at the field its rule constrains: both `after_id` rules report `after_id`, the time-range rules report the `*_before_datetime` field. The same form works for `PageWindowRequest`, `PaginationRequest`, `TimeFilterRequest`, `TrgmSearchRequest` and your own subclasses (`query_dependency(MyTableViewRequest)`). The endpoint can still declare further query parameters next to it; undeclared query parameters (such as a cache-busting `?_=123`) are ignored.

::: warning Don't switch to `Annotated[TableViewRequest, Query()]`
A Pydantic query model only works when it is the endpoint's **only** query parameter: add one more ordinary query parameter to the endpoint (e.g. `status: str | None = None`) and FastAPI no longer hands the whole query string to the model, so requests fail outright with 422 `Field required`. On top of that, `SQLModelBase`'s `extra='forbid'` turns any undeclared query parameter (such as `?_=123`) into a 422.
:::

## 2. Call `get_with_count()` in the endpoint

```python
from sqlmodel_ext import ListResponse

@router.get("", response_model=ListResponse[ArticleResponse])
async def list_articles(
    session: SessionDep,
    table_view: TableViewDep,
) -> ListResponse[Article]:
    return await Article.get_with_count(
        session,
        Article.is_published == True,
        table_view=table_view,
    )
```

`get_with_count()` first runs `SELECT ... LIMIT N OFFSET M` (or the keyset condition), then `COUNT(*)`, and assembles them into a `ListResponse[T]`. `count` is always the size of the whole filtered set.

## 3. How clients call it

```http
GET /articles?offset=0&limit=20&desc=true&order=created_at&created_after_datetime=2026-01-01T00:00:00Z
```

Returns:

```json
{
  "count": 142,
  "items": [
    { "id": "...", "title": "...", "created_at": "...", "..." : "..." }
  ]
}
```

For sequential traversal (infinite scroll, export), switch to a keyset cursor: pass the `id` of the last item of the previous page back as `after_id`, and **stop passing `offset`**. See [Keyset cursor pagination](./keyset-pagination).

## Defaults

| Parameter | Default | Cap / values |
|-----------|---------|------|
| `offset` | `0` | `MAX_TABLE_VIEW_OFFSET` (`2**53 - 1 - 1000`) |
| `limit` | `50` | `100` |
| `desc` | `True` | — |
| `order` | `"created_at"` | `"created_at"` / `"updated_at"` / `"id"` |
| `after_id` | `None` | UUID |

Sorting always appends `id` in the same direction as a tiebreaker column, so rows created in the same batch with identical `created_at` are never duplicated or skipped at page boundaries. To sort by another field, either override the `order` `Literal` in a `PaginationRequest` subclass to add your domain column, or skip `table_view`'s sorting and pass `order_by=` yourself.

## Common pitfalls

- **`response_model` must be `ListResponse[ArticleResponse]`**, not `list[ArticleResponse]`.
- **Time parameters must carry a timezone** (`...Z` or `+08:00`). All time bounds are `AwareDatetime`; a value without a timezone is a 422 outright — otherwise it would be silently interpreted in the database's timezone.
- **Time intervals are half-open** `[after, before)`. `created_after_datetime=2026-01-01T00:00:00Z` + `created_before_datetime=2026-02-01T00:00:00Z` means "all of January (UTC)".
- **`order` only accepts values in the `Literal`**; any other string makes FastAPI return `422`.
- **`after_id` can't be used with a non-zero `offset`**, nor with `order=updated_at` — both are rejected when `TableViewRequest` is constructed (a 422 at `["query", "after_id"]` through `query_dependency()`).

## Related reference

- [`TableViewRequest` / `ListResponse` field details](/en/reference/pagination-types)
- [`get_with_count()` full signature](/en/reference/crud-methods#get-with-count)
- [Keyset cursor pagination](./keyset-pagination)
