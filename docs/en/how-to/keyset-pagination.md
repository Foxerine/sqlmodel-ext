# Keyset cursor pagination

**Goal**: iterate sequentially over a list that is subject to concurrent inserts / deletes (infinite scroll, exports, batch scans) without duplicating or skipping records because earlier rows changed.

**Prerequisites**:

- The model inherits `UUIDTableBaseMixin` (keyset cursors only support UUID primary keys)
- You already know how to do offset pagination with `table_view` (see [Paginate a list endpoint](./paginate-a-list-endpoint))

## Why offset is not enough

Offset means "skip the first N rows". After you read page 1, if someone inserts or deletes a row in front of it, the start of page 2 shifts by one — you either get a duplicate or a row you will never see. A keyset cursor instead anchors on "**the last row I read**": the next page = the records ordered after the anchor, and changes in front of it don't affect it (except when the anchor itself is deleted, see below).

Offset is still the right tool for random access ("jump to page 7"); keyset is for sequential traversal.

## 1. First page: no `after_id`

```python
from sqlmodel_ext import TableViewRequest

page = await Article.get(
    session,
    fetch_mode='all',
    table_view=TableViewRequest(limit=20, desc=True),   # default order='created_at'
)
```

## 2. Next page: pass the `id` of the last row of the previous page as `after_id`

```python
next_page = await Article.get(
    session,
    fetch_mode='all',
    table_view=TableViewRequest(limit=20, desc=True, after_id=page[-1].id),  # [!code highlight]
)
```

Repeat until the returned list is empty. `get_with_count()` accepts `after_id` as well; the `count` it returns is the size of the whole filtered set and is **not affected by the cursor**.

In FastAPI the endpoint code doesn't change: with `TableViewRequest` as a dependency, the client simply passes `?after_id=<id of the last row of the previous page>&limit=20` (remember to register the handler that maps `ValidationError` to 422, see the constraints table below).

## How it guarantees no duplicates and no gaps

- The ordering is always the composite key `(sort column, id)`: `id` is the tie-breaker, so rows created in the same millisecond with the same `created_at` are deterministically split across pages, with no gaps and no duplicates.
- The anchor's sort value is looked up server-side from `after_id`; the client **only sends the id**, never a timestamp (avoiding precision loss between database microseconds and transport-layer milliseconds). With `order='id'` the id itself is the anchor value, so no extra query is issued.
- The generated condition is a row-value comparison: with `desc=True` it is `(col < anchor value) OR (col = anchor value AND id < after_id)`.

## 3. Constraints: which combinations are rejected

| Combination | When rejected | Error |
|------|---------|------|
| `after_id` + non-zero `offset` | When constructing `TableViewRequest` | `ValidationError` (in a FastAPI dependency it must be mapped to 422, see [Paginate a list endpoint](./paginate-a-list-endpoint#map-cross-field-validation-errors-to-422)) |
| `after_id` + `order='updated_at'` (or any mutable domain sort column) | At construction | `ValidationError` |
| `after_id` + explicit `order_by=` | At `get()` | `KeysetCursorUnsupportedError` |
| `after_id` + `join=` | At `get()` | `KeysetCursorUnsupportedError` |
| Anchor does not exist, or is outside this query's visible scope (`condition` + `filter` + STI filter) | At `get()` | `KeysetCursorInvalidError` |
| Table without a UUID primary key | At `get()` | `ValueError` (programming error; use offset instead) |

Why so strict:

- **`offset` stacks rather than being an either/or**. If the records after anchor B are C D E F and a leftover `offset=2` remains, the query returns E F — C and D are silently skipped and can never be reached. `offset` defaults to 0, and "forgot to reset it" is the most natural misuse, so this combination is made unrepresentable at construction time.
- **Only immutable sort columns are allowed** (`created_at` / `id`). When ordering by `updated_at`, an anchor that gets updated moves elsewhere, and the meaning of "after the anchor" changes with it.
- **The anchor must be visible**. Otherwise anyone holding the UUID of a row outside your scope could use the difference between "returns a page" and "raises an error" to probe whether that row exists and when it was created (a UUID is an identifier, not a credential). That is why "does not exist" and "not visible" deliberately produce the same error. Time filters do not apply to the anchor (a time window is a page boundary, not a visibility boundary).
- **When the anchor is deleted or no longer matches the filter**, the cursor becomes invalid and `get()` raises instead of returning an empty page that looks like "reached the end" — the client should restart from the first page.

## 4. Map cursor errors to HTTP responses

The three cursor exceptions share the parent class `KeysetCursorError` (which inherits `ValueError`, `status_code = 422`); `str(e)` is a safe message that can be returned to the client directly:

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlmodel_ext.mixins import KeysetCursorError

@app.exception_handler(KeysetCursorError)
async def keyset_cursor_error_handler(request: Request, exc: KeysetCursorError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
```

The message of `KeysetCursorUnsupportedError` does not reveal the specific reason for the rejection (`order_by` or `join`); the reason is only written to the `INFO` log.

## 5. Window only, no cursor: `PageWindowRequest`

If your query fixes its own ordering (e.g. "by price descending") and only needs `offset` / `limit` / `desc`, don't let an `order` / `after_id` that nobody consumes appear in the public schema:

```python
from sqlmodel import col
from sqlmodel_ext.pagination import PageWindowRequest

items = await Article.get(
    session,
    fetch_mode='all',
    order_by=[col(Article.price).desc()],
    table_view=PageWindowRequest(offset=0, limit=20),
)
```

## Related reference

- [`PaginationRequest` / `PageWindowRequest` fields and validation](/en/reference/pagination-types)
- [Cursor-related exceptions of `get()`](/en/reference/crud-methods#get)
- Keyset pagination on cached models: `after_id` is part of the query cache key, so different cursors never share the same cache entry (see [Cache transparency inside transactions](/en/explanation/transactional-cache-transparency))
