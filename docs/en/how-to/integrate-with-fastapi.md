# Integrate with FastAPI

**Goal**: write a complete set of CRUD endpoints (GET single / GET list / POST / PATCH / DELETE) for a typical RESTful resource.

**Prerequisites**:

- You already have a table-backed model (inheriting `UUIDTableBaseMixin` or `TableBaseMixin`)
- You already have an `AsyncSession` dependency wired up (commonly `SessionDep`), with sessions built by `SessionFactory(engine, class_=sqlmodel_ext.AsyncSession)`
- You already have an `XxxBase` data model + `XxxResponse` DTO

## 1. Prepare the DTOs: declare each fact only once

```python
from uuid import UUID
from sqlmodel import Field
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, UUIDIdDatetimeInfoMixin,
    NonEmptyStrippedStr64, Str64, Text10K,
)

class ArticleBase(SQLModelBase):
    title: NonEmptyStrippedStr64
    body: Text10K
    subtitle: Str64 | None = None

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id')

class ArticleCreateRequest(ArticleBase):
    """POST body: all fields and constraints are inherited from ArticleBase"""

class ArticleUpdateRequest(ArticleBase, partial=True):
    """PATCH body: inherited fields automatically become ``Unset | T = Unset``, constraints preserved as-is"""

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    """Response DTO: id and timestamps are guaranteed to exist"""
    author_id: UUID
```

Field constraints (length, non-empty, whitespace stripping) are written only once, in `ArticleBase`; the table model, create request, PATCH request and response DTO are all derived from it. Change one constraint and all four change together — there is no second place for it to drift.

The PATCH DTO derived with `partial=True` distinguishes three states:

| Client sends | Field value | In `model_dump()` |
|------|------|------|
| No `subtitle` key | `Unset` | **Absent** — not written |
| `"subtitle": null` | `None` | `{'subtitle': None}` — written as NULL (when the field allows `None`) |
| `"subtitle": "x"` | `'x'` | `{'subtitle': 'x'}` |

`title` does not allow `None` in `ArticleBase`, so `"title": null` in a PATCH is still a 422 — "may be omitted" and "may be null" are two different things. To check in code whether "the client sent it", use `is not Unset`, not `is not None`:

```python
from sqlmodel_ext import Unset

if patch.subtitle is not Unset:
    ...   # the client sent subtitle (possibly None, meaning clear it)
```

`UUIDIdDatetimeInfoMixin` adds three **required** fields, `id: UUID`, `created_at: datetime`, `updated_at: datetime` — reflecting "these fields always have values in a response", as opposed to the table model, where they may be empty before INSERT.

## 2. The five endpoints

```python
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlmodel_ext import ListResponse, TableViewRequest

router = APIRouter(prefix="/articles", tags=["articles"])
TableViewDep = Annotated[TableViewRequest, Depends()]

@router.post("", response_model=ArticleResponse)
async def create_article(
    session: SessionDep,
    current_user: CurrentUserDep,
    data: ArticleCreateRequest,
) -> Article:
    article = Article(**data.model_dump(), author_id=current_user.id)
    return await article.save(session)

@router.get("", response_model=ListResponse[ArticleResponse])
async def list_articles(
    session: SessionDep,
    table_view: TableViewDep,
) -> ListResponse[Article]:
    return await Article.get_with_count(session, table_view=table_view)

@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(
    session: SessionDep,
    article_id: UUID,
) -> Article:
    return await Article.get_exist_one(session, article_id)

@router.patch("/{article_id}", response_model=ArticleResponse)
async def update_article(
    session: SessionDep,
    article_id: UUID,
    data: ArticleUpdateRequest,
) -> Article:
    article = await Article.get_exist_one(session, article_id)
    return await article.update(session, data)

@router.delete("/{article_id}")
async def delete_article(
    session: SessionDep,
    article_id: UUID,
) -> dict[str, int]:
    article = await Article.get_exist_one(session, article_id, with_for_update=True)
    deleted = await Article.delete(session, article)
    return {"deleted": deleted}
```

The `with_for_update=True` in the `delete` endpoint closes the TOCTOU window between "existence check → delete": a concurrent second request blocks, and after the first one commits it finds no row, getting the same 404 as a serial second delete would.

## 3. Application-level exception mapping

```python
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlmodel_ext import OptimisticLockError
from sqlmodel_ext.mixins import KeysetCursorError, ResourceReferencedError

app = FastAPI()

@app.exception_handler(ValidationError)          # cross-field validation of query-parameter DTOs (after_id + offset, etc.)
async def dto_validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={
        "detail": jsonable_encoder(exc.errors(include_url=False, include_context=False)),
    })

@app.exception_handler(KeysetCursorError)        # cursor invalid / unsupported, status_code = 422
async def keyset_error_handler(request: Request, exc: KeysetCursorError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

@app.exception_handler(ResourceReferencedError)  # deleting a row still referenced by a foreign key, status_code = 409
async def referenced_handler(request: Request, exc: ResourceReferencedError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.friendly_message})

@app.exception_handler(OptimisticLockError)      # optimistic lock retries exhausted
async def oplock_handler(request: Request, exc: OptimisticLockError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "Record was modified by someone else. Please refresh and retry."})
```

Why the first handler is needed: FastAPI validates the query parameters of a `Depends()` class dependency one by one (single-field errors are 422), but a cross-field `model_validator` only fires when FastAPI calls `TableViewRequest(...)` to construct the object — left unhandled, that is a 500.

## Key conventions

| Convention | Reason |
|------------|--------|
| Every mutation endpoint uses `await xxx.save(session)` and **uses the return value** | After `commit()` the object is expired; you must use the refreshed instance |
| `get_exist_one()` instead of `get_one()` | Auto-raises `HTTPException(404)` when not found (with FastAPI installed); `detail=` customizes the message |
| List endpoints return `ListResponse[T]` instead of `list[T]` | The `count` field lets the frontend build pagination UI |
| PATCH uses a `partial=True` DTO + `update()` | `Unset` fields don't appear in `model_dump()`, so only the fields the client sent are written |
| Permission checks read with `authoritative=True` | The basis for authorization must be the latest committed value, bypassing the identity map and Redis |

## On permissions and scoping

The code above assumes `CurrentUserDep` already handles authentication. PATCH/DELETE endpoints usually also need to check "is the current user allowed to operate on this record" — that's business logic, and you should check `article.author_id == current_user.id` inside the endpoint yourself. Pass `authoritative=True` on reads used for authorization decisions:

```python
article = await Article.get_one(session, article_id, authoritative=True)
if article.author_id != current_user.id:
    raise HTTPException(403)
```

## On responses containing relation fields

If `ArticleResponse` includes a relation field (e.g. `author: UserResponse`), you must preload it via `load=` at query time, or it will trigger MissingGreenlet. See [Prevent MissingGreenlet errors](./prevent-missing-greenlet).

```python
from sqlmodel_ext import rel

# rel() casts the Relationship field to QueryableAttribute --
# type checkers infer Article.author as User, not a loadable attribute
return await Article.get_exist_one(session, article_id, load=rel(Article.author))
```

## Related reference

- [Full CRUD method signatures](/en/reference/crud-methods)
- [Info response mixins](/en/reference/pagination-types#info-response-mixins-dto)
- [Keyset cursor pagination](./keyset-pagination)
- [Handle deletes of still-referenced rows](./handle-referenced-deletes)
