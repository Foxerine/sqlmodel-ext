# Integrate with FastAPI

**Goal**: write a complete set of CRUD endpoints (GET single / GET list / POST / PATCH / DELETE) for a typical RESTful resource.

**Prerequisites**:

- You already have a table-backed model (inheriting `UUIDTableBaseMixin` or `TableBaseMixin`)
- You already have an `AsyncSession` dependency wired up (commonly `SessionDep`)
- You already have an `XxxBase` data model + `XxxResponse` DTO

## 1. Prepare the DTOs

```python
from sqlmodel_ext import SQLModelBase, UUIDIdDatetimeInfoMixin, NonEmptyStrippedStr64, Text10K

class ArticleBase(SQLModelBase):
    title: NonEmptyStrippedStr64
    body: Text10K

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id')

class ArticleCreateRequest(ArticleBase):
    """POST body: all fields are required"""
    pass

class ArticleUpdateRequest(ArticleBase, all_fields_optional=True):
    """PATCH body: inherited fields auto-convert to ``T | None = None``, constraints preserved (no hand-writing)"""
    pass

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    """Response DTO: id and timestamps are guaranteed to exist"""
    author_id: UUID
```

`UUIDIdDatetimeInfoMixin` adds **required** `id: UUID`, `created_at: datetime`, `updated_at: datetime` — reflecting "these fields always have values in API responses", in contrast to `id: UUID | None` on the table model (which is None before INSERT).

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
    article = await Article.get_exist_one(session, article_id)
    deleted = await Article.delete(session, article)
    return {"deleted": deleted}
```

## Key conventions

| Convention | Reason |
|------------|--------|
| Every mutation endpoint uses `await xxx.save(session)` and **uses the return value** | After `commit()` the object is expired; you must use the refreshed instance |
| `get_exist_one()` instead of `get_one()` | Auto-raises `HTTPException(404)` when not found (with FastAPI installed) |
| List endpoints return `ListResponse[T]` instead of `list[T]` | The `count` field lets the frontend build pagination UI |
| `PATCH` uses `update(other)` instead of `save()` | `update()` defaults to `exclude_unset=True`, i.e. PATCH semantics |

## On permissions and scoping

The code above assumes `CurrentUserDep` already handles authentication. In real projects, PATCH/DELETE endpoints typically also need to check "does the current user own this record" — that's business logic, sqlmodel-ext doesn't manage it directly. Check `article.author_id == current_user.id` inside the endpoint.

## On responses containing relation fields

If `ArticleResponse` includes a relation field (e.g. `author: UserResponse`), you **must** preload it via `load=` at query time, or the response will trigger MissingGreenlet. See [Prevent MissingGreenlet errors](./prevent-missing-greenlet).

```python
from sqlmodel_ext import rel

# rel() casts the Relationship field to QueryableAttribute --
# type checkers infer Article.author as User, not a loadable attribute
return await Article.get_exist_one(session, article_id, load=rel(Article.author))
```

## Related reference

- [Full CRUD method signatures](/en/reference/crud-methods)
- [Info response mixins](/en/reference/pagination-types#info-response-mixins-dto)
