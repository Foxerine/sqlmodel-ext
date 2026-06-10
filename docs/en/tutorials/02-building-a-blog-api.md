# 02 · Building a blog API

Tutorial 01 taught you the basic CRUD round-trip. This time you're building a **complete project**: a blog backend with users, articles, and comments — wired up with FastAPI endpoints, pagination, relation preloading, and PATCH-style partial updates.

Estimate: 60 minutes. When you're done you'll have a real HTTP service you can hit with `curl`.

## What you'll build

```
┌──────────┐      ┌──────────┐      ┌──────────┐
│  User    │ 1—N  │ Article  │ 1—N  │ Comment  │
│          │←─────│ author_id│←─────│article_id│
└──────────┘      └──────────┘      └──────────┘
```

Each resource gets a complete RESTful surface:

| Method | Path | What it does |
|--------|------|--------------|
| POST | `/users` | Register a user |
| GET | `/users/{id}` | Get a user |
| POST | `/articles` | Publish an article |
| GET | `/articles` | List articles (paginated + time-filtered) |
| GET | `/articles/{id}` | Get an article |
| PATCH | `/articles/{id}` | Partial update |
| POST | `/articles/{id}/comments` | Comment on an article |

## 0. Prep

Continue from the tutorial 01 directory or start fresh:

```bash
pip install sqlmodel-ext aiosqlite "fastapi[standard]"
```

`fastapi[standard]` pulls in uvicorn and other common deps.

## 1. The data layer

Create `models.py`:

```python
from datetime import datetime
from uuid import UUID

from pydantic import EmailStr  # requires: pip install 'pydantic[email]'
from sqlmodel import Field, Relationship
from sqlmodel_ext import (
    SQLModelBase,
    UUIDTableBaseMixin,
    UUIDIdDatetimeInfoMixin,
    NonEmptyStrippedStr64,
    NonEmptyStrippedStr256,
    Text10K,
)


# ============ User ============

class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    """User name (rejects "" and whitespace-only)"""
    email: EmailStr
    """Email"""


class User(UserBase, UUIDTableBaseMixin, table=True):
    articles: list["Article"] = Relationship(back_populates="author")


class UserCreateRequest(UserBase):
    pass


class UserResponse(UserBase, UUIDIdDatetimeInfoMixin):
    pass


# ============ Article ============

class ArticleBase(SQLModelBase):
    title: NonEmptyStrippedStr256
    """Article title (rejects "" and whitespace-only)"""
    body: Text10K
    """Article body"""
    is_published: bool = False
    """Whether the article is published"""


class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key="user.id", index=True)
    author: User = Relationship(back_populates="articles")
    comments: list["Comment"] = Relationship(back_populates="article")


class ArticleCreateRequest(ArticleBase):
    pass


class ArticleUpdateRequest(SQLModelBase):
    title: NonEmptyStrippedStr256 | None = None
    body: Text10K | None = None
    is_published: bool | None = None


class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID


# ============ Comment ============

class CommentBase(SQLModelBase):
    body: Text10K
    """Comment body"""


class Comment(CommentBase, UUIDTableBaseMixin, table=True):
    article_id: UUID = Field(foreign_key="article.id", index=True)
    author_id: UUID = Field(foreign_key="user.id", index=True)
    article: Article = Relationship(back_populates="comments")
    author: User = Relationship()


class CommentCreateRequest(CommentBase):
    pass


class CommentResponse(CommentBase, UUIDIdDatetimeInfoMixin):
    article_id: UUID
    author_id: UUID
```

::: info Why split Base / Table / CreateRequest / UpdateRequest / Response
- **`XxxBase`**: the greatest common factor across every variant (fields needed by both "create" and "response")
- **`Xxx`**: the table model, with foreign keys and `Relationship` added
- **`XxxCreateRequest`**: POST body (inherits Base, all fields required)
- **`XxxUpdateRequest`**: PATCH body (every field optional, defined separately)
- **`XxxResponse`**: response DTO (inherits Base + `UUIDIdDatetimeInfoMixin` to add id and timestamps)

This layering means validation rules are **written once** — the `max_length=256` + non-empty-stripped constraints carried by `NonEmptyStrippedStr256` automatically apply to every subclass of `ArticleBase`.
:::

::: warning Foreign key indexes
`Field(foreign_key=..., index=True)` — PostgreSQL doesn't automatically index FK columns! Add `index=True` manually to avoid full-table scans on reverse queries.
:::

## 2. Database lifespan

Create `db.py`:

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

# Note: SQLite is used for zero-config tutorials; real projects should use PostgreSQL
engine = create_async_engine("sqlite+aiosqlite:///blog.db")
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    # Shutdown: release the connection pool
    await engine.dispose()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
```

## 3. The endpoints

Create `main.py`:

```python
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI
from sqlmodel_ext import ListResponse, TableViewRequest

from db import SessionDep, lifespan
from models import (
    Article, ArticleCreateRequest, ArticleResponse, ArticleUpdateRequest,
    Comment, CommentCreateRequest, CommentResponse,
    User, UserCreateRequest, UserResponse,
)

app = FastAPI(lifespan=lifespan)
TableViewDep = Annotated[TableViewRequest, Depends()]


# ============ Users ============

users = APIRouter(prefix="/users", tags=["users"])

@users.post("", response_model=UserResponse)
async def create_user(session: SessionDep, data: UserCreateRequest) -> User:
    user = User(**data.model_dump())
    return await user.save(session)

@users.get("/{user_id}", response_model=UserResponse)
async def get_user(session: SessionDep, user_id: UUID) -> User:
    return await User.get_exist_one(session, user_id)


# ============ Articles ============

articles = APIRouter(prefix="/articles", tags=["articles"])

@articles.post("", response_model=ArticleResponse)
async def create_article(
    session: SessionDep,
    author_id: UUID,
    data: ArticleCreateRequest,
) -> Article:
    article = Article(**data.model_dump(), author_id=author_id)
    return await article.save(session)

@articles.get("", response_model=ListResponse[ArticleResponse])
async def list_articles(
    session: SessionDep,
    table_view: TableViewDep,
) -> ListResponse[Article]:
    return await Article.get_with_count(
        session,
        Article.is_published == True,
        table_view=table_view,
    )

@articles.get("/{article_id}", response_model=ArticleResponse)
async def get_article(session: SessionDep, article_id: UUID) -> Article:
    return await Article.get_exist_one(session, article_id)

@articles.patch("/{article_id}", response_model=ArticleResponse)
async def update_article(
    session: SessionDep,
    article_id: UUID,
    data: ArticleUpdateRequest,
) -> Article:
    article = await Article.get_exist_one(session, article_id)
    return await article.update(session, data)


# ============ Comments ============

@articles.post("/{article_id}/comments", response_model=CommentResponse)
async def add_comment(
    session: SessionDep,
    article_id: UUID,
    author_id: UUID,
    data: CommentCreateRequest,
) -> Comment:
    # Make sure the article exists first
    await Article.get_exist_one(session, article_id)
    comment = Comment(
        **data.model_dump(),
        article_id=article_id,
        author_id=author_id,
    )
    return await comment.save(session)


app.include_router(users)
app.include_router(articles)
```

## 4. Run it and try

```bash
fastapi dev main.py
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — FastAPI's auto-generated Swagger UI.

Or `curl`:

```bash
# Register a user
curl -X POST http://127.0.0.1:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice","email":"alice@example.com"}'
# → {"id":"550e...","name":"Alice","email":"alice@example.com",...}

# Create an article using the returned id
curl -X POST "http://127.0.0.1:8000/articles?author_id=550e..." \
  -H "Content-Type: application/json" \
  -d '{"title":"Hello sqlmodel-ext","body":"This is my first post.","is_published":true}'

# List articles (paginated)
curl "http://127.0.0.1:8000/articles?offset=0&limit=10"
# → {"count":1,"items":[...]}

# Partial update
curl -X PATCH http://127.0.0.1:8000/articles/<article_id> \
  -H "Content-Type: application/json" \
  -d '{"title":"Updated title"}'
# Note: body and is_published aren't sent, so they keep their original values (PATCH semantics)
```

## 5. Key patterns recap

**"Use the return value"**: every `save()` / `update()` captures its return value. You learned this in tutorial 01, and it's the most common review nit on real PRs.

**`get_exist_one()` vs `get_one()` vs `get()`**:

| Method | When not found |
|--------|----------------|
| `get(condition)` | returns `None` |
| `get_one(id)` | raises `NoResultFound` |
| `get_exist_one(id)` | raises `HTTPException(404)` (with FastAPI installed) |

In endpoints, **always** use `get_exist_one()` — it converts "not found" into a proper 404 response automatically.

**`update(other)`'s PATCH semantics**:

```python
return await article.update(session, data)
```

`update()` defaults to `exclude_unset=True`: only the fields **explicitly set** on `data` are written to the database. If the client only sent `{"title": "new"}`, then `body` and `is_published` are completely untouched. That's exactly HTTP PATCH semantics.

**`ListResponse[T]` instead of `list[T]`**:

```python
@articles.get("", response_model=ListResponse[ArticleResponse])
```

Returns `{count, items}` — the frontend can build pagination UI from `count`. Tutorial 03 uses this.

**`index=True` on foreign keys**:

```python
author_id: UUID = Field(foreign_key="user.id", index=True)
```

PostgreSQL doesn't auto-index foreign keys! This is a "must remember" detail across the project — forgetting it leads to full-table scans on reverse queries.

## 6. What your project looks like now

```
hello-sqlmodel-ext/
├── models.py    # 9 DTOs + 3 table models
├── db.py        # engine + lifespan + SessionDep
├── main.py      # 7 endpoints
└── blog.db      # SQLite database (auto-created)
```

## But there's a hidden trap

If you add `author: UserResponse` to `ArticleResponse`, the list endpoint would explode immediately:

```
greenlet_spawn has not been called; can't call await_only() here.
```

That's the famous `MissingGreenlet` error — accessing an unloaded relation in async land triggers an implicit synchronous query. Tutorial 03 introduces Redis caching and **along the way** teaches you how to handle it (short answer: use `load=`). The full guide lives at [Prevent MissingGreenlet errors](/en/how-to/prevent-missing-greenlet).

## What you just learned

| Concept | Where it appeared |
|---------|-------------------|
| The 5-piece set: Base / Table / CreateRequest / UpdateRequest / Response | every resource in `models.py` |
| Bidirectional `Relationship` + `back_populates` | User ↔ Article ↔ Comment |
| FK `index=True` | `author_id` / `article_id` |
| FastAPI lifespan + `async_sessionmaker` | `db.py` |
| `Annotated[..., Depends()]` for SessionDep / TableViewDep | `db.py` / `main.py` |
| `get_exist_one()` auto-404 | every GET/PATCH/DELETE endpoint |
| `update()`'s PATCH semantics | `update_article` |
| `get_with_count()` + `ListResponse[T]` | `list_articles` |

## Next

Tutorial 03 plugs Redis caching into this project — `Article.get_one()` will hit cache with zero SQL — and shows how to use `load=` for relation preloading to avoid MissingGreenlet.

[Continue to 03 · Adding Redis caching →](./03-adding-redis-cache)
