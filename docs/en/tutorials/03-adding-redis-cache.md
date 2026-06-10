# 03 · Adding Redis caching

The blog API from tutorial 02 works, but every `GET /articles/{id}` hits the database — a viral article would make your DB cry. Now we add a Redis cache layer to `Article` so repeat reads are zero-SQL, and along the way we resolve the `MissingGreenlet` cliffhanger from tutorial 02.

Estimate: 30 minutes.

## What you'll do

1. Start a local Redis
2. Add `CachedTableBaseMixin` to the `Article` model
3. Configure the Redis client at app startup
4. Verify cache hits (using the SQL log)
5. Add `author: UserResponse` to `ArticleResponse` and use `load=` to fix MissingGreenlet
6. See cache invalidation in action (modifying an article causes the next read to refetch)

## 0. Start Redis

The fastest way is Docker:

```bash
docker run --name blog-redis -p 6379:6379 -d redis:7
```

Confirm it's reachable:

```bash
docker exec -it blog-redis redis-cli ping
# → PONG
```

## 1. Install the async Redis client

Continue from the tutorial 02 directory:

```bash
pip install "redis[hiredis]>=5"
```

## 2. Add caching to `Article`

Open `models.py` and modify the `Article` section:

```python
from sqlmodel_ext import (
    SQLModelBase,
    UUIDTableBaseMixin,
    UUIDIdDatetimeInfoMixin,
    CachedTableBaseMixin,    # ← new
    Str64,
    Str256,
    Text10K,
)

# ... User / UserBase / UserCreateRequest / UserResponse unchanged ...

class Article(
    CachedTableBaseMixin,    # ← new (must be first)  // [!code highlight]
    ArticleBase,
    UUIDTableBaseMixin,
    table=True,
    cache_ttl=600,           # 10 minutes  // [!code highlight]
):
    author_id: UUID = Field(foreign_key="user.id", index=True)
    author: User = Relationship(back_populates="articles")
    comments: list["Comment"] = Relationship(back_populates="article")
```

::: warning MRO order
`CachedTableBaseMixin` **must** appear before `UUIDTableBaseMixin`. It needs to come earlier in the MRO chain so its `get()` / `save()` / `update()` / `delete()` overrides win.
:::

`cache_ttl=600` is a metaclass-handled keyword argument that is translated to `__cache_ttl__: ClassVar[int] = 600`. Default is 3600 seconds.

## 3. Configure Redis in lifespan

Modify `db.py`:

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated

import redis.asyncio as redis                        # ← new
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel_ext import AsyncSession, CachedTableBaseMixin  # ← new (note: the enhanced AsyncSession)

engine = create_async_engine("sqlite+aiosqlite:///blog.db")
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

::: warning Must use sqlmodel_ext.AsyncSession
Since 0.4.0 cache invalidation is orchestrated by the enhanced `AsyncSession.commit()`. The session factory's `class_=` must point at `sqlmodel_ext.AsyncSession` (not `sqlmodel.ext.asyncio.session.AsyncSession`), otherwise invalidation degrades to the fire-and-forget compensation hook with a brief stale window.
:::

```python


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: create tables + configure Redis
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    redis_client = redis.from_url(                      # [!code ++]
        "redis://localhost:6379",                        # [!code ++]
        decode_responses=False,    # ← critical, do not change  // [!code ++]
    )                                                    # [!code ++]
    CachedTableBaseMixin.configure_redis(redis_client)  # [!code ++]
    CachedTableBaseMixin.check_cache_config()           # [!code ++]

    yield

    # Shutdown
    await redis_client.aclose()                          # [!code ++]
    await engine.dispose()


# ... get_session / SessionDep unchanged ...
```

::: danger `decode_responses=False`
Cached values are `bytes` (from `model_dump_json().encode()`). Setting it to `True` makes redis-py decode bytes into str, breaking deserialization.
:::

`check_cache_config()` validates that every subclass of `CachedTableBaseMixin` has a valid positive integer `__cache_ttl__`, and registers SQLAlchemy `after_commit` event hooks (used by the `commit=False` invalidation compensation path).

## 4. Verify cache hits

Restart the server:

```bash
fastapi dev main.py
```

Pick the article ID from tutorial 02 and call `curl` twice in a row:

```bash
curl http://127.0.0.1:8000/articles/<article_id>
curl http://127.0.0.1:8000/articles/<article_id>
```

Watch the SQL log in the `fastapi dev` terminal:

- **First call** logs `SELECT ... FROM article WHERE article.id = ?`
- **Second call** has **no SELECT in the log** — the request hit the ID cache directly (key shaped like `id:Article:550e...`)

::: info Inspect what's in the cache
```bash
docker exec -it blog-redis redis-cli
> KEYS id:Article:*
1) "id:Article:550e8400-..."
> GET id:Article:550e8400-...
"{\"_t\":\"single\",\"_data\":{...},\"_c\":\"Article\"}"
> TTL id:Article:550e8400-...
(integer) 597
```

`_t` is the result type (single / list / none), `_c` is the actual class name (polymorphic safety), `_data` is the result of `model_dump_json()`.
:::

## 5. Verify automatic invalidation

```bash
curl -X PATCH http://127.0.0.1:8000/articles/<article_id> \
  -H "Content-Type: application/json" \
  -d '{"title":"new title"}'

curl http://127.0.0.1:8000/articles/<article_id>
```

The second `curl` hits the database again — why? Because `update()` internally called `_invalidate_for_model()`, which deletes `id:Article:<id>` and bumps the query cache version. The next read is a cache miss → DB query → new entry written.

Business code is completely unaware.

## 6. Resolve the MissingGreenlet cliffhanger from tutorial 02

Now let's make `ArticleResponse` include author info:

```python
# models.py
class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID
    author: UserResponse    # ← new
```

If you don't change the endpoint and just try:

```bash
curl http://127.0.0.1:8000/articles/<article_id>
```

The server explodes:

```
sqlalchemy.exc.InvalidRequestError: 'Article.author' is not available
due to lazy='raise_on_sql'
```

::: info The third defense
Since 0.2.0 sqlmodel-ext sets every `Relationship`'s default `lazy` to `'raise_on_sql'` — accessing an unloaded relation **raises a clear error immediately** instead of triggering an implicit synchronous query that would cause `MissingGreenlet`. It converts a confusing greenlet error into a readable `InvalidRequestError`.
:::

The fix is simple — tell the query to preload `author`:

```python
# main.py
@articles.get("/{article_id}", response_model=ArticleResponse)
async def get_article(session: SessionDep, article_id: UUID) -> Article:
    return await Article.get_exist_one(
        session,
        article_id,
        load=Article.author,    # ← new
    )
```

`load=Article.author` makes sqlmodel-ext use `selectinload(Article.author)` under the hood to pull the author back in one query.

Try it again:

```bash
curl http://127.0.0.1:8000/articles/<article_id>
# → {"id":"...","title":"...","author":{"id":"...","name":"Alice","email":"..."}}
```

::: tip Nested relations also work
If you want the author **and** something related to the author, write `load=[Article.author, User.profile]` — sqlmodel-ext will build the `selectinload(author).selectinload(profile)` chain automatically.
:::

## 7. About bypassing the cache

Some scenarios you don't want to use the cache — for example, immediately after a PATCH you want the freshest read. `get()` accepts `no_cache=True`:

```python
fresh = await Article.get_one(session, article_id, no_cache=True)
```

Usually you don't need it though — `save()` / `update()` already invalidated the cache, so the next normal read picks up the new data.

**Auto-bypass scenarios** (you don't have to specify these):

- `with_for_update=True` (row lock requires fresh data)
- `populate_existing=True`
- non-empty `options=` / `join=` (cannot be hashed stably)
- pending invalidation in the current transaction

## 8. What you just learned

| Concept | Action |
|---------|--------|
| `CachedTableBaseMixin` must be first in the MRO | `class Article(CachedTableBaseMixin, ..., table=True)` |
| `cache_ttl` is a metaclass-only kwarg | `cache_ttl=600` |
| `configure_redis()` is called once at startup | inside lifespan |
| `check_cache_config()` validates every subclass | inside lifespan |
| `decode_responses=False` is non-negotiable | redis client config |
| Cache invalidation is fully automatic | handled inside `save()` / `update()` / `delete()` |
| `lazy='raise_on_sql'` is the MissingGreenlet safety net | no setup needed; on by default |
| `load=` preloads relations | `Article.get_exist_one(..., load=Article.author)` |

## What you can already do

Congratulations — you've finished all three tutorials. This skill set covers about 80% of real-world projects:

- Define models (the Base / Table / CreateRequest / UpdateRequest / Response 5-piece set)
- Full CRUD endpoints (with pagination + PATCH semantics)
- Relations + preloading (avoiding MissingGreenlet)
- Redis caching (with automatic invalidation)

## Where to go next

- **Have a specific task**? Browse the [how-to guides](/en/how-to/) — like "handle concurrent updates" or "define STI polymorphic models".
- **Looking for the precise signature of an API**? Browse the [reference](/en/reference/).
- **Curious about internals**? See [explanation](/en/explanation/) — like [what the metaclass does](/en/explanation/metaclass) or [how the Redis cache implements automatic invalidation](/en/explanation/cached-table).
- **Hit a bug or have a suggestion**? Drop by [GitHub Issues](https://github.com/Foxerine/sqlmodel-ext/issues).
