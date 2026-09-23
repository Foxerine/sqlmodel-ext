# 02 · 构建博客 API

教程 01 教了你最基础的 CRUD。这次我们要做一个**完整的项目**：博客后端，包含用户、文章、评论三个资源，配上 FastAPI 端点、分页、关系预加载、PATCH 局部更新。

预计 60 分钟。完成后你会得到一个能用 `curl` 真实调用的 HTTP 服务。

## 你将构建什么

```
┌──────────┐      ┌──────────┐      ┌──────────┐
│  User    │ 1—N  │ Article  │ 1—N  │ Comment  │
│          │←─────│ author_id│←─────│article_id│
└──────────┘      └──────────┘      └──────────┘
```

每个资源都有完整的 RESTful 端点：

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/users` | 注册用户 |
| GET | `/users/{id}` | 获取用户（含文章列表） |
| POST | `/articles` | 发布文章 |
| GET | `/articles` | 列出文章（分页 + 时间过滤） |
| GET | `/articles/{id}` | 获取文章（含作者和评论） |
| PATCH | `/articles/{id}` | 局部更新文章 |
| POST | `/articles/{id}/comments` | 评论文章 |

## 0. 准备

延续教程 01 的目录或新建一个：

```bash
pip install sqlmodel-ext aiosqlite "fastapi[standard]"
```

`fastapi[standard]` 会顺带装 uvicorn 和其他常用依赖。

## 1. 数据模型层

新建 `models.py`：

```python
from datetime import datetime
from uuid import UUID

from pydantic import EmailStr  # 需要：pip install 'pydantic[email]'
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
    """用户名（拒绝空串与纯空白）"""
    email: EmailStr
    """邮箱"""


class User(UserBase, UUIDTableBaseMixin, table=True):
    articles: list["Article"] = Relationship(back_populates="author")


class UserCreateRequest(UserBase):
    pass


class UserResponse(UserBase, UUIDIdDatetimeInfoMixin):
    pass


# ============ Article ============

class ArticleBase(SQLModelBase):
    title: NonEmptyStrippedStr256
    """文章标题（拒绝空串与纯空白）"""
    body: Text10K
    """正文"""
    is_published: bool = False
    """是否已发布"""


class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key="user.id", index=True)
    author: User = Relationship(back_populates="articles")
    comments: list["Comment"] = Relationship(back_populates="article")


class ArticleCreateRequest(ArticleBase):
    pass


class ArticleUpdateRequest(ArticleBase, partial=True):
    # 继承字段变为 ``Unset | T = Unset``（"没传"是 Unset，不是 None）；
    # max_length / 非空 strip 等约束与 docstring 原样保留
    pass


class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID


# ============ Comment ============

class CommentBase(SQLModelBase):
    body: Text10K
    """评论内容"""


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

::: info 拆 Base / Table / CreateRequest / UpdateRequest / Response 的好处
- **`XxxBase`**：所有变体的最大公约数（"创建"和"响应"都需要的字段）
- **`Xxx`**：表模型，加上外键和 Relationship
- **`XxxCreateRequest`**：POST 请求体（继承 Base，所有字段必填）
- **`XxxUpdateRequest`**：PATCH 请求体（`partial=True` 从 Base 派生：每个字段都可省略，约束与可空性保留）
- **`XxxResponse`**：响应 DTO（继承 Base + `UUIDIdDatetimeInfoMixin` 自动加 id 和时间戳）

这个分层让验证规则**只写一遍**——`NonEmptyStrippedStr256` 携带的 `max_length=256` + 非空 strip 约束自动适用于 `ArticleBase` 的所有子类。
:::

::: warning 外键索引
`Field(foreign_key=..., index=True)`——PostgreSQL 不会自动给外键列建索引！手动加 `index=True` 避免反向查询全表扫描。
:::

## 2. 数据库 lifespan

新建 `db.py`：

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel_ext import AsyncSession

# 注意：教程用 SQLite 是为了零配置；真实项目应该用 PostgreSQL
engine = create_async_engine("sqlite+aiosqlite:///blog.db")
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 启动：建表
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    # 关闭：释放连接池
    await engine.dispose()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
```

## 3. 端点实现

新建 `main.py`：

```python
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI
from sqlmodel_ext import ListResponse, TableViewRequest, query_dependency

from db import SessionDep, lifespan
from models import (
    Article, ArticleCreateRequest, ArticleResponse, ArticleUpdateRequest,
    Comment, CommentCreateRequest, CommentResponse,
    User, UserCreateRequest, UserResponse,
)

app = FastAPI(lifespan=lifespan)
TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]


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
    # 先确认文章存在
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

## 4. 启动并试用

```bash
fastapi dev main.py
```

打开浏览器访问 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)——FastAPI 自动生成的 Swagger UI。

或者用 `curl`：

```bash
# 注册用户
curl -X POST http://127.0.0.1:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice","email":"alice@example.com"}'
# → {"id":"550e...","name":"Alice","email":"alice@example.com",...}

# 用上面返回的 id 发文章
curl -X POST "http://127.0.0.1:8000/articles?author_id=550e..." \
  -H "Content-Type: application/json" \
  -d '{"title":"Hello sqlmodel-ext","body":"This is my first post.","is_published":true}'

# 列文章（分页）
curl "http://127.0.0.1:8000/articles?offset=0&limit=10"
# → {"count":1,"items":[...]}

# 局部更新
curl -X PATCH http://127.0.0.1:8000/articles/<article_id> \
  -H "Content-Type: application/json" \
  -d '{"title":"Updated title"}'
# 注意：body 和 is_published 不传，所以保持原值（PATCH 语义）

# 对基类声明为不可空的字段传 null
curl -X PATCH http://127.0.0.1:8000/articles/<article_id> \
  -H "Content-Type: application/json" \
  -d '{"title":null}'
# → 422：ArticleBase.title 不可为 null，PATCH 请求体继承了这个事实

# 用 keyset 游标取下一页：传入你手上最后一条的 id
curl "http://127.0.0.1:8000/articles?limit=10&after_id=<last_article_id>"
```

## 5. 关键模式回顾

**"用返回值"**：所有 `save()` / `update()` 都接收返回值。这是教程 01 学过的，PR 审查时也是最常被提醒的点。

**`get_exist_one()` vs `get_one()` vs `get()`**：

| 方法 | 找不到时 |
|------|---------|
| `get(condition)` | 返回 `None` |
| `get_one(id)` | 抛 `NoResultFound` |
| `get_exist_one(id)` | 抛 `HTTPException(404)`（FastAPI 已装时） |

端点里**总是**用 `get_exist_one()`——它把"找不到"自动转成 404 响应。

**`update(other)` 的 PATCH 语义**：

```python
return await article.update(session, data)
```

客户端没传的字段在 `data` 上是 `Unset`，而 `Unset` 字段永远不出现在 `model_dump()` 中——所以 `update()` 只写客户端传了的字段。如果客户端只传了 `{"title": "new"}`，那么 `body` 和 `is_published` 完全不动。这正是 HTTP PATCH 的语义，而你无需做任何 `exclude_unset` 之类的簿记。

三种状态，三种行为：**没传**的字段不动；**传了 `null`** 的字段写成 `NULL`——前提是基类声明它可空，否则请求以 422 拒绝；**传了值**的字段按与创建时相同的约束校验。如果需要手动检查某个字段，与 `Unset` 比较（`if data.title is not Unset:`），永远不要与 `None` 比较。

**`ListResponse[T]` 而不是 `list[T]`**：

```python
@articles.get("", response_model=ListResponse[ArticleResponse])
```

返回 `{count, items}`——前端可以基于 `count` 实现分页 UI。教程 03 我们会用到这个。

**offset 分页还是 keyset 分页**：`TableViewRequest` 两者都支持。"跳到第 7 页"用 `offset` 就好；"加载更多" / 无限滚动则传 `after_id`（已展示的最后一条的 id）——并发插入和删除不会让下一页错位。`after_id` 要求 `order=created_at`（默认）或 `order=id`，且不能与非零 `offset` 同用；这两种错误都会在请求模型校验时被拒绝。时间过滤参数（`created_after_datetime` 等）必须带时区，例如 `2026-01-01T00:00:00Z`。

**外键 `index=True`**：

```python
author_id: UUID = Field(foreign_key="user.id", index=True)
```

PostgreSQL 不自动给外键建索引！这是 sqlmodel-ext 项目里反复强调的"不得不手写"细节，因为忘了它会导致反向查询全表扫描。

## 6. 现在你的项目长什么样

```
hello-sqlmodel-ext/
├── models.py    # 9 个 DTO + 3 个表模型
├── db.py        # 引擎 + lifespan + SessionDep
├── main.py      # 7 个端点
└── blog.db      # SQLite 数据库（自动创建）
```

## 但是有个隐患

如果你在 `ArticleResponse` 中加上 `author: UserResponse`，端点会立刻爆炸：

```
sqlalchemy.exc.InvalidRequestError: 'Article.author' is not available due to lazy='raise_on_sql'
```

在异步上下文里访问没预加载的关系字段需要一次隐式同步查询——也就是著名的 `MissingGreenlet` 错误。sqlmodel-ext 默认把每个关系设为 `lazy='raise_on_sql'`，所以你会在访问的那一行得到这个清晰的错误。教程 03 我们会引入 Redis 缓存，**顺便**学怎么处理这个问题（短答：用 `load=`）。完整的避坑指南在 [防止 MissingGreenlet 错误](/how-to/prevent-missing-greenlet)。

## 更进一步

有真实用户之后，两处小改动马上就会派上用场：

**并发编辑**——两个编辑同时 PATCH 同一篇文章。加上 `OptimisticLockMixin`（放在基类列表最前，位于 `SQLModelBase` / `ArticleBase` 之前），表会多一个 `oplock_version` 列，每次 UPDATE 都会校验它。发生冲突时 `update()` 重新读取该行、只重新应用本次请求的变更，默认最多重试 3 次，之后才抛出 `OptimisticLockError`：

```python
from sqlmodel_ext import OptimisticLockMixin

class Article(OptimisticLockMixin, ArticleBase, UUIDTableBaseMixin, table=True):
    ...
```

**删除仍被引用的记录**——在 PostgreSQL 上，删除一篇仍有评论（且外键为 `RESTRICT`）的文章时，`delete()` 抛出 `ResourceReferencedError`（从 `sqlmodel_ext.mixins` 导入）；把它映射为 HTTP 409。参见 [配置级联删除](/how-to/configure-cascade-delete)。

在 PostgreSQL 上，你还会用 `JSON100K`（`from sqlmodel_ext.field_types.dialects.postgresql import JSON100K`）保存自由格式的文章元数据：它存为 JSONB，拒绝超过 100,000 字符的载荷，返回给客户端时是 JSON 对象而不是字符串。

## 你刚才学到了什么

| 概念 | 出现在 |
|------|------|
| Base / Table / CreateRequest / UpdateRequest / Response 五件套 | `models.py` 的每个资源 |
| 双向 `Relationship` + `back_populates` | User ↔ Article ↔ Comment |
| 外键 `index=True` | `author_id` / `article_id` |
| FastAPI lifespan + `async_sessionmaker` | `db.py` |
| `Annotated[..., Depends(...)]` 创建 SessionDep / TableViewDep（`query_dependency()`：跨字段错误是 422） | `db.py` / `main.py` |
| `get_exist_one()` 自动 404 | 所有 GET/PATCH/DELETE 端点 |
| `partial=True` + `Unset`：无需簿记的 PATCH 语义 | `ArticleUpdateRequest` / `update_article` |
| `get_with_count()` + `ListResponse[T]`，offset 或 `after_id` keyset | `list_articles` |

## 下一步

教程 03 会在这个项目上接入 Redis 缓存，让 `Article.get_one()` 命中缓存时零 SQL；同时学怎么用 `load=` 处理关系预加载，避免 MissingGreenlet。

[继续到 03 · 给博客加 Redis 缓存 →](./03-adding-redis-cache)
