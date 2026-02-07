# sqlmodel-ext

[![PyPI version](https://img.shields.io/pypi/v/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![Python versions](https://img.shields.io/pypi/pyversions/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | **中文**

SQLModel 增强基础设施：智能元类、异步 CRUD Mixin、多态继承、乐观锁、关系预加载、可复用字段类型。

**sqlmodel-ext** 消除了使用 [SQLModel](https://sqlmodel.tiangolo.com/) 构建异步数据库应用时的样板代码。定义模型、继承一个 Mixin，即可获得完整的异步 CRUD API -- 分页、关系加载、多态查询、乐观锁一应俱全。

## 特性

| 特性 | 说明 |
|------|------|
| **SQLModelBase** | 智能元类，自动设置 `table=True`、合并 `mapper_args`，兼容 Python 3.14 (PEP 649) |
| **TableBaseMixin / UUIDTableBaseMixin** | 完整异步 CRUD：`add()`、`save()`、`update()`、`delete()`、`get()`、`count()`、`get_with_count()`、`get_exist_one()` |
| **PolymorphicBaseMixin** | 简化联表继承 (JTI) 和单表继承 (STI) 配置 |
| **AutoPolymorphicIdentityMixin** | 根据类名自动生成 `polymorphic_identity` |
| **OptimisticLockMixin** | 基于版本号的乐观锁，支持自动重试 |
| **RelationPreloadMixin** | 基于装饰器的关系自动预加载（防止 `MissingGreenlet` 错误） |
| **ListResponse[T]** | 泛型分页响应模型，适用于列表接口 |
| **字段类型** | 可复用的约束类型：`Str64`、`Port`、`IPAddress`、`HttpUrl`、`SafeHttpUrl` 等 |
| **响应 DTO Mixin** | 预构建的 API 响应模型 Mixin，包含 id/时间戳字段 |

## 安装

```bash
pip install sqlmodel-ext
```

配合 [FastAPI](https://fastapi.tiangolo.com/) 使用（启用 `get_exist_one()` 中的 `HTTPException`）：

```bash
pip install sqlmodel-ext[fastapi]
```

## 快速开始

### 定义模型

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64

# Base 类 -- 仅定义字段，不创建数据库表
class UserBase(SQLModelBase):
    name: Str64
    email: str

# Table 类 -- 继承字段 + 获得异步 CRUD + UUID 主键
class User(UserBase, UUIDTableBaseMixin, table=True):
    pass
```

`SQLModelBase` 是所有模型的基础，其元类自动：
- 检测到继承链中有 `TableBaseMixin` 时自动设置 `table=True`
- 合并父类的 `__mapper_args__`
- 从 `Annotated` 元数据中提取 `sa_type` 用于列映射
- 应用 Python 3.14 (PEP 649) 兼容性补丁

### 异步 CRUD

所有 CRUD 方法均为异步，需要 `AsyncSession`：

```python
from sqlmodel.ext.asyncio.session import AsyncSession

async def demo(session: AsyncSession):
    # 创建
    user = User(name="Alice", email="alice@example.com")
    user = await user.save(session)  # 必须使用返回值！

    # 查询 -- 单条记录
    user = await User.get(session, User.email == "alice@example.com")

    # 查询 -- 所有记录
    all_users = await User.get(session, fetch_mode="all")

    # 查询 -- 分页和排序
    recent_users = await User.get(
        session,
        fetch_mode="all",
        offset=0,
        limit=20,
        order_by=[User.created_at.desc()],
    )

    # 更新
    user = await user.update(session, UserUpdateRequest(name="Bob"))

    # 删除 -- 按实例
    await User.delete(session, user)

    # 删除 -- 按条件
    await User.delete(session, condition=User.email == "old@example.com")
```

> **重要**：`save()` 和 `update()` 会在 commit 后使所有 session 对象过期。务必使用返回值。

### FastAPI 示例

完整的 REST API -- 模型、DTO 和五个端点：

```python
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlmodel import Field
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, Str64, Text10K,
    ListResponse, TableViewRequest, UUIDIdDatetimeInfoMixin,
)

# ── 依赖注入（定义一次，处处复用）─────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_session)]
TableViewDep = Annotated[TableViewRequest, Depends()]

# ── 模型 ─────────────────────────────────────────────────────────

class ArticleBase(SQLModelBase):
    title: Str64
    body: Text10K
    is_published: bool = False

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id')

class ArticleCreate(ArticleBase):
    pass

class ArticleUpdate(ArticleBase):
    title: Str64 | None = None       # 覆盖为可选，
    body: Text10K | None = None      # 同时保留 Base 中的
    is_published: bool | None = None  # 原始类型约束

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID

# ── 端点 ─────────────────────────────────────────────────────────

router = APIRouter(prefix="/articles", tags=["articles"])

@router.post("", response_model=ArticleResponse)
async def create_article(
        session: SessionDep, data: ArticleCreate, user: CurrentUserDep,
) -> Article:
    article = Article(**data.model_dump(), author_id=user.id)
    return await article.save(session)

@router.get("", response_model=ListResponse[ArticleResponse])
async def list_articles(
        session: SessionDep, table_view: TableViewDep,
) -> ListResponse[Article]:
    return await Article.get_with_count(
        session,
        Article.is_published == True,
        table_view=table_view,
    )

@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(session: SessionDep, article_id: UUID) -> Article:
    return await Article.get_exist_one(session, article_id)

@router.patch("/{article_id}", response_model=ArticleResponse)
async def update_article(
        session: SessionDep, article_id: UUID, data: ArticleUpdate,
) -> Article:
    article = await Article.get_exist_one(session, article_id)
    return await article.update(session, data)

@router.delete("/{article_id}")
async def delete_article(session: SessionDep, article_id: UUID) -> None:
    article = await Article.get_exist_one(session, article_id)
    await Article.delete(session, article)
```

无需手写 SQL、无需手工分页逻辑、无需样板 session 管理。`TableViewDep` 开箱即用地为客户端提供 `offset`、`limit`、`desc`、`order` 和四个时间过滤参数。

**客户端请求 `GET /articles?offset=0&limit=10&desc=true` 得到的响应：**

```json
{
  "count": 42,
  "items": [
    {
      "id": "a1b2c3d4-...",
      "title": "Hello World",
      "body": "...",
      "is_published": true,
      "author_id": "e5f6g7h8-...",
      "created_at": "2024-06-15T10:30:00",
      "updated_at": "2024-06-15T10:30:00"
    }
  ]
}
```

#### 传统写法（不使用 sqlmodel-ext）

同样的五个端点，使用原生 SQLModel + SQLAlchemy 编写：

```python
from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, desc as sa_desc, asc as sa_asc
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

# ── 模型 ─────────────────────────────────────────────────────────

class ArticleBase(SQLModel):
    title: str = Field(max_length=64)
    body: str = Field(max_length=10000)
    is_published: bool = False

class Article(ArticleBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    author_id: UUID = Field(foreign_key='user.id')
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ArticleCreate(ArticleBase):
    pass

class ArticleUpdate(SQLModel):
    title: str | None = Field(default=None, max_length=64)
    body: str | None = Field(default=None, max_length=10000)
    is_published: bool | None = None

class ArticleResponse(ArticleBase):
    id: UUID
    author_id: UUID
    created_at: datetime
    updated_at: datetime

class ArticleListResponse(SQLModel):
    count: int
    items: list[ArticleResponse]

# ── 端点 ─────────────────────────────────────────────────────────

router = APIRouter(prefix="/articles", tags=["articles"])

@router.post("", response_model=ArticleResponse)
async def create_article(
        session: SessionDep, data: ArticleCreate, user: CurrentUserDep,
) -> Article:
    article = Article(**data.model_dump(), author_id=user.id)
    session.add(article)
    await session.commit()
    await session.refresh(article)
    return article

@router.get("", response_model=ArticleListResponse)
async def list_articles(
        session: SessionDep,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, le=100),
        desc: bool = True,
        order: str = Query(default="created_at", pattern="^(created_at|updated_at)$"),
        created_after: datetime | None = None,
        created_before: datetime | None = None,
) -> ArticleListResponse:
    # 计数查询
    count_stmt = select(func.count()).select_from(Article).where(Article.is_published == True)
    if created_after:
        count_stmt = count_stmt.where(Article.created_at >= created_after)
    if created_before:
        count_stmt = count_stmt.where(Article.created_at < created_before)
    total = await session.scalar(count_stmt) or 0

    # 数据查询
    stmt = select(Article).where(Article.is_published == True)
    if created_after:
        stmt = stmt.where(Article.created_at >= created_after)
    if created_before:
        stmt = stmt.where(Article.created_at < created_before)
    order_col = Article.created_at if order == "created_at" else Article.updated_at
    stmt = stmt.order_by(sa_desc(order_col) if desc else sa_asc(order_col))
    stmt = stmt.offset(offset).limit(limit)
    result = await session.exec(stmt)
    items = list(result.all())

    return ArticleListResponse(count=total, items=items)

@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(session: SessionDep, article_id: UUID) -> Article:
    article = await session.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Not found")
    return article

@router.patch("/{article_id}", response_model=ArticleResponse)
async def update_article(
        session: SessionDep, article_id: UUID, data: ArticleUpdate,
) -> Article:
    article = await session.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Not found")
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(article, key, value)
    article.updated_at = datetime.now()
    session.add(article)
    await session.commit()
    await session.refresh(article)
    return article

@router.delete("/{article_id}")
async def delete_article(session: SessionDep, article_id: UUID) -> None:
    article = await session.get(Article, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Not found")
    await session.delete(article)
    await session.commit()
```

**对比：**

| 关注点 | 传统写法 | sqlmodel-ext |
|--------|----------|------------|
| 主键 + 时间戳 | 手动定义 4 个字段 | 继承 `UUIDTableBaseMixin` |
| 分页 + 排序 | 每个列表端点约 20 行 | `table_view=table_view`（一个参数） |
| 计数 + 分页数据 | 两次查询，手动组装 | `get_with_count()`（一次调用） |
| 查找或 404 | `session.get()` + `if not` + `raise HTTPException` | `get_exist_one()`（一次调用） |
| 局部更新 | `model_dump(exclude_unset)` + `for/setattr` 循环 + 手动更新 `updated_at` | `article.update(session, data)` |
| 时间过滤 | 每个字段手写 `if/where` | 内置于 `TableViewRequest` |
| 响应 DTO 时间戳 | 手动定义 `id`、`created_at`、`updated_at` 字段 | 继承 `UUIDIdDatetimeInfoMixin` |
| 乐观锁 | 未包含（需要大量额外工作） | 模型添加 `OptimisticLockMixin` |

**多态端点** 同样简洁：

```python
from abc import ABC, abstractmethod
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin, create_subclass_id_mixin,
    ListResponse, TableViewRequest,
)

# ── 多态模型 ─────────────────────────────────────────────────────

class NotificationBase(SQLModelBase):
    user_id: UUID = Field(foreign_key='user.id')
    message: str

class Notification(NotificationBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
    @abstractmethod
    def summary(self) -> str: ...

NotifSubclassId = create_subclass_id_mixin('notification')

class EmailNotification(NotifSubclassId, Notification, AutoPolymorphicIdentityMixin, table=True):
    email_to: str

    def summary(self) -> str:
        return f"Email to {self.email_to}: {self.message}"

class PushNotification(NotifSubclassId, Notification, AutoPolymorphicIdentityMixin, table=True):
    device_token: str

    def summary(self) -> str:
        return f"Push to {self.device_token}: {self.message}"

# ── 一个端点返回所有通知类型 ──────────────────────────────────────

@router.get("/notifications", response_model=ListResponse[NotificationBase])
async def list_notifications(
        session: SessionDep, user: CurrentUserDep, table_view: TableViewDep,
) -> ListResponse[Notification]:
    return await Notification.get_with_count(
        session,
        Notification.user_id == user.id,
        table_view=table_view,
    )
    # 透明返回 EmailNotification 和 PushNotification 实例
```

---

## 详细指南

### TableBaseMixin 与 UUIDTableBaseMixin

这两个 Mixin 提供异步 CRUD 接口。`TableBaseMixin` 使用自增整数主键；`UUIDTableBaseMixin` 使用 UUID4 主键。

两者均自动添加 `id`、`created_at` 和 `updated_at` 字段。

```python
from sqlmodel_ext import SQLModelBase, TableBaseMixin, UUIDTableBaseMixin

# 整数主键
class LogEntry(SQLModelBase, TableBaseMixin, table=True):
    message: str

# UUID 主键（大多数场景推荐）
class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str
```

#### `add()` -- 批量插入

```python
users = [User(name="Alice", email="a@x.com"), User(name="Bob", email="b@x.com")]
users = await User.add(session, users)

# 也支持单条
user = await User.add(session, User(name="Alice", email="a@x.com"))
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session` | `AsyncSession` | 必填 | 异步数据库会话 |
| `instances` | `T \| list[T]` | 必填 | 要插入的实例 |
| `refresh` | `bool` | `True` | commit 后是否 refresh 以同步数据库生成的值 |

#### `save()` -- 插入或更新

```python
# 基本保存
user = await user.save(session)

# 保存后预加载关系
user = await user.save(session, load=User.profile)

# 乐观锁自动重试
user = await user.save(session, optimistic_retry_count=3)

# 跳过 refresh（不从数据库重新获取）
user = await user.save(session, refresh=False)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session` | `AsyncSession` | 必填 | 异步数据库会话 |
| `load` | `RelationshipInfo \| list` | `None` | 保存后预加载的关系 |
| `refresh` | `bool` | `True` | 保存后是否从数据库刷新对象 |
| `commit` | `bool` | `True` | 是否提交事务。批量操作时设为 `False` |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | 多态子类加载（需配合 `load`） |
| `optimistic_retry_count` | `int` | `0` | 乐观锁冲突时的自动重试次数 |

**使用 `commit=False` 的批量操作：**

插入多条记录时，可延迟 commit 以减少数据库往返：

```python
await user1.save(session, commit=False)  # 仅 flush
await user2.save(session, commit=False)  # 仅 flush
user3 = await user3.save(session)        # 一次性 commit 全部三条
```

#### `update()` -- 从模型实例局部更新

```python
class UserUpdate(SQLModelBase):
    name: str | None = None
    email: str | None = None

# 仅更新显式设置的字段
user = await user.update(session, UserUpdate(name="Charlie"))

# 附加更新模型以外的字段
user = await user.update(
    session,
    update_request,
    extra_data={"updated_by": current_user.id},
)

# 排除指定字段
user = await user.update(session, data, exclude={"role", "is_admin"})
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session` | `AsyncSession` | 必填 | 异步数据库会话 |
| `other` | `SQLModelBase` | 必填 | 数据来源模型实例，其已设置字段将合并到 self |
| `extra_data` | `dict` | `None` | `other` 之外的额外更新字段 |
| `exclude_unset` | `bool` | `True` | 为 `True` 时跳过 `other` 中未显式设置的字段 |
| `exclude` | `set[str]` | `None` | 要排除的字段名称集合 |
| `load` | `RelationshipInfo \| list` | `None` | 更新后预加载的关系 |
| `refresh` | `bool` | `True` | 更新后是否从数据库刷新对象 |
| `commit` | `bool` | `True` | 是否提交事务 |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | 多态子类加载（需配合 `load`） |
| `optimistic_retry_count` | `int` | `0` | 乐观锁冲突时的自动重试次数 |

#### `delete()` -- 按实例或条件删除

```python
# 按实例删除
deleted_count = await User.delete(session, user)

# 按列表删除
deleted_count = await User.delete(session, [user1, user2])

# 按条件批量删除
deleted_count = await User.delete(session, condition=User.is_active == False)

# 不提交（用于事务批量操作）
await User.delete(session, user, commit=False)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session` | `AsyncSession` | 必填 | 异步数据库会话 |
| `instances` | `T \| list[T]` | `None` | 要删除的实例 |
| `condition` | `BinaryExpression` | `None` | 批量删除的 WHERE 条件 |
| `commit` | `bool` | `True` | 是否提交事务 |

`instances` 和 `condition` 二选一，不可同时提供。

#### `get()` -- 灵活查询

`get()` 是核心查询方法，支持过滤、分页、排序、JOIN、关系加载、多态查询、时间过滤和行锁。

```python
# 按条件查询单条
user = await User.get(session, User.email == "alice@example.com")

# 多条件（使用 & 运算符）
user = await User.get(
    session,
    (User.name == "Alice") & (User.is_active == True),
)

# 查询所有
users = await User.get(session, fetch_mode="all")

# 预加载关系
user = await User.get(
    session,
    User.id == user_id,
    load=[User.profile, User.orders],
)

# JOIN 查询
orders = await Order.get(
    session,
    Order.total > 100,
    join=User,
    fetch_mode="all",
)

# FOR UPDATE 行锁
user = await User.get(
    session,
    User.id == user_id,
    with_for_update=True,
)

# 时间过滤
recent = await User.get(
    session,
    fetch_mode="all",
    created_after_datetime=datetime(2024, 1, 1),
    created_before_datetime=datetime(2024, 12, 31),
)
```

**fetch_mode 模式：**

| 模式 | 返回值 | 行为 |
|------|--------|------|
| `"first"`（默认） | `T \| None` | 返回第一条结果或 `None` |
| `"one"` | `T` | 返回恰好一条结果；未找到或多条时抛异常 |
| `"all"` | `list[T]` | 返回所有匹配记录 |

#### `count()` -- 高效计数

```python
total = await User.count(session)
active = await User.count(session, User.is_active == True)

# 带时间过滤
from sqlmodel_ext import TimeFilterRequest
recent_count = await User.count(
    session,
    time_filter=TimeFilterRequest(
        created_after_datetime=datetime(2024, 1, 1),
    ),
)
```

#### `get_with_count()` -- 分页响应

返回 `ListResponse[T]`，同时包含总数和分页数据：

```python
from sqlmodel_ext import ListResponse, TableViewRequest

result = await User.get_with_count(
    session,
    User.is_active == True,
    table_view=TableViewRequest(offset=0, limit=20, desc=True),
)
# result.count -> 匹配总数（如 150）
# result.items -> 20 条 User 实例
```

#### `get_exist_one()` -- 查找或 404

```python
# 安装了 FastAPI 时抛出 HTTPException(404)，否则抛出 RecordNotFoundError
user = await User.get_exist_one(session, user_id)
```

---

### 分页模型

sqlmodel-ext 提供开箱即用的分页和时间过滤请求模型：

```python
from sqlmodel_ext import ListResponse, TableViewRequest, TimeFilterRequest, PaginationRequest
```

**`TableViewRequest`** 组合了分页 + 排序 + 时间过滤：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `offset` | `int` | `0` | 跳过前 N 条记录 |
| `limit` | `int` | `50` | 每页最大记录数（上限 100） |
| `desc` | `bool` | `True` | 是否降序排列 |
| `order` | `"created_at" \| "updated_at"` | `"created_at"` | 排序字段 |
| `created_after_datetime` | `datetime \| None` | `None` | 过滤 `created_at >= 值` |
| `created_before_datetime` | `datetime \| None` | `None` | 过滤 `created_at < 值` |
| `updated_after_datetime` | `datetime \| None` | `None` | 过滤 `updated_at >= 值` |
| `updated_before_datetime` | `datetime \| None` | `None` | 过滤 `updated_at < 值` |

**`ListResponse[T]`** 是标准分页响应：

```python
from sqlmodel_ext import ListResponse

# 配合 FastAPI 使用
@router.get("", response_model=ListResponse[UserResponse])
async def list_users(session: SessionDep, table_view: TableViewDep) -> ListResponse[User]:
    return await User.get_with_count(session, table_view=table_view)
```

---

### 多态继承

sqlmodel-ext 同时支持联表继承 (JTI) 和单表继承 (STI)，简化了 SQLAlchemy 繁琐的多态配置。

#### 联表继承 (JTI)

每个子类拥有独立的数据库表，通过外键关联父表。适用于子类字段差异较大的场景。

```python
from abc import ABC, abstractmethod
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
)

# 1. Base 类（仅定义字段，无表）
class ToolBase(SQLModelBase):
    name: str

# 2. 抽象父类（创建父表）
class Tool(ToolBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
    @abstractmethod
    async def execute(self) -> str: ...

# 3. 创建子类的外键 Mixin
ToolSubclassIdMixin = create_subclass_id_mixin('tool')

# 4. 具体子类（各自拥有独立表）
class WebSearchTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    search_url: str

    async def execute(self) -> str:
        return f"Searching {self.search_url}"

class CalculatorTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    precision: int = 2

    async def execute(self) -> str:
        return "Calculating..."
```

**核心组件：**

| 组件 | 作用 |
|------|------|
| `PolymorphicBaseMixin` | 自动配置 `polymorphic_on`，添加 `_polymorphic_name` 鉴别列 |
| `create_subclass_id_mixin(table)` | 创建含 FK+PK `id` 字段的 Mixin，指向父表 |
| `AutoPolymorphicIdentityMixin` | 根据类名（小写）自动生成 `polymorphic_identity` |

**MRO 顺序很重要：** `SubclassIdMixin` 必须放在继承列表第一位以正确覆盖 `id` 字段：

```python
# 正确
class MyTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True): ...

# 错误 -- id 字段不会被正确覆盖
class MyTool(Tool, ToolSubclassIdMixin, AutoPolymorphicIdentityMixin, table=True): ...
```

#### 单表继承 (STI)

所有子类共享父表。子类特有的列作为 nullable 添加到父表。适用于子类额外字段较少的场景。

```python
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)

class UserFile(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    filename: str

class PendingFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    upload_deadline: datetime | None = None  # 作为 nullable 列添加到 userfile 表

class CompletedFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    file_size: int | None = None  # 作为 nullable 列添加到 userfile 表

# 所有模型定义完成后，在 configure_mappers() 之前调用：
register_sti_columns_for_all_subclasses()
# 在 configure_mappers() 之后调用：
register_sti_column_properties_for_all_subclasses()
```

#### 查询多态模型

```python
# 获取所有工具（返回具体子类实例）
tools = await Tool.get(session, fetch_mode="all")
# tools[0] 可能是 WebSearchTool，tools[1] 可能是 CalculatorTool

# 加载多态关系
from sqlmodel_ext import UUIDTableBaseMixin
from sqlmodel import Relationship

class ToolSet(SQLModelBase, UUIDTableBaseMixin, table=True):
    tools: list[Tool] = Relationship(back_populates="tool_set")

# 加载工具及所有子类数据
tool_set = await ToolSet.get(
    session,
    ToolSet.id == ts_id,
    load=ToolSet.tools,
    jti_subclasses='all',  # 加载所有子类特有的列
)
```

#### 多态工具方法

```python
# 获取所有具体（非抽象）子类
subclasses = Tool.get_concrete_subclasses()
# [WebSearchTool, CalculatorTool]

# 获取 identity 到类的映射
mapping = Tool.get_identity_to_class_map()
# {'websearchtool': WebSearchTool, 'calculatortool': CalculatorTool}

# 检查继承类型
Tool._is_joined_table_inheritance()  # JTI 返回 True，STI 返回 False
```

---

### 乐观锁

利用 SQLAlchemy 的 `version_id_col` 机制防止并发环境下的更新丢失。

```python
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    OptimisticLockMixin, OptimisticLockError,
)

# OptimisticLockMixin 在 MRO 中必须位于 TableBaseMixin 之前
class Order(OptimisticLockMixin, UUIDTableBaseMixin, table=True):
    status: str
    amount: int
```

该 Mixin 添加一个 `version` 整数字段（初始值为 0）。每次 `UPDATE` 生成如下 SQL：

```sql
UPDATE "order" SET status=?, amount=?, version=version+1
WHERE id=? AND version=?
```

如果 `WHERE` 条件不匹配（另一个事务修改了该记录），更新影响 0 行，抛出 `OptimisticLockError`。

#### 手动处理冲突

```python
try:
    order = await order.save(session)
except OptimisticLockError as e:
    print(f"冲突: {e.model_class} id={e.record_id}")
    print(f"期望版本: {e.expected_version}")
    # 重新查询并重试...
```

#### 自动重试（推荐）

```python
# 冲突时最多重试 3 次：
# 1. 重新从数据库获取最新记录
# 2. 重新应用你的修改
# 3. 再次尝试保存
order = await order.save(session, optimistic_retry_count=3)

# update() 同样支持
order = await order.update(session, update_data, optimistic_retry_count=3)
```

**适用场景：**
- 状态转换（待支付 -> 已支付 -> 已发货）
- 并发修改的数值字段（余额、库存）

**不适用场景：**
- 日志/审计表（仅插入）
- 简单计数器（`UPDATE SET count = count + 1` 即可）

---

### 关系预加载

`RelationPreloadMixin` 和 `@requires_relations` 装饰器在方法执行前自动加载关系，防止异步 SQLAlchemy 中的 `MissingGreenlet` 错误。

```python
from sqlmodel import Relationship
from sqlmodel_ext import UUIDTableBaseMixin, SQLModelBase
from sqlmodel_ext.mixins import RelationPreloadMixin, requires_relations

class GeneratorConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    price: int

class Generator(SQLModelBase, UUIDTableBaseMixin, table=True):
    config: GeneratorConfig = Relationship()

class MyFunction(SQLModelBase, UUIDTableBaseMixin, RelationPreloadMixin, table=True):
    generator: Generator = Relationship()

    @requires_relations('generator', Generator.config)
    async def calculate_cost(self, session) -> int:
        # generator 和 generator.config 在执行前自动加载
        return self.generator.config.price * 10
```

**工作原理：**

1. `@requires_relations` 声明方法需要的关系
2. 方法执行前，装饰器通过 `sqlalchemy.inspect` 检查哪些关系已加载
3. 未加载的关系通过单次查询获取
4. 已加载的关系被跳过（增量加载）

**支持的参数格式：**

```python
@requires_relations(
    'generator',           # 字符串：本类的属性名
    Generator.config,      # RelationshipInfo：外部类属性（嵌套关系）
)
```

**也支持异步生成器：**

```python
@requires_relations('items')
async def stream_items(self, session):
    for item in self.items:
        yield item
```

**导入时验证：** 字符串关系名在类创建时即被验证。如果声明 `@requires_relations('nonexistent')`，会立即得到 `AttributeError`，而非等到运行时。

**手动预加载 API**（通常不需要）：

```python
# 为指定方法预加载关系
await instance.preload_for(session, 'calculate_cost', 'validate')

# 获取方法的关系列表（用于构建查询）
rels = MyFunction.get_relations_for_method('calculate_cost')
rels = MyFunction.get_relations_for_methods('calculate_cost', 'validate')
```

---

### 字段类型

sqlmodel-ext 提供可复用的 `Annotated` 类型别名，同时兼容 Pydantic 验证和 SQLAlchemy 列映射。

#### 字符串约束

| 类型 | 最大长度 | 用途 |
|------|----------|------|
| `Str24` | 24 | 短编码 |
| `Str32` | 32 | Token、哈希 |
| `Str36` | 36 | UUID 字符串 |
| `Str48` | 48 | 短标签 |
| `Str64` | 64 | 名称、标题 |
| `Str100` | 100 | 简短描述 |
| `Str128` | 128 | 路径、标识符 |
| `Str255` | 255 | 标准 VARCHAR |
| `Str256` | 256 | 标准 VARCHAR |
| `Text1K` | 1,000 | 短文本 |
| `Text1024` | 1,024 | 短文本（2的幂） |
| `Text2K` | 2,000 | 中等文本 |
| `Text2500` | 2,500 | 中等文本 |
| `Text3K` | 3,000 | 中等文本 |
| `Text10K` | 10,000 | 长文本 |
| `Text60K` | 60,000 | 超长文本 |
| `Text64K` | 65,536 | TEXT 列 |
| `Text100K` | 100,000 | 大文本 |

```python
from sqlmodel_ext import Str64, Text10K

class Article(SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    content: Text10K
```

#### 数值约束

| 类型 | 范围 | 用途 |
|------|------|------|
| `Port` | 1--65535 | 网络端口 |
| `Percentage` | 0--100 | 百分比 |
| `PositiveInt` | >= 1 | 计数、数量 |
| `NonNegativeInt` | >= 0 | 索引、计数器 |
| `PositiveFloat` | > 0.0 | 价格、重量 |

```python
from sqlmodel_ext import Port, Percentage

class ServerConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    port: Port = 8080
    cpu_threshold: Percentage = 80
```

#### URL 类型

| 类型 | 验证 | SSRF 防护 |
|------|------|-----------|
| `Url` | 任意 URL 协议 | 无 |
| `HttpUrl` | 仅 HTTP/HTTPS | 无 |
| `WebSocketUrl` | 仅 WS/WSS | 无 |
| `SafeHttpUrl` | 仅 HTTP/HTTPS | 有 |

所有 URL 类型均为 `str` 子类 -- 在数据库中存储为 `VARCHAR`，在 Python 中表现为普通字符串，同时提供 Pydantic 赋值验证。

```python
from sqlmodel_ext import HttpUrl, SafeHttpUrl, WebSocketUrl

class APIConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    api_url: HttpUrl
    callback_url: SafeHttpUrl    # 阻止内网 IP、localhost
    ws_endpoint: WebSocketUrl
```

**`SafeHttpUrl` 阻止的地址：**
- 内网 IP（10.x、172.16-31.x、192.168.x）
- 回环地址（127.x、::1、localhost）
- 链路本地地址（169.254.x）
- 非 HTTP 协议（file://、gopher:// 等）

```python
from sqlmodel_ext import SafeHttpUrl, UnsafeURLError, validate_not_private_host

# 验证器也可单独使用
try:
    validate_not_private_host("192.168.1.1")
except UnsafeURLError:
    print("已阻止内网 IP")
```

#### IP 地址类型

```python
from sqlmodel_ext import IPAddress

class Server(SQLModelBase, UUIDTableBaseMixin, table=True):
    ip: IPAddress

server = Server(ip="192.168.1.1")
server.ip.is_private()  # True
```

#### 路径类型

```python
from sqlmodel_ext import FilePathType, DirectoryPathType

class FileRecord(SQLModelBase, UUIDTableBaseMixin, table=True):
    file_path: FilePathType      # 必须包含文件名
    output_dir: DirectoryPathType  # 不能包含文件扩展名
```

---

### 响应 DTO Mixin

为 API 响应模型预构建的 Mixin，包含 id 和时间戳字段：

```python
from sqlmodel_ext import (
    SQLModelBase,
    UUIDIdDatetimeInfoMixin,  # UUID id + created_at + updated_at
    IntIdDatetimeInfoMixin,    # int id + created_at + updated_at
    UUIDIdInfoMixin,           # 仅 UUID id
    IntIdInfoMixin,            # 仅 int id
    DatetimeInfoMixin,         # 仅 created_at + updated_at
)

class UserResponse(UserBase, UUIDIdDatetimeInfoMixin):
    """API 响应模型 -- id、created_at、updated_at 始终存在。"""
    pass
```

这些 Mixin 将字段定义为**必填**（非可选），因为从数据库返回的 API 响应中这些字段始终有值。这与表模型中插入前 `id=None` 的设计不同。

---

## 架构

```
sqlmodel_ext/
    __init__.py              # 公共 API 重导出
    base.py                  # SQLModelBase + __DeclarativeMeta 元类
    _compat.py               # Python 3.14 (PEP 649) 猴子补丁
    _sa_type.py              # 从 Annotated 元数据提取 sa_type
    _utils.py                # now()、now_date() 时间戳工具
    _exceptions.py           # RecordNotFoundError
    pagination.py            # ListResponse、TimeFilterRequest、PaginationRequest、TableViewRequest
    mixins/
        __init__.py          # Mixin 重导出
        table.py             # TableBaseMixin、UUIDTableBaseMixin（异步 CRUD）
        polymorphic.py       # PolymorphicBaseMixin、AutoPolymorphicIdentityMixin、create_subclass_id_mixin
        optimistic_lock.py   # OptimisticLockMixin、OptimisticLockError
        relation_preload.py  # RelationPreloadMixin、@requires_relations
        info_response.py     # Id/Datetime DTO Mixin
    field_types/
        __init__.py          # 类型别名重导出（Str64、Port 等）
        _ssrf.py             # UnsafeURLError、validate_not_private_host
        ip_address.py        # IPAddress 类型
        url.py               # Url、HttpUrl、WebSocketUrl、SafeHttpUrl
        _internal/path.py    # 路径类型处理器
        mixins/              # ModuleNameMixin
```

## 环境要求

- **Python** >= 3.12（已在 3.12、3.13、3.14 上测试）
- **sqlmodel** >= 0.0.22
- **pydantic** >= 2.0
- **sqlalchemy** >= 2.0
- （可选）**fastapi** >= 0.100.0

## AI 使用披露

本项目使用了 AI 辅助编码（Claude）进行开发。代码中约一半由人类编写，一半由 AI 编写，所有代码均经过人类开发者审查和验证。

## 许可证

MIT
