# 快速开始

## sqlmodel-ext 是什么？

**sqlmodel-ext** 是构建在 [SQLModel](https://sqlmodel.tiangolo.com/) 之上的增强库，消除异步数据库应用中的大量样板代码。

**定义模型，继承 Mixin，获得完整的异步 CRUD API。**

## 技术栈

```
你的应用代码
    ↓ 使用
sqlmodel-ext          ← 本项目（增强层）
    ↓ 构建于
SQLModel              ← Pydantic + SQLAlchemy 的融合
    ↓ 包装
┌─────────┐  ┌──────────────┐
│ Pydantic │  │  SQLAlchemy   │
│ 数据验证  │  │  ORM 映射     │
└─────────┘  └──────────────┘
    ↓               ↓
         数据库（SQLite / PostgreSQL / ...）
```

## 解决了什么痛点？

用原生 SQLModel 写 CRUD API，每个端点都要重复大量代码：

```python
# 每次创建都要写三行
session.add(user)
await session.commit()
await session.refresh(user)

# 查询列表：COUNT + SELECT + offset/limit/order_by + 时间过滤
# 局部更新：model_dump + setattr 循环 + commit
```

sqlmodel-ext 把这些操作封装成**一行调用**：

```python
user = await user.save(session)                              # 创建/更新
users = await User.get(session, fetch_mode="all")            # 查询
result = await User.get_with_count(session, table_view=tv)   # 分页列表
user = await user.update(session, update_data)               # 局部更新
```

## 安装

```bash
pip install sqlmodel-ext
```

## 基本用法

### 1. 定义模型

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64

# 纯数据模型（不建表，用于 API 输入/输出）
class UserBase(SQLModelBase):
    name: Str64
    email: str

# 表模型（建表，拥有 CRUD 能力）
class User(UserBase, UUIDTableBaseMixin, table=True): # [!code highlight]
    pass
```

### 2. CRUD 操作

```python
from sqlmodel.ext.asyncio.session import AsyncSession

async def demo(session: AsyncSession):
    # 创建
    user = User(name="Alice", email="alice@example.com")
    user = await user.save(session)

    # 查询
    user = await User.get_exist_one(session, user.id)  # 找不到自动 404

    # 更新
    user = await user.update(session, UserUpdate(name="Bob"))

    # 删除
    await User.delete(session, user)

    # 分页列表
    result = await User.get_with_count(session, table_view=table_view)
    # result.count = 42, result.items = [...]
```

### 3. 在 FastAPI 中使用

```python
from fastapi import APIRouter, Depends
from typing import Annotated
from sqlmodel_ext import ListResponse, TableViewRequest

router = APIRouter()
TableViewDep = Annotated[TableViewRequest, Depends()]

@router.post("", response_model=UserResponse)
async def create_user(session: SessionDep, data: UserCreate):
    user = User(**data.model_dump())
    return await user.save(session)

@router.get("", response_model=ListResponse[UserResponse])
async def list_users(session: SessionDep, table_view: TableViewDep):
    return await User.get_with_count(session, table_view=table_view)

@router.get("/{id}", response_model=UserResponse)
async def get_user(session: SessionDep, id: UUID):
    return await User.get_exist_one(session, id)

@router.patch("/{id}", response_model=UserResponse)
async def update_user(session: SessionDep, id: UUID, data: UserUpdate):
    user = await User.get_exist_one(session, id)
    return await user.update(session, data)

@router.delete("/{id}")
async def delete_user(session: SessionDep, id: UUID):
    user = await User.get_exist_one(session, id)
    await User.delete(session, user)
```

## 功能一览

| 功能 | 说明 | 详细文档 |
|------|------|---------|
| [字段类型](./field-types) | `Str64`、`Port`、`HttpUrl`、`SafeHttpUrl` 等预定义类型 | 类型列表与用法 |
| [CRUD 操作](./crud) | `save`、`get`、`update`、`delete`、`count`、`get_with_count` | 方法详解 |
| [分页与列表](./pagination) | `ListResponse`、`TableViewRequest`、DTO Mixin | 分页集成 |
| [多态继承](./polymorphic) | JTI 联表继承、STI 单表继承 | 配置指南 |
| [乐观锁](./optimistic-lock) | 基于版本号的并发控制 | 使用模式 |
| [关系预加载](./relation-preload) | `@requires_relations` 声明式关系加载、`@requires_for_update` 锁验证 | 装饰器用法 |
| [Redis 缓存](./cached-table) | `CachedTableBaseMixin` 自动缓存查询结果到 Redis | 缓存集成 |
| [静态分析器](./relation-load-checker) | 启动时检测潜在的 MissingGreenlet 问题 | 配置方式 |

## 其他基类

### `ExtraIgnoreModelBase` — 处理外部数据

与 `SQLModelBase`（`extra='forbid'`，未知字段报错）不同，`ExtraIgnoreModelBase` 静默忽略未知字段，同时记录 WARNING 日志：

```python
from sqlmodel_ext import ExtraIgnoreModelBase

class ThirdPartyResponse(ExtraIgnoreModelBase):
    status: str
    data: dict
    # 第三方 API 新增的字段会被忽略，但会记录日志
```

适用于：第三方 API 响应、WebSocket 消息、外部 JSON 输入等 schema 可能变化的场景。

::: tip 想了解实现原理？
如果你对框架的内部机制感兴趣，可以阅读[实现原理](/internals/)部分。
:::
