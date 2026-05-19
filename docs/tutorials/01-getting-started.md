# 01 · 快速上手

这是你和 sqlmodel-ext 的第一次对话。15 分钟后，你会完成：

- 装好 sqlmodel-ext
- 定义你的第一个模型
- 跑通一次完整的 CRUD：插入、查询、更新、删除
- 理解"模型 + Mixin = 表"这个核心范式

::: tip 不需要事先懂 SQLAlchemy 或 SQLModel
本教程会按需引入这些概念。你只需要 Python 3.12+ 和基本的 `async` / `await` 知识。如果你完全没接触过 ORM，建议在开始之前先扫一眼 [前置知识](/explanation/prerequisites)。
:::

## 0. 准备环境

新建一个目录，建一个虚拟环境：

```bash
mkdir hello-sqlmodel-ext
cd hello-sqlmodel-ext
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

安装 sqlmodel-ext 和异步 SQLite 驱动：

```bash
pip install sqlmodel-ext aiosqlite
```

## 1. 定义第一个模型

新建 `app.py`：

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64

class UserBase(SQLModelBase):
    name: Str64
    """用户名"""
    email: Str64
    """邮箱"""

class User(UserBase, UUIDTableBaseMixin, table=True):
    pass
```

发生了什么？

- **`UserBase`** 继承 `SQLModelBase`——这是一个**纯数据模型**，不建表。它只声明了字段。`Str64` 是 sqlmodel-ext 提供的字符串类型别名，等于 `Annotated[str, Field(max_length=64)]`，同时给 Pydantic 加约束、给 SQLAlchemy 创建 `VARCHAR(64)` 列。
- **`User`** 同时继承 `UserBase`（拿到字段）和 `UUIDTableBaseMixin`（拿到 UUID 主键 + `created_at` / `updated_at` + 全套 CRUD 方法）。`table=True` 告诉 SQLModel "建一张表"。

::: info 为什么要拆 Base 和 Table
等你写 API 时，`UserBase` 可以作为 POST 请求体（不需要 `id`），`User` 是数据库表。后面教程 02 会用到这个模式。现在先记住"Base 不建表，Table 建表"。
:::

## 2. 创建数据库引擎和 session

继续在 `app.py` 添加：

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel

engine = create_async_engine("sqlite+aiosqlite:///hello.db", echo=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
```

这里 `echo=True` 让 SQLAlchemy 把每条 SQL 打印到终端——非常适合学习时观察实际发生了什么。

## 3. 跑一次 CRUD

```python
async def main() -> None:
    await init_db()

    async with SessionLocal() as session:
        # CREATE
        alice = User(name="Alice", email="alice@example.com")
        alice = await alice.save(session)            # [!code highlight]
        print(f"创建: id={alice.id}")

        # READ
        fetched = await User.get_one(session, alice.id)
        print(f"读取: name={fetched.name}")

        # UPDATE
        alice.name = "Alice Cooper"
        alice = await alice.save(session)
        print(f"更新: name={alice.name}")

        # LIST
        users = await User.get(session, fetch_mode="all")
        print(f"列表: {len(users)} 个用户")

        # DELETE
        deleted = await User.delete(session, alice)
        print(f"删除: {deleted} 条")


if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
python app.py
```

预期输出（除去 SQL 日志）：

```
创建: id=550e8400-e29b-41d4-a716-446655440000
读取: name=Alice
更新: name=Alice Cooper
列表: 1 个用户
删除: 1 条
```

## 4. 关键点解读

**保存必须用返回值**：

```python
alice = await alice.save(session)    # ✅ 正确
await alice.save(session)            # ❌ 错误
```

为什么？`session.commit()` 让 session 中所有对象**过期**——这是 SQLAlchemy 的设计。`save()` 返回经过刷新的新鲜对象，而原 `alice` 变量已经过期。如果你不接收返回值，下一行访问 `alice.name` 会触发"过期对象重新查询"，在异步环境下变成 `MissingGreenlet` 错误。

::: tip 这条规则很重要
**所有** `save()` / `update()` 调用都要用返回值。养成肌肉记忆：`x = await x.save(session)`。
:::

**`get_one` vs `get`**：

```python
user = await User.get_one(session, user_id)    # 找不到 → 异常
user = await User.get(session, User.id == user_id)  # 找不到 → None
```

在端点中通常用 `get_exist_one()` —— 找不到自动抛 HTTP 404。教程 02 会用到。

**`fetch_mode`**：

```python
await User.get(session, fetch_mode="first")  # T | None
await User.get(session, fetch_mode="one")    # T，0 条或多条都抛异常
await User.get(session, fetch_mode="all")    # list[T]
```

## 5. 你刚才学到了什么

| 概念 | 作用 |
|------|------|
| `SQLModelBase` | 所有 sqlmodel-ext 模型的根类 |
| `UUIDTableBaseMixin` | 加 UUID 主键 + 时间戳 + CRUD 方法 |
| `Str64` 等类型别名 | 同时满足 Pydantic 验证和 SQLAlchemy 列类型 |
| `save()` / `get()` / `get_one()` / `delete()` | 异步 CRUD |
| "用返回值" 规则 | commit 后对象过期，必须用刷新后的实例 |

## 下一步

教程 02 会带你用同一套范式构建一个完整的博客 API：用户、文章、评论，配上 FastAPI 端点、分页、JOIN、关系预加载。

[继续到 02 · 构建博客 API →](./02-building-a-blog-api)
