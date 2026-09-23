# 01 · 快速上手

这是你和 sqlmodel-ext 的第一次对话。15 分钟后，你会完成：

- 装好 sqlmodel-ext，并让 basedpyright 检查你的代码
- 定义你的第一个模型
- 跑通一次完整的 CRUD：插入、查询、更新、PATCH、删除
- 理解"模型 + Mixin = 表"与"声明一次，其余派生"这两个核心范式

::: tip 不需要事先懂 SQLAlchemy 或 SQLModel
本教程会按需引入这些概念。你只需要 Python 3.12+ 和基本的 `async` / `await` 知识。如果你完全没接触过 ORM，建议在开始之前先扫一眼 [前置知识](/explanation/prerequisites)。
:::

::: warning 开发中
sqlmodel-ext 正在积极开发中，API 可能在版本之间不经通知地变化。请锁定你所使用的版本，并自行承担使用风险。
:::

## 0. 准备环境

新建一个目录，建一个虚拟环境：

```bash
mkdir hello-sqlmodel-ext
cd hello-sqlmodel-ext
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

安装 sqlmodel-ext、异步 SQLite 驱动、下文用到的邮箱校验器以及 basedpyright：

```bash
pip install sqlmodel-ext aiosqlite "pydantic[email]" "basedpyright>=1.40.1"
```

## 1. 配置 basedpyright

sqlmodel-ext 把字段约束和"这个字段传了没有"的状态都放进了**类型**，所以类型检查器能在运行前抓到大多数误用。先把它配好，让它盯着本教程的每一步。新建 `pyrightconfig.json`：

```jsonc
{
  "pythonVersion": "3.12",
  "typeCheckingMode": "recommended",
  // SQLAlchemy / Pydantic stubs expose `Any` and partially-unknown types everywhere;
  // these rules would drown the diagnostics that matter.
  "reportAny": false,
  "reportExplicitAny": false,
  "reportUnknownMemberType": false,
  "reportUnknownVariableType": false,
  "reportUnknownArgumentType": false,
  // Optional dependencies (redis, pgvector, ...) ship without stubs.
  "reportMissingTypeStubs": false
}
```

::: info 为什么要 ≥ 1.40.1
basedpyright 1.40.1 是第一个能正确收窄 `x is Unset` 的版本。你会在第 5 步遇到 `Unset`。
:::

## 2. 定义第一个模型

新建 `app.py`：

```python
from pydantic import EmailStr  # 需要：pip install 'pydantic[email]'
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, NonEmptyStrippedStr64

class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    """用户名（拒绝空串与纯空白）"""
    email: EmailStr
    """邮箱"""

class User(UserBase, UUIDTableBaseMixin, table=True):
    pass
```

发生了什么？

- **`UserBase`** 继承 `SQLModelBase`——这是一个**纯数据模型**，不建表。它只声明了字段。`NonEmptyStrippedStr64` 是 sqlmodel-ext 提供的字符串类型别名（`Str64` 家族的命名字段变体）：限制 64 字符、自动 strip 首尾空白、拒绝空串与纯空白——这一份声明同时给 Pydantic 加约束、给 SQLAlchemy 创建 `VARCHAR(64)` 列、在 JSON Schema 中发布 `maxLength: 64`。`EmailStr` 是 Pydantic 的邮箱格式类型。
- **`User`** 同时继承 `UserBase`（拿到字段）和 `UUIDTableBaseMixin`（拿到 UUIDv7 主键 + `created_at` / `updated_at` + 全套 CRUD 方法）。`table=True` 告诉 SQLModel "建一张表"。

::: info 为什么要拆 Base 和 Table
等你写 API 时，`UserBase` 可以作为 POST 请求体（不需要 `id`），`User` 是数据库表。其他形状——PATCH 请求体、响应体——都从 `UserBase` **派生**，而不是重新声明。现在先记住"Base 不建表，Table 建表"。
:::

## 3. 创建数据库引擎和 session

继续在 `app.py` 添加：

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel_ext import AsyncSession

engine = create_async_engine("sqlite+aiosqlite:///hello.db", echo=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
```

这里 `echo=True` 让 SQLAlchemy 把每条 SQL 打印到终端——非常适合学习时观察实际发生了什么。

`sqlmodel_ext.AsyncSession` 是 sqlmodel `AsyncSession` 的子类。现在还用不到它的增强功能，但教程 03（缓存）需要它，所以一开始就用它。

## 4. 跑一次 CRUD

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

预期输出（除去 SQL 日志；你的 id 会不同）：

```
创建: id=01a0cd5e-36eb-7859-b8c1-57c960506ae0
读取: name=Alice
更新: name=Alice Cooper
列表: 1 个用户
删除: 1 条
```

这个 id 是 **UUIDv7**：前 48 位是毫秒时间戳，所以后创建的 id 排序更靠后。

## 5. 用派生 DTO 做 PATCH

HTTP PATCH 请求体的意思是"改这些字段，其余不动"。不必再写一个把每个字段都重复成可选的类，直接派生：

```python
class UserUpdate(UserBase, partial=True):
    """PATCH 请求体：UserBase 的每个字段，都可省略。"""
```

`partial=True` 把每个继承字段变为 `Unset | T = Unset`。`Unset` 表示"没传"——与 `None`（"传了 null"）是不同的状态。约束和 docstring 原样继承。

在 `main()` 里 UPDATE 与 LIST 之间加入 PATCH 步骤：

```python
        # PATCH（只写提交了的字段）
        patch = UserUpdate.model_validate({"email": "alice@cooper.dev"})
        print(f"提交: {patch.model_dump()}")
        alice = await alice.update(session, patch)
        print(f"PATCH 后: name={alice.name}, email={alice.email}")
```

并在 `async with` 块之后，看看客户端对一个基类不允许为 null 的字段传 `null` 会怎样：

```python
    try:
        _ = UserUpdate.model_validate({"name": None})
    except ValidationError as e:     # from pydantic import ValidationError
        print(f"拒绝: 'name' 有 {e.error_count()} 个错误")
```

新增的输出：

```
提交: {'email': 'alice@cooper.dev'}
PATCH 后: name=Alice Cooper, email=alice@cooper.dev
拒绝: 'name' 有 2 个错误
```

`name` 没传，所以是 `Unset`——它不出现在 `model_dump()` 中，`update()` 也永远不会碰它。而 `null` 被拒绝，因为 `UserBase.name` 不可为 null；PATCH 请求体继承了这个事实，而不是重新声明一遍。

## 6. 让 basedpyright 检查

```bash
basedpyright
```

```
0 errors, 0 warnings, 0 notes
```

现在故意改坏一处——比如在 LIST 之后写 `print(users.name)`（`users` 是 `list[User]`）——再跑一次。basedpyright 不运行程序就报出 `Cannot access attribute "name" for class "list[User]"`。从现在起，让它常驻在你的编辑器里。

## 7. 关键点解读

**保存必须用返回值**：

```python
alice = await alice.save(session)    # ✅ 正确
await alice.save(session)            # ❌ 错误
```

为什么？`session.commit()` 让 session 中所有对象**过期**（我们设置了 `expire_on_commit=True`，这也是 SQLAlchemy 的默认值）。`save()` 返回经过刷新的新鲜对象，而原 `alice` 变量已经过期。如果你不接收返回值，下一行访问 `alice.name` 会触发"过期对象重新查询"，在异步环境下变成 `MissingGreenlet` 错误。

::: tip 这条规则很重要
**所有** `save()` / `update()` 调用都要用返回值。养成肌肉记忆：`x = await x.save(session)`。
:::

**"没传"用 `Unset` 表示，不是 `None`**。手动检查 PATCH 字段时写 `if patch.email is not Unset:`（`from sqlmodel_ext import Unset`）。`None` 是客户端可以传的真实值。

**`get_one` vs `get`**：

```python
user = await User.get_one(session, user_id)    # 找不到 → 异常
user = await User.get(session, User.id == user_id)  # 找不到 → None
```

在端点中通常用 `get_exist_one()` —— 找不到自动抛 HTTP 404。教程 02 会用到。

**`fetch_mode`**（返回类型随之变化，类型检查器知道你拿到的是什么）：

```python
await User.get(session, fetch_mode="first")  # T | None
await User.get(session, fetch_mode="one")    # T，0 条或多条都抛异常
await User.get(session, fetch_mode="all")    # list[T]
```

## 8. 你刚才学到了什么

| 概念 | 作用 |
|------|------|
| `SQLModelBase` | 所有 sqlmodel-ext 模型的根类 |
| `UUIDTableBaseMixin` | 加 UUIDv7 主键 + 时间戳 + CRUD 方法 |
| `Str64` 等类型别名 | 一份声明同时驱动 Pydantic 校验、列类型和 JSON Schema |
| `partial=True` + `Unset` | 从 Base 派生 PATCH DTO；"没传"与 `null` 区分开 |
| `save()` / `get()` / `get_one()` / `update()` / `delete()` | 异步 CRUD |
| basedpyright | 在运行前抓到误用 |
| "用返回值" 规则 | commit 后对象过期，必须用刷新后的实例 |

## 下一步

教程 02 会带你用同一套范式构建一个完整的博客 API：用户、文章、评论，配上 FastAPI 端点、分页、JOIN、关系预加载。

[继续到 02 · 构建博客 API →](./02-building-a-blog-api)
