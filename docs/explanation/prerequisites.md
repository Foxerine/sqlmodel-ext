# 前置知识

本章为你补齐阅读源码所需的背景概念。如果你已经熟悉 ORM 和 SQLAlchemy，可以跳过。

::: tip 这是讲解（Explanation）类文档
本章介绍背景概念，**不是**操作步骤。如果你想学着写代码，请去 [教程](/tutorials/)。
:::

## ORM 是什么？

ORM（Object-Relational Mapping）让你用 Python 类和对象代替写原始 SQL。

```python
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=64)
    email: str

# 插入
user = User(name="Alice", email="alice@example.com")
session.add(user)
await session.commit()

# 查询
statement = select(User).where(User.email == "alice@example.com")
result = await session.exec(statement)
user = result.first()
```

## SQLAlchemy 核心概念

### Session（会话）

Session 是你和数据库之间的"对话通道"。所有数据库操作都通过 Session 进行：

```python
async def demo(session: AsyncSession):
    session.add(user)       # 标记对象"需要保存"（不立即执行 SQL）
    await session.flush()   # 把挂起的操作发送给数据库（不提交）
    await session.commit()  # 提交事务（永久写入）
    await session.refresh(user)  # 从数据库重新读取最新状态
```

::: danger 关键理解
`session.add()` 不执行 SQL，它只把对象放入"待处理队列"。
`session.commit()` 才真正执行 SQL，并且**会让 Session 中所有对象过期**。
过期的对象在下次访问属性时会触发新的 SQL 查询。在异步环境中，这会导致 `MissingGreenlet` 错误。
:::

### select 语句构建

```python
from sqlmodel import select

select(User)                                                # SELECT * FROM user
select(User).where(User.email == "alice@example.com")       # WHERE email = ?
select(User).order_by(User.created_at.desc()).limit(20)     # ORDER BY ... LIMIT 20
select(func.count()).select_from(User)                      # SELECT COUNT(*)
```

每个方法返回新的语句对象（不可变链式调用），通过 `session.exec(statement)` 执行。

### Relationship（关系）

关系描述表之间的关联：

```python
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    articles: list["Article"] = Relationship(back_populates="author")

class Article(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    author_id: int = Field(foreign_key="user.id")
    author: User = Relationship(back_populates="articles")
```

访问 `user.articles` 时，SQLAlchemy 自动执行 `SELECT * FROM article WHERE author_id = ?`。这叫**懒加载**。

### 懒加载在异步中的问题

```python
async def get_user_articles(session: AsyncSession):
    user = await session.get(User, 1)
    print(user.articles)  # MissingGreenlet! // [!code error]
```

解决办法是**预加载**：

```python
from sqlalchemy.orm import selectinload

statement = select(User).options(selectinload(User.articles)) # [!code highlight]
result = await session.exec(statement)
user = result.first()
print(user.articles)  # 已加载，不触发额外查询 // [!code highlight]
```

sqlmodel-ext 的 `load` 参数和 `RelationPreloadMixin` 就是对这个问题的封装。

## 元类（Metaclass）

Python 用 `type()` 创建类对象。`type` 就是所有类的"元类"——**创建类的类**。

| 概念 | 类比 |
|------|------|
| 实例 | 饼干 |
| 类 | 饼干模具 |
| 元类 | **制造模具的机器**，在模具被造出来时可以修改模具 |

自定义元类让你能**拦截类的创建过程**：

```python
class MyMeta(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        print(f"正在创建类: {name}")
        return super().__new__(cls, name, bases, attrs, **kwargs)

class MyClass(metaclass=MyMeta):
    pass
# 输出: 正在创建类: MyClass
```

SQLModel 用了元类 `SQLModelMetaclass`。sqlmodel-ext 继承它，加入更多自动化逻辑——这就是 `__DeclarativeMeta`。详见 [元类与 SQLModelBase](./metaclass)。

## `Annotated` 类型

Python 3.9+ 引入 `Annotated`，在类型注解上附加额外元数据：

```python
from typing import Annotated
from sqlmodel import Field

# 这两种写法等价：
name: str = Field(max_length=64)
name: Annotated[str, Field(max_length=64)]
```

优势是可以定义**可复用的类型别名**：

```python
Str64 = Annotated[str, Field(max_length=64)]

class User(SQLModel, table=True):
    name: Str64    # Pydantic 验证 + SQLAlchemy VARCHAR(64)
    title: Str64   # 复用同一个约束
```

这正是 sqlmodel-ext "只声明一次"的基本构件：约束写在类型里，校验、列类型和 OpenAPI schema 都从它来。见 [单点真相](./single-source-of-truth)。

## 哨兵值：当 `None` 本身是一个有意义的值

**哨兵值（sentinel）**是一个独一无二的对象，唯一的作用是表示"某种特殊情况"，用 `is` 比较。Python 里早就有这种用法（`dataclasses.MISSING`、私有的 `_NOTSET = object()` 常量），[PEP 661](https://peps.python.org/pep-0661/) 把它标准化了。Pydantic 2.12 提供了一个：`pydantic.experimental.missing_sentinel.MISSING`，sqlmodel-ext 以 `Unset` 的名字导出它。

为什么需要它：在 PATCH 请求体里，`None` 必须能表示"清空这一列"，于是"这个字段没传"就需要另一个载体。有了 `Unset`，三种状态——没传 / `null` / 有值——就能区分开：

```python
from sqlmodel_ext import SQLModelBase, Unset


class Patch(SQLModelBase):
    nickname: Unset | str | None = Unset


assert Patch().nickname is Unset                          # 没传
assert Patch(nickname=None).nickname is None              # 传了 null
assert Patch().model_dump() == {}                         # Unset 的键永远不会被输出
```

详见 [Unset 三态](./unset-three-state)。

## 静态类型检查器

静态类型检查器（pyright、basedpyright、mypy）不运行代码，只读注解，报告"不可能正确"的用法：可能为 `None` 的值被当成一定存在、参数名写错、把列表当单个对象用。它还能在判断之后**收窄**联合类型——在 `if x is Unset: return` 之后，函数余下部分知道 `x` 已经不是 `Unset`。

sqlmodel-ext 的设计目标之一是让尽可能多的误用成为类型错误，库本身也以 basedpyright 为门禁。见 [用 basedpyright 做类型检查](/how-to/type-check-with-basedpyright)。

## 多态继承的数据库概念

不同类型的对象共享基础字段，但各自有专属字段：

| 方式 | 表结构 | 适用场景 |
|------|--------|---------|
| **联表继承 (JTI)** | 父类一张表 + 每个子类一张表，外键关联 | 子类字段差异大 |
| **单表继承 (STI)** | 所有子类共用一张表，子类字段为 nullable | 子类额外字段少 |

详见 [多态继承机制](./polymorphic-internals)。
