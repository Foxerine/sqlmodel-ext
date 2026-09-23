# sqlmodel-ext

[![PyPI version](https://img.shields.io/pypi/v/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![Python versions](https://img.shields.io/pypi/pyversions/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | **中文**

> **警告**：本项目正在积极开发中。API 可能在版本之间发生不兼容变更，且不提供任何稳定性或向后兼容性保证。请自行承担使用风险。
>
> **升级到 0.5.0？** 本版本包含不兼容变更（`all_fields_optional` → `partial=True`、乐观锁列改名为 `oplock_version`、UUIDv7 主键等）。请参阅 [CHANGELOG](CHANGELOG.md) 与 [0.5.0 迁移指南](docs/how-to/migrate-to-0-5.md)。

面向异步应用的 SQLModel 增强基础设施：智能元类、`Unset` 三态 DTO、约束字段类型、异步 CRUD Mixin、多态继承、行锁与乐观锁、关系预加载，以及事务透明的 Redis 缓存。

## 设计哲学

> **每个事实只在一个地方声明，其余一切由它派生。**
>
> 字段的约束写在类型里一次——同一份声明同时产出 Pydantic 校验、数据库列类型和 OpenAPI schema；更新 DTO 从表模型派生；"没传 / 传了 null / 传了值"是三个不同的状态，由类型系统区分而不是靠约定。
>
> 这在 AI 辅助编码时代格外重要：AI 最擅长生成"看起来对"的代码，而最常见的错误正是在第二个地方重复声明同一个事实，然后两处慢慢漂移。sqlmodel-ext 让重复声明变得不必要，并让剩下的错误尽量成为类型错误——配合 basedpyright，绝大多数误用在运行前就被标红。仓库附带一套给 AI 编码助手的规则，放进你的项目，让 Claude / Codex 等按正确方式使用本库。
>
> 失败要响亮：非法状态在构造时就被拒绝，而不是在生产环境里被静默兜底。

## 哲学落地

### 1. 用 `Unset` 实现三态 PATCH

`field: T | None = None` 无法区分"客户端没传这个字段"和"客户端要把它清空"。`partial=True` 把每个继承字段变成 `Unset | T = Unset`，三种状态保持可区分——`update()` 只写真正提交了的字段。

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, Text10K, Unset

class ArticleBase(SQLModelBase):
    title: Str64
    """文章标题（必填，不可为 null）"""
    summary: Str64 | None = None
    """可选摘要——null 是真实值："没有摘要\""""
    body: Text10K
    """正文"""

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    pass

class ArticleUpdate(ArticleBase, partial=True):
    """PATCH 请求体：派生而来，不重复声明。约束与 docstring 全部保留。"""

omitted = ArticleUpdate.model_validate({})                     # 什么都没传
cleared = ArticleUpdate.model_validate({'summary': None})      # 清空摘要
changed = ArticleUpdate.model_validate({'title': 'Hello v2'})  # 修改标题

omitted.title is Unset   # True
omitted.model_dump()     # {}                  -- Unset 永远不会出现在 dump 里
cleared.model_dump()     # {'summary': None}
changed.model_dump()     # {'title': 'Hello v2'}

ArticleUpdate.model_validate({'title': None})      # ValidationError：title 在基类中不可为 null
ArticleUpdate.model_validate({'title': 'x' * 65})  # ValidationError：Str64 约束依然生效

# PATCH {"summary": null} -> 只 UPDATE `summary`，title 与 body 不动
article = await article.update(session, cleared)
```

不需要 `exclude_unset=True`，不需要手工维护一份每个字段都 `Optional` 的副本，可空性沿用基类：基类里不可为 null 的字段在 PATCH 请求体里依然不可为 null。`partial` 在类创建时改写注解，因此类型检查器在派生类上看到的仍是基类注解（`title: str`）；如果处理函数需要按三种状态分支，就把该字段显式声明为 `Unset | T | None = Unset`（类体里的声明优先于 `partial`），并用 `is Unset` 判断——见示例 3；或者用实验性的 `check_derived` 让检查器直接看见三态（见[配合 basedpyright 效果最佳](#配合-basedpyright-效果最佳)）。

### 2. 类型即约束——一个别名，贯穿每一层

约束类型别名只声明一次，每一层都读它。需要长度上限的代码用 `max_length_of()` 问别名，而不是把数字抄一遍。

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, NonEmptyStrippedStr64, Str64, max_length_of

class ProjectBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    """显示名称"""
    slug: Str64
    """由名称派生的 URL slug"""

class Project(ProjectBase, UUIDTableBaseMixin, table=True):
    pass

ProjectBase(name='   ', slug='ok')                    # ValidationError (string_too_short)：去空白后为空
Project.__table__.c.name.type                         # VARCHAR(64)
ProjectBase.model_json_schema()['properties']['slug'] # {..., 'maxLength': 64, ...}

def make_slug(name: str) -> str:
    # 上限从别名反推——不存在第二个会漂移的 "64"。
    return name.lower().replace(' ', '-')[: max_length_of(Str64)]
```

`max_length_of()` 严格对齐 Pydantic 实际执行的规则（后声明的约束生效、自动拆开 `X | None`、`Array[T, N]` 返回元素个数上限）；别名没有声明上限时抛 `TypeError`，而不是编造一个。

### 3. basedpyright 在运行前抓到误用

三种状态、`get()` 的返回形状、`delete()` 的调用契约都写进了类型，所以 AI（或疲惫的人）最常犯的错误会被检查器标出来：

```python
from sqlmodel_ext import AsyncSession, SQLModelBase, Str64, UUIDTableBaseMixin, Unset


class Article(SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    category: Str64 | None = None


class ArticleFilter(SQLModelBase):
    category: Unset | Str64 | None = Unset
    """Omitted: no filter. null: uncategorized only. Value: that category."""


def label(f: ArticleFilter) -> str:
    if f.category is not None:
        return f.category.upper()        # forgot the Unset state
    return "uncategorized"


async def handler(session: AsyncSession) -> None:
    articles = await Article.get(session, fetch_mode="all")
    print(articles.title)                # a list, not an Article
    await Article.delete(session)        # neither instances nor condition
```

`basedpyright 1.40.1` 的真实输出（文件路径已缩短）：

```text
misuse.py:16:27 - error: Cannot access attribute "upper" for class "MISSING"
    Attribute "upper" is unknown (reportAttributeAccessIssue)
misuse.py:22:20 - error: Cannot access attribute "title" for class "list[Article]"
    Attribute "title" is unknown (reportAttributeAccessIssue)
misuse.py:23:11 - error: No overloads for "delete" match the provided arguments
    Argument types: (AsyncSession) (reportCallIssue)
3 errors, 0 warnings, 0 notes
```

正确写法对每个状态显式收窄，类型检查零报错：

```python
def label(f: ArticleFilter) -> str:
    if f.category is Unset:
        return "any"
    if f.category is None:
        return "uncategorized"
    return f.category.upper()
```

## 亮点

链接指向 [`docs/`](docs/) 中的文档（英文：[`docs/en/`](docs/en/)）。

**单点真相与类型**

| 特性 | 你得到什么 |
|------|-----------|
| [`Unset` 三态字段](docs/reference/base-classes.md) | "没传" / `null` / 有值，基于 Pydantic 官方 `MISSING` 哨兵；`Unset` 字段永远不出现在 dump 中。无法省略键的调用方（如严格模式的 LLM 工具调用）可按模型开启 `SQLModelExtConfig(omitted_sentinel=True)`。 |
| [`partial=True` PATCH DTO](docs/reference/base-classes.md) | 从基模型派生 PATCH 请求体：每个继承字段变为 `Unset \| T = Unset`，约束、docstring 与可空性全部保留。 |
| [约束类型别名 + `max_length_of()`](docs/reference/field-types.md) | `Str64`、`Text10K`、`Port`、`NonEmptyStrippedStr64` 等一次同时驱动校验、列类型和 OpenAPI；`max_length_of()` 反推上限而不是重复数字。 |
| [智能元类](docs/explanation/metaclass.md) | 自动 `table=True`、合并 `mapper_args`、从 `Annotated` 提取 `sa_type`、属性 docstring 继承、支持 Python 3.14 (PEP 649)。 |
| [正确的 `Decimal`](docs/reference/field-types.md) | `NUMERIC(p, s)` 类型：校验整数位、拒绝 `float` 输入、序列化为精确的 JSON 字符串，并提供 `Write`（为 `SUM()` 预留余量）与 `Sum` 变体。 |
| [`select()` 类型重载到 9 列](docs/reference/crud-methods.md) | `sqlmodel_ext.select` 运行时就是 `sqlmodel.select`，为 5–9 列投影补上重载，不再报类型错误。 |
| [basedpyright 零报错](#配合-basedpyright-效果最佳) | 库本身以 basedpyright 0 error 为门禁；下文给出适用于你项目的推荐配置。 |

**CRUD 与查询**

| 特性 | 你得到什么 |
|------|-----------|
| [统一的 `get(condition)`](docs/reference/crud-methods.md) | 一个方法覆盖过滤、`fetch_mode`（带类型重载）、分页、JOIN、关系加载、多态加载、时间过滤与行锁。 |
| [聚合](docs/reference/crud-methods.md) | `count(distinct_column=...)`、`distinct_column()`、`group_sum()` 在数据库中执行，并遵守 STI 过滤。 |
| [Keyset 分页](docs/how-to/paginate-a-list-endpoint.md) | `PaginationRequest` 上的 `after_id` 游标（并发写入下无缺漏、无重复）；排序固定的端点用 `PageWindowRequest`。 |
| [`ResourceReferencedError`](docs/how-to/configure-cascade-delete.md) | `delete()` 把外键 `RESTRICT` 违例转换为带类型的异常，附带已注册的面向用户文案。 |
| [UUIDv7 主键](docs/reference/mixins.md) | `UUIDTableBaseMixin` 的 id 按时间有序（RFC 9562），3.14 上走标准库快路径。 |
| [JTI / STI 多态](docs/how-to/define-jti-models.md) | 联表与单表继承，自动鉴别列、子类注册与 `DeferredIndex`。 |

**并发与事务**

| 特性 | 你得到什么 |
|------|-----------|
| [`with_for_update`](docs/how-to/handle-concurrent-updates.md) | 加锁读总是刷新 identity map；工作队列用 `skip_locked=True`；锁跟踪随 savepoint 回滚。 |
| [事务契约装饰器](docs/reference/decorators.md) | `@requires_for_update`、`@requires_locked_param`、`@requires_read_committed`、`@requires_repeatable_read`——对加锁与隔离级别假设做 fail-closed 运行时检查。 |
| [事务辅助](docs/how-to/handle-concurrent-updates.md) | `SessionFactory.run_in_repeatable_read()`（`40001` 重试）、post-commit 回调、`set_local_timeouts()`、有预算的 `rollback(best_effort_budget_seconds=...)`。 |
| [乐观锁](docs/explanation/optimistic-lock.md) | `OptimisticLockMixin` 添加 `oplock_version` 列；冲突默认重试 3 次，`delete()` 冲突统一为 `OptimisticLockError`。 |

**缓存与关系**

| 特性 | 你得到什么 |
|------|-----------|
| [事务透明的 Redis 缓存](docs/how-to/cache-queries.md) | 两级（ID + 查询）缓存，从不发布未提交状态，commit 时失效；ORM 以外的写入用 `invalidate_on_commit()`、`invalidate_all()`、`register_raw_dml_write()`。 |
| [关系预加载](docs/how-to/prevent-missing-greenlet.md) | `@requires_relations` 按方法所需加载；`ensure_relations_loaded_bulk()` 对异构集合批量加载。 |
| [`RelationLoadChecker`](docs/explanation/relation-load-checker.md) | 启动期 AST 静态分析（RLC001–RLC014），找出 `MissingGreenlet` 隐患，识别 session 子类，commit 方法集可配置。 |

另外还有：`ResourceQuotaMixin`、`TrgmSearchableMixin`（PostgreSQL trigram 搜索）、`MixinTableScanMixin`，以及由 Alembic 迁移驱动缓存失效的 `run_pending_migration_cache_invalidations()`。

## 与 AI 编码助手一起用

仓库附带一套规则，教 AI 助手遵循本库的约定——用 `partial=True` 而不是手写可选 DTO、判断未传字段用 `is Unset` 而不是 `is None`、用 `max_length_of()` 反推上限、始终使用 `save()` / `update()` 的返回值、读-改-写之前先加锁，等等。规则位于 [`ai-rules/`](ai-rules/)：

| 文件 | 适用于 |
|------|--------|
| `ai-rules/CLAUDE.md` | Claude Code（项目指令） |
| `ai-rules/AGENTS.md` | Codex、Copilot 等读取 `AGENTS.md` 的工具——规则的唯一正文 |

`CLAUDE.md` 只是导入 `AGENTS.md`，所以需要维护的规则只有一份。安装方法（无论你的项目是否已有自己的 `AGENTS.md` / `CLAUDE.md`）见 [`ai-rules/README.md`](ai-rules/README.md)。

规则与类型检查器互补：规则把助手引向正确的 API，basedpyright 拦下漏网的误用。

## 配合 basedpyright 效果最佳

sqlmodel-ext 把约束和字段的三种状态放进了类型，因此类型检查器能查出几乎所有误用：把 `Unset` 字段当值读、把 `fetch_mode="all"` 的结果当单行用、调用 `delete()` 却不给目标、在需要 `Decimal` 的地方传 `float`、10 列的 `select()` 静默退化成 `Any`，等等。请在编辑器和 CI 中运行 [basedpyright](https://docs.basedpyright.com/)。

**请使用 basedpyright ≥ 1.40.1**（pyright ≥ 1.1.414）：这是第一个能正确收窄 `x is Unset`（PEP 661 哨兵）的版本；更早的版本无法收窄 `Unset | T`。

最小 `pyrightconfig.json`（即上面示例 3 所用的配置）：

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

```bash
pip install "basedpyright>=1.40.1"
basedpyright
```

**一个缺口，以及补上它的实验性工具。** `partial=True` 在运行时改写字段类型，因此单独运行 basedpyright 时，派生的 PATCH DTO 上看到的仍是基类注解：`if patch.subtitle is not None: patch.subtitle.strip()` 不报错，继承来的、守卫挡不住 `Unset` 的 validator 也不报错。`python -m sqlmodel_ext.check_derived <你的包>` 在项目的一次性副本里展开派生类（字段与继承来的方法），在副本上跑 basedpyright，只报告展开新增的错误——不会改动你的工作树。作为 pre-commit 钩子时，它必须排在**所有其他静态检查之前**。见 [检查 partial DTO 的误用](docs/how-to/check-partial-dtos.md) 与 [`examples/check_derived_demo`](examples/check_derived_demo)。

## 安装

```bash
pip install sqlmodel-ext
```

可选 extra：

| Extra | 启用 |
|-------|------|
| `sqlmodel-ext[fastapi]` | `get_exist_one()` 抛出 `HTTPException(404)` |
| `sqlmodel-ext[postgresql]` | `JSON100K` / `JSONList100K` JSONB 类型（需要 `orjson`） |
| `sqlmodel-ext[cache]` | `CachedTableBaseMixin`（Redis + `orjson`） |
| `sqlmodel-ext[pgvector]` | `NumpyVector`（包含 `[postgresql]`，并加入 NumPy + pgvector） |
| `sqlmodel-ext[alembic]` | `run_pending_migration_cache_invalidations()` 从 Alembic 迁移脚本中发现任务 |

```bash
pip install "sqlmodel-ext[fastapi,cache]"
```

sqlmodel-ext 要求 **pydantic ≥ 2.12**（第一个提供 `Unset` 背后 `MISSING` 哨兵的版本）。

## 快速开始

### 定义模型

```python
from pydantic import EmailStr  # 需要: pip install 'pydantic[email]'
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, NonEmptyStrippedStr64

# Base 类 -- 仅定义字段，不创建数据库表
class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64   # 用户可见名称：拒绝 "" 和纯空白
    email: EmailStr

# Table 类 -- 继承字段 + 获得异步 CRUD + UUIDv7 主键
class User(UserBase, UUIDTableBaseMixin, table=True):
    pass

# PATCH 请求体 -- 从 Base 派生，每个字段都可省略
class UserUpdateRequest(UserBase, partial=True):
    pass
```

`SQLModelBase` 是所有模型的基础。它的元类会自动：
- 检测到继承链中有 `TableBaseMixin` 时设置 `table=True`
- 合并父类的 `__mapper_args__`
- 从 `Annotated` 元数据中提取 `sa_type`，正确映射列
- 派生 `partial=True` PATCH DTO（字段为 `Unset | T = Unset`）
- 应用 Python 3.14 (PEP 649) 兼容补丁

`SQLModelBase` 使用 `extra='forbid'`：未知键会被拒绝。对可能新增字段的第三方载荷，改为继承 `ExtraIgnoreModelBase`（未知键被丢弃并记录警告）。

### 异步 CRUD

所有 CRUD 方法都是异步的，需要一个 `AsyncSession`。请使用增强版 `sqlmodel_ext.AsyncSession`（sqlmodel `AsyncSession` 的子类）——缓存场景必需，其他场景无副作用：

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel_ext import AsyncSession

engine = create_async_engine("postgresql+asyncpg://localhost/app")
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)

async def demo(session: AsyncSession):
    # 创建
    user = User(name="Alice", email="alice@example.com")
    user = await user.save(session)  # 务必使用返回值！

    # 查询 -- 单条
    user = await User.get(session, User.email == "alice@example.com")

    # 查询 -- 全部
    all_users = await User.get(session, fetch_mode="all")

    # 查询 -- 分页与排序
    recent_users = await User.get(
        session,
        fetch_mode="all",
        offset=0,
        limit=20,
        order_by=[User.created_at.desc()],
    )

    # 更新 -- 只写提交了的字段
    user = await user.update(session, UserUpdateRequest(name="Bob"))

    # 删除 -- 按实例
    await User.delete(session, user)

    # 删除 -- 按条件
    await User.delete(session, condition=User.email == "old@example.com")
```

> **重要**：`save()` 和 `update()` 在 commit 后会使 session 中所有对象过期。务必使用返回值。

### FastAPI 示例

一个完整的 REST API——模型、DTO 和五个端点：

```python
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import Field
from sqlmodel_ext import (
    AsyncSession, SQLModelBase, UUIDTableBaseMixin, Str64, Text10K,
    ListResponse, TableViewRequest, UUIDIdDatetimeInfoMixin, query_dependency,
)

# ── 依赖注入层：声明一次（例如放在共享的 deps 模块），
#    所有 router 导入这些类型别名 ──────────────────────────────────

_engine = create_async_engine("postgresql+asyncpg://localhost/app")
_Session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=True)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _Session() as session:
        yield session

SessionDep = Annotated[AsyncSession, Depends(get_session)]
"""请求级 AsyncSession。端点访问数据库的唯一途径。"""

# query_dependency() 把 TableViewRequest 的每个字段绑定为查询参数（无需手写
# offset/limit/order 的管道代码），并把跨字段错误（如 after_id + offset）变成
# 422 而不是 500。
TableViewRequestDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]

async def get_current_user(session: SessionDep) -> "User":
    ...  # 解析 bearer token、加载用户——沿用你自己的认证逻辑

CurrentUserDep = Annotated["User", Depends(get_current_user)]

# ── 模型：DTO 阶梯（Base → Create → Update → Response）───────────

class ArticleBase(SQLModelBase):
    title: Str64
    """文章标题"""
    body: Text10K
    """文章正文（最多 1 万字符）"""
    is_published: bool = False
    """是否公开可见"""

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id', index=True)

class ArticleCreate(ArticleBase):
    pass

# partial=True 把每个继承字段变为 ``Unset | T = Unset``，
# 同时保留约束和属性 docstring——无需手工维护逐字段覆盖。
# 未提交的字段永远不会被写入。
class ArticleUpdate(ArticleBase, partial=True):
    pass

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID

# ── 资源即依赖："按 id 加载，否则 404" 只写一次，
#    取到的 ORM 实例就是又一个注入参数 ────────────────────────────

async def get_article(session: SessionDep, article_id: UUID) -> Article:
    return await Article.get_exist_one(session, article_id)

ArticleDep = Annotated[Article, Depends(get_article)]
"""{article_id} 对应的 Article，否则 404——在处理函数运行前取好。"""

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
        session: SessionDep, table_view: TableViewRequestDep,
) -> ListResponse[Article]:
    return await Article.get_with_count(
        session,
        Article.is_published == True,
        table_view=table_view,
    )

# article: ArticleDep —— 404 与查询由依赖负责，
# 单资源处理函数完全没有查找样板。
@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article_detail(article: ArticleDep) -> Article:
    return article

@router.patch("/{article_id}", response_model=ArticleResponse)
async def update_article(
        session: SessionDep, article: ArticleDep, data: ArticleUpdate,
) -> Article:
    return await article.update(session, data)

@router.delete("/{article_id}")
async def delete_article(session: SessionDep, article: ArticleDep) -> None:
    await Article.delete(session, article)
```

没有手写 SQL，没有手写分页逻辑，没有 session 管理样板。`TableViewRequestDep` 开箱即为客户端提供 `offset`、`limit`、`desc`、`order`、`after_id` keyset 游标以及四个时间过滤参数。

**客户端调用 `GET /articles?offset=0&limit=10&desc=true` 得到：**

```json
{
  "count": 42,
  "items": [
    {
      "id": "0199a3b2-7c4e-7d1a-9f3b-2c5d8e6f1a40",
      "title": "Hello World",
      "body": "...",
      "is_published": true,
      "author_id": "0199a3b1-1e2f-7a3b-8c4d-5e6f7a8b9c0d",
      "created_at": "2026-06-15T10:30:00Z",
      "updated_at": "2026-06-15T10:30:00Z"
    }
  ]
}
```

#### 传统写法（不使用 sqlmodel-ext）

同样五个端点，用原生 SQLModel + SQLAlchemy 编写：

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
    author_id: UUID = Field(foreign_key='user.id', index=True)
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

注意传统写法的 `ArticleUpdate` 把每个字段和每条约束都重写了一遍，而且 `{"title": null}` 能从它那里漏过去，把 `NULL` 写进一个模型声明为必填的列。

**对比一览：**

| 关注点 | 传统写法 | sqlmodel-ext |
|--------|---------|-------------|
| 主键 + 时间戳 | 4 个字段，手动定义 | 继承自 `UUIDTableBaseMixin`（UUIDv7 + 带时区的时间戳） |
| 分页 + 排序 | 每个列表端点约 20 行 | `table_view=table_view`（一个参数） |
| 计数 + 分页数据 | 两次独立查询，手动拼装 | `get_with_count()`（一次调用） |
| 查找或 404 | `session.get()` + `if not` + `raise HTTPException` | `get_exist_one()`（一次调用） |
| PATCH DTO | 每个字段和约束都重写为 `T \| None` | `partial=True`（派生，约束保留） |
| "没传" vs `null` | 类型上无法区分；必填列的 `null` 会漏过 | `Unset` vs `None`；基类禁止处拒绝 `null` |
| 局部更新 | `model_dump(exclude_unset)` + `for/setattr` 循环 + 手动 `updated_at` | `article.update(session, data)` |
| 时间过滤 | 每个字段手写 `if/where` | 内置于 `TableViewRequest`（必须带时区） |
| 响应 DTO 时间戳 | 手动定义 `id`、`created_at`、`updated_at` | 继承 `UUIDIdDatetimeInfoMixin` |
| 乐观锁 | 不包含（额外工作量大） | 模型加上 `OptimisticLockMixin` |

**多态端点**同样简洁：

```python
from abc import ABC, abstractmethod
from pydantic import EmailStr
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin, create_subclass_id_mixin,
    ListResponse, TableViewRequest,
    Str512, Text1K,
)

# ── 多态模型 ─────────────────────────────────────────────────────

class NotificationBase(SQLModelBase):
    user_id: UUID = Field(foreign_key='user.id', index=True)
    message: Text1K

class Notification(NotificationBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
    @abstractmethod
    def summary(self) -> str: ...

NotifSubclassId = create_subclass_id_mixin('notification')

class EmailNotification(NotifSubclassId, Notification, AutoPolymorphicIdentityMixin, table=True):
    email_to: EmailStr

    def summary(self) -> str:
        return f"Email to {self.email_to}: {self.message}"

class PushNotification(NotifSubclassId, Notification, AutoPolymorphicIdentityMixin, table=True):
    device_token: Str512

    def summary(self) -> str:
        return f"Push to {self.device_token}: {self.message}"

# ── 一个端点返回所有通知类型 ──────────────────────────────────────

@router.get("/notifications", response_model=ListResponse[NotificationBase])
async def list_notifications(
        session: SessionDep, user: CurrentUserDep, table_view: TableViewRequestDep,
) -> ListResponse[Notification]:
    return await Notification.get_with_count(
        session,
        Notification.user_id == user.id,
        table_view=table_view,
    )
    # 透明地返回 EmailNotification 与 PushNotification 实例
```

---

## 详细指南

### TableBaseMixin 与 UUIDTableBaseMixin

这两个 Mixin 提供异步 CRUD 接口。`TableBaseMixin` 使用自增整数主键；`UUIDTableBaseMixin` 使用 **UUIDv7** 主键（RFC 9562：前 48 位是毫秒时间戳，因此 id 大致按创建时间排序，索引插入保持局部性）。

两者都会自动添加 `id`、`created_at`、`updated_at` 字段（带时区的 UTC 时间戳）。

```python
from sqlmodel_ext import SQLModelBase, TableBaseMixin, UUIDTableBaseMixin, NonEmptyStrippedStr64, Text1K

# 整数主键
class LogEntry(SQLModelBase, TableBaseMixin, table=True):
    message: Text1K

# UUIDv7 主键（大多数场景推荐）
class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: NonEmptyStrippedStr64
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
| `session` | `AsyncSession` | 必填 | 异步数据库 session |
| `instances` | `T \| list[T]` | 必填 | 要插入的实例 |
| `refresh` | `bool` | `True` | commit 后是否刷新实例以同步数据库生成的值 |
| `commit` | `bool` | `True` | 是否提交；`False` 时只 flush |

#### `save()` -- 插入或更新

```python
# 基本保存
user = await user.save(session)

# 保存后预加载关系
# rel() 把 Relationship 字段 cast 为 QueryableAttribute——
# 否则类型检查器把 User.profile 推断为 Profile 而非可加载属性
user = await user.save(session, load=rel(User.profile))

# 显式指定乐观锁重试次数（默认沿用模型策略）
user = await user.save(session, optimistic_retry_count=5)

# 跳过 refresh（不从数据库重新获取）
user = await user.save(session, refresh=False)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session` | `AsyncSession` | 必填 | 异步数据库 session |
| `load` | `QueryableAttribute \| list` | `None` | 保存后要预加载的关系 |
| `refresh` | `bool` | `True` | 保存后是否从数据库刷新 |
| `commit` | `bool` | `True` | 是否提交事务。批量操作时设为 `False` |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | 多态子类加载（需配合 `load`） |
| `optimistic_retry_count` | `int \| None` | `None` | 乐观锁冲突时的重试次数。`None` = 模型策略（`OptimisticLockMixin` 模型为 3，其他为 0）；`0` = 不重试 |

**用 `commit=False` 批量操作：**

插入多条记录时，可以推迟提交以减少往返：

```python
await user1.save(session, commit=False)  # 只 flush
await user2.save(session, commit=False)  # 只 flush
user3 = await user3.save(session)        # 三条一起提交
```

#### `update()` -- 从模型实例局部更新

```python
# partial=True 派生 PATCH 请求体：每个继承字段变为
# ``Unset | T = Unset``，约束与可空性保留
class UserUpdate(UserBase, partial=True):
    pass

# 只写提交了的字段；Unset 字段永远不会到达数据库
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
| `session` | `AsyncSession` | 必填 | 异步数据库 session |
| `other` | `SQLModelBase` | 必填 | 其已提交字段将合并到 self 的模型实例 |
| `extra_data` | `dict` | `None` | `other` 以外额外要更新的字段 |
| `exclude_unset` | `bool` | `True` | 只应用 `other.model_fields_set` 中的字段。`Unset` 字段无论如何都不进入 dump，所以 `partial=True` DTO 无需额外处理 |
| `exclude` | `set[str]` | `None` | 从更新中排除的字段名 |
| `load` | `QueryableAttribute \| list` | `None` | 更新后要预加载的关系 |
| `refresh` | `bool` | `True` | 更新后是否从数据库刷新 |
| `commit` | `bool` | `True` | 是否提交事务 |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | 多态子类加载（需配合 `load`） |
| `optimistic_retry_count` | `int \| None` | `None` | 同 `save()`；重试时重新读行并重新应用 `other` 的变更 |

要在共享的更新请求体中拒绝特权字段，`submitted_fields_among()` 返回属于给定模型的已提交字段：

```python
forbidden = body.submitted_fields_among(ItemAdminOnlyFields)
if forbidden and not user.is_admin:
    raise PermissionError(sorted(forbidden))
```

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
| `session` | `AsyncSession` | 必填 | 异步数据库 session |
| `instances` | `T \| list[T]` | `None` | 要删除的实例 |
| `condition` | `ColumnElement[bool]` | `None` | 批量删除的 WHERE 条件（仅关键字） |
| `commit` | `bool` | `True` | 是否提交事务（仅关键字） |

`instances` 与 `condition` 二选一，不能同时提供——重载让"两者都不给"成为类型错误。`delete()` 会抛出：

- `ResourceReferencedError`：仍有 `RESTRICT` / `NO ACTION` 外键引用该行（PostgreSQL）。用 `TableBaseMixin.register_fk_delete_restrict_message(constraint_name, message)` 为每个约束注册面向用户的文案。
- `OptimisticLockError`：flush 遇到乐观锁冲突（不重试）。

```python
from sqlmodel_ext.mixins import ResourceReferencedError

try:
    await Folder.delete(session, folder)
except ResourceReferencedError as e:
    raise HTTPException(409, detail=str(e))
```

#### `get()` -- 灵活查询

`get()` 是主要的查询方法，支持过滤、分页、排序、JOIN、关系加载、多态查询、时间过滤与行锁。`fetch_mode` 选择带类型的重载，因此返回类型分别是 `T | None`、`T` 或 `list[T]`。

```python
from datetime import datetime, timezone

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
    load=[rel(User.profile), rel(User.orders)],
)

# JOIN 查询
orders = await Order.get(
    session,
    Order.total > 100,
    join=User,
    fetch_mode="all",
)

# FOR UPDATE 行锁（总是刷新 identity map）
user = await User.get(
    session,
    User.id == user_id,
    with_for_update=True,
)

# 工作队列：每个 worker 领取不同的行
job = await Job.get(session, Job.status == "pending", with_for_update=True, skip_locked=True)

# 鉴权读：绕过 identity map 与缓存
user = await User.get(session, User.id == user_id, authoritative=True)

# 时间过滤（带时区的 datetime）
recent = await User.get(
    session,
    fetch_mode="all",
    created_after_datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
    created_before_datetime=datetime(2026, 12, 31, tzinfo=timezone.utc),
)
```

**fetch_mode：**

| 模式 | 返回 | 行为 |
|------|------|------|
| `"first"`（默认） | `T \| None` | 返回第一条或 `None` |
| `"one"` | `T` | 恰好返回一条；不存在或多条时抛异常 |
| `"all"` | `list[T]` | 返回所有匹配记录 |

`get_one(session, id)` 是"必须存在"的快捷方式（否则 `NoResultFound`）；`get_exist_one(session, id)` 在安装了 FastAPI 时抛 `HTTPException(404)`，否则抛 `RecordNotFoundError`。

#### `count()`、`distinct_column()`、`group_sum()` -- 聚合

```python
from datetime import datetime, timezone
from sqlmodel import col
from sqlmodel_ext import TimeFilterRequest

total = await User.count(session)
active = await User.count(session, User.is_active == True)

# COUNT(DISTINCT user_id)
buyers = await Order.count(session, distinct_column=col(Order.user_id))

# 带时间过滤
recent_count = await User.count(
    session,
    time_filter=TimeFilterRequest(
        created_after_datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ),
)

# SELECT DISTINCT
countries = await User.distinct_column(session, col(User.country), limit=100)

# 按组求和（一次查询：COUNT(*) + 每列 COALESCE(SUM(col), 0)）
rows = await Order.group_sum(session, [col(Order.amount)], group_by=col(Order.status))
for row in rows:
    print(row.key, row.count, row.totals[0])
```

#### `get_with_count()` -- 分页响应

返回同时包含总数和分页数据的 `ListResponse[T]`：

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

#### `select()` -- 带类型的投影

`sqlmodel.select` 的类型重载止于 4 列。`sqlmodel_ext.select` 运行时是同一个函数对象，重载扩展到 9 列：

```python
from sqlmodel import col
from sqlmodel_ext import select

stmt = select(col(User.id), col(User.name), col(User.email), col(User.created_at), col(User.updated_at))
```

---

### 分页模型

sqlmodel-ext 提供开箱即用的分页与时间过滤请求模型：

```python
from sqlmodel_ext import ListResponse, TableViewRequest, TimeFilterRequest, PaginationRequest
from sqlmodel_ext.pagination import PageWindowRequest
```

| 模型 | 字段 |
|------|------|
| `PageWindowRequest` | `offset`、`limit`、`desc`——排序由领域语义固定的端点使用 |
| `PaginationRequest` | `PageWindowRequest` + `order` + `after_id`（keyset 游标） |
| `TimeFilterRequest` | 四个 `created_*` / `updated_*` 时间边界 |
| `TableViewRequest` | `TimeFilterRequest` + `PaginationRequest` |

**`TableViewRequest`** 字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `offset` | `int \| None` | `0` | 跳过前 N 条（`0` ≤ offset ≤ `2**53 - 1001`） |
| `limit` | `int \| None` | `50` | 每页最多条数（1–100） |
| `desc` | `bool \| None` | `True` | 是否降序 |
| `order` | `"created_at" \| "updated_at" \| "id"` | `"created_at"` | 排序字段；总会追加 `id` 作为同向决胜列 |
| `after_id` | `UUID \| None` | `None` | keyset 游标：返回该记录之后的数据。要求 `order` 为 `created_at` 或 `id`；不能与非零 `offset` 同用 |
| `created_after_datetime` | `AwareDatetime \| None` | `None` | 过滤 `created_at >= value` |
| `created_before_datetime` | `AwareDatetime \| None` | `None` | 过滤 `created_at < value` |
| `updated_after_datetime` | `AwareDatetime \| None` | `None` | 过滤 `updated_at >= value` |
| `updated_before_datetime` | `AwareDatetime \| None` | `None` | 过滤 `updated_at < value` |

时间边界必须带时区：无时区（naive）的 datetime 在校验时即被拒绝，而不是被静默按数据库时区解释。

**Keyset 分页**——传入上一页最后一条的 id；并发插入和删除不会让下一页错位：

```python
page1 = await Article.get_with_count(session, table_view=TableViewRequest(limit=20))
page2 = await Article.get_with_count(
    session, table_view=TableViewRequest(limit=20, after_id=page1.items[-1].id),
)
```

如果锚点已被删除或不再匹配查询，`get()` 抛出 `KeysetCursorInvalidError`，而不是返回一个看起来像"到底了"的空页。

**`ListResponse[T]`** 是标准分页响应：

```python
from sqlmodel_ext import ListResponse

@router.get("", response_model=ListResponse[UserResponse])
async def list_users(session: SessionDep, table_view: TableViewRequestDep) -> ListResponse[User]:
    return await User.get_with_count(session, table_view=table_view)
```

---

### 多态继承

sqlmodel-ext 同时支持联表继承 (JTI) 和单表继承 (STI)，简化了 SQLAlchemy 冗长的多态配置。

#### 联表继承 (JTI)

每个子类拥有独立的表，并通过外键指向父表。适用于子类字段差异较大的场景。

```python
from abc import ABC, abstractmethod
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
    HttpUrl, NonEmptyStrippedStr64, NonNegativeInt,
)

# 1. Base 类（仅定义字段，无表）
class ToolBase(SQLModelBase):
    name: NonEmptyStrippedStr64

# 2. 抽象父类（创建父表）
class Tool(ToolBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
    @abstractmethod
    async def execute(self) -> str: ...

# 3. 创建子类的外键 Mixin
ToolSubclassIdMixin = create_subclass_id_mixin('tool')

# 4. 具体子类（各自拥有独立表）
class WebSearchTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    search_url: HttpUrl

    async def execute(self) -> str:
        return f"Searching {self.search_url}"

class CalculatorTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    precision: NonNegativeInt = 2

    async def execute(self) -> str:
        return "Calculating..."
```

**关键组件：**

| 组件 | 作用 |
|------|------|
| `PolymorphicBaseMixin` | 自动配置 `polymorphic_on`，添加 `_polymorphic_name` 鉴别列 |
| `create_subclass_id_mixin(table)` | 创建带有指向父表的 FK+PK `id` 字段（默认 UUIDv7）的 Mixin |
| `AutoPolymorphicIdentityMixin` | 根据类名（小写）自动生成 `polymorphic_identity` |

**MRO 顺序很重要：** `SubclassIdMixin` 必须放在最前，才能正确覆盖 `id` 字段：

```python
# 正确
class MyTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True): ...

# 错误 -- id 字段不会被正确覆盖
class MyTool(Tool, ToolSubclassIdMixin, AutoPolymorphicIdentityMixin, table=True): ...
```

#### 单表继承 (STI)

所有子类共享父表。子类特有的列以可空列的形式加到父表上。适用于子类附加字段较少的场景。

```python
from datetime import datetime
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
    NonNegativeBigInt, Str256,
)

class UserFile(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    filename: Str256

class PendingFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    upload_deadline: datetime | None = None  # 以可空列加到 userfile 表

class CompletedFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    file_size: NonNegativeBigInt | None = None  # 以可空列加到 userfile 表

# 所有模型定义完成后，在 configure_mappers() 之前调用：
register_sti_columns_for_all_subclasses()
# 在 configure_mappers() 之后调用：
register_sti_column_properties_for_all_subclasses()
```

在 STI 子类上调用 `get()`、`count()`、`delete(condition=...)` 与各聚合方法时都会自动加上鉴别过滤，子类级删除永远不会删掉兄弟子类的行。

#### 查询多态模型

```python
# 获取所有工具（返回具体子类实例）
tools = await Tool.get(session, fetch_mode="all")
# tools[0] 可能是 WebSearchTool，tools[1] 可能是 CalculatorTool

# 加载多态关系
from sqlmodel import Relationship

class ToolSet(SQLModelBase, UUIDTableBaseMixin, table=True):
    tools: list[Tool] = Relationship(back_populates="tool_set")

# 加载工具及所有子类数据
tool_set = await ToolSet.get(
    session,
    ToolSet.id == ts_id,
    load=rel(ToolSet.tools),
    jti_subclasses='all',  # 加载所有子类特有列
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
Tool.is_joined_table_inheritance()  # JTI 为 True，STI 为 False
```

---

### 行锁与事务契约

读-改-写必须先锁行。`get(with_for_update=True)` 发出 `SELECT ... FOR UPDATE`，总是刷新 identity map（否则内存中的陈旧对象会导致更新丢失），并把锁记录在 session 上。依赖锁的方法声明这一点，声明在运行时被检查（fail-closed）：

```python
from sqlmodel_ext import requires_for_update
from sqlmodel_ext.mixins import requires_repeatable_read, requires_read_committed, requires_locked_param

class Account(SQLModelBase, UUIDTableBaseMixin, table=True):
    balance: NonNegativeDecimal38_18

    @requires_for_update
    async def withdraw(self, session: AsyncSession, *, amount: Decimal) -> None:
        if amount > self.balance:
            raise ValueError("insufficient balance")
        self.balance -= amount

account = await Account.get(session, Account.id == account_id, with_for_update=True)
await account.withdraw(session, amount=Decimal("10"))   # 若 `account` 未加锁则 RuntimeError
account = await account.save(session)
```

`@requires_locked_param` 检查某个参数中传入的实例已加锁；`@requires_repeatable_read` / `@requires_read_committed` 检查隔离级别。锁跟踪随 savepoint：savepoint 回滚后，其中获得的锁也会被遗忘。

增强 session 上的**事务辅助**（PostgreSQL）：

```python
from sqlmodel_ext.session import SessionFactory

session_factory = SessionFactory(engine, class_=AsyncSession, expire_on_commit=True)

async def transfer(session: AsyncSession) -> None:
    ...  # 必须幂等：可能被执行多次
    await session.commit()

# 独立的 REPEATABLE READ session；串行化失败（40001）时整体重试
await session_factory.run_in_repeatable_read(transfer, description="transfer")

async with session_factory() as session:
    await session.set_local_timeouts(lock_timeout_ms=2_000, statement_timeout_ms=10_000)
    session.add_post_commit_callback(notify_downstream)   # 仅在 commit 成功后运行
    ...
```

---

### 乐观锁

利用 SQLAlchemy 的 `version_id_col` 机制，防止并发环境下的更新丢失。

```python
from enum import StrEnum

from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    OptimisticLockMixin, OptimisticLockError,
    NonNegativeDecimal38_18,
)

class OrderStatusEnum(StrEnum):
    pending = 'pending'
    paid = 'paid'

# OptimisticLockMixin 必须在 MRO 中位于 SQLModelBase / TableBaseMixin 之前
class Order(OptimisticLockMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    status: OrderStatusEnum = OrderStatusEnum.pending
    amount: NonNegativeDecimal38_18
```

该 Mixin 添加一个 `oplock_version` BIGINT 列（每次写入递增，带 `server_default` 以兼容滚动部署，不出现在 `model_dump()` 中）。这个列名是保留的，因此永远不会与领域上的 `version` 字段冲突——自己声明 `oplock_version` 会抛 `TypeError`。每条 `UPDATE` 生成类似如下的 SQL：

```sql
UPDATE "order" SET status=?, amount=?, oplock_version=oplock_version+1
WHERE id=? AND oplock_version=?
```

如果 `WHERE` 不匹配（其他事务已修改该记录），更新影响 0 行，冲突即被检测到。

#### 自动重试（默认）

`OptimisticLockMixin` 模型**默认重试 3 次**：每次重试重新读取最新行、只重新应用你改过的列，再次保存。只有重试耗尽才抛出 `OptimisticLockError`。

```python
order = await order.save(session)                              # 最多重试 3 次
order = await order.update(session, update_data)               # 同样的策略
order = await order.save(session, optimistic_retry_count=5)    # 显式次数
```

#### 手动处理错误

```python
try:
    order = await order.save(session, optimistic_retry_count=0)   # 不重试
except OptimisticLockError as e:
    print(f"Conflict on {e.model_class} id={e.record_id}")
    print(f"Expected version: {e.expected_version}")
```

`OptimisticLockMixin` 模型上的 `delete()` 遇到冲突同样抛 `OptimisticLockError`（不重试；`record_id` 为 `None`，因为冲突属于整个 flush）。

**适合使用乐观锁的场景：**
- 状态流转（pending -> paid -> shipped）
- 并发修改的数值字段（余额、库存）

**不适合的场景：**
- 日志/审计表（只插入）
- 简单计数器（`UPDATE SET count = count + 1` 即可）

---

### 关系预加载

`RelationPreloadMixin` 与 `@requires_relations` 装饰器在方法执行前自动加载关系，防止异步 SQLAlchemy 中的 `MissingGreenlet` 错误。

```python
from decimal import Decimal

from sqlmodel import Relationship
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel_ext import (
    UUIDTableBaseMixin, SQLModelBase, NonNegativeDecimal38_18,
    RelationPreloadMixin, requires_relations,
)

class GeneratorConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    price: NonNegativeDecimal38_18

class Generator(SQLModelBase, UUIDTableBaseMixin, table=True):
    config: GeneratorConfig = Relationship()

class MyFunction(SQLModelBase, UUIDTableBaseMixin, RelationPreloadMixin, table=True):
    generator: Generator = Relationship()

    @requires_relations('generator', Generator.config)
    async def calculate_cost(self, session: AsyncSession) -> Decimal:
        # 运行前 generator 与 generator.config 已自动加载
        return self.generator.config.price * 10
```

**工作原理：**

1. `@requires_relations` 声明方法需要哪些关系
2. 方法运行前，装饰器检查哪些关系已加载（使用 `sqlalchemy.inspect`）
3. 未加载的关系在一次查询中取回
4. 已加载的关系跳过（增量加载）

**支持的参数格式：**

```python
@requires_relations(
    'generator',           # 字符串：本类上的属性名
    Generator.config,      # QueryableAttribute：外部类属性（嵌套）
)
```

**也支持异步生成器：**

```python
@requires_relations('items')
async def stream_items(self, session):
    for item in self.items:
        yield item
```

**导入期校验：** 字符串形式的关系名在类创建时即被校验。如果声明了 `@requires_relations('nonexistent')`，会立即得到 `AttributeError`，而不是等到运行时。

**手动与批量预加载 API**（通常不需要）：

```python
# 为指定方法预加载关系
await instance.preload_for(session, 'calculate_cost', 'validate')

# 获取方法所需的关系列表（构造查询时有用）
rels = MyFunction.get_relations_for_method('calculate_cost')
rels = MyFunction.get_relations_for_methods('calculate_cost', 'validate')

# 批量：对整个（可能异构的）列表，每个不同的目标根类只查一次
await MyFunction.ensure_relations_loaded_bulk(session, functions, {MyFunction: ('generator',)})
```

每个 `Relationship` 默认 `lazy='raise_on_sql'`：访问未加载的关系会立即抛出清晰的 `InvalidRequestError`，而不是晦涩的 `MissingGreenlet`。启动时 `RelationLoadChecker` 会静态找出导致这类问题的写法（见[讲解](docs/explanation/relation-load-checker.md)）。

---

### 字段类型

sqlmodel-ext 提供可复用的 `Annotated` 类型别名，同时适用于 Pydantic 校验和 SQLAlchemy 列映射。所有字符串别名都拒绝 NUL 字节（PostgreSQL 无法存储）。

#### 字符串约束

| 类型 | 最大长度 | 用途 |
|------|---------|------|
| `Str1` / `Str16` / `Str24` / `Str32` | 1 / 16 / 24 / 32 | 标志、短码、token |
| `Str36` | 36 | UUID 字符串 |
| `Str48` / `Str64` / `Str100` / `Str128` | 48 / 64 / 100 / 128 | 标签、名称、标题、标识符 |
| `Str255` / `Str256` / `Str500` / `Str512` / `Str2048` | 255 / 256 / 500 / 512 / 2048 | 标准 VARCHAR、URL |
| `Text1K` / `Text1024` / `Text2K` / `Text2500` / `Text3K` / `Text3072` | 1,000 – 3,072 | 短文本 |
| `Text4K` / `Text5K` / `Text8K` / `Text10K` / `Text16K` | 4,000 – 16,000 | 中等文本 |
| `Text32K` / `Text48K` / `Text60K` / `Text64K`（65,536） | 32,000 – 65,536 | 长文本 |
| `Text100K` / `Text128K`（131,072） / `Text1M` | 100,000 – 1,000,000 | 超长文本 |
| `NonEmptyStr64` / `128` / `256` | 1–N | 非空（不去空白） |
| `NonEmptyStrippedStr32` / `64` / `128` / `256` | 1–N | 用户可见名称：去空白，拒绝空串 |
| `SingleLineStr64` / `SearchQueryStr64` | 64 | 单行名称 / 搜索关键词（去空白后至少 2 字符） |
| `Sha256Hex` | 恰好 64 | 小写十六进制 SHA-256 摘要 |
| `BCP47LanguageCode` | 16 | `zh-CN`、`en-US` 等 |
| `HttpHeaderName` | — | RFC 9110 token |

```python
from sqlmodel_ext import Str64, Text10K, max_length_of

class Article(SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    content: Text10K

max_length_of(Str64)   # 64 -- 反推上限，而不是重复数字
```

#### 数值约束

| 类型 | 范围 | 列类型 |
|------|------|--------|
| `Port` | 1 – 65535 | INTEGER |
| `Percentage` | 0 – 100 | INTEGER |
| `PositiveInt` / `NonNegativeInt` | ≥ 1 / ≥ 0，≤ 2³¹−1 | INTEGER |
| `PositiveBigInt` / `NonNegativeBigInt` / `SignedBigInt` | 最大 ±(2⁵³−1)（JS 安全） | BIGINT |
| `PositiveFloat` / `NonNegativeFloat` | > 0 / ≥ 0，有限值（拒绝 inf / nan） | FLOAT |

```python
from sqlmodel_ext import Port, Percentage

class ServerConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    port: Port = 8080
    cpu_threshold: Percentage = 80
```

#### Decimal 类型

用于金额与费率的 `NUMERIC(p, s)` 别名。它们校验列能容纳的整数位数，拒绝 `float` / `bool` 输入（精度已经丢失），并序列化为定点 JSON **字符串**（无科学计数法、无尾随零），让 JavaScript 客户端不丢精度。`model_dump()` 保留 `Decimal` 对象。

| 系列 | 列类型 | 说明 |
|------|--------|------|
| `SignedDecimal38_18` / `NonNegativeDecimal38_18` / `PositiveDecimal38_18` | `NUMERIC(38, 18)` | 20 位整数 + 18 位小数 |
| `Optional…Decimal38_18` | `NUMERIC(38, 18)` | `… \| None` |
| `…WriteDecimal38_18` | `NUMERIC(38, 18)` | 写入限制为 35 位，为 `SUM()` 预留 1000 倍余量 |
| `SignedSumDecimal38_18` | — | 读取聚合求和结果 |
| `SignedDecimal20_10` / `NonNegativeDecimal20_10` / `OptionalNonNegativeDecimal20_10` / `NullableNonNegativeDecimal20_10` | `NUMERIC(20, 10)` | 费率、比例 |

```python
from decimal import Decimal
from sqlmodel_ext import NonNegativeDecimal38_18

class WalletBase(SQLModelBase):
    balance: NonNegativeDecimal38_18 = Decimal(0)

class Wallet(WalletBase, UUIDTableBaseMixin, table=True):
    pass

WalletBase.model_validate({"balance": "12.50"})   # OK
WalletBase.model_validate({"balance": 12.5})      # ValidationError：拒绝 float
WalletBase(balance=Decimal("0.1")).model_dump_json()   # '{"balance":"0.1"}'
```

#### 有界列表

`List1` … `List1024` 限制元素个数：`tags: List20[Str32]`。`max_length_of()` 同样能反推这个上限。

#### URL 类型

| 类型 | 校验 | SSRF 防护 |
|------|------|-----------|
| `Url` | 任意 URL scheme | 否 |
| `HttpUrl` | 仅 HTTP/HTTPS | 否 |
| `WebSocketUrl` | 仅 WS/WSS | 否 |
| `SafeHttpUrl` | 仅 HTTP/HTTPS | 是 |

所有 URL 类型都是 `str` 子类——在数据库中存为 `VARCHAR`，在 Python 代码中表现为普通字符串，同时在赋值时提供 Pydantic 校验。

```python
from sqlmodel_ext import HttpUrl, SafeHttpUrl, WebSocketUrl

class APIConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    api_url: HttpUrl
    callback_url: SafeHttpUrl    # 阻止私有 IP、localhost
    ws_endpoint: WebSocketUrl
```

**`SafeHttpUrl` 阻止：**
- 私有 IP（10.x、172.16-31.x、192.168.x）
- 回环地址（127.x、::1、localhost）
- 链路本地地址（169.254.x）
- 非 HTTP 协议（file://、gopher:// 等）

```python
from sqlmodel_ext import SafeHttpUrl, UnsafeURLError, validate_not_private_host

# 校验函数也可单独使用
try:
    validate_not_private_host("192.168.1.1")
except UnsafeURLError:
    print("Blocked private IP")
```

#### IP 地址类型

```python
from sqlmodel_ext import IPAddress, ClientIPAddress

class Server(SQLModelBase, UUIDTableBaseMixin, table=True):
    ip: IPAddress          # 存储列（VARCHAR，行为同 str）

server = Server(ip="192.168.1.1")
server.ip.is_private()  # True
```

`ClientIPAddress` 是面向不可信输入（如代理头）的解析期类型：校验为 `ipaddress` 对象并拒绝 IPv6 scope id。结果存入 `IPAddress` 列。

#### 路径类型

```python
from sqlmodel_ext import FilePathType, DirectoryPathType

class FileRecord(SQLModelBase, UUIDTableBaseMixin, table=True):
    file_path: FilePathType      # 必须包含文件名部分
    output_dir: DirectoryPathType  # 不能带文件扩展名
```

---

### PostgreSQL 类型

PostgreSQL 专用类型位于 `sqlmodel_ext.field_types.dialects.postgresql`。它们**不**从顶层 `sqlmodel_ext` 包导出，因为需要 PostgreSQL 专用依赖。

```python
from sqlmodel_ext.field_types.dialects.postgresql import (
    Array,          # pip install sqlmodel-ext  （使用 sqlalchemy.dialects.postgresql）
    JSON100K,       # pip install sqlmodel-ext[postgresql]  （需要 orjson）
    JSONList100K,   # pip install sqlmodel-ext[postgresql]  （需要 orjson）
    NumpyVector,    # pip install sqlmodel-ext[pgvector]  （需要 numpy + pgvector）
)
```

#### `Array[T]` -- PostgreSQL ARRAY

把 Python `list[T]` 映射到 PostgreSQL 原生 `ARRAY` 列的泛型数组类型。

```python
from uuid import UUID
from sqlmodel import Field
from sqlmodel_ext.field_types.dialects.postgresql import Array

class Article(SQLModelBase, UUIDTableBaseMixin, table=True):
    tags: Array[str] = Field(default_factory=list)
    """在 PostgreSQL 中存为 TEXT[]"""

    scores: Array[int] = Field(default_factory=list)
    """在 PostgreSQL 中存为 INTEGER[]"""

    metadata_list: Array[dict] = Field(default_factory=list)
    """在 PostgreSQL 中存为 JSONB[]"""

    refs: Array[UUID] = Field(default_factory=list)
    """在 PostgreSQL 中存为 UUID[]"""
```

**限制长度：**

```python
class Config(SQLModelBase, UUIDTableBaseMixin, table=True):
    version_vector: Array[dict, 20] = Field(default_factory=list)
    """最多 20 个元素，由 Pydantic 校验"""
```

**支持的元素类型：**

| Python 类型 | PostgreSQL 类型 |
|-------------|----------------|
| `str` | `TEXT[]` |
| `int` | `INTEGER[]` |
| `dict` | `JSONB[]` |
| `UUID` | `UUID[]` |
| `Enum` 子类 | `ENUM[]`（滚动部署期间读取可容忍未知值） |

#### `JSON100K` / `JSONList100K` -- 限长 JSONB

无论输入形式如何，规范化 JSON 编码都不超过 100,000 字符的 JSONB 类型。

```python
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K, JSONList100K

class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
    canvas: JSON100K
    """画布数据，存为 JSONB（最多 100K 字符）"""

    messages: JSONList100K
    """消息列表，存为 JSONB（最多 100K 字符）"""
```

**行为——对象进，对象出：**

| 特性 | `JSON100K` | `JSONList100K` |
|------|-----------|---------------|
| Python 类型 | `dict[str, Any]` | `list[dict[str, Any]]` |
| 接受 | `dict`（推荐）或 JSON 字符串 | `list`（推荐）或 JSON 字符串 |
| PostgreSQL 类型 | `JSONB` | `JSONB` |
| 限制 | 规范化 JSON 不超过 100,000 字符；必须可序列化（嵌套深度） | 同左 |
| API 序列化 | JSON 对象本身 | JSON 数组本身 |

`model_dump()`、`model_dump(mode='json')` 和 `model_dump_json()` 都以嵌套 JSON 输出该值，从不输出转义字符串。这些限制在 `table=True` 模型（会跳过 Pydantic 校验器）构造时同样生效。

#### `NumpyVector` -- pgvector + NumPy 集成

在 PostgreSQL 中以 pgvector 的 `Vector` 类型存储向量，在 Python 中以 `numpy.ndarray` 暴露。支持固定维度与 dtype 约束。

```python
import numpy as np
from sqlmodel import Field
from sqlmodel_ext.field_types.dialects.postgresql import NumpyVector

class SpeakerInfo(SQLModelBase, UUIDTableBaseMixin, table=True):
    embedding: NumpyVector[1024, np.float32] = Field(...)
    """1024 维 float32 嵌入向量"""

# 默认 dtype 为 float32
class Document(SQLModelBase, UUIDTableBaseMixin, table=True):
    embedding: NumpyVector[768] = Field(...)
    """768 维向量（默认 float32）"""
```

**API 序列化格式**（base64 编码以提高效率）：

```json
{
    "dtype": "float32",
    "shape": 1024,
    "data_b64": "AAABAAA..."
}
```

**接受的输入格式：**

| 格式 | 示例 |
|------|------|
| `numpy.ndarray` | `np.zeros(1024, dtype=np.float32)` |
| `list` / `tuple` | `[0.1, 0.2, ...]` |
| base64 字典 | `{"dtype": "float32", "shape": 1024, "data_b64": "..."}` |
| pgvector 字符串 | `"[0.1, 0.2, ...]"`（来自数据库） |

**向量相似度搜索**（pgvector 运算符）：

```python
from sqlalchemy import select

# L2 距离（欧氏）
stmt = select(SpeakerInfo).order_by(
    SpeakerInfo.embedding.l2_distance(query_vector)
).limit(10)

# 余弦距离
stmt = select(SpeakerInfo).order_by(
    SpeakerInfo.embedding.cosine_distance(query_vector)
).limit(10)

# 最大内积
stmt = select(SpeakerInfo).order_by(
    SpeakerInfo.embedding.max_inner_product(query_vector)
).limit(10)
```

**向量异常：**

| 异常 | 触发时机 |
|------|---------|
| `VectorError` | 所有向量错误的基类 |
| `VectorDimensionError` | 数组维度与声明不符 |
| `VectorDTypeError` | dtype 转换失败 |
| `VectorDecodeError` | base64 或数据库格式解码失败 |

```python
from sqlmodel_ext.field_types.dialects.postgresql import (
    VectorError, VectorDimensionError, VectorDTypeError, VectorDecodeError,
)
```

---

### Info 响应 DTO Mixin

为 API 响应模型预置的 Mixin，总是包含 id 与时间戳字段：

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

这些 Mixin 把字段定义为**必填**（非可选），因为来自数据库的 API 响应中这些字段总有值。这与表模型不同——表模型在插入前 `id=None`。

---

### Redis 缓存（CachedTableBaseMixin）

为任意表模型添加两级 Redis 缓存。查询先走 Redis；未命中再落到数据库。

```bash
pip install sqlmodel-ext[cache]  # 安装 redis + orjson
```

**配置（应用启动时一次）：**

```python
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel_ext import AsyncSession, CachedTableBaseMixin

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=False)
CachedTableBaseMixin.configure_redis(redis)
CachedTableBaseMixin.check_cache_config()   # 校验每个缓存模型

# 必需：增强 session 在 commit 时使缓存失效
session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)
```

**定义缓存模型：**

```python
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=1800):
    pass  # 缓存 TTL 30 分钟
```

就这些。`get()` 先查 Redis，所有已提交的写入——CRUD 方法、裸 `session.add()` / 属性修改 / `session.delete()`——都在 `commit()` 时失效。

**事务透明：** 如果查询结果依赖的某张表在当前事务中有未提交写入（包括子查询引用的表、`load` 目标，以及 raw DML 写入的表），该查询既不读缓存也不写缓存。未提交状态永远不会被发布给其他请求。

**缓存架构：**

| 层级 | Key 格式 | 失效方式 |
|------|---------|---------|
| ID 缓存 | `id:{Model}:{id}` | commit 时行级 DEL |
| 查询缓存 | `query:{Model}:v{version}:{hash}` | 版本号递增（O(1) INCR）使旧 key 不可达 |
| 版本号 | `ver:{Model}` | 任何写入都 INCR；STI 子类变更会递增所有祖先的版本 |

**ORM 以外的写入：**

```python
# 与本事务耦合的触发器或 raw SQL：commit 后失效
Character.invalidate_on_commit(session, character_id)
await session.commit()

# 批量修数：清空该模型所有缓存条目
await Character.invalidate_all()

# 不经过增强 session execute() 的 raw DML
CachedTableBaseMixin.register_raw_dml_write(session, statement)
```

对于必须随数据迁移进行的缓存失效，`run_pending_migration_cache_invalidations()` 执行 Alembic 迁移中声明的失效任务（安装 `[alembic]`，或显式传入 `tasks=`）。

**可选的指标回调：**

```python
CachedTableBaseMixin.on_cache_hit = lambda name: print(f"HIT: {name}")
CachedTableBaseMixin.on_cache_miss = lambda name: print(f"MISS: {name}")
```

**自动跳过缓存的条件：**
- `no_cache=True`（显式绕过）
- `authoritative=True`（鉴权读）
- 查询依赖的任一表存在未提交写入
- `with_for_update=True` / `populate_existing=True`
- 设置了 `join`（JOIN 目标变更不会触发失效）
- 设置了 `options`（自定义加载选项）
- `load` 中包含 ID 缓存无法提供的关系

**不用 Redis？没问题。** 如果不调用 `configure_redis()`、也不继承 `CachedTableBaseMixin`，就完全不依赖 Redis。缓存层完全按需启用。

---

### `partial=True` 与 `Unset`

`partial=True` 从基模型派生 PATCH DTO：

```python
class ArticleBase(SQLModelBase):
    title: Str64
    """文章标题"""
    body: Text10K
    """正文"""
    summary: Str64 | None = None
    """可选摘要"""

class ArticleUpdateRequest(ArticleBase, partial=True):
    pass
    # title:   Unset | Str64        = Unset   （拒绝 null）
    # body:    Unset | Text10K      = Unset   （拒绝 null）
    # summary: Unset | Str64 | None = Unset   （null 清空该列）
```

**它做了什么：**
- 对继承字段，把 `T` 变为 `Unset | T = Unset`，把 `T | None` 变为 `Unset | T | None = Unset`
- 保留 `Annotated` 约束（`max_length`、`ge` 等）与属性 docstring
- 把字段级属性（`alias`、`exclude` 等）提升到 union 外层，使其继续生效
- 跳过 `Literal` 字段（鉴别字段必须保持必填）以及类自己声明的字段
- 省略的 `default_factory` 字段为 `Unset`，而不是工厂返回值
- `partial=True` 不能与 `table=True` 同用（`Unset` 无法存储）；已移除的 `all_fields_optional=True` 关键字会抛 `TypeError` 并附迁移说明

**手动检查字段**——用 `is Unset`，不要用 `is None`：

```python
from sqlmodel_ext import Unset

if body.summary is not Unset:        # 已提交（值或 null）
    ...
```

**几种写法：**

| 注解 | 含义 |
|------|------|
| `Unset \| T = Unset` | 可省略；拒绝 `null` |
| `Unset \| T \| None = Unset` | 可省略；`null` 是真实值 |
| `T \| None` | 必须提供；可以为 `null` |
| `T = <默认值>` | 可省略，有自然默认值，拒绝 `null` |

**无法省略键的调用方**（如严格模式的 LLM 函数调用）可以按模型开启一个线上取值：

```python
from sqlmodel_ext import SQLModelExtConfig

class UpdateArticleToolArgs(ArticleUpdateRequest):
    model_config = SQLModelExtConfig(omitted_sentinel=True)
    # 入站的 "__omitted__" 被归一化为 Unset；JSON Schema 增加一个 const 分支
```

`Unset` 就是 Pydantic 官方的 `MISSING` 哨兵（PEP 661），以一个名字导出供整个代码库使用。静态收窄需要 basedpyright ≥ 1.40.1。

---

### 属性 Docstring 继承

使用 Pydantic 的 `use_attribute_docstrings=True`（`SQLModelBase` 默认开启）时，字段描述会出现在 OpenAPI schema 中。但 Pydantic 基于 AST 的 docstring 解析在子类覆盖字段时不会继承描述。

**sqlmodel-ext 自动修复了这一点。** 元类沿 MRO 从父类继承缺失的描述，`__get_pydantic_json_schema__` 为裸 `$ref` 属性补上描述。

```python
class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    """用户显示名称"""     # ← 由 Pydantic 解析

class UserUpdateRequest(UserBase, partial=True):
    pass
    # name: Unset | NonEmptyStrippedStr64 = Unset —— 描述 "用户显示名称" 被继承
    # 在 OpenAPI/Swagger 文档中正确显示
```

---

### 增强 AsyncSession

`sqlmodel_ext.AsyncSession` 是 sqlmodel `AsyncSession` 的子类。把 session 工厂指向它：

```python
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel_ext import AsyncSession

session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)
# 需要 run_in_repeatable_read() 时改用 sqlmodel_ext.session.SessionFactory(...)
```

它做了什么：

- **`commit()`** 登记 session 中每个 `CachedTableBaseMixin` 变更（包括裸 `session.add()` / 属性修改 / `session.delete()`），commit 后同步使缓存失效，然后运行 post-commit 回调。`commit_count` 报告成功提交的次数。
- **`rollback()`** 丢弃待执行的 post-commit 回调；`best_effort_budget_seconds=` 为尽力回滚设定时间预算。
- **`begin()`**——`async with session.begin():` 退出时经过增强版 `commit()` / `rollback()`。
- **`reset()` / `close()`** 释放连接，并清除锁跟踪、REPEATABLE READ 标记、回调与缓存跟踪状态。
- **`refresh()`** 总是读数据库（从不读缓存）。
- **`execute()` / `exec()` / `scalar()` / `stream()` / `stream_scalars()`** 把 raw DML 写入的表登记为"未提交"（依赖它们的查询因此跳过缓存），并在 raw `UPDATE` / `DELETE` 命中缓存表却未登记失效时发出警告。
- **`set_local_timeouts()` / `enter_repeatable_read()`**——PostgreSQL 事务辅助。

不使用 `CachedTableBaseMixin` 的模型不受影响——每个钩子都退化为上游行为。如果使用了 `CachedTableBaseMixin`，**必须**用 `class_=sqlmodel_ext.AsyncSession` 构造 session（普通 session 会退化到 fire-and-forget 的 `after_commit` 补偿钩子，存在短暂的缓存陈旧窗口）。

---

## 架构

```
sqlmodel_ext/
    __init__.py              # 公共 API 再导出
    base.py                  # SQLModelBase、SQLModelExtConfig、partial=True、元类
    unset.py                 # Unset（pydantic MISSING）、OMITTED_SENTINEL
    constants.py             # EXCLUDE_IF_NONE、乐观锁列名
    select.py                # 带 5-9 列重载的 select()
    session.py               # 增强 AsyncSession、SessionFactory
    pagination.py            # ListResponse、PageWindowRequest、PaginationRequest、TimeFilterRequest、TableViewRequest
    relation_load_checker.py # RelationLoadChecker（AST 静态分析，RLC001-RLC014）
    _compat.py               # Python 3.14 (PEP 649) 兼容
    _sa_type.py              # 从 Annotated 元数据提取 sa_type
    _type_unwrap.py          # Annotated / union 拆解辅助
    _exceptions.py           # RecordNotFoundError
    mixins/
        table.py             # TableBaseMixin、UUIDTableBaseMixin（异步 CRUD、聚合、keyset）
        cached_table.py      # CachedTableBaseMixin（事务透明 Redis 缓存）
        polymorphic.py       # PolymorphicBaseMixin、AutoPolymorphicIdentityMixin、create_subclass_id_mixin、DeferredIndex
        optimistic_lock.py   # OptimisticLockMixin、OptimisticLockError
        relation_preload.py  # RelationPreloadMixin、@requires_relations、事务契约装饰器
        exceptions.py        # ResourceReferencedError、keyset 游标异常
        info_response.py     # Id/Datetime DTO Mixin
        resource_quota.py    # ResourceQuotaMixin
        trgm_searchable.py   # TrgmSearchableMixin
        mixin_table_scan.py  # MixinTableScanMixin
        migration_cache_invalidation.py  # run_pending_migration_cache_invalidations
        _uuid.py             # UUIDv7 生成
    field_types/
        __init__.py          # 类型别名（Str64、Port、Decimal、List* 等）、max_length_of
        _ssrf.py             # UnsafeURLError、validate_not_private_host
        ip_address.py        # IPAddress、ClientIPAddress
        url.py               # Url、HttpUrl、WebSocketUrl、SafeHttpUrl
        dialects/postgresql/ # Array[T]、JSON100K / JSONList100K、NumpyVector
```

## 依赖要求

- **Python** >= 3.12（在 3.12、3.13、3.14 上测试）
- **sqlmodel** >= 0.0.32
- **pydantic** >= 2.12
- **sqlalchemy** >= 2.0
- **typing-extensions** >= 4.14.1
- （可选）**fastapi** >= 0.100.0
- （可选）**redis** >= 5.0 -- 用于 `CachedTableBaseMixin`
- （可选）**orjson** >= 3.0 -- 用于 `CachedTableBaseMixin` 与 `JSON100K` / `JSONList100K`
- （可选）**alembic** >= 1.13 -- 用于迁移驱动的缓存失效
- （可选）**numpy** >= 1.24 与 **pgvector** >= 0.3 -- 用于 `NumpyVector`
- （推荐）**basedpyright** >= 1.40.1

## AI 声明

本项目在 AI 辅助编码（Claude）下开发。约一半代码由人类编写、一半由 AI 编写，所有代码均经人类开发者审查与验证。

## 许可证

MIT
