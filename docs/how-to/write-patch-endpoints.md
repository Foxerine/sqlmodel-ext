# 编写 PATCH 端点

**目标**：写一个部分更新（PATCH）端点，满足三条语义——

1. **没传的字段不动**；
2. **显式 `null` 清空可空字段**；
3. **不可空字段传 `null` 被拒绝**（422），约束（长度、范围）照常生效。

并且不重复声明任何字段。

**前置条件**：

- 已安装 `sqlmodel-ext[fastapi]`（本页示例额外用 `aiosqlite` 跑 SQLite、用 `httpx` 发请求）
- 了解 [Unset 三态](/explanation/unset-three-state) 的基本概念

## 1. 模型：基类声明一次，其余全部派生

```python
from uuid import UUID

from sqlmodel_ext import (
    NonEmptyStrippedStr128,
    NonNegativeInt,
    SQLModelBase,
    Str255,
    UUIDIdDatetimeInfoMixin,
    UUIDTableBaseMixin,
)


class ArticleBase(SQLModelBase):
    title: NonEmptyStrippedStr128
    """标题（必填，不可为 null）。"""

    subtitle: Str255 | None = None
    """副标题；null 表示没有副标题。"""

    view_limit: NonNegativeInt | None = None
    """最大浏览次数；null 表示不限。"""


class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    is_pinned: bool = False
    """置顶标记——只有管理员能改。"""


class ArticleCreate(ArticleBase):
    pass


class ArticleUpdate(ArticleBase, partial=True):
    """PATCH 请求体：每个继承字段都变成 ``Unset | T``（可空字段变成 ``Unset | T | None``）。"""


class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    is_pinned: bool
```

`ArticleUpdate` 一行字段都没写，但：

| 基类字段 | `ArticleUpdate` 中的类型 | 不传 | 传 `null` | 传值 |
|---|---|---|---|---|
| `title: NonEmptyStrippedStr128` | `Unset \| NonEmptyStrippedStr128` | 不动 | **422** | 校验长度/空白 |
| `subtitle: Str255 \| None` | `Unset \| Str255 \| None` | 不动 | 清空 | 校验长度 |
| `view_limit: NonNegativeInt \| None` | `Unset \| NonNegativeInt \| None` | 不动 | 清空（不限） | 校验 `>= 0` |

::: tip 不要再写 `T | None = None` 的更新 DTO
0.4.x 的 `all_fields_optional=True` 会把 `title` 变成 `str | None = None`：`{"title": null}` 能通过校验，然后在数据库 NOT NULL 约束处炸成 500；而"没传"和"传了 null"无法区分。`partial=True` 把这两个问题都在校验层解决了。
:::

## 2. 会话与应用

```python
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import create_async_engine

from sqlmodel_ext import AsyncSession
from sqlmodel_ext.session import SessionFactory

engine = create_async_engine("sqlite+aiosqlite:///:memory:")
session_factory = SessionFactory(engine, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
app = FastAPI()
```

## 3. 端点

```python
@app.post("/articles", response_model=ArticleResponse)
async def create_article(session: SessionDep, body: ArticleCreate) -> Article:
    return await Article(**body.model_dump()).save(session)


@app.patch("/articles/{article_id}", response_model=ArticleResponse)
async def patch_article(session: SessionDep, article_id: UUID, body: ArticleUpdate) -> Article:
    article = await Article.get_exist_one(session, article_id)
    return await article.update(session, body)
```

`update()` 内部是 `body.model_dump(exclude_unset=True)`，然后 `sqlmodel_update()`：

- 值为 `Unset` 的字段**根本不在** `model_dump()` 的输出里 → 数据库里对应的列不动；
- 显式 `null` 在输出里是 `None` → 写成 `NULL`；
- 不可空字段的 `null` 在进入端点之前就被 FastAPI 以 422 拒绝了。

端点里**不需要任何 `if body.x is not None`**。

## 4. 验证三条语义

```python
import httpx
from sqlmodel import SQLModel


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = (await client.post(
            "/articles", json={"title": "Hello", "subtitle": "first draft", "view_limit": 100},
        )).json()
        url = f"/articles/{created['id']}"

        # 1. 只传 title：其余字段不动
        r = await client.patch(url, json={"title": "Hello, world"})
        assert r.status_code == 200
        assert (r.json()["title"], r.json()["subtitle"], r.json()["view_limit"]) == ("Hello, world", "first draft", 100)

        # 2. 显式 null 清空可空字段，其余不动
        r = await client.patch(url, json={"subtitle": None})
        assert (r.json()["title"], r.json()["subtitle"], r.json()["view_limit"]) == ("Hello, world", None, 100)

        # 空请求体 = 什么都不改
        r = await client.patch(url, json={})
        assert r.status_code == 200 and r.json()["subtitle"] is None and r.json()["view_limit"] == 100

        # 3. 不可空字段传 null → 422
        r = await client.patch(url, json={"title": None})
        assert r.status_code == 422

        # 约束仍然生效 → 422
        r = await client.patch(url, json={"view_limit": -1})
        assert r.status_code == 422

    await engine.dispose()
```

用 `asyncio.run(main())` 运行即可。

## 5. OpenAPI 里看到的契约

PATCH 请求体的 schema 就是一个"所有字段都可选"的对象；不可空字段的 schema 里**没有** `null`，所以生成的前端 SDK 也不会允许给 `title` 传 `null`：

```python
schema = app.openapi()["components"]["schemas"]["ArticleUpdate"]
assert "required" not in schema
assert schema["properties"]["title"]["type"] == "string"                       # 没有 null 分支
assert {"type": "null"} in schema["properties"]["subtitle"]["anyOf"]           # 可空字段保留 null
```

## 6. 可选：只有管理员能改的字段

把"只有管理员能改的字段"声明成一个模型，与更新 DTO 组合，再用 `submitted_fields_among()` 判断请求有没有碰它们——不需要维护字段名字符串列表：

```python
from fastapi import HTTPException


class ArticleAdminOnlyFields(SQLModelBase):
    is_pinned: bool


class ArticleAdminUpdate(ArticleAdminOnlyFields, ArticleUpdate, partial=True):
    pass


@app.patch("/admin/articles/{article_id}", response_model=ArticleResponse)
async def admin_patch_article(
        session: SessionDep, article_id: UUID, body: ArticleAdminUpdate, is_admin: bool = False,
) -> Article:
    forbidden = body.submitted_fields_among(ArticleAdminOnlyFields)
    if forbidden and not is_admin:
        raise HTTPException(status_code=403, detail=f"admin-only fields: {sorted(forbidden)}")
    article = await Article.get_exist_one(session, article_id)
    return await article.update(session, body)
```

（示例里的 `is_admin` 查询参数只是占位，真实项目换成你的认证依赖。）

## 7. 需要逐字段处理时

偶尔需要根据"某个字段传了没有"做额外动作（例如标题变了要重建 slug）。判断一律用 `is Unset` / `is not Unset`：

```python
from sqlmodel_ext import Unset

body = ArticleUpdate.model_validate({"subtitle": None})
assert body.subtitle is not Unset and body.subtitle is None   # 传了，而且是 null
assert body.title is Unset                                     # 没传
```

::: warning 不要用 `is None` 判断"有没有传"
`body.subtitle is None` 在"传了 null"时为真，在"没传"时为假（此时是 `Unset`）——它回答的是另一个问题。另外注意：`partial=True` 派生出的字段在静态类型检查器眼里仍是基类类型（`title: str`），检查器**不会**提醒你收窄；这正是推荐把整个 DTO 交给 `update()` 的原因。需要静态强制时，把字段在 partial 类体里显式声明为 `Unset | T = Unset`，或用实验性的 `python -m sqlmodel_ext.check_derived`（见 [检查 partial DTO 的误用](./check-partial-dtos)）。详见 [Unset 三态](/explanation/unset-three-state#类型检查器的收窄)。
:::

## 常见陷阱

- **在 partial DTO 上 `model_dump(exclude_unset=True)`**：不再需要，但也无害。`Unset` 字段本来就不会被输出。
- **给 partial DTO 加 `table=True`**：类创建时 `TypeError`——`Unset` 不能入库。
- **更新 DTO 里想保留某个字段"必填"**：在 partial 类体里重新声明它即可，`partial` 不会改动类自己声明的字段。
- **判别字段（`Literal`）**：`partial` 会跳过它，discriminated union 照常工作。
