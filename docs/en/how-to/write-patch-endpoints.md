# Write PATCH endpoints

**Goal**: write a partial-update (PATCH) endpoint with three semantics —

1. **fields that were not sent are left alone**;
2. **an explicit `null` clears a nullable field**;
3. **`null` for a non-nullable field is rejected** (422), and constraints (length, range) still apply.

And re-declare no field at all.

**Prerequisites**:

- `sqlmodel-ext[fastapi]` installed (this page additionally uses `aiosqlite` for SQLite and `httpx` to send requests)
- the basic idea of [Unset](/en/explanation/unset-three-state)

## 1. Models: declare once in the base, derive everything else

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
    """Title (required, cannot be null)."""

    subtitle: Str255 | None = None
    """Subtitle; null means "no subtitle"."""

    view_limit: NonNegativeInt | None = None
    """Maximum number of views; null means unlimited."""


class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    is_pinned: bool = False
    """Pinned flag -- only admins may change it."""


class ArticleCreate(ArticleBase):
    pass


class ArticleUpdate(ArticleBase, partial=True):
    """PATCH body: every inherited field becomes ``Unset | T`` (nullable ones ``Unset | T | None``)."""


class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    is_pinned: bool
```

`ArticleUpdate` declares no field, yet:

| Base field | Type in `ArticleUpdate` | Not sent | `null` | A value |
|---|---|---|---|---|
| `title: NonEmptyStrippedStr128` | `Unset \| NonEmptyStrippedStr128` | unchanged | **422** | length / blank checked |
| `subtitle: Str255 \| None` | `Unset \| Str255 \| None` | unchanged | cleared | length checked |
| `view_limit: NonNegativeInt \| None` | `Unset \| NonNegativeInt \| None` | unchanged | cleared (unlimited) | `>= 0` checked |

::: tip Stop writing `T | None = None` update DTOs
0.4.x's `all_fields_optional=True` turned `title` into `str | None = None`: `{"title": null}` passed validation and then blew up as a 500 at the database NOT NULL constraint, and "not sent" could not be told apart from "null". `partial=True` solves both at the validation layer.
:::

## 2. Session and app

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

## 3. Endpoints

```python
@app.post("/articles", response_model=ArticleResponse)
async def create_article(session: SessionDep, body: ArticleCreate) -> Article:
    return await Article(**body.model_dump()).save(session)


@app.patch("/articles/{article_id}", response_model=ArticleResponse)
async def patch_article(session: SessionDep, article_id: UUID, body: ArticleUpdate) -> Article:
    article = await Article.get_exist_one(session, article_id)
    return await article.update(session, body)
```

Internally `update()` does `body.model_dump(exclude_unset=True)` followed by `sqlmodel_update()`:

- fields whose value is `Unset` are **not in** the `model_dump()` output at all → their columns are left untouched;
- an explicit `null` is `None` in the output → written as `NULL`;
- `null` for a non-nullable field is rejected by FastAPI with a 422 before the endpoint runs.

The endpoint needs **no `if body.x is not None`** at all.

## 4. Verify the three semantics

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

        # 1. only title is sent: everything else is left alone
        r = await client.patch(url, json={"title": "Hello, world"})
        assert r.status_code == 200
        assert (r.json()["title"], r.json()["subtitle"], r.json()["view_limit"]) == ("Hello, world", "first draft", 100)

        # 2. an explicit null clears a nullable field; the rest is untouched
        r = await client.patch(url, json={"subtitle": None})
        assert (r.json()["title"], r.json()["subtitle"], r.json()["view_limit"]) == ("Hello, world", None, 100)

        # an empty body changes nothing
        r = await client.patch(url, json={})
        assert r.status_code == 200 and r.json()["subtitle"] is None and r.json()["view_limit"] == 100

        # 3. null for a non-nullable field -> 422
        r = await client.patch(url, json={"title": None})
        assert r.status_code == 422

        # constraints still apply -> 422
        r = await client.patch(url, json={"view_limit": -1})
        assert r.status_code == 422

    await engine.dispose()
```

Run it with `asyncio.run(main())`.

## 5. The contract visible in OpenAPI

The PATCH body schema is an object whose fields are all optional; a non-nullable field's schema has **no** `null`, so a generated frontend SDK will not allow `null` for `title` either:

```python
schema = app.openapi()["components"]["schemas"]["ArticleUpdate"]
assert "required" not in schema
assert schema["properties"]["title"]["type"] == "string"                       # no null branch
assert {"type": "null"} in schema["properties"]["subtitle"]["anyOf"]           # nullable field keeps null
```

## 6. Optional: fields only admins may change

Declare the admin-only fields as a model, combine it with the update DTO, and use `submitted_fields_among()` to check whether a request touched them — no list of field-name strings to maintain:

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

(The `is_admin` query parameter is a placeholder; use your authentication dependency in a real project.)

## 7. When you need per-field handling

Occasionally you need to act on "was this field sent" (e.g. rebuild a slug when the title changes). Always check with `is Unset` / `is not Unset`:

```python
from sqlmodel_ext import Unset

body = ArticleUpdate.model_validate({"subtitle": None})
assert body.subtitle is not Unset and body.subtitle is None   # sent, and it is null
assert body.title is Unset                                     # not sent
```

::: warning Do not use `is None` to check "was it sent"
`body.subtitle is None` is true for "sent as null" and false for "not sent" (it is `Unset` then) — it answers a different question. Also note that fields derived by `partial=True` still have the base type in static type checkers (`title: str`), so the checker will **not** remind you to narrow; that is exactly why handing the whole DTO to `update()` is recommended. Where static enforcement matters, declare the field explicitly as `Unset | T = Unset` in the partial class body, or use the experimental `python -m sqlmodel_ext.check_derived` (see [Check partial DTOs for misuse](./check-partial-dtos)). See [Unset](/en/explanation/unset-three-state#narrowing-in-the-type-checker).
:::

## Common pitfalls

- **`model_dump(exclude_unset=True)` on a partial DTO**: no longer needed, but harmless. `Unset` fields are never output anyway.
- **`table=True` on a partial DTO**: `TypeError` at class creation — `Unset` cannot be stored.
- **Keeping one field required in the update DTO**: re-declare it in the partial class body; `partial` never touches fields the class declares itself.
- **Discriminator fields (`Literal`)**: `partial` skips them, discriminated unions keep working.
