# sqlmodel-ext

[![PyPI version](https://img.shields.io/pypi/v/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![Python versions](https://img.shields.io/pypi/pyversions/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**English** | [中文](README_zh.md)

> **Warning**: This project is under active development. APIs may change without notice between releases. No stability or backward-compatibility guarantees are provided at this stage. Use at your own risk.
>
> **Upgrading to 0.5.0?** It contains breaking changes (`all_fields_optional` → `partial=True`, the optimistic-lock column is now `oplock_version`, UUIDv7 primary keys, ...). See the [CHANGELOG](CHANGELOG.md) and the [0.5.0 migration guide](docs/en/how-to/migrate-to-0-5.md).

Extended SQLModel infrastructure for async applications: a smart metaclass, three-state `Unset` DTOs, constrained field types, async CRUD mixins, polymorphic inheritance, row and optimistic locking, relation preloading, and a transaction-transparent Redis cache.

## Design philosophy

> **Every fact is declared in exactly one place; everything else is derived from it.**
>
> A field's constraints are written once, in its type — that single declaration produces the Pydantic validation, the database column type and the OpenAPI schema. Update DTOs are derived from the table model. "Not sent", "sent as null" and "sent a value" are three different states, told apart by the type system instead of by convention.
>
> This matters even more in the age of AI-assisted coding. AI is best at producing code that *looks* right, and its most common mistake is restating a fact in a second place — after which the two copies slowly drift apart. sqlmodel-ext makes the restatement unnecessary, and turns as many of the remaining mistakes as possible into type errors: with basedpyright, most misuse is flagged before the code ever runs. The repository ships a rule set for AI coding assistants — drop it into your project and Claude, Codex and friends will use the library the intended way.
>
> Fail loudly: illegal states are rejected at construction time instead of being silently patched over in production.

## The philosophy in practice

### 1. Three-state PATCH with `Unset`

`field: T | None = None` cannot tell "the client did not send this field" from "the client wants it cleared". With `partial=True`, every inherited field becomes `Unset | T = Unset`, so the three states stay distinct — and `update()` writes only the fields that were actually submitted.

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, Text10K, Unset

class ArticleBase(SQLModelBase):
    title: Str64
    """Article title (required, never null)"""
    summary: Str64 | None = None
    """Optional summary -- null is a real value: "no summary\""""
    body: Text10K
    """Article body"""

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    pass

class ArticleUpdate(ArticleBase, partial=True):
    """PATCH body: derived, not re-declared. Constraints and docstrings are kept."""

omitted = ArticleUpdate.model_validate({})                     # nothing sent
cleared = ArticleUpdate.model_validate({'summary': None})      # clear the summary
changed = ArticleUpdate.model_validate({'title': 'Hello v2'})  # change the title

omitted.title is Unset   # True
omitted.model_dump()     # {}                  -- Unset never appears in a dump
cleared.model_dump()     # {'summary': None}
changed.model_dump()     # {'title': 'Hello v2'}

ArticleUpdate.model_validate({'title': None})      # ValidationError: title is not nullable in the base
ArticleUpdate.model_validate({'title': 'x' * 65})  # ValidationError: Str64 still applies

# PATCH {"summary": null} -> UPDATE only `summary`; title and body are untouched
article = await article.update(session, cleared)
```

No `exclude_unset=True`, no hand-maintained `Optional` copy of every field, and nullability follows the base: a field that was not nullable stays non-nullable in the PATCH body. `partial` rewrites annotations at class-creation time, so type checkers still see the base annotation (`title: str`) on the derived class; when handler code must branch on the three states, declare that field explicitly as `Unset | T | None = Unset` (the class body wins over `partial`) and check it with `is Unset` — see example 3; or use the experimental `check_derived` so the checker sees the tri-state directly (see [Works best with basedpyright](#works-best-with-basedpyright)).

### 2. The type is the constraint — one alias, every layer

A constrained type alias is declared once and read by every layer. Code that needs the bound asks the alias with `max_length_of()` instead of copying the number.

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, NonEmptyStrippedStr64, Str64, max_length_of

class ProjectBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    """Display name"""
    slug: Str64
    """URL slug derived from the name"""

class Project(ProjectBase, UUIDTableBaseMixin, table=True):
    pass

ProjectBase(name='   ', slug='ok')                    # ValidationError (string_too_short): stripped, then empty
Project.__table__.c.name.type                         # VARCHAR(64)
ProjectBase.model_json_schema()['properties']['slug'] # {..., 'maxLength': 64, ...}

def make_slug(name: str) -> str:
    # The bound is reflected from the alias -- there is no second "64" to drift.
    return name.lower().replace(' ', '-')[: max_length_of(Str64)]
```

`max_length_of()` follows exactly what Pydantic enforces (last constraint wins, `X | None` is unwrapped, `Array[T, N]` reports its element bound) and raises `TypeError` when the alias declares no bound instead of inventing one.

### 3. basedpyright catches the misuse before it runs

The three states, the `get()` return shape and the `delete()` contract are all in the types, so the checker flags the mistakes an AI (or a tired human) typically makes:

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

Real `basedpyright 1.40.1` output (file paths shortened):

```text
misuse.py:16:27 - error: Cannot access attribute "upper" for class "MISSING"
    Attribute "upper" is unknown (reportAttributeAccessIssue)
misuse.py:22:20 - error: Cannot access attribute "title" for class "list[Article]"
    Attribute "title" is unknown (reportAttributeAccessIssue)
misuse.py:23:11 - error: No overloads for "delete" match the provided arguments
    Argument types: (AsyncSession) (reportCallIssue)
3 errors, 0 warnings, 0 notes
```

The correct version narrows each state explicitly and type-checks cleanly:

```python
def label(f: ArticleFilter) -> str:
    if f.category is Unset:
        return "any"
    if f.category is None:
        return "uncategorized"
    return f.category.upper()
```

## Highlights

Links point to the documentation in [`docs/en/`](docs/en/) (Chinese: [`docs/`](docs/)).

**Single source of truth & types**

| Feature | What you get |
|---------|--------------|
| [`Unset` three-state fields](docs/en/reference/base-classes.md) | "Not sent" vs `null` vs a value, as Pydantic's official `MISSING` sentinel; `Unset` fields never appear in dumps. Opt-in `SQLModelExtConfig(omitted_sentinel=True)` for callers that cannot omit keys (e.g. strict LLM tool calls). |
| [`partial=True` PATCH DTOs](docs/en/reference/base-classes.md) | Derive the PATCH body from the base model: every inherited field becomes `Unset \| T = Unset`, constraints, docstrings and nullability preserved. |
| [Constrained type aliases + `max_length_of()`](docs/en/reference/field-types.md) | `Str64`, `Text10K`, `Port`, `NonEmptyStrippedStr64`, ... drive validation, column type and OpenAPI at once; `max_length_of()` reflects the bound instead of repeating it. |
| [Smart metaclass](docs/en/explanation/metaclass.md) | Automatic `table=True`, `mapper_args` merging, `sa_type` from `Annotated`, attribute-docstring inheritance, Python 3.14 (PEP 649) support. |
| [Correct `Decimal`](docs/en/reference/field-types.md) | `NUMERIC(p, s)` types that enforce integer digits, reject `float` input, serialize to exact JSON strings, with `Write` (headroom for `SUM()`) and `Sum` variants. |
| [Typed `select()` up to 9 columns](docs/en/reference/crud-methods.md) | `sqlmodel_ext.select` is `sqlmodel.select` at runtime, with overloads for 5–9 column projections instead of a type error. |
| [basedpyright-clean](#works-best-with-basedpyright) | The library itself is gated at 0 basedpyright errors; a recommended config for your project is below. |

**CRUD & queries**

| Feature | What you get |
|---------|--------------|
| [Unified `get(condition)`](docs/en/reference/crud-methods.md) | One method for filtering, `fetch_mode` (typed overloads), pagination, joins, relation loading, polymorphic loading, time filters and row locks. |
| [Aggregates](docs/en/reference/crud-methods.md) | `count(distinct_column=...)`, `distinct_column()` and `group_sum()` run in the database, honoring the STI filter. |
| [Keyset pagination](docs/en/how-to/paginate-a-list-endpoint.md) | `after_id` cursor on `PaginationRequest` (no gaps or duplicates under concurrent writes); `PageWindowRequest` for fixed-order endpoints. |
| [`ResourceReferencedError`](docs/en/how-to/configure-cascade-delete.md) | `delete()` turns FK `RESTRICT` violations into a typed error with a registered user-facing message. |
| [UUIDv7 primary keys](docs/en/reference/mixins.md) | `UUIDTableBaseMixin` ids are time-ordered (RFC 9562), with a stdlib fast path on 3.14. |
| [JTI / STI polymorphism](docs/en/how-to/define-jti-models.md) | Joined and single table inheritance with automatic discriminators, subclass registration and `DeferredIndex`. |

**Concurrency & transactions**

| Feature | What you get |
|---------|--------------|
| [`with_for_update`](docs/en/how-to/handle-concurrent-updates.md) | Locking reads always refresh the identity map; `skip_locked=True` for work queues; lock tracking follows savepoints. |
| [Transaction-contract decorators](docs/en/reference/decorators.md) | `@requires_for_update`, `@requires_locked_param`, `@requires_read_committed`, `@requires_repeatable_read` — fail-closed runtime checks of locking and isolation assumptions. |
| [Transaction helpers](docs/en/how-to/handle-concurrent-updates.md) | `SessionFactory.run_in_repeatable_read()` (retries `40001`), post-commit callbacks, `set_local_timeouts()`, bounded `rollback(best_effort_budget_seconds=...)`. |
| [Optimistic locking](docs/en/explanation/optimistic-lock.md) | `OptimisticLockMixin` adds an `oplock_version` column; conflicts are retried 3 times by default and `delete()` conflicts become `OptimisticLockError`. |

**Caching & relations**

| Feature | What you get |
|---------|--------------|
| [Transaction-transparent Redis cache](docs/en/how-to/cache-queries.md) | Two-tier (ID + query) cache that never publishes uncommitted state and invalidates on commit; `invalidate_on_commit()`, `invalidate_all()`, `register_raw_dml_write()` for writes outside the ORM. |
| [Relation preloading](docs/en/how-to/prevent-missing-greenlet.md) | `@requires_relations` loads what a method needs; `ensure_relations_loaded_bulk()` batches it for heterogeneous collections. |
| [`RelationLoadChecker`](docs/en/explanation/relation-load-checker.md) | Startup AST analysis (RLC001–RLC014) that finds `MissingGreenlet` hazards, with session-subclass awareness and configurable commit-method sets. |

Also included: `ResourceQuotaMixin`, `TrgmSearchableMixin` (PostgreSQL trigram search), `MixinTableScanMixin`, and `run_pending_migration_cache_invalidations()` for cache invalidation driven by Alembic migrations.

## Using it with AI coding assistants

The repository ships a rule set that teaches AI assistants the library's conventions — `partial=True` instead of hand-written optional DTOs, `is Unset` instead of `is None` for omitted fields, reflecting bounds with `max_length_of()`, always using the return value of `save()` / `update()`, locking before read-modify-write, and so on. It lives in [`ai-rules/`](ai-rules/):

| File | For |
|------|-----|
| `ai-rules/CLAUDE.md` | Claude Code (project instructions) |
| `ai-rules/AGENTS.md` | Codex, Copilot and other tools that read `AGENTS.md` — the single copy of the rules |

`CLAUDE.md` only imports `AGENTS.md`, so there is one copy of the rules to keep current. Installation (for a project with or without its own `AGENTS.md` / `CLAUDE.md`) is described in [`ai-rules/README.md`](ai-rules/README.md).

The rules and the type checker complement each other: the rules steer the assistant toward the intended API, and basedpyright rejects what slips through.

## Works best with basedpyright

sqlmodel-ext puts constraints and the three field states into types, which means a type checker can find almost every misuse: reading an `Unset` field as a value, treating a `fetch_mode="all"` result as a single row, calling `delete()` without a target, passing a `float` where a `Decimal` is expected, a 10-column `select()` that silently degrades to `Any`, and so on. Run [basedpyright](https://docs.basedpyright.com/) in your editor and CI.

**Use basedpyright ≥ 1.40.1** (pyright ≥ 1.1.414): it is the first release that narrows `x is Unset` correctly (PEP 661 sentinels); older versions cannot narrow `Unset | T`.

A minimal `pyrightconfig.json` (the one used for example 3 above):

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

**One gap, and an experimental way to close it.** `partial=True` rewrites field types at runtime, so basedpyright alone sees the base annotations on a derived PATCH DTO: `if patch.subtitle is not None: patch.subtitle.strip()` passes, and so does an inherited validator whose guard is not enough for `Unset`. `python -m sqlmodel_ext.check_derived <your package>` expands the derived classes (fields and inherited methods) in a throwaway copy of the project, runs basedpyright there and reports only the errors the expansion introduced — your working tree is never touched. As a pre-commit hook it must run **before** every other static check. See [Check partial DTOs for misuse](docs/en/how-to/check-partial-dtos.md) and [`examples/check_derived_demo`](examples/check_derived_demo).

## Installation

```bash
pip install sqlmodel-ext
```

Optional extras:

| Extra | Enables |
|-------|---------|
| `sqlmodel-ext[fastapi]` | `get_exist_one()` raises `HTTPException(404)` |
| `sqlmodel-ext[postgresql]` | `JSON100K` / `JSONList100K` JSONB types (requires `orjson`) |
| `sqlmodel-ext[cache]` | `CachedTableBaseMixin` (Redis + `orjson`) |
| `sqlmodel-ext[pgvector]` | `NumpyVector` (includes `[postgresql]`, adds NumPy + pgvector) |
| `sqlmodel-ext[alembic]` | `run_pending_migration_cache_invalidations()` discovers tasks from Alembic revision scripts |

```bash
pip install "sqlmodel-ext[fastapi,cache]"
```

sqlmodel-ext requires **pydantic ≥ 2.12** (the first release with the `MISSING` sentinel behind `Unset`).

## Quick Start

### Define Models

```python
from pydantic import EmailStr  # requires: pip install 'pydantic[email]'
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, NonEmptyStrippedStr64

# Base class -- fields only, no database table
class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64   # user-visible name: rejects "" and whitespace-only
    email: EmailStr

# Table class -- inherits fields + gains async CRUD + UUIDv7 primary key
class User(UserBase, UUIDTableBaseMixin, table=True):
    pass

# PATCH body -- derived from the base, every field omissible
class UserUpdateRequest(UserBase, partial=True):
    pass
```

`SQLModelBase` is the foundation for all models. Its metaclass automatically:
- Sets `table=True` when it detects `TableBaseMixin` in the inheritance chain
- Merges `__mapper_args__` from parent classes
- Extracts `sa_type` from `Annotated` metadata for proper column mapping
- Derives `partial=True` PATCH DTOs (`Unset | T = Unset` fields)
- Applies Python 3.14 (PEP 649) compatibility patches

`SQLModelBase` uses `extra='forbid'`: unknown keys are rejected. For third-party payloads that may grow new fields, inherit `ExtraIgnoreModelBase` instead (unknown keys are dropped with a warning).

### Async CRUD

All CRUD methods are async and require an `AsyncSession`. Use the enhanced `sqlmodel_ext.AsyncSession` (a subclass of sqlmodel's) — it is required for caching and harmless otherwise:

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel_ext import AsyncSession

engine = create_async_engine("postgresql+asyncpg://localhost/app")
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)

async def demo(session: AsyncSession):
    # Create
    user = User(name="Alice", email="alice@example.com")
    user = await user.save(session)  # Always use the return value!

    # Read -- single record
    user = await User.get(session, User.email == "alice@example.com")

    # Read -- all records
    all_users = await User.get(session, fetch_mode="all")

    # Read -- with pagination and sorting
    recent_users = await User.get(
        session,
        fetch_mode="all",
        offset=0,
        limit=20,
        order_by=[User.created_at.desc()],
    )

    # Update -- only the submitted fields are written
    user = await user.update(session, UserUpdateRequest(name="Bob"))

    # Delete -- by instance
    await User.delete(session, user)

    # Delete -- by condition
    await User.delete(session, condition=User.email == "old@example.com")
```

> **Important**: `save()` and `update()` cause all session objects to expire after commit. Always use the return value.

### FastAPI Example

A complete REST API -- models, DTOs, and five endpoints:

```python
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import Field
from sqlmodel_ext import (
    AsyncSession, SQLModelBase, UUIDTableBaseMixin, Str64, Text10K,
    ListResponse, TableViewRequest, UUIDIdDatetimeInfoMixin,
)

# ── Dependency-injection layer: declare once (e.g. in a shared
#    deps module), every router imports the type aliases ────────────

_engine = create_async_engine("postgresql+asyncpg://localhost/app")
_Session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=True)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _Session() as session:
        yield session

SessionDep = Annotated[AsyncSession, Depends(get_session)]
"""Request-scoped AsyncSession. The single way endpoints touch the DB."""

# TableViewRequest is a Pydantic model → FastAPI binds its fields as
# query params automatically. No hand-written offset/limit/order plumbing.
TableViewRequestDep = Annotated[TableViewRequest, Depends()]

async def get_current_user(session: SessionDep) -> "User":
    ...  # decode the bearer token, load the user — your auth, unchanged

CurrentUserDep = Annotated["User", Depends(get_current_user)]

# ── Models: the DTO ladder (Base → Create → Update → Response) ─────

class ArticleBase(SQLModelBase):
    title: Str64
    """Article title"""
    body: Text10K
    """Article body (max 10k chars)"""
    is_published: bool = False
    """Whether the article is publicly visible"""

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id', index=True)

class ArticleCreate(ArticleBase):
    pass

# partial=True turns every inherited field into ``Unset | T = Unset``
# while preserving its constraints AND attribute docstrings — no
# hand-maintained per-field overrides. Omitted fields are never written.
class ArticleUpdate(ArticleBase, partial=True):
    pass

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID

# ── Resource-as-dependency: wrap "load by id or 404" once, then the
#    fetched ORM instance is just another injected parameter ────────

async def get_article(session: SessionDep, article_id: UUID) -> Article:
    return await Article.get_exist_one(session, article_id)

ArticleDep = Annotated[Article, Depends(get_article)]
"""The Article for {article_id}, or 404 — fetched before the handler runs."""

# ── Endpoints ─────────────────────────────────────────────────────

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

# article: ArticleDep — the 404 + fetch is the dependency's job, so the
# single-resource handlers carry no lookup boilerplate at all.
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

No manual SQL, no hand-written pagination logic, no boilerplate session management. The `TableViewRequestDep` gives clients `offset`, `limit`, `desc`, `order`, the `after_id` keyset cursor and four time filters out of the box.

**What the client gets from `GET /articles?offset=0&limit=10&desc=true`:**

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

#### The Traditional Way (without sqlmodel-ext)

The same five endpoints written with plain SQLModel + SQLAlchemy:

```python
from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, desc as sa_desc, asc as sa_asc
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

# ── Models ────────────────────────────────────────────────────────

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

# ── Endpoints ─────────────────────────────────────────────────────

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
    # Count query
    count_stmt = select(func.count()).select_from(Article).where(Article.is_published == True)
    if created_after:
        count_stmt = count_stmt.where(Article.created_at >= created_after)
    if created_before:
        count_stmt = count_stmt.where(Article.created_at < created_before)
    total = await session.scalar(count_stmt) or 0

    # Data query
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

Note how the traditional `ArticleUpdate` repeats every field and every constraint, and how `{"title": null}` slips through it and writes `NULL` into a column the model says is required.

**Side-by-side comparison:**

| Concern | Traditional | sqlmodel-ext |
|---------|-------------|------------|
| Primary key + timestamps | 4 fields, manually defined | Inherited from `UUIDTableBaseMixin` (UUIDv7 + timezone-aware timestamps) |
| Pagination + sorting | ~20 lines per list endpoint | `table_view=table_view` (one arg) |
| Count + paginated items | Two separate queries, manual wiring | `get_with_count()` (one call) |
| Get-or-404 | `session.get()` + `if not` + `raise HTTPException` | `get_exist_one()` (one call) |
| PATCH DTO | Every field and constraint re-declared as `T \| None` | `partial=True` (derived, constraints kept) |
| "Omitted" vs `null` | Indistinguishable by type; `null` for a required column slips through | `Unset` vs `None`; `null` rejected where the base forbids it |
| Partial update | `model_dump(exclude_unset)` + `for/setattr` loop + manual `updated_at` | `article.update(session, data)` |
| Time filtering | Manual `if/where` per field | Built into `TableViewRequest` (timezone required) |
| Response DTO timestamps | Manually define `id`, `created_at`, `updated_at` fields | Inherit `UUIDIdDatetimeInfoMixin` |
| Optimistic locking | Not included (significant extra work) | Add `OptimisticLockMixin` to the model |

**Polymorphic endpoints** are just as clean:

```python
from abc import ABC, abstractmethod
from pydantic import EmailStr
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin, create_subclass_id_mixin,
    ListResponse, TableViewRequest,
    Str512, Text1K,
)

# ── Polymorphic models ────────────────────────────────────────────

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

# ── One endpoint returns all notification types ───────────────────

@router.get("/notifications", response_model=ListResponse[NotificationBase])
async def list_notifications(
        session: SessionDep, user: CurrentUserDep, table_view: TableViewRequestDep,
) -> ListResponse[Notification]:
    return await Notification.get_with_count(
        session,
        Notification.user_id == user.id,
        table_view=table_view,
    )
    # Returns EmailNotification and PushNotification instances transparently
```

---

## Detailed Guide

### TableBaseMixin & UUIDTableBaseMixin

These mixins provide the async CRUD interface. `TableBaseMixin` uses an auto-increment integer primary key; `UUIDTableBaseMixin` uses a **UUIDv7** primary key (RFC 9562: the first 48 bits are a millisecond timestamp, so ids sort roughly by creation time and index inserts stay local).

Both mixins automatically add `id`, `created_at`, and `updated_at` fields (timezone-aware UTC timestamps).

```python
from sqlmodel_ext import SQLModelBase, TableBaseMixin, UUIDTableBaseMixin, NonEmptyStrippedStr64, Text1K

# Integer primary key
class LogEntry(SQLModelBase, TableBaseMixin, table=True):
    message: Text1K

# UUIDv7 primary key (recommended for most use cases)
class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: NonEmptyStrippedStr64
```

#### `add()` -- Batch Insert

```python
users = [User(name="Alice", email="a@x.com"), User(name="Bob", email="b@x.com")]
users = await User.add(session, users)

# Or a single instance
user = await User.add(session, User(name="Alice", email="a@x.com"))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `AsyncSession` | required | Async database session |
| `instances` | `T \| list[T]` | required | Instance(s) to insert |
| `refresh` | `bool` | `True` | Whether to refresh instances after commit to sync DB-generated values |
| `commit` | `bool` | `True` | Whether to commit; `False` only flushes |

#### `save()` -- Insert or Update

```python
# Basic save
user = await user.save(session)

# Save with relationship preloading
# (rel() casts the Relationship field for type checkers, which would
#  otherwise infer User.profile as Profile instead of a loadable attribute)
user = await user.save(session, load=rel(User.profile))

# Explicit optimistic-lock retry count (default: the model policy)
user = await user.save(session, optimistic_retry_count=5)

# Skip refresh (return self without re-fetching from DB)
user = await user.save(session, refresh=False)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `AsyncSession` | required | Async database session |
| `load` | `QueryableAttribute \| list` | `None` | Relationship(s) to eagerly load after save |
| `refresh` | `bool` | `True` | Whether to refresh the object from DB after save |
| `commit` | `bool` | `True` | Whether to commit the transaction. Set `False` for batch operations |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | Polymorphic subclass loading (requires `load`) |
| `optimistic_retry_count` | `int \| None` | `None` | Retries on optimistic-lock conflicts. `None` = model policy (3 for `OptimisticLockMixin` models, 0 otherwise); `0` = no retry |

**Batch operations with `commit=False`:**

When inserting multiple records, you can defer the commit to reduce round-trips:

```python
await user1.save(session, commit=False)  # flush only
await user2.save(session, commit=False)  # flush only
user3 = await user3.save(session)        # commits all three
```

#### `update()` -- Partial Update from a Model

```python
# partial=True derives the PATCH body: every inherited field becomes
# ``Unset | T = Unset``, constraints and nullability preserved
class UserUpdate(UserBase, partial=True):
    pass

# Only the submitted fields are written; Unset fields never reach the database
user = await user.update(session, UserUpdate(name="Charlie"))

# With extra data not in the update model
user = await user.update(
    session,
    update_request,
    extra_data={"updated_by": current_user.id},
)

# Exclude specific fields
user = await user.update(session, data, exclude={"role", "is_admin"})
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `AsyncSession` | required | Async database session |
| `other` | `SQLModelBase` | required | Model instance whose submitted fields are merged into self |
| `extra_data` | `dict` | `None` | Additional fields to update beyond those in `other` |
| `exclude_unset` | `bool` | `True` | Only apply fields in `other.model_fields_set`. `Unset` fields are excluded from dumps regardless, so `partial=True` DTOs need nothing extra |
| `exclude` | `set[str]` | `None` | Field names to exclude from the update |
| `load` | `QueryableAttribute \| list` | `None` | Relationship(s) to eagerly load after update |
| `refresh` | `bool` | `True` | Whether to refresh the object from DB after update |
| `commit` | `bool` | `True` | Whether to commit the transaction |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | Polymorphic subclass loading (requires `load`) |
| `optimistic_retry_count` | `int \| None` | `None` | As in `save()`; a retry re-reads the row and re-applies `other`'s changes |

To reject privileged fields in a shared update body, `submitted_fields_among()` returns the submitted fields that belong to given models:

```python
forbidden = body.submitted_fields_among(ItemAdminOnlyFields)
if forbidden and not user.is_admin:
    raise PermissionError(sorted(forbidden))
```

#### `delete()` -- Instance or Condition Delete

```python
# Delete by instance
deleted_count = await User.delete(session, user)

# Delete by list
deleted_count = await User.delete(session, [user1, user2])

# Delete by condition (bulk)
deleted_count = await User.delete(session, condition=User.is_active == False)

# Delete without committing (for transactional batch operations)
await User.delete(session, user, commit=False)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `AsyncSession` | required | Async database session |
| `instances` | `T \| list[T]` | `None` | Instance(s) to delete |
| `condition` | `ColumnElement[bool]` | `None` | WHERE condition for bulk delete (keyword-only) |
| `commit` | `bool` | `True` | Whether to commit the transaction (keyword-only) |

Provide either `instances` or `condition`, not both — the overloads make "neither" a type error. `delete()` raises:

- `ResourceReferencedError` when a foreign key with `RESTRICT` / `NO ACTION` still references the row (PostgreSQL). Register a user-facing message per constraint with `TableBaseMixin.register_fk_delete_restrict_message(constraint_name, message)`.
- `OptimisticLockError` when the flush hits an optimistic-lock conflict (never retried).

```python
from sqlmodel_ext.mixins import ResourceReferencedError

try:
    await Folder.delete(session, folder)
except ResourceReferencedError as e:
    raise HTTPException(409, detail=str(e))
```

#### `get()` -- Flexible Queries

`get()` is the primary query method. It supports filtering, pagination, sorting, joins, relationship loading, polymorphic queries, time filtering, and row locking. `fetch_mode` selects a typed overload, so the return type is `T | None`, `T` or `list[T]`.

```python
from datetime import datetime, timezone

# Single record by condition
user = await User.get(session, User.email == "alice@example.com")

# Multiple conditions (use & operator)
user = await User.get(
    session,
    (User.name == "Alice") & (User.is_active == True),
)

# All records
users = await User.get(session, fetch_mode="all")

# With relationship preloading
user = await User.get(
    session,
    User.id == user_id,
    load=[rel(User.profile), rel(User.orders)],
)

# With JOIN
orders = await Order.get(
    session,
    Order.total > 100,
    join=User,
    fetch_mode="all",
)

# With FOR UPDATE row locking (always refreshes the identity map)
user = await User.get(
    session,
    User.id == user_id,
    with_for_update=True,
)

# Work queue: each worker claims a different row
job = await Job.get(session, Job.status == "pending", with_for_update=True, skip_locked=True)

# Authorization read: bypass identity map and cache
user = await User.get(session, User.id == user_id, authoritative=True)

# Time-based filtering (timezone-aware datetimes)
recent = await User.get(
    session,
    fetch_mode="all",
    created_after_datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
    created_before_datetime=datetime(2026, 12, 31, tzinfo=timezone.utc),
)
```

**Fetch modes:**

| Mode | Returns | Behavior |
|------|---------|----------|
| `"first"` (default) | `T \| None` | Returns the first result or `None` |
| `"one"` | `T` | Returns exactly one result; raises if not found or multiple |
| `"all"` | `list[T]` | Returns all matching records |

`get_one(session, id)` is the "must exist" shortcut (`NoResultFound` otherwise); `get_exist_one(session, id)` raises `HTTPException(404)` with FastAPI installed, `RecordNotFoundError` otherwise.

#### `count()`, `distinct_column()`, `group_sum()` -- Aggregates

```python
from datetime import datetime, timezone
from sqlmodel import col
from sqlmodel_ext import TimeFilterRequest

total = await User.count(session)
active = await User.count(session, User.is_active == True)

# COUNT(DISTINCT user_id)
buyers = await Order.count(session, distinct_column=col(Order.user_id))

# With time filter
recent_count = await User.count(
    session,
    time_filter=TimeFilterRequest(
        created_after_datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ),
)

# SELECT DISTINCT
countries = await User.distinct_column(session, col(User.country), limit=100)

# SUM per group (one query: COUNT(*) + every COALESCE(SUM(col), 0))
rows = await Order.group_sum(session, [col(Order.amount)], group_by=col(Order.status))
for row in rows:
    print(row.key, row.count, row.totals[0])
```

#### `get_with_count()` -- Paginated Response

Returns a `ListResponse[T]` containing both the total count and the paginated items:

```python
from sqlmodel_ext import ListResponse, TableViewRequest

result = await User.get_with_count(
    session,
    User.is_active == True,
    table_view=TableViewRequest(offset=0, limit=20, desc=True),
)
# result.count -> total matching records (e.g. 150)
# result.items -> list of 20 User instances
```

#### `select()` -- Typed Projections

`sqlmodel.select` stops its type overloads at 4 columns. `sqlmodel_ext.select` is the same function object at runtime, with overloads up to 9 columns:

```python
from sqlmodel import col
from sqlmodel_ext import select

stmt = select(col(User.id), col(User.name), col(User.email), col(User.created_at), col(User.updated_at))
```

---

### Pagination Models

sqlmodel-ext provides ready-to-use pagination and time filtering request models:

```python
from sqlmodel_ext import ListResponse, TableViewRequest, TimeFilterRequest, PaginationRequest
from sqlmodel_ext.pagination import PageWindowRequest
```

| Model | Fields |
|-------|--------|
| `PageWindowRequest` | `offset`, `limit`, `desc` — for endpoints whose sort order is fixed by the domain |
| `PaginationRequest` | `PageWindowRequest` + `order` + `after_id` (keyset cursor) |
| `TimeFilterRequest` | the four `created_*` / `updated_*` bounds |
| `TableViewRequest` | `TimeFilterRequest` + `PaginationRequest` |

**`TableViewRequest`** fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offset` | `int \| None` | `0` | Skip first N records (`0` ≤ offset ≤ `2**53 - 1001`) |
| `limit` | `int \| None` | `50` | Max records per page (1–100) |
| `desc` | `bool \| None` | `True` | Sort descending |
| `order` | `"created_at" \| "updated_at" \| "id"` | `"created_at"` | Sort field; `id` is always appended as a tie-break |
| `after_id` | `UUID \| None` | `None` | Keyset cursor: return records after this one. Requires `order` = `created_at` or `id`; cannot be combined with a non-zero `offset` |
| `created_after_datetime` | `AwareDatetime \| None` | `None` | Filter `created_at >= value` |
| `created_before_datetime` | `AwareDatetime \| None` | `None` | Filter `created_at < value` |
| `updated_after_datetime` | `AwareDatetime \| None` | `None` | Filter `updated_at >= value` |
| `updated_before_datetime` | `AwareDatetime \| None` | `None` | Filter `updated_at < value` |

Time bounds must carry a timezone: a naive datetime is rejected at validation instead of being silently interpreted in the database's timezone.

**Keyset pagination** — pass the id of the last item of the previous page; concurrent inserts and deletes do not shift the next page:

```python
page1 = await Article.get_with_count(session, table_view=TableViewRequest(limit=20))
page2 = await Article.get_with_count(
    session, table_view=TableViewRequest(limit=20, after_id=page1.items[-1].id),
)
```

If the anchor was deleted or no longer matches the query, `get()` raises `KeysetCursorInvalidError` instead of returning an empty page that looks like the end.

**`ListResponse[T]`** is the standard paginated response:

```python
from sqlmodel_ext import ListResponse

@router.get("", response_model=ListResponse[UserResponse])
async def list_users(session: SessionDep, table_view: TableViewRequestDep) -> ListResponse[User]:
    return await User.get_with_count(session, table_view=table_view)
```

---

### Polymorphic Inheritance

sqlmodel-ext supports both Joined Table Inheritance (JTI) and Single Table Inheritance (STI), simplifying SQLAlchemy's verbose polymorphic configuration.

#### Joined Table Inheritance (JTI)

Each subclass gets its own database table with a foreign key to the parent table. Use this when subclasses have significantly different fields.

```python
from abc import ABC, abstractmethod
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
    HttpUrl, NonEmptyStrippedStr64, NonNegativeInt,
)

# 1. Base class (fields only, no table)
class ToolBase(SQLModelBase):
    name: NonEmptyStrippedStr64

# 2. Abstract parent (creates the parent table)
class Tool(ToolBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
    @abstractmethod
    async def execute(self) -> str: ...

# 3. Create FK mixin for subclasses
ToolSubclassIdMixin = create_subclass_id_mixin('tool')

# 4. Concrete subclasses (each gets its own table)
class WebSearchTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    search_url: HttpUrl

    async def execute(self) -> str:
        return f"Searching {self.search_url}"

class CalculatorTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    precision: NonNegativeInt = 2

    async def execute(self) -> str:
        return "Calculating..."
```

**Key components:**

| Component | Purpose |
|-----------|---------|
| `PolymorphicBaseMixin` | Auto-configures `polymorphic_on`, adds `_polymorphic_name` discriminator column |
| `create_subclass_id_mixin(table)` | Creates a mixin with a FK+PK `id` field (UUIDv7 default) pointing to the parent table |
| `AutoPolymorphicIdentityMixin` | Auto-generates `polymorphic_identity` from class name (lowercase) |

**MRO order matters:** `SubclassIdMixin` must come first to properly override the `id` field:

```python
# Correct
class MyTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True): ...

# Wrong -- id field won't be overridden correctly
class MyTool(Tool, ToolSubclassIdMixin, AutoPolymorphicIdentityMixin, table=True): ...
```

#### Single Table Inheritance (STI)

All subclasses share the parent's table. Subclass-specific columns are added to the parent table as nullable. Use this when subclasses have few additional fields.

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
    upload_deadline: datetime | None = None  # Added to userfile table as nullable

class CompletedFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    file_size: NonNegativeBigInt | None = None  # Added to userfile table as nullable

# After all models are defined, before configure_mappers():
register_sti_columns_for_all_subclasses()
# After configure_mappers():
register_sti_column_properties_for_all_subclasses()
```

`get()`, `count()`, `delete(condition=...)` and the aggregates all add the discriminator filter automatically when called on an STI subclass, so a subclass-level delete never removes sibling rows.

#### Querying Polymorphic Models

```python
# Get all tools (returns concrete subclass instances)
tools = await Tool.get(session, fetch_mode="all")
# tools[0] might be WebSearchTool, tools[1] might be CalculatorTool

# Load polymorphic relationships
from sqlmodel import Relationship

class ToolSet(SQLModelBase, UUIDTableBaseMixin, table=True):
    tools: list[Tool] = Relationship(back_populates="tool_set")

# Load tools with all subclass data
tool_set = await ToolSet.get(
    session,
    ToolSet.id == ts_id,
    load=rel(ToolSet.tools),
    jti_subclasses='all',  # Loads all subclass-specific columns
)
```

#### Polymorphic Utility Methods

```python
# Get all concrete (non-abstract) subclasses
subclasses = Tool.get_concrete_subclasses()
# [WebSearchTool, CalculatorTool]

# Get identity-to-class mapping
mapping = Tool.get_identity_to_class_map()
# {'websearchtool': WebSearchTool, 'calculatortool': CalculatorTool}

# Check inheritance type
Tool._is_joined_table_inheritance()  # True for JTI, False for STI
```

---

### Row Locks and Transaction Contracts

A read-modify-write must lock the row first. `get(with_for_update=True)` issues `SELECT ... FOR UPDATE`, always refreshes the identity map (a stale in-memory object would otherwise cause a lost update), and records the lock on the session. Methods that rely on the lock declare it, and the declaration is checked at runtime (fail-closed):

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
await account.withdraw(session, amount=Decimal("10"))   # RuntimeError if `account` was not locked
account = await account.save(session)
```

`@requires_locked_param` checks that instances passed in a parameter are locked; `@requires_repeatable_read` / `@requires_read_committed` check the isolation level. Lock tracking follows savepoints: a rolled-back savepoint also forgets the locks taken inside it.

**Transaction helpers** on the enhanced session (PostgreSQL):

```python
from sqlmodel_ext.session import SessionFactory

session_factory = SessionFactory(engine, class_=AsyncSession, expire_on_commit=True)

async def transfer(session: AsyncSession) -> None:
    ...  # idempotent: may run more than once
    await session.commit()

# Own REPEATABLE READ session; serialization failures (40001) retry the whole operation
await session_factory.run_in_repeatable_read(transfer, description="transfer")

async with session_factory() as session:
    await session.set_local_timeouts(lock_timeout_ms=2_000, statement_timeout_ms=10_000)
    session.add_post_commit_callback(notify_downstream)   # runs only after a successful commit
    ...
```

---

### Optimistic Locking

Prevents lost updates in concurrent environments using SQLAlchemy's `version_id_col` mechanism.

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

# OptimisticLockMixin MUST come before SQLModelBase / TableBaseMixin in MRO
class Order(OptimisticLockMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    status: OrderStatusEnum = OrderStatusEnum.pending
    amount: NonNegativeDecimal38_18
```

The mixin adds an `oplock_version` BIGINT column (incremented by every write, with a `server_default` for rolling deploys, excluded from `model_dump()`). The name is reserved so it never collides with a domain `version` field — declaring `oplock_version` yourself raises `TypeError`. Every `UPDATE` generates SQL like:

```sql
UPDATE "order" SET status=?, amount=?, oplock_version=oplock_version+1
WHERE id=? AND oplock_version=?
```

If the `WHERE` clause doesn't match (another transaction modified the record), the update affects 0 rows and a conflict is detected.

#### Automatic Retry (Default)

`OptimisticLockMixin` models retry conflicts **3 times by default**: each retry re-reads the latest row and re-applies only the columns you changed, then saves again. `OptimisticLockError` is raised only when retries are exhausted.

```python
order = await order.save(session)                              # up to 3 retries
order = await order.update(session, update_data)               # same policy
order = await order.save(session, optimistic_retry_count=5)    # explicit count
```

#### Manual Error Handling

```python
try:
    order = await order.save(session, optimistic_retry_count=0)   # no retry
except OptimisticLockError as e:
    print(f"Conflict on {e.model_class} id={e.record_id}")
    print(f"Expected version: {e.expected_version}")
```

`delete()` on an `OptimisticLockMixin` model also raises `OptimisticLockError` on conflict (never retried; `record_id` is `None` because the conflict belongs to the whole flush).

**When to use optimistic locking:**
- State transitions (pending -> paid -> shipped)
- Numeric fields modified concurrently (balance, inventory)

**When NOT to use it:**
- Log/audit tables (insert-only)
- Simple counters (`UPDATE SET count = count + 1` is sufficient)

---

### Relation Preloading

The `RelationPreloadMixin` and `@requires_relations` decorator automatically load relationships before method execution, preventing `MissingGreenlet` errors in async SQLAlchemy.

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
        # generator and generator.config are auto-loaded before this runs
        return self.generator.config.price * 10
```

**How it works:**

1. `@requires_relations` declares which relationships a method needs
2. Before the method runs, the decorator checks which relationships are already loaded (using `sqlalchemy.inspect`)
3. Unloaded relationships are fetched in a single query
4. Already-loaded relationships are skipped (incremental loading)

**Supported argument formats:**

```python
@requires_relations(
    'generator',           # String: attribute name on this class
    Generator.config,      # QueryableAttribute: external class attribute (nested)
)
```

**Works with async generators too:**

```python
@requires_relations('items')
async def stream_items(self, session):
    for item in self.items:
        yield item
```

**Import-time validation:** String relationship names are verified at class creation time. If you declare `@requires_relations('nonexistent')`, you get an `AttributeError` immediately, not at runtime.

**Manual and batch preload API** (usually not needed):

```python
# Preload relationships for specific methods
await instance.preload_for(session, 'calculate_cost', 'validate')

# Get relationship list for a method (useful for query building)
rels = MyFunction.get_relations_for_method('calculate_cost')
rels = MyFunction.get_relations_for_methods('calculate_cost', 'validate')

# Batch: 1 query per distinct target root for a whole (possibly heterogeneous) list
await MyFunction.ensure_relations_loaded_bulk(session, functions, {MyFunction: ('generator',)})
```

Every `Relationship` defaults to `lazy='raise_on_sql'`: touching an unloaded relation raises a clear `InvalidRequestError` instead of an opaque `MissingGreenlet`. At startup, `RelationLoadChecker` statically finds the patterns that lead there (see [the explanation](docs/en/explanation/relation-load-checker.md)).

---

### Field Types

sqlmodel-ext provides reusable `Annotated` type aliases that work with both Pydantic validation and SQLAlchemy column mapping. All string aliases reject NUL bytes (PostgreSQL cannot store them).

#### String Constraints

| Type | Max Length | Use Case |
|------|-----------|----------|
| `Str1` / `Str16` / `Str24` / `Str32` | 1 / 16 / 24 / 32 | Flags, short codes, tokens |
| `Str36` | 36 | UUID strings |
| `Str48` / `Str64` / `Str100` / `Str128` | 48 / 64 / 100 / 128 | Labels, names, titles, identifiers |
| `Str255` / `Str256` / `Str500` / `Str512` / `Str2048` | 255 / 256 / 500 / 512 / 2048 | Standard VARCHAR, URLs |
| `Text1K` / `Text1024` / `Text2K` / `Text2500` / `Text3K` / `Text3072` | 1,000 – 3,072 | Short text |
| `Text4K` / `Text5K` / `Text8K` / `Text10K` / `Text16K` | 4,000 – 16,000 | Medium text |
| `Text32K` / `Text48K` / `Text60K` / `Text64K` (65,536) | 32,000 – 65,536 | Long text |
| `Text100K` / `Text128K` (131,072) / `Text1M` | 100,000 – 1,000,000 | Very long text |
| `NonEmptyStr64` / `128` / `256` | 1–N | Non-empty (not stripped) |
| `NonEmptyStrippedStr32` / `64` / `128` / `256` | 1–N | User-visible names: strips whitespace, rejects empty |
| `SingleLineStr64` / `SearchQueryStr64` | 64 | One-line names / search keywords (≥ 2 chars after stripping) |
| `Sha256Hex` | exactly 64 | Lowercase hex SHA-256 digest |
| `BCP47LanguageCode` | 16 | `zh-CN`, `en-US`, ... |
| `HttpHeaderName` | — | RFC 9110 token |

```python
from sqlmodel_ext import Str64, Text10K, max_length_of

class Article(SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    content: Text10K

max_length_of(Str64)   # 64 -- reflect the bound instead of repeating it
```

#### Numeric Constraints

| Type | Range | Column |
|------|-------|--------|
| `Port` | 1 – 65535 | INTEGER |
| `Percentage` | 0 – 100 | INTEGER |
| `PositiveInt` / `NonNegativeInt` | ≥ 1 / ≥ 0, ≤ 2³¹−1 | INTEGER |
| `PositiveBigInt` / `NonNegativeBigInt` / `SignedBigInt` | up to ±(2⁵³−1) (JS-safe) | BIGINT |
| `PositiveFloat` / `NonNegativeFloat` | > 0 / ≥ 0, finite (rejects inf / nan) | FLOAT |

```python
from sqlmodel_ext import Port, Percentage

class ServerConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    port: Port = 8080
    cpu_threshold: Percentage = 80
```

#### Decimal Types

`NUMERIC(p, s)` aliases for money and rates. They enforce the integer digits the column can hold, reject `float` / `bool` input (precision is already lost), and serialize to a fixed-point JSON **string** (no scientific notation, no trailing zeros) so JavaScript clients do not lose precision. `model_dump()` keeps `Decimal` objects.

| Family | Column | Notes |
|--------|--------|-------|
| `SignedDecimal38_18` / `NonNegativeDecimal38_18` / `PositiveDecimal38_18` | `NUMERIC(38, 18)` | 20 integer + 18 fractional digits |
| `Optional…Decimal38_18` | `NUMERIC(38, 18)` | `… \| None` |
| `…WriteDecimal38_18` | `NUMERIC(38, 18)` | Writes limited to 35 digits, leaving 1000x headroom for `SUM()` |
| `SignedSumDecimal38_18` | — | Reading aggregated sums |
| `SignedDecimal20_10` / `NonNegativeDecimal20_10` / `OptionalNonNegativeDecimal20_10` / `NullableNonNegativeDecimal20_10` | `NUMERIC(20, 10)` | Rates, ratios |

```python
from decimal import Decimal
from sqlmodel_ext import NonNegativeDecimal38_18

class WalletBase(SQLModelBase):
    balance: NonNegativeDecimal38_18 = Decimal(0)

class Wallet(WalletBase, UUIDTableBaseMixin, table=True):
    pass

WalletBase.model_validate({"balance": "12.50"})   # OK
WalletBase.model_validate({"balance": 12.5})      # ValidationError: float rejected
WalletBase(balance=Decimal("0.1")).model_dump_json()   # '{"balance":"0.1"}'
```

#### Bounded Lists

`List1` … `List1024` bound the number of elements: `tags: List20[Str32]`. `max_length_of()` reflects the bound.

#### URL Types

| Type | Validates | SSRF Protection |
|------|-----------|-----------------|
| `Url` | Any URL scheme | No |
| `HttpUrl` | HTTP/HTTPS only | No |
| `WebSocketUrl` | WS/WSS only | No |
| `SafeHttpUrl` | HTTP/HTTPS only | Yes |

All URL types are `str` subclasses -- they store as `VARCHAR` in the database and behave as plain strings in Python code, while providing Pydantic validation on assignment.

```python
from sqlmodel_ext import HttpUrl, SafeHttpUrl, WebSocketUrl

class APIConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    api_url: HttpUrl
    callback_url: SafeHttpUrl    # Blocks private IPs, localhost
    ws_endpoint: WebSocketUrl
```

**`SafeHttpUrl` blocks:**
- Private IPs (10.x, 172.16-31.x, 192.168.x)
- Loopback (127.x, ::1, localhost)
- Link-local (169.254.x)
- Non-HTTP protocols (file://, gopher://, etc.)

```python
from sqlmodel_ext import SafeHttpUrl, UnsafeURLError, validate_not_private_host

# The validator is also available standalone
try:
    validate_not_private_host("192.168.1.1")
except UnsafeURLError:
    print("Blocked private IP")
```

#### IP Address Types

```python
from sqlmodel_ext import IPAddress, ClientIPAddress

class Server(SQLModelBase, UUIDTableBaseMixin, table=True):
    ip: IPAddress          # storage column (VARCHAR, behaves like str)

server = Server(ip="192.168.1.1")
server.ip.is_private()  # True
```

`ClientIPAddress` is the parse-time counterpart for untrusted input (e.g. a proxy header): it validates to an `ipaddress` object and rejects IPv6 scope ids. Store the result in an `IPAddress` column.

#### Path Types

```python
from sqlmodel_ext import FilePathType, DirectoryPathType

class FileRecord(SQLModelBase, UUIDTableBaseMixin, table=True):
    file_path: FilePathType      # Must have a filename component
    output_dir: DirectoryPathType  # Must not have a file extension
```

---

### PostgreSQL Types

PostgreSQL-specific types live in `sqlmodel_ext.field_types.dialects.postgresql`. They are **not** imported from the top-level `sqlmodel_ext` package because they require PostgreSQL-specific dependencies.

```python
from sqlmodel_ext.field_types.dialects.postgresql import (
    Array,          # pip install sqlmodel-ext  (uses sqlalchemy.dialects.postgresql)
    JSON100K,       # pip install sqlmodel-ext[postgresql]  (requires orjson)
    JSONList100K,   # pip install sqlmodel-ext[postgresql]  (requires orjson)
    NumpyVector,    # pip install sqlmodel-ext[pgvector]  (requires numpy + pgvector)
)
```

#### `Array[T]` -- PostgreSQL ARRAY

A generic array type that maps Python `list[T]` to PostgreSQL's native `ARRAY` column type.

```python
from uuid import UUID
from sqlmodel import Field
from sqlmodel_ext.field_types.dialects.postgresql import Array

class Article(SQLModelBase, UUIDTableBaseMixin, table=True):
    tags: Array[str] = Field(default_factory=list)
    """String array stored as TEXT[] in PostgreSQL"""

    scores: Array[int] = Field(default_factory=list)
    """Integer array stored as INTEGER[] in PostgreSQL"""

    metadata_list: Array[dict] = Field(default_factory=list)
    """JSONB array stored as JSONB[] in PostgreSQL"""

    refs: Array[UUID] = Field(default_factory=list)
    """UUID array stored as UUID[] in PostgreSQL"""
```

**With max length:**

```python
class Config(SQLModelBase, UUIDTableBaseMixin, table=True):
    version_vector: Array[dict, 20] = Field(default_factory=list)
    """Max 20 elements, validated by Pydantic"""
```

**Supported inner types:**

| Python Type | PostgreSQL Type |
|-------------|----------------|
| `str` | `TEXT[]` |
| `int` | `INTEGER[]` |
| `dict` | `JSONB[]` |
| `UUID` | `UUID[]` |
| `Enum` subclass | `ENUM[]` (reads tolerate unknown values during rolling deploys) |

#### `JSON100K` / `JSONList100K` -- Size-Limited JSONB

JSONB types whose canonical JSON encoding never exceeds 100,000 characters, whatever the input form.

```python
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K, JSONList100K

class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
    canvas: JSON100K
    """Canvas data stored as JSONB (max 100K chars)"""

    messages: JSONList100K
    """Message list stored as JSONB (max 100K chars)"""
```

**Behavior — object in, object out:**

| Feature | `JSON100K` | `JSONList100K` |
|---------|-----------|---------------|
| Python type | `dict[str, Any]` | `list[dict[str, Any]]` |
| Accepts | `dict` (preferred) or a JSON string | `list` (preferred) or a JSON string |
| PostgreSQL type | `JSONB` | `JSONB` |
| Limits | 100,000 chars of canonical JSON; must be serializable (nesting depth) | same |
| API serialization | the JSON object itself | the JSON array itself |

`model_dump()`, `model_dump(mode='json')` and `model_dump_json()` all emit the value as nested JSON, never as an escaped string. The limits are also enforced on `table=True` models (which skip Pydantic validators) at construction.

#### `NumpyVector` -- pgvector + NumPy Integration

Stores vectors as pgvector's `Vector` type in PostgreSQL while exposing them as `numpy.ndarray` in Python. Supports fixed-dimension vectors with dtype enforcement.

```python
import numpy as np
from sqlmodel import Field
from sqlmodel_ext.field_types.dialects.postgresql import NumpyVector

class SpeakerInfo(SQLModelBase, UUIDTableBaseMixin, table=True):
    embedding: NumpyVector[1024, np.float32] = Field(...)
    """1024-dimensional float32 embedding vector"""

# Default dtype is float32
class Document(SQLModelBase, UUIDTableBaseMixin, table=True):
    embedding: NumpyVector[768] = Field(...)
    """768-dimensional vector (float32 by default)"""
```

**API serialization format** (base64-encoded for efficiency):

```json
{
    "dtype": "float32",
    "shape": 1024,
    "data_b64": "AAABAAA..."
}
```

**Accepted input formats:**

| Format | Example |
|--------|---------|
| `numpy.ndarray` | `np.zeros(1024, dtype=np.float32)` |
| `list` / `tuple` | `[0.1, 0.2, ...]` |
| base64 dict | `{"dtype": "float32", "shape": 1024, "data_b64": "..."}` |
| pgvector string | `"[0.1, 0.2, ...]"` (from database) |

**Vector similarity search** with pgvector operators:

```python
from sqlalchemy import select

# L2 distance (Euclidean)
stmt = select(SpeakerInfo).order_by(
    SpeakerInfo.embedding.l2_distance(query_vector)
).limit(10)

# Cosine distance
stmt = select(SpeakerInfo).order_by(
    SpeakerInfo.embedding.cosine_distance(query_vector)
).limit(10)

# Max inner product
stmt = select(SpeakerInfo).order_by(
    SpeakerInfo.embedding.max_inner_product(query_vector)
).limit(10)
```

**Vector exceptions:**

| Exception | When |
|-----------|------|
| `VectorError` | Base class for all vector errors |
| `VectorDimensionError` | Array dimensions don't match the declared size |
| `VectorDTypeError` | dtype conversion fails |
| `VectorDecodeError` | base64 or database format decoding fails |

```python
from sqlmodel_ext.field_types.dialects.postgresql import (
    VectorError, VectorDimensionError, VectorDTypeError, VectorDecodeError,
)
```

---

### Info Response DTO Mixins

Pre-built mixins for API response models that always include id and timestamp fields:

```python
from sqlmodel_ext import (
    SQLModelBase,
    UUIDIdDatetimeInfoMixin,  # UUID id + created_at + updated_at
    IntIdDatetimeInfoMixin,    # int id + created_at + updated_at
    UUIDIdInfoMixin,           # UUID id only
    IntIdInfoMixin,            # int id only
    DatetimeInfoMixin,         # created_at + updated_at only
)

class UserResponse(UserBase, UUIDIdDatetimeInfoMixin):
    """API response model -- id, created_at, updated_at are always present."""
    pass
```

These mixins define the fields as **required** (non-optional), because in API responses from the database, these fields are always populated. This is different from table models where `id=None` before insertion.

---

### Redis Caching (CachedTableBaseMixin)

Add a two-tier Redis cache to any table model. Queries go through Redis first; cache misses fall through to the database.

```bash
pip install sqlmodel-ext[cache]  # installs redis + orjson
```

**Setup (once at application startup):**

```python
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel_ext import AsyncSession, CachedTableBaseMixin

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=False)
CachedTableBaseMixin.configure_redis(redis)
CachedTableBaseMixin.check_cache_config()   # validates every cached model

# Required: the enhanced session invalidates the cache on commit
session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)
```

**Define a cached model:**

```python
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=1800):
    pass  # 30-minute cache TTL
```

That's it. `get()` checks Redis first, and every committed write — CRUD methods, bare `session.add()` / attribute changes / `session.delete()` — invalidates on `commit()`.

**Transaction transparency:** a query whose result depends on a table with uncommitted writes in the current transaction (including tables referenced in subqueries and `load` targets, and tables written by raw DML) neither reads from nor writes to the cache. Uncommitted state is never published to other requests.

**Cache architecture:**

| Layer | Key format | Invalidation |
|-------|-----------|--------------|
| ID cache | `id:{Model}:{id}` | Row-level DEL on commit |
| Query cache | `query:{Model}:v{version}:{hash}` | Version bump (O(1) INCR) makes old keys unreachable |
| Version | `ver:{Model}` | INCR on any write; STI subclass changes bump all ancestor versions |

**Writes outside the ORM:**

```python
# A trigger or raw SQL coupled to this transaction: invalidate after commit
Character.invalidate_on_commit(session, character_id)
await session.commit()

# Bulk data fix: drop every entry of the model
await Character.invalidate_all()

# Raw DML through another path than the enhanced session's execute()
CachedTableBaseMixin.register_raw_dml_write(session, statement)
```

For cache invalidation that must follow a data migration, `run_pending_migration_cache_invalidations()` runs the invalidation tasks declared by Alembic revisions (install `[alembic]`, or pass `tasks=` explicitly).

**Optional metrics callbacks:**

```python
CachedTableBaseMixin.on_cache_hit = lambda name: print(f"HIT: {name}")
CachedTableBaseMixin.on_cache_miss = lambda name: print(f"MISS: {name}")
```

**Cache skip conditions** (automatically detected):
- `no_cache=True` (explicit bypass)
- `authoritative=True` (authorization reads)
- uncommitted writes on any table the query depends on
- `with_for_update=True` / `populate_existing=True`
- `join` is set (join target changes don't trigger invalidation)
- `options` is set (custom loading options)
- `load` contains a relationship the ID cache cannot serve

**No Redis? No problem.** If you don't call `configure_redis()` and don't inherit `CachedTableBaseMixin`, there is zero Redis dependency. The caching layer is entirely opt-in.

---

### `partial=True` and `Unset`

`partial=True` derives a PATCH DTO from a base model:

```python
class ArticleBase(SQLModelBase):
    title: Str64
    """Article title"""
    body: Text10K
    """Article body"""
    summary: Str64 | None = None
    """Optional summary"""

class ArticleUpdateRequest(ArticleBase, partial=True):
    pass
    # title:   Unset | Str64        = Unset   (null rejected)
    # body:    Unset | Text10K      = Unset   (null rejected)
    # summary: Unset | Str64 | None = Unset   (null clears the column)
```

**What it does:**
- Converts `T` to `Unset | T = Unset` and `T | None` to `Unset | T | None = Unset` for inherited fields
- Preserves `Annotated` constraints (`max_length`, `ge`, ...) and attribute docstrings
- Hoists field-level attributes (`alias`, `exclude`, ...) so they keep working
- Skips `Literal` fields (discriminators must remain required) and fields the class declares itself
- An omitted `default_factory` field is `Unset`, not the factory value
- `partial=True` cannot be combined with `table=True` (`Unset` cannot be stored); the removed `all_fields_optional=True` keyword raises `TypeError` with migration instructions

**Checking fields by hand** — use `is Unset`, never `is None`:

```python
from sqlmodel_ext import Unset

if body.summary is not Unset:        # submitted (value or null)
    ...
```

**The spellings:**

| Annotation | Meaning |
|------------|---------|
| `Unset \| T = Unset` | may be omitted; `null` is rejected |
| `Unset \| T \| None = Unset` | may be omitted; `null` is a real value |
| `T \| None` | must be provided; may be `null` |
| `T = <default>` | may be omitted, has a natural default, `null` is rejected |

**Callers that cannot omit keys** (e.g. LLM function calling in strict mode) can enable a wire value per model:

```python
from sqlmodel_ext import SQLModelExtConfig

class UpdateArticleToolArgs(ArticleUpdateRequest):
    model_config = SQLModelExtConfig(omitted_sentinel=True)
    # inbound "__omitted__" is normalized to Unset; the JSON Schema gains a const branch
```

`Unset` is Pydantic's official `MISSING` sentinel (PEP 661), exported under one name for the whole codebase. Static narrowing requires basedpyright ≥ 1.40.1.

---

### Attribute Docstring Inheritance

When using Pydantic's `use_attribute_docstrings=True` (enabled by default in `SQLModelBase`), field descriptions appear in OpenAPI schemas. However, Pydantic's AST-based docstring parser doesn't inherit descriptions when subclasses override fields.

**sqlmodel-ext fixes this automatically.** The metaclass inherits missing descriptions from parent classes via MRO traversal, and `__get_pydantic_json_schema__` patches bare `$ref` properties to include descriptions.

```python
class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    """User display name"""     # ← parsed by Pydantic

class UserUpdateRequest(UserBase, partial=True):
    pass
    # name: Unset | NonEmptyStrippedStr64 = Unset — description "User display name" is inherited
    # Shows up correctly in OpenAPI/Swagger docs
```

---

### Enhanced AsyncSession

`sqlmodel_ext.AsyncSession` subclasses sqlmodel's `AsyncSession`. Point your session factory at it:

```python
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel_ext import AsyncSession

session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=True)
# or sqlmodel_ext.session.SessionFactory(...) for run_in_repeatable_read()
```

What it does:

- **`commit()`** registers every `CachedTableBaseMixin` mutation in the session (including bare `session.add()` / attribute mutation / `session.delete()`), synchronously invalidates the cache after commit, then runs post-commit callbacks. `commit_count` reports how many commits succeeded.
- **`rollback()`** discards pending post-commit callbacks; `best_effort_budget_seconds=` bounds a best-effort abort.
- **`begin()`** — `async with session.begin():` exits through the enhanced `commit()` / `rollback()`.
- **`reset()` / `close()`** release the connection and clear lock tracking, the REPEATABLE READ marker, callbacks and cache tracking.
- **`refresh()`** always reads the database (never the cache).
- **`execute()` / `exec()` / `scalar()` / `stream()` / `stream_scalars()`** register the tables written by raw DML as uncommitted (so dependent queries skip the cache) and warn when a raw `UPDATE` / `DELETE` hits a cached table without a registered invalidation.
- **`set_local_timeouts()` / `enter_repeatable_read()`** — PostgreSQL transaction helpers.

Models that don't use `CachedTableBaseMixin` are unaffected — every hook degrades to upstream behavior. If you use `CachedTableBaseMixin`, you **must** construct sessions with `class_=sqlmodel_ext.AsyncSession` (plain sessions fall back to a fire-and-forget `after_commit` compensation hook with a brief stale-cache window).

---

## Architecture

```
sqlmodel_ext/
    __init__.py              # Public API re-exports
    base.py                  # SQLModelBase, SQLModelExtConfig, partial=True, metaclass
    unset.py                 # Unset (pydantic MISSING), OMITTED_SENTINEL
    constants.py             # EXCLUDE_IF_NONE, optimistic-lock column name
    select.py                # select() with 5-9 column overloads
    session.py               # Enhanced AsyncSession, SessionFactory
    pagination.py            # ListResponse, PageWindowRequest, PaginationRequest, TimeFilterRequest, TableViewRequest
    relation_load_checker.py # RelationLoadChecker (static AST analysis, RLC001-RLC014)
    _compat.py               # Python 3.14 (PEP 649) compatibility
    _sa_type.py              # sa_type extraction from Annotated metadata
    _type_unwrap.py          # Annotated / union unwrapping helpers
    _exceptions.py           # RecordNotFoundError
    mixins/
        table.py             # TableBaseMixin, UUIDTableBaseMixin (async CRUD, aggregates, keyset)
        cached_table.py      # CachedTableBaseMixin (transaction-transparent Redis cache)
        polymorphic.py       # PolymorphicBaseMixin, AutoPolymorphicIdentityMixin, create_subclass_id_mixin, DeferredIndex
        optimistic_lock.py   # OptimisticLockMixin, OptimisticLockError
        relation_preload.py  # RelationPreloadMixin, @requires_relations, transaction-contract decorators
        exceptions.py        # ResourceReferencedError, keyset cursor errors
        info_response.py     # Id/Datetime DTO mixins
        resource_quota.py    # ResourceQuotaMixin
        trgm_searchable.py   # TrgmSearchableMixin
        mixin_table_scan.py  # MixinTableScanMixin
        migration_cache_invalidation.py  # run_pending_migration_cache_invalidations
        _uuid.py             # UUIDv7 generation
    field_types/
        __init__.py          # Type aliases (Str64, Port, Decimal, List*, ...), max_length_of
        _ssrf.py             # UnsafeURLError, validate_not_private_host
        ip_address.py        # IPAddress, ClientIPAddress
        url.py               # Url, HttpUrl, WebSocketUrl, SafeHttpUrl
        dialects/postgresql/ # Array[T], JSON100K / JSONList100K, NumpyVector
```

## Requirements

- **Python** >= 3.12 (tested on 3.12, 3.13, 3.14)
- **sqlmodel** >= 0.0.32
- **pydantic** >= 2.12
- **sqlalchemy** >= 2.0
- **typing-extensions** >= 4.14.1
- (optional) **fastapi** >= 0.100.0
- (optional) **redis** >= 5.0 -- for `CachedTableBaseMixin`
- (optional) **orjson** >= 3.0 -- for `CachedTableBaseMixin` and `JSON100K` / `JSONList100K`
- (optional) **alembic** >= 1.13 -- for migration-driven cache invalidation
- (optional) **numpy** >= 1.24 and **pgvector** >= 0.3 -- for `NumpyVector`
- (recommended) **basedpyright** >= 1.40.1

## AI Disclosure

This project was developed with AI-assisted coding (Claude). Approximately half of the code was written by humans and half by AI, with all code reviewed and validated by human developers.

## License

MIT
