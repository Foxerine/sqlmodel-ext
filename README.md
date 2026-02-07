# sqlmodel-ext

[![PyPI version](https://img.shields.io/pypi/v/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![Python versions](https://img.shields.io/pypi/pyversions/sqlmodel-ext.svg)](https://pypi.org/project/sqlmodel-ext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**English** | [中文](README_zh.md)

Extended SQLModel infrastructure: smart metaclass, async CRUD mixins, polymorphic inheritance, optimistic locking, relation preloading, and reusable field types.

**sqlmodel-ext** eliminates the boilerplate of building async database applications with [SQLModel](https://sqlmodel.tiangolo.com/). Define your models, inherit a mixin, and get a full async CRUD API -- pagination, relationship loading, polymorphic queries, and optimistic locking included.

## Features

| Feature | Description |
|---------|-------------|
| **SQLModelBase** | Smart metaclass with automatic `table=True`, `mapper_args` merging, and Python 3.14 (PEP 649) compatibility |
| **TableBaseMixin / UUIDTableBaseMixin** | Full async CRUD: `add()`, `save()`, `update()`, `delete()`, `get()`, `count()`, `get_with_count()`, `get_exist_one()` |
| **PolymorphicBaseMixin** | Simplified Joined Table Inheritance (JTI) and Single Table Inheritance (STI) |
| **AutoPolymorphicIdentityMixin** | Auto-generated `polymorphic_identity` from class names |
| **OptimisticLockMixin** | Version-based optimistic locking with automatic retry |
| **RelationPreloadMixin** | Decorator-based automatic relationship preloading (prevents `MissingGreenlet` errors) |
| **ListResponse[T]** | Generic paginated response model for list endpoints |
| **Field Types** | Reusable constrained types: `Str64`, `Port`, `IPAddress`, `HttpUrl`, `SafeHttpUrl`, and more |
| **Info Response DTOs** | Pre-built mixins for API response models with id/timestamp fields |

## Installation

```bash
pip install sqlmodel-ext
```

With [FastAPI](https://fastapi.tiangolo.com/) support (enables `HTTPException` in `get_exist_one()`):

```bash
pip install sqlmodel-ext[fastapi]
```

## Quick Start

### Define Models

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64

# Base class -- fields only, no database table
class UserBase(SQLModelBase):
    name: Str64
    email: str

# Table class -- inherits fields + gains async CRUD + UUID primary key
class User(UserBase, UUIDTableBaseMixin, table=True):
    pass
```

`SQLModelBase` is the foundation for all models. Its metaclass automatically:
- Sets `table=True` when it detects `TableBaseMixin` in the inheritance chain
- Merges `__mapper_args__` from parent classes
- Extracts `sa_type` from `Annotated` metadata for proper column mapping
- Applies Python 3.14 (PEP 649) compatibility patches

### Async CRUD

All CRUD methods are async and require an `AsyncSession`:

```python
from sqlmodel.ext.asyncio.session import AsyncSession

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

    # Update
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
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlmodel import Field
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, Str64, Text10K,
    ListResponse, TableViewRequest, UUIDIdDatetimeInfoMixin,
)

# ── Dependencies (defined once, reused everywhere) ────────────────

SessionDep = Annotated[AsyncSession, Depends(get_session)]
TableViewDep = Annotated[TableViewRequest, Depends()]

# ── Models ────────────────────────────────────────────────────────

class ArticleBase(SQLModelBase):
    title: Str64
    body: Text10K
    is_published: bool = False

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id')

class ArticleCreate(ArticleBase):
    pass

class ArticleUpdate(ArticleBase):
    title: Str64 | None = None       # Override to optional,
    body: Text10K | None = None      # preserving the original
    is_published: bool | None = None  # type constraints from Base

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID

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

No manual SQL, no hand-written pagination logic, no boilerplate session management. The `TableViewDep` gives clients `offset`, `limit`, `desc`, `order`, and four time filters out of the box.

**What the client gets from `GET /articles?offset=0&limit=10&desc=true`:**

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

**Side-by-side comparison:**

| Concern | Traditional | sqlmodel-ext |
|---------|-------------|------------|
| Primary key + timestamps | 4 fields, manually defined | Inherited from `UUIDTableBaseMixin` |
| Pagination + sorting | ~20 lines per list endpoint | `table_view=table_view` (one arg) |
| Count + paginated items | Two separate queries, manual wiring | `get_with_count()` (one call) |
| Get-or-404 | `session.get()` + `if not` + `raise HTTPException` | `get_exist_one()` (one call) |
| Partial update | `model_dump(exclude_unset)` + `for/setattr` loop + manual `updated_at` | `article.update(session, data)` |
| Time filtering | Manual `if/where` per field | Built into `TableViewRequest` |
| Response DTO timestamps | Manually define `id`, `created_at`, `updated_at` fields | Inherit `UUIDIdDatetimeInfoMixin` |
| Optimistic locking | Not included (significant extra work) | Add `OptimisticLockMixin` to model |

**Polymorphic endpoints** are just as clean:

```python
from abc import ABC, abstractmethod
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin, create_subclass_id_mixin,
    ListResponse, TableViewRequest,
)

# ── Polymorphic models ────────────────────────────────────────────

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

# ── One endpoint returns all notification types ───────────────────

@router.get("/notifications", response_model=ListResponse[NotificationBase])
async def list_notifications(
        session: SessionDep, user: CurrentUserDep, table_view: TableViewDep,
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

These mixins provide the async CRUD interface. `TableBaseMixin` uses an auto-increment integer primary key; `UUIDTableBaseMixin` uses a UUID4 primary key.

Both mixins automatically add `id`, `created_at`, and `updated_at` fields.

```python
from sqlmodel_ext import SQLModelBase, TableBaseMixin, UUIDTableBaseMixin

# Integer primary key
class LogEntry(SQLModelBase, TableBaseMixin, table=True):
    message: str

# UUID primary key (recommended for most use cases)
class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str
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

#### `save()` -- Insert or Update

```python
# Basic save
user = await user.save(session)

# Save with relationship preloading
user = await user.save(session, load=User.profile)

# Save with optimistic lock retry
user = await user.save(session, optimistic_retry_count=3)

# Skip refresh (return self without re-fetching from DB)
user = await user.save(session, refresh=False)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `AsyncSession` | required | Async database session |
| `load` | `RelationshipInfo \| list` | `None` | Relationship(s) to eagerly load after save |
| `refresh` | `bool` | `True` | Whether to refresh the object from DB after save |
| `commit` | `bool` | `True` | Whether to commit the transaction. Set `False` for batch operations |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | Polymorphic subclass loading (requires `load`) |
| `optimistic_retry_count` | `int` | `0` | Auto-retry count on optimistic lock conflicts |

**Batch operations with `commit=False`:**

When inserting multiple records, you can defer the commit to reduce round-trips:

```python
await user1.save(session, commit=False)  # flush only
await user2.save(session, commit=False)  # flush only
user3 = await user3.save(session)        # commits all three
```

#### `update()` -- Partial Update from a Model

```python
class UserUpdate(SQLModelBase):
    name: str | None = None
    email: str | None = None

# Only updates fields that were explicitly set
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
| `other` | `SQLModelBase` | required | Model instance whose set fields will be merged into self |
| `extra_data` | `dict` | `None` | Additional fields to update beyond those in `other` |
| `exclude_unset` | `bool` | `True` | If `True`, skip fields not explicitly set in `other` |
| `exclude` | `set[str]` | `None` | Field names to exclude from the update |
| `load` | `RelationshipInfo \| list` | `None` | Relationship(s) to eagerly load after update |
| `refresh` | `bool` | `True` | Whether to refresh the object from DB after update |
| `commit` | `bool` | `True` | Whether to commit the transaction |
| `jti_subclasses` | `list[type] \| 'all'` | `None` | Polymorphic subclass loading (requires `load`) |
| `optimistic_retry_count` | `int` | `0` | Auto-retry count on optimistic lock conflicts |

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
| `condition` | `BinaryExpression` | `None` | WHERE condition for bulk delete |
| `commit` | `bool` | `True` | Whether to commit the transaction |

Provide either `instances` or `condition`, not both.

#### `get()` -- Flexible Queries

`get()` is the primary query method. It supports filtering, pagination, sorting, joins, relationship loading, polymorphic queries, time filtering, and row locking.

```python
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
    load=[User.profile, User.orders],
)

# With JOIN
orders = await Order.get(
    session,
    Order.total > 100,
    join=User,
    fetch_mode="all",
)

# With FOR UPDATE row locking
user = await User.get(
    session,
    User.id == user_id,
    with_for_update=True,
)

# Time-based filtering
recent = await User.get(
    session,
    fetch_mode="all",
    created_after_datetime=datetime(2024, 1, 1),
    created_before_datetime=datetime(2024, 12, 31),
)
```

**Fetch modes:**

| Mode | Returns | Behavior |
|------|---------|----------|
| `"first"` (default) | `T \| None` | Returns the first result or `None` |
| `"one"` | `T` | Returns exactly one result; raises if not found or multiple |
| `"all"` | `list[T]` | Returns all matching records |

#### `count()` -- Efficient Record Counting

```python
total = await User.count(session)
active = await User.count(session, User.is_active == True)

# With time filter
from sqlmodel_ext import TimeFilterRequest
recent_count = await User.count(
    session,
    time_filter=TimeFilterRequest(
        created_after_datetime=datetime(2024, 1, 1),
    ),
)
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

#### `get_exist_one()` -- Get or 404

```python
# Raises HTTPException(404) if FastAPI is installed, else RecordNotFoundError
user = await User.get_exist_one(session, user_id)
```

---

### Pagination Models

sqlmodel-ext provides ready-to-use pagination and time filtering request models:

```python
from sqlmodel_ext import ListResponse, TableViewRequest, TimeFilterRequest, PaginationRequest
```

**`TableViewRequest`** combines pagination + sorting + time filtering:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offset` | `int` | `0` | Skip first N records |
| `limit` | `int` | `50` | Max records per page (max 100) |
| `desc` | `bool` | `True` | Sort descending |
| `order` | `"created_at" \| "updated_at"` | `"created_at"` | Sort field |
| `created_after_datetime` | `datetime \| None` | `None` | Filter `created_at >= value` |
| `created_before_datetime` | `datetime \| None` | `None` | Filter `created_at < value` |
| `updated_after_datetime` | `datetime \| None` | `None` | Filter `updated_at >= value` |
| `updated_before_datetime` | `datetime \| None` | `None` | Filter `updated_at < value` |

**`ListResponse[T]`** is the standard paginated response:

```python
from sqlmodel_ext import ListResponse

# Use with FastAPI
@router.get("", response_model=ListResponse[UserResponse])
async def list_users(session: SessionDep, table_view: TableViewRequest):
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
)

# 1. Base class (fields only, no table)
class ToolBase(SQLModelBase):
    name: str

# 2. Abstract parent (creates the parent table)
class Tool(ToolBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
    @abstractmethod
    async def execute(self) -> str: ...

# 3. Create FK mixin for subclasses
ToolSubclassIdMixin = create_subclass_id_mixin('tool')

# 4. Concrete subclasses (each gets its own table)
class WebSearchTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    search_url: str

    async def execute(self) -> str:
        return f"Searching {self.search_url}"

class CalculatorTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    precision: int = 2

    async def execute(self) -> str:
        return "Calculating..."
```

**Key components:**

| Component | Purpose |
|-----------|---------|
| `PolymorphicBaseMixin` | Auto-configures `polymorphic_on`, adds `_polymorphic_name` discriminator column |
| `create_subclass_id_mixin(table)` | Creates a mixin with a FK+PK `id` field pointing to the parent table |
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
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)

class UserFile(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    filename: str

class PendingFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    upload_deadline: datetime | None = None  # Added to userfile table as nullable

class CompletedFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    file_size: int | None = None  # Added to userfile table as nullable

# After all models are defined, before configure_mappers():
register_sti_columns_for_all_subclasses()
# After configure_mappers():
register_sti_column_properties_for_all_subclasses()
```

#### Querying Polymorphic Models

```python
# Get all tools (returns concrete subclass instances)
tools = await Tool.get(session, fetch_mode="all")
# tools[0] might be WebSearchTool, tools[1] might be CalculatorTool

# Load polymorphic relationships
from sqlmodel_ext import UUIDTableBaseMixin
from sqlmodel import Relationship

class ToolSet(SQLModelBase, UUIDTableBaseMixin, table=True):
    tools: list[Tool] = Relationship(back_populates="tool_set")

# Load tools with all subclass data
tool_set = await ToolSet.get(
    session,
    ToolSet.id == ts_id,
    load=ToolSet.tools,
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

### Optimistic Locking

Prevents lost updates in concurrent environments using SQLAlchemy's `version_id_col` mechanism.

```python
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    OptimisticLockMixin, OptimisticLockError,
)

# OptimisticLockMixin MUST come before TableBaseMixin in MRO
class Order(OptimisticLockMixin, UUIDTableBaseMixin, table=True):
    status: str
    amount: int
```

The mixin adds a `version` integer field (starting at 0). Every `UPDATE` generates SQL like:

```sql
UPDATE "order" SET status=?, amount=?, version=version+1
WHERE id=? AND version=?
```

If the `WHERE` clause doesn't match (another transaction modified the record), the update affects 0 rows, and an `OptimisticLockError` is raised.

#### Manual Error Handling

```python
try:
    order = await order.save(session)
except OptimisticLockError as e:
    print(f"Conflict on {e.model_class} id={e.record_id}")
    print(f"Expected version: {e.expected_version}")
    # Re-fetch and retry...
```

#### Automatic Retry (Recommended)

```python
# Retries up to 3 times on conflict:
# 1. Re-fetches the latest record from DB
# 2. Re-applies your changes
# 3. Attempts save again
order = await order.save(session, optimistic_retry_count=3)

# Also works with update()
order = await order.update(session, update_data, optimistic_retry_count=3)
```

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
    Generator.config,      # RelationshipInfo: external class attribute (nested)
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

**Manual preload API** (usually not needed):

```python
# Preload relationships for specific methods
await instance.preload_for(session, 'calculate_cost', 'validate')

# Get relationship list for a method (useful for query building)
rels = MyFunction.get_relations_for_method('calculate_cost')
rels = MyFunction.get_relations_for_methods('calculate_cost', 'validate')
```

---

### Field Types

sqlmodel-ext provides reusable `Annotated` type aliases that work with both Pydantic validation and SQLAlchemy column mapping.

#### String Constraints

| Type | Max Length | Use Case |
|------|-----------|----------|
| `Str24` | 24 | Short codes |
| `Str32` | 32 | Tokens, hashes |
| `Str36` | 36 | UUID strings |
| `Str48` | 48 | Short labels |
| `Str64` | 64 | Names, titles |
| `Str100` | 100 | Descriptions |
| `Str128` | 128 | Paths, identifiers |
| `Str255` | 255 | Standard VARCHAR |
| `Str256` | 256 | Standard VARCHAR |
| `Text1K` | 1,000 | Short text |
| `Text1024` | 1,024 | Short text (power of 2) |
| `Text2K` | 2,000 | Medium text |
| `Text2500` | 2,500 | Medium text |
| `Text3K` | 3,000 | Medium text |
| `Text10K` | 10,000 | Long text |
| `Text60K` | 60,000 | Very long text |
| `Text64K` | 65,536 | TEXT column |
| `Text100K` | 100,000 | Large text |

```python
from sqlmodel_ext import Str64, Text10K

class Article(SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    content: Text10K
```

#### Numeric Constraints

| Type | Range | Use Case |
|------|-------|----------|
| `Port` | 1--65535 | Network ports |
| `Percentage` | 0--100 | Percentages |
| `PositiveInt` | >= 1 | Counts, quantities |
| `NonNegativeInt` | >= 0 | Indices, counters |
| `PositiveFloat` | > 0.0 | Prices, weights |

```python
from sqlmodel_ext import Port, Percentage

class ServerConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    port: Port = 8080
    cpu_threshold: Percentage = 80
```

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

#### IP Address Type

```python
from sqlmodel_ext import IPAddress

class Server(SQLModelBase, UUIDTableBaseMixin, table=True):
    ip: IPAddress

server = Server(ip="192.168.1.1")
server.ip.is_private()  # True
```

#### Path Types

```python
from sqlmodel_ext import FilePathType, DirectoryPathType

class FileRecord(SQLModelBase, UUIDTableBaseMixin, table=True):
    file_path: FilePathType      # Must have a filename component
    output_dir: DirectoryPathType  # Must not have a file extension
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

## Architecture

```
sqlmodel_ext/
    __init__.py              # Public API re-exports
    base.py                  # SQLModelBase + __DeclarativeMeta metaclass
    _compat.py               # Python 3.14 (PEP 649) monkey-patches
    _sa_type.py              # sa_type extraction from Annotated metadata
    _utils.py                # now(), now_date() timestamp utilities
    _exceptions.py           # RecordNotFoundError
    pagination.py            # ListResponse, TimeFilterRequest, PaginationRequest, TableViewRequest
    mixins/
        __init__.py          # Mixin re-exports
        table.py             # TableBaseMixin, UUIDTableBaseMixin (async CRUD)
        polymorphic.py       # PolymorphicBaseMixin, AutoPolymorphicIdentityMixin, create_subclass_id_mixin
        optimistic_lock.py   # OptimisticLockMixin, OptimisticLockError
        relation_preload.py  # RelationPreloadMixin, @requires_relations
        info_response.py     # Id/Datetime DTO mixins
    field_types/
        __init__.py          # Type alias re-exports (Str64, Port, etc.)
        _ssrf.py             # UnsafeURLError, validate_not_private_host
        ip_address.py        # IPAddress type
        url.py               # Url, HttpUrl, WebSocketUrl, SafeHttpUrl
        _internal/path.py    # Path type handlers
        mixins/              # ModuleNameMixin
```

## Requirements

- **Python** >= 3.12 (tested on 3.12, 3.13, 3.14)
- **sqlmodel** >= 0.0.22
- **pydantic** >= 2.0
- **sqlalchemy** >= 2.0
- (optional) **fastapi** >= 0.100.0

## AI Disclosure

This project was developed with AI-assisted coding (Claude). Approximately half of the code was written by humans and half by AI, with all code reviewed and validated by human developers.

## License

MIT
