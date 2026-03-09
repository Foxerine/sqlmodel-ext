# Prerequisites

This chapter covers the background concepts needed for reading the source code. If you're already familiar with ORMs and SQLAlchemy, feel free to skip ahead.

## What is an ORM?

ORM (Object-Relational Mapping) lets you use Python classes and objects instead of writing raw SQL.

```python
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=64)
    email: str

# Insert
user = User(name="Alice", email="alice@example.com")
session.add(user)
await session.commit()

# Query
statement = select(User).where(User.email == "alice@example.com")
result = await session.exec(statement)
user = result.first()
```

## SQLAlchemy Core Concepts

### Session

A Session is the "conversation channel" between you and the database. All database operations go through a Session:

```python
async def demo(session: AsyncSession):
    session.add(user)       # Mark object as "needs saving" (no SQL executed yet)
    await session.flush()   # Send pending operations to the database (no commit)
    await session.commit()  # Commit the transaction (permanent write)
    await session.refresh(user)  # Re-read the latest state from the database
```

::: danger Danger
`session.add()` does not execute SQL — it only places the object in a "pending queue".
`session.commit()` is what actually executes SQL, and it **expires all objects in the Session**.
Expired objects trigger new SQL queries when their attributes are accessed. In async environments, this causes `MissingGreenlet` errors.
:::

### Building select Statements

```python
from sqlmodel import select

select(User)                                                # SELECT * FROM user
select(User).where(User.email == "alice@example.com")       # WHERE email = ?
select(User).order_by(User.created_at.desc()).limit(20)     # ORDER BY ... LIMIT 20
select(func.count()).select_from(User)                      # SELECT COUNT(*)
```

Each method returns a new statement object (immutable chaining), executed via `session.exec(statement)`.

### Relationship

Relationships describe associations between tables:

```python
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    articles: list["Article"] = Relationship(back_populates="author")

class Article(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    author_id: int = Field(foreign_key="user.id")
    author: User = Relationship(back_populates="articles")
```

Accessing `user.articles` triggers SQLAlchemy to automatically execute `SELECT * FROM article WHERE author_id = ?`. This is called **lazy loading**.

### Lazy Loading Issues in Async

```python
async def get_user_articles(session: AsyncSession):
    user = await session.get(User, 1)
    print(user.articles)  # MissingGreenlet! // [!code error]
```

The solution is **eager loading**:

```python
from sqlalchemy.orm import selectinload

statement = select(User).options(selectinload(User.articles)) # [!code highlight]
result = await session.exec(statement)
user = result.first()
print(user.articles)  # Already loaded, no extra query triggered // [!code highlight]
```

sqlmodel-ext's `load` parameter and `RelationPreloadMixin` are abstractions over this pattern.

## Metaclass

Python uses `type()` to create class objects. `type` is the "metaclass" of all classes — **a class that creates classes**.

| Concept | Analogy |
|---------|---------|
| Instance | Cookie |
| Class | Cookie mold |
| Metaclass | **The machine that makes molds** — can modify the mold as it's being created |

Custom metaclasses let you **intercept the class creation process**:

```python
class MyMeta(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, attrs, **kwargs)

class MyClass(metaclass=MyMeta):
    pass
# Output: Creating class: MyClass
```

SQLModel uses the metaclass `SQLModelMetaclass`. sqlmodel-ext inherits it and adds more automation logic — that's `__DeclarativeMeta`.

## `Annotated` Types

Python 3.9+ introduced `Annotated`, which attaches extra metadata to type annotations:

```python
from typing import Annotated
from sqlmodel import Field

# These two notations are equivalent:
name: str = Field(max_length=64)
name: Annotated[str, Field(max_length=64)]
```

The advantage is defining **reusable type aliases**:

```python
Str64 = Annotated[str, Field(max_length=64)]

class User(SQLModel, table=True):
    name: Str64    # Pydantic validation + SQLAlchemy VARCHAR(64)
    title: Str64   # Reuses the same constraints
```

## Polymorphic Inheritance Database Concepts

Different types of objects share base fields but each has specialized fields:

| Approach | Table Structure | Use Case |
|----------|----------------|----------|
| **Joined Table Inheritance (JTI)** | One table for parent + one table per subclass, linked by FK | Large field differences between subclasses |
| **Single Table Inheritance (STI)** | All subclasses share one table, subclass fields are nullable | Few extra fields per subclass |

See [Polymorphic Inheritance](./polymorphic) for details.
