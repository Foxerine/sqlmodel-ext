# 01 · Getting started

This is your first conversation with sqlmodel-ext. In 15 minutes you'll have:

- sqlmodel-ext installed
- Your first model defined
- A complete CRUD round-trip running: insert, query, update, delete
- A clear mental model for "model + Mixin = table"

::: tip You don't need prior SQLAlchemy or SQLModel experience
This tutorial introduces those concepts on demand. You only need Python 3.12+ and basic `async` / `await` knowledge. If you've never touched an ORM, skim [Prerequisites](/en/explanation/prerequisites) first.
:::

## 0. Set up the environment

Create a new directory and a virtual environment:

```bash
mkdir hello-sqlmodel-ext
cd hello-sqlmodel-ext
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

Install sqlmodel-ext and the async SQLite driver:

```bash
pip install sqlmodel-ext aiosqlite
```

## 1. Define your first model

Create `app.py`:

```python
from pydantic import EmailStr  # requires: pip install 'pydantic[email]'
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, NonEmptyStrippedStr64

class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    """User name (rejects "" and whitespace-only)"""
    email: EmailStr
    """Email"""

class User(UserBase, UUIDTableBaseMixin, table=True):
    pass
```

What just happened?

- **`UserBase`** inherits `SQLModelBase` — this is a **pure data model** with no table. It only declares fields. `NonEmptyStrippedStr64` is a string type alias provided by sqlmodel-ext (the naming-field variant of the `Str64` family): max 64 characters, auto-strips surrounding whitespace, rejects empty and whitespace-only strings — constraining Pydantic and creating a `VARCHAR(64)` column in SQLAlchemy in one step. `EmailStr` is Pydantic's email-format type.
- **`User`** inherits both `UserBase` (gets the fields) and `UUIDTableBaseMixin` (gets a UUID primary key + `created_at` / `updated_at` + the full set of CRUD methods). `table=True` tells SQLModel "create a table".

::: info Why split Base and Table
When you start writing APIs, `UserBase` becomes a useful POST request body (no `id` needed) while `User` is the database table. Tutorial 02 will use this pattern. For now, just remember: "Base — no table, Table — yes table".
:::

## 2. Create the engine and session factory

Add this to `app.py`:

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

`echo=True` makes SQLAlchemy print every SQL statement to the terminal — perfect for learning, since you can see exactly what's happening.

## 3. Run a CRUD round-trip

```python
async def main() -> None:
    await init_db()

    async with SessionLocal() as session:
        # CREATE
        alice = User(name="Alice", email="alice@example.com")
        alice = await alice.save(session)            # [!code highlight]
        print(f"Created: id={alice.id}")

        # READ
        fetched = await User.get_one(session, alice.id)
        print(f"Read: name={fetched.name}")

        # UPDATE
        alice.name = "Alice Cooper"
        alice = await alice.save(session)
        print(f"Updated: name={alice.name}")

        # LIST
        users = await User.get(session, fetch_mode="all")
        print(f"List: {len(users)} users")

        # DELETE
        deleted = await User.delete(session, alice)
        print(f"Deleted: {deleted} rows")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python app.py
```

Expected output (apart from the SQL log):

```
Created: id=550e8400-e29b-41d4-a716-446655440000
Read: name=Alice
Updated: name=Alice Cooper
List: 1 users
Deleted: 1 rows
```

## 4. Key takeaways

**Always use the return value of `save()`**:

```python
alice = await alice.save(session)    # ✅ correct
await alice.save(session)            # ❌ wrong
```

Why? `session.commit()` **expires** every object in the session — that's SQLAlchemy's design. `save()` returns a freshly-loaded object while the original `alice` variable is now expired. If you don't capture the return value, the next access to `alice.name` would trigger a re-fetch on an expired object — which in async land becomes a `MissingGreenlet` error.

::: tip This rule matters
**Every** `save()` / `update()` call must use the return value. Build the muscle memory: `x = await x.save(session)`.
:::

**`get_one` vs `get`**:

```python
user = await User.get_one(session, user_id)             # not found → exception
user = await User.get(session, User.id == user_id)      # not found → None
```

In endpoints you usually use `get_exist_one()` — it auto-raises HTTP 404 when not found. Tutorial 02 will use it.

**`fetch_mode`**:

```python
await User.get(session, fetch_mode="first")  # T | None
await User.get(session, fetch_mode="one")    # T, raises on 0 or multiple rows
await User.get(session, fetch_mode="all")    # list[T]
```

## 5. What you just learned

| Concept | Role |
|---------|------|
| `SQLModelBase` | Root class for all sqlmodel-ext models |
| `UUIDTableBaseMixin` | Adds UUID PK + timestamps + CRUD methods |
| `Str64` and friends | Type aliases satisfying both Pydantic validation and SQLAlchemy column types |
| `save()` / `get()` / `get_one()` / `delete()` | Async CRUD |
| The "use the return value" rule | After commit objects expire; you must work with the refreshed instance |

## Next

Tutorial 02 builds a full blog API on the same pattern: users, articles, comments, with FastAPI endpoints, pagination, JOINs, and relation preloading.

[Continue to 02 · Building a blog API →](./02-building-a-blog-api)
