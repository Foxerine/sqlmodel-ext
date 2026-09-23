# 01 · Getting started

This is your first conversation with sqlmodel-ext. In 15 minutes you'll have:

- sqlmodel-ext installed, with basedpyright checking your code
- Your first model defined
- A complete CRUD round-trip running: insert, query, update, PATCH, delete
- A clear mental model for "model + Mixin = table" and "declare once, derive the rest"

::: tip You don't need prior SQLAlchemy or SQLModel experience
This tutorial introduces those concepts on demand. You only need Python 3.12+ and basic `async` / `await` knowledge. If you've never touched an ORM, skim [Prerequisites](/en/explanation/prerequisites) first.
:::

::: warning Work in progress
sqlmodel-ext is under active development; APIs may change between releases without notice. Pin the version you build against, and use it at your own risk.
:::

## 0. Set up the environment

Create a new directory and a virtual environment:

```bash
mkdir hello-sqlmodel-ext
cd hello-sqlmodel-ext
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

Install sqlmodel-ext, the async SQLite driver, the email validator used below, and basedpyright:

```bash
pip install sqlmodel-ext aiosqlite "pydantic[email]" "basedpyright>=1.40.1"
```

## 1. Configure basedpyright

sqlmodel-ext puts field constraints and the "was this field sent?" state into **types**, so a type checker catches most misuse before the code runs. Set it up first, so it watches every step of this tutorial. Create `pyrightconfig.json`:

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

::: info Why ≥ 1.40.1
basedpyright 1.40.1 is the first release that narrows `x is Unset` correctly. You'll meet `Unset` in step 5.
:::

## 2. Define your first model

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

- **`UserBase`** inherits `SQLModelBase` — this is a **pure data model** with no table. It only declares fields. `NonEmptyStrippedStr64` is a string type alias provided by sqlmodel-ext (the naming-field variant of the `Str64` family): max 64 characters, auto-strips surrounding whitespace, rejects empty and whitespace-only strings — constraining Pydantic, creating a `VARCHAR(64)` column in SQLAlchemy and publishing `maxLength: 64` in the JSON Schema, all from this one declaration. `EmailStr` is Pydantic's email-format type.
- **`User`** inherits both `UserBase` (gets the fields) and `UUIDTableBaseMixin` (gets a UUIDv7 primary key + `created_at` / `updated_at` + the full set of CRUD methods). `table=True` tells SQLModel "create a table".

::: info Why split Base and Table
When you start writing APIs, `UserBase` becomes a useful POST request body (no `id` needed) while `User` is the database table. Every other shape — the PATCH body, the response — is **derived** from `UserBase` rather than re-declared. For now, just remember: "Base — no table, Table — yes table".
:::

## 3. Create the engine and session factory

Add this to `app.py`:

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

`echo=True` makes SQLAlchemy print every SQL statement to the terminal — perfect for learning, since you can see exactly what's happening.

`sqlmodel_ext.AsyncSession` is a subclass of sqlmodel's `AsyncSession`. You don't need its extras yet, but tutorial 03 (caching) requires it, so start with it.

## 4. Run a CRUD round-trip

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

Expected output (apart from the SQL log; your id differs):

```
Created: id=01a0cd5e-36eb-7859-b8c1-57c960506ae0
Read: name=Alice
Updated: name=Alice Cooper
List: 1 users
Deleted: 1 rows
```

The id is a **UUIDv7**: its first 48 bits are a millisecond timestamp, so ids created later sort later.

## 5. PATCH with a derived DTO

An HTTP PATCH body says "change these fields, leave the rest alone". Instead of writing a second class that repeats every field as optional, derive it:

```python
class UserUpdate(UserBase, partial=True):
    """PATCH body: every field of UserBase, each one omissible."""
```

`partial=True` turns each inherited field into `Unset | T = Unset`. `Unset` means "not sent" — a different state from `None` ("sent as null"). Constraints and docstrings carry over unchanged.

Add the PATCH step to `main()`, between UPDATE and LIST:

```python
        # PATCH (only the submitted fields are written)
        patch = UserUpdate.model_validate({"email": "alice@cooper.dev"})
        print(f"Submitted: {patch.model_dump()}")
        alice = await alice.update(session, patch)
        print(f"Patched: name={alice.name}, email={alice.email}")
```

and, after the `async with` block, see what happens when a client sends `null` for a field the base does not allow to be null:

```python
    try:
        _ = UserUpdate.model_validate({"name": None})
    except ValidationError as e:     # from pydantic import ValidationError
        print(f"Rejected: {e.error_count()} errors for 'name'")
```

New output lines:

```
Submitted: {'email': 'alice@cooper.dev'}
Patched: name=Alice Cooper, email=alice@cooper.dev
Rejected: 2 errors for 'name'
```

`name` was not sent, so it is `Unset` — it does not appear in `model_dump()` and `update()` never touches it. And `null` is rejected because `UserBase.name` is not nullable; the PATCH body inherited that fact instead of re-stating it.

## 6. Let basedpyright check it

```bash
basedpyright
```

```
0 errors, 0 warnings, 0 notes
```

Now break something on purpose — for example `print(users.name)` after the LIST step (`users` is a `list[User]`) — and run it again. basedpyright reports `Cannot access attribute "name" for class "list[User]"` without running the program. Keep it running in your editor from now on.

## 7. Key takeaways

**Always use the return value of `save()`**:

```python
alice = await alice.save(session)    # ✅ correct
await alice.save(session)            # ❌ wrong
```

Why? `session.commit()` **expires** every object in the session (we set `expire_on_commit=True`, SQLAlchemy's default). `save()` returns a freshly-loaded object while the original `alice` variable is now expired. If you don't capture the return value, the next access to `alice.name` would trigger a re-fetch on an expired object — which in async land becomes a `MissingGreenlet` error.

::: tip This rule matters
**Every** `save()` / `update()` call must use the return value. Build the muscle memory: `x = await x.save(session)`.
:::

**`Unset`, not `None`, means "not sent"**. When you check a PATCH field by hand, write `if patch.email is not Unset:` (`from sqlmodel_ext import Unset`). `None` is a real value that a client can send.

**`get_one` vs `get`**:

```python
user = await User.get_one(session, user_id)             # not found → exception
user = await User.get(session, User.id == user_id)      # not found → None
```

In endpoints you usually use `get_exist_one()` — it auto-raises HTTP 404 when not found. Tutorial 02 will use it.

**`fetch_mode`** (the return type follows it, so the type checker knows what you got):

```python
await User.get(session, fetch_mode="first")  # T | None
await User.get(session, fetch_mode="one")    # T, raises on 0 or multiple rows
await User.get(session, fetch_mode="all")    # list[T]
```

## 8. What you just learned

| Concept | Role |
|---------|------|
| `SQLModelBase` | Root class for all sqlmodel-ext models |
| `UUIDTableBaseMixin` | Adds UUIDv7 PK + timestamps + CRUD methods |
| `Str64` and friends | One declaration drives Pydantic validation, the column type and the JSON Schema |
| `partial=True` + `Unset` | PATCH DTO derived from the base; "not sent" is distinct from `null` |
| `save()` / `get()` / `get_one()` / `update()` / `delete()` | Async CRUD |
| basedpyright | Catches misuse before the code runs |
| The "use the return value" rule | After commit objects expire; you must work with the refreshed instance |

## Next

Tutorial 02 builds a full blog API on the same pattern: users, articles, comments, with FastAPI endpoints, pagination, JOINs, and relation preloading.

[Continue to 02 · Building a blog API →](./02-building-a-blog-api)
