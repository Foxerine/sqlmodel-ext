# CRUD Operations

Inherit `TableBaseMixin` (auto-increment ID) or `UUIDTableBaseMixin` (UUID ID) to get a full set of async CRUD methods.

## Utility Functions

### `rel()` — Type-safe Relationship Reference

basedpyright infers SQLModel's Relationship fields as the annotated type (e.g., `LLM`) rather than `QueryableAttribute`. `rel()` fixes this type issue:

```python
from sqlmodel_ext import rel

# Without rel: basedpyright reports type error
user = await User.get(session, load=User.profile) # [!code --]

# With rel: correct type
user = await User.get(session, load=rel(User.profile)) # [!code ++]
```

### `cond()` — Type-safe Condition Composition

Similar issue: basedpyright infers `Model.field == value` as `bool`, causing `&` / `|` operator errors.

```python
from sqlmodel_ext import cond

scope = cond(UserFile.user_id == current_user.id) # [!code highlight]
condition = scope & cond(UserFile.status == FileStatusEnum.uploaded) # [!code highlight]
users = await UserFile.get(session, condition, fetch_mode="all")
```

### `sanitize_integrity_error()` — User-friendly Error Messages

Extracts user-safe error messages from `IntegrityError`. Specifically supports PostgreSQL trigger SQLSTATE 23514 errors:

```python
from sqlalchemy.exc import IntegrityError

try:
    await order.save(session)
except IntegrityError as e:
    msg = Order.sanitize_integrity_error(e, "Operation failed")
    # PostgreSQL trigger messages are extracted; other constraint errors return the default message
```

## Built-in Fields

Three fields are automatically included upon inheritance:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` or `UUID` | Primary key, auto-generated |
| `created_at` | `datetime` | Creation time, auto-set |
| `updated_at` | `datetime` | Update time, auto-refreshed on every UPDATE |

## `save()` — Create or Update

```python{3}
user = User(name="Alice", email="alice@example.com")
await user.save(session)            # Don't do this — causes expiration // [!code --]
user = await user.save(session)     # Correct usage // [!code ++]
print(user.id)  # Has a value now

# Modify then save again = UPDATE
user.name = "Bob"
user = await user.save(session)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `commit` | `True` | When `False`, only flushes without committing — useful for batch operations |
| `load` | `None` | Preload specified relations after saving |
| `optimistic_retry_count` | `0` | Number of retries on optimistic lock conflicts |

```python
# Batch operations: only flush earlier ones, commit with the last
await user1.save(session, commit=False)
await user2.save(session, commit=False)
user3 = await user3.save(session)  # Commits all at once // [!code highlight]

# Preload relations after save
user = await user.save(session, load=User.profile)
```

## `add()` — Bulk Insert

```python
users = [User(name="Alice"), User(name="Bob")]
users = await User.add(session, users)

# Single item works too
user = await User.add(session, User(name="Eve"))
```

## `update()` — Partial Update (PATCH Semantics)

```python
class UserUpdate(SQLModelBase):
    name: str | None = None
    email: str | None = None

data = UserUpdate(name="Bob")  # Only name is set // [!code highlight]
user = await user.update(session, data)
# Only name is updated, email remains unchanged (exclude_unset=True)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exclude_unset` | `True` | Only update explicitly set fields (PATCH semantics) |
| `exclude` | `None` | Exclude certain fields from being updated |
| `extra_data` | `None` | Append additional fields beyond the update model |

```python
# Append extra fields
user = await user.update(session, data, extra_data={"updated_by": admin.id})

# Exclude sensitive fields
user = await user.update(session, data, exclude={"role", "is_admin"})
```

## `delete()` — Delete

Two modes, mutually exclusive:

```python
# Delete by instance
await User.delete(session, user)
await User.delete(session, [user1, user2])

# Bulk delete by condition (⚠️ deletes ALL matching records)
count = await User.delete(session, condition=User.is_active == False) # [!code warning]
```

Returns the number of deleted records.

## `get()` — Query

A versatile query method that covers all query scenarios through different parameter combinations.

### Basic Queries

```python
# Get first match by condition (default fetch_mode="first")
user = await User.get(session, User.email == "alice@example.com")

# Get all
users = await User.get(session, fetch_mode="all") # [!code highlight]

# Get exactly one (errors on 0 or multiple results)
user = await User.get(session, User.id == some_id, fetch_mode="one") # [!code highlight]
```

### `fetch_mode` Return Values

| `fetch_mode` | Return Type | On 0 results | On multiple results |
|---|---|---|---|
| `"first"` (default) | `T \| None` | `None` | Returns first |
| `"one"` | `T` | Raises exception | Raises exception |
| `"all"` | `list[T]` | Empty list | Returns all |

### Pagination & Sorting

```python
from sqlmodel_ext import TableViewRequest

tv = TableViewRequest(offset=0, limit=20, desc=True, order="created_at")
users = await User.get(session, fetch_mode="all", table_view=tv)
# → SELECT ... ORDER BY created_at DESC LIMIT 20 OFFSET 0
```

### Time Filtering

```python
users = await User.get(
    session,
    fetch_mode="all",
    created_after=datetime(2024, 1, 1),
    created_before=datetime(2024, 12, 31),
)
```

### Relation Preloading

```python
user = await User.get(
    session,
    User.id == user_id,
    load=[User.profile, Profile.avatar], # [!code highlight]
)
# Automatically builds: selectinload(User.profile).selectinload(Profile.avatar)
```

### Other Parameters

| Parameter | Purpose |
|-----------|---------|
| `join` | JOIN another table |
| `options` | Custom SQLAlchemy query options |
| `filter` | Additional WHERE conditions |
| `with_for_update` | SELECT ... FOR UPDATE (row lock) |

## `count()` — Count

```python
total = await User.count(session)
active = await User.count(session, User.is_active == True)
```

## `get_with_count()` — Paginated List

Combination of `count()` + `get(fetch_mode="all")`, returns `ListResponse`:

```python
result = await User.get_with_count(session, table_view=table_view)
# result.count = 42
# result.items = [User(...), User(...), ...]
```

## `get_one()` — Guaranteed Existence Query

Similar to `get(fetch_mode="one")` but with a simpler interface — pass the ID directly:

```python
user = await User.get_one(session, user_id)
# Record not found → NoResultFound // [!code error]
# Multiple records → MultipleResultsFound // [!code error]

# Query with lock
user = await User.get_one(session, user_id, with_for_update=True)
```

::: tip Tip
`get_one()` raises SQLAlchemy exceptions (`NoResultFound`), while `get_exist_one()` raises HTTP 404 (with FastAPI) or `RecordNotFoundError`.
:::

## `get_exist_one()` — Find or 404

```python
user = await User.get_exist_one(session, user_id) # [!code highlight]
# Not found → HTTPException(404) (with FastAPI)
# Or RecordNotFoundError (without FastAPI)

# With relation preloading
user = await User.get_exist_one(session, user_id, load=User.profile)
```

## Method Quick Reference

| Method | Type | SQL Equivalent | Return Value |
|--------|------|---------------|--------------|
| `add()` | Class method | INSERT | Instance or list |
| `save()` | Instance method | INSERT or UPDATE | Refreshed instance |
| `update()` | Instance method | UPDATE (partial) | Refreshed instance |
| `delete()` | Class method | DELETE | Delete count |
| `get()` | Class method | SELECT + WHERE + ... | Instance / list / None |
| `get_one()` | Class method | SELECT + guaranteed existence | Instance |
| `count()` | Class method | SELECT COUNT(*) | int |
| `get_with_count()` | Class method | COUNT + SELECT | ListResponse |
| `get_exist_one()` | Class method | SELECT + 404 | Instance |
