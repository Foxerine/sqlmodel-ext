# Redis Caching

`CachedTableBaseMixin` adds a Redis cache layer to the model's `get()` queries, with automatic invalidation on CRUD operations.

## Basic Usage

```python
from sqlmodel_ext import CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin

class CharacterBase(SQLModelBase):
    name: str

class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=1800):
    pass  # Cache for 30 minutes
```

::: warning Warning
`CachedTableBaseMixin` must be placed **before** `UUIDTableBaseMixin` in the inheritance list.
:::

### Configure Redis

Configure once at startup:

```python
import redis.asyncio as redis

redis_client = redis.from_url("redis://localhost:6379")
CachedTableBaseMixin.configure_redis(redis_client)
```

### Startup Check (Recommended)

```python
CachedTableBaseMixin.check_cache_config()
```

Verifies configuration correctness for all cached models: Redis client is set, `__cache_ttl__` is a positive integer, no forbidden direct calls.

## Cache Behavior

### Automatic Caching

`get()` query results are automatically cached to Redis; subsequent identical queries read directly from cache:

```python
# First time: queries database + writes cache
char = await Character.get(session, Character.id == char_id) # [!code highlight]

# Second time: reads from cache directly, zero SQL
char = await Character.get(session, Character.id == char_id) # [!code highlight]
```

### Automatic Invalidation

CRUD operations automatically clean related caches:

| Operation | Invalidation Strategy |
|-----------|----------------------|
| `save()` / `update()` | Delete the record's ID cache + all query caches for the model |
| `delete(instances)` | Delete each instance's ID cache + all query caches |
| `delete(condition)` | Delete all caches for the model (ID + query) |
| `add()` | Delete all query caches (new objects have no old cache) |

### Skipping Cache

```python
# Explicitly skip cache
char = await Character.get(session, Character.id == char_id, no_cache=True) # [!code highlight]
```

::: details Details
Cache is automatically skipped in these scenarios:
- `with_for_update=True` (needs latest data)
- `populate_existing=True`
- `options` parameter is non-empty
- `join` parameter is non-empty
- Transaction has uncommitted pending invalidation data
:::

## Dual-Layer Cache Architecture

```
1. ID Cache (id:{ModelName}:{id_value})
   → Single-row ID equality queries
   → Row-level invalidation O(1)

2. Query Cache (query:{ModelName}:{hash})
   → Conditional/list queries
   → Model-level invalidation SCAN+DEL
```

ID queries (`Character.id == some_id`) use precise ID cache keys, requiring only one key deletion for invalidation. Other queries use parameter hashes as cache keys.

## Manual Invalidation

```python
# Invalidate specific ID
await Character.invalidate_by_id(char_id)
await Character.invalidate_by_id(id1, id2, id3)  # Multiple

# Invalidate all caches for the model
await Character.invalidate_all()
```

## Cache invalidation on cascade deletes

When a parent row is deleted, ``CachedTableBaseMixin`` follows one of two
paths depending on the relationship's ``passive_deletes`` flag, and both paths
correctly clean up the child model's cache.

| `passive_deletes` | Who removes the children | How the cache is invalidated |
|-------------------|--------------------------|------------------------------|
| `False` (default) | SA issues one `DELETE` per child during flush | `persistent_to_deleted` event fires automatically |
| `True` | The database `ON DELETE CASCADE` removes them silently | `delete()` pre-queries child IDs, then explicitly invalidates after `DELETE` |

```python
class User(CachedTableBaseMixin, UserBase, UUIDTableBaseMixin, table=True):
    # passive_deletes=True: rely on the DB CASCADE; SA will not load children
    files: list['UserFile'] = Relationship(
        back_populates='user',
        cascade_delete=True,
        passive_deletes=True,
    )
```

``passive_deletes=True`` is especially useful under high write pressure
because it keeps SA from pulling thousands of child rows into memory. Older
releases rejected this combination outright. Starting with 0.3.0, ``delete()``
pre-queries the child IDs before the DB CASCADE runs, then explicitly calls
``_invalidate_id_cache`` + ``_invalidate_query_caches`` on the target class,
so the child model's Redis cache can never serve stale data. Condition-based
deletes (``delete(condition=...)``) cannot enumerate parent IDs and therefore
perform a model-level full invalidation on the target.

## Polymorphic Inheritance Support

STI subclass caches automatically cascade to ancestor classes: when subclass data changes, ancestor class query caches are also cleaned.

## TTL Configuration

::: code-group

```python [Keyword argument (recommended)]
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin,
                table=True, cache_ttl=1800):  # [!code highlight]
    pass  # 30 minutes

class Config(CachedTableBaseMixin, ConfigBase, UUIDTableBaseMixin,
             table=True, cache_ttl=86400):  # [!code highlight]
    pass  # 24 hours
```

```python [Class variable]
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True):
    __cache_ttl__: ClassVar[int] = 1800  # [!code highlight]
```

:::

Default TTL is 3600 seconds (1 hour).

## Graceful Degradation

::: tip Tip
When Redis is unavailable, the system automatically degrades to database queries without affecting business logic:
- Read failure → log + query database
- Write failure → log + continue (non-critical path)
- Delete failure → log + TTL provides eventual consistency
:::

## Raw SQL Scenarios

::: warning Warning
When bypassing the ORM, you need to manage cache manually:
:::

```python
from sqlmodel_ext.mixins.cached_table import CachedTableBaseMixin

# Register before raw SQL operations
CachedTableBaseMixin._register_pending_invalidation(session, Character, char_id) # [!code warning]

# Commit and invalidate
await instance._commit_and_invalidate(session)
```
