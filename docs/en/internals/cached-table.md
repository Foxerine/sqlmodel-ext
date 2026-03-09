# Redis Caching

::: tip Tip
`src/sqlmodel_ext/mixins/cached_table.py` — `CachedTableBaseMixin` (~1700 lines)
:::

This is the largest module added in 0.2.0, providing a Redis cache layer for query results.

## Dual-Layer Cache Architecture

```
1. ID Cache (id:{ModelName}:{id_value})
   - For cls.id == value single-row exact queries
   - Row-level invalidation O(1)

2. Query Cache (query:{ModelName}:{md5_hash})
   - For conditional queries, list queries
   - Model-level invalidation SCAN+DEL
```

### Cache Key Generation

ID cache keys are directly concatenated: `id:Character:550e8400-...`

Query cache keys normalize all parameters (conditions, pagination, sorting, filtering, time) and compute an MD5 hash (first 16 characters), ensuring semantically identical queries produce the same key.

## Core Class Structure

```python
class CachedTableBaseMixin(TableBaseMixin):
    __cache_ttl__: ClassVar[int] = 3600

    # Redis client (shared at class level)
    _redis_client: ClassVar[Any] = None

    @classmethod
    def configure_redis(cls, client: Any) -> None: ...

    # Cache primitives
    @classmethod
    async def _cache_get(cls, key: str) -> bytes | None: ...
    @classmethod
    async def _cache_set(cls, key: str, value: bytes, ttl: int) -> None: ...
    @classmethod
    async def _cache_delete(cls, key: str) -> None: ...
    @classmethod
    async def _cache_delete_pattern(cls, pattern: str) -> None: ...
```

## `get()` Override

Overrides `TableBaseMixin.get()`, adding cache logic before and after the database query:

```python
@classmethod
async def get(cls, session, condition, *, no_cache=False, ...):
    # 1. Determine if cache can be used
    if no_cache or with_for_update or populate_existing or ...:
        return await super().get(session, condition, ...)

    # 2. Check for pending invalidation data in transaction
    if session.info has pending invalidation for this model:
        return await super().get(session, condition, ...)

    # 3. Detect if this is an ID query
    id_value = cls._extract_id_from_condition(condition) # [!code focus]

    # 4. Multi-ID cache joint query (load + MANYTOONE relations)
    if id_value and load contains only cacheable MANYTOONE:
        result = await cls._try_load_from_id_caches(...) # [!code focus]
        if result is not _LOAD_CACHE_MISS:
            return result

    # 5. Build cache key + try reading
    cache_key = cls._build_cache_key(condition, fetch_mode, ...)
    cached = await cls._cache_get(cache_key) # [!code focus]
    if cached:
        return cls._deserialize_result(cached, fetch_mode) # [!code highlight]

    # 6. Cache miss, query database
    result = await super().get(session, condition, ...) # [!code warning]

    # 7. Write to cache
    serialized = cls._serialize_result(result)
    await cls._cache_set(cache_key, serialized, cls.__cache_ttl__) # [!code focus]

    return result
```

### ID Query Detection

```python
@classmethod
def _extract_id_from_condition(cls, condition):
    """Detects pure ID equality queries, returns ID value or None"""
```

Detects `cls.id == value` form conditions, using precise ID cache keys instead of query hashes.

### Multi-ID Cache Joint Query

When `load` specifies only cacheable MANYTOONE relations, attempts to read the primary object and relation objects from their respective ID caches — returning zero-SQL results if all hit.

```python
@classmethod
async def _try_load_from_id_caches(cls, session, id_value, rel_info):
    # 1. Read primary model ID cache
    # 2. Read each relation target's ID cache
    # 3. All hit → assemble and return
    # 4. Any miss → return _LOAD_CACHE_MISS
```

## Serialization Scheme

```python
# Wrapper format
{
    "_t": "none|single|list",   # Result type
    "_data": {...},             # Single item data (result of model_dump_json)
    "_items": [{...}, ...],     # List data
    "_c": "ClassName"           # Polymorphic safety: records actual class name
}
```

Serialization uses `model_dump_json()` → JSON → `json.loads()`. Deserialization uses `model_validate()` (not `model_validate_json` to avoid UUID stringification issues with table=True models).

Optional orjson support for faster serialization.

## Cache Invalidation

### Invalidation in CRUD Methods

Each CRUD method is overridden to perform invalidation before/after commit:

```python
async def save(self, session, ...):
    result = await super().save(session, ...)

    # Immediately invalidate when commit=True
    await self._invalidate_for_model(instance_id) # [!code focus]

    # Write-through refresh: write latest data to ID cache
    serialized = cls._serialize_result(result)
    await cls._cache_set(id_cache_key, serialized, cls.__cache_ttl__) # [!code focus]

    return result
```

### Invalidation Granularity

| Operation | Strategy |
|-----------|----------|
| `save/update` | `DEL id:{cls}:{id}` + `SCAN+DEL query:{cls}:*` |
| `delete(instances)` | Per-instance `DEL id:*` + `SCAN+DEL query:*` |
| `delete(condition)` | `SCAN+DEL id:*` + `SCAN+DEL query:*` (full model) |
| `add()` | Only `SCAN+DEL query:*` |

### Polymorphic Inheritance Cascading

When STI subclass data changes, traverses MRO to invalidate all ancestor caches:

```python
async def _invalidate_id_cache(cls, instance_id):
    await cls._cache_delete(f"id:{cls.__name__}:{instance_id}")
    # Traverse ancestors
    for ancestor in cls.__mro__:
        if issubclass(ancestor, CachedTableBaseMixin):
            await ancestor._cache_delete(f"id:{ancestor.__name__}:{instance_id}")
```

## Invalidation Compensation Mechanism

Handles `commit=False` scenarios (deferred commit):

### session.info State Tracking

```python
session.info['_cache_pending']  # Pending invalidation: dict[type, set[id]]
session.info['_cache_synced']   # Already synced: dict[type, set[id]]
```

### Two Paths

1. **Synchronous path** (CRUD method with `commit=True`): directly `await` invalidation
2. **Async compensation path** (`commit=False`):
   - Record pending invalidation types and IDs in `session.info`
   - Register SQLAlchemy `after_commit` event hook
   - On commit, the compensation function invalidates what synced path didn't cover

### Sentinel Objects

```python
_QUERY_ONLY_INVALIDATION  # add() scenario: only invalidate query cache
_FULL_MODEL_INVALIDATION  # delete(condition) scenario: full model invalidation
_LOAD_CACHE_MISS          # Multi-ID cache joint query miss
```

## MissingGreenlet Avoidance

::: danger Danger
After commit, SQLAlchemy resets object association states. Directly accessing attributes triggers synchronous queries.
:::

Solutions:

- Extract IDs with `getattr()` before commit
- After commit, read from identity map via `sa_inspect()` (no DB query)
- External SQL methods use `_register_pending_invalidation()` + `_commit_and_invalidate()`

## `check_cache_config()` Static Check

```python
@classmethod
def check_cache_config(cls) -> None:
```

Validates:
1. Redis client has been set via `configure_redis()`
2. All subclasses' `__cache_ttl__` are positive integers
3. AST check: forbids direct calls to `invalidate_by_id()` etc. in non-cache methods (prevents MissingGreenlet)

## Graceful Degradation

All Redis operations are wrapped in try/except:
- Read failure → return None (degrade to database)
- Write failure → log + continue
- Delete failure → log (TTL provides eventual consistency)
