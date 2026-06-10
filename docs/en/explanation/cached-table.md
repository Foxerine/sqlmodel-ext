# Redis cache mechanism

::: tip Source location
`src/sqlmodel_ext/mixins/cached_table.py` — `CachedTableBaseMixin` (~1900 lines)
:::

`CachedTableBaseMixin` provides a transparent Redis cache layer for `get()` queries with automatic invalidation on CRUD operations. This chapter explains **how the internals work**; to plug it into your own project, see [Cache queries with Redis](/en/how-to/cache-queries).

## Dual-layer cache architecture

```
1. ID Cache (id:{ModelName}:{id_value})
   - For cls.id == value single-row exact queries
   - Row-level invalidation O(1)

2. Query Cache (query:{ModelName}:v{version}:{md5_hash})
   - For conditional and list queries
   - Model-level invalidation: version bump O(1) (old keys expire via TTL)
```

::: info 0.3 version-based invalidation
Starting with 0.3.0, model-level query cache invalidation switched from `SCAN+DEL` to **version bumping**: every model has a `ver:{ModelName}` counter, the cache key embeds the version, and invalidation only requires a single `INCR`. Old-version keys can no longer be read and disappear naturally via TTL. This drops model-level invalidation cost from O(N keys) to O(1).
:::

### Cache key generation

ID cache keys are directly concatenated: `id:Character:550e8400-...`

Query cache keys normalize all parameters (conditions, pagination, sorting, filtering, time) and compute an MD5 hash (first 16 characters), ensuring semantically identical queries produce the same key. Full format: `query:Character:v3:abcdef0123456789`.

## Core class structure

```python
class CachedTableBaseMixin(TableBaseMixin):
    __cache_ttl__: ClassVar[int] = 3600

    # Redis client (shared at class level)
    _redis_client: ClassVar[Any] = None

    # Optional metric hooks
    on_cache_hit: ClassVar[Callable[[str], None] | None] = None
    on_cache_miss: ClassVar[Callable[[str], None] | None] = None

    @classmethod
    def configure_redis(cls, client: Any) -> None: ...

    @classmethod
    def check_cache_config(cls) -> None: ...

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

`on_cache_hit` / `on_cache_miss` are optional metric hooks — set callbacks at startup to feed hit ratios into Prometheus / Grafana.

## `get()` override

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

### ID query detection

```python
@classmethod
def _extract_id_from_condition(cls, condition):
    """Detects pure ID equality queries, returns ID value or None"""
```

Detects `cls.id == value` form conditions, using precise ID cache keys instead of query hashes.

### Multi-ID cache joint query

When `load` specifies only cacheable MANYTOONE relations, attempts to read the primary object and relation objects from their respective ID caches — returning zero-SQL results if all hit.

```python
@classmethod
async def _try_load_from_id_caches(cls, session, id_value, rel_info):
    # 1. Read primary model ID cache
    # 2. Read each relation target's ID cache
    # 3. All hit → assemble and return
    # 4. Any miss → return _LOAD_CACHE_MISS
```

## Serialization scheme

```python
# Wrapper format
{
    "_t": "none|single|list",   # Result type
    "_data": {...},             # Single item data (result of model_dump_json)
    "_items": [{...}, ...],     # List data
    "_c": "ClassName"           # Polymorphic safety: records actual class name
}
```

Serialization uses `model_dump_json()` → JSON → `json.loads()`. Deserialization uses `model_validate()` (not `model_validate_json` to avoid UUID stringification issues with `table=True` models).

Optional orjson support for faster serialization.

## Cache invalidation

### CRUD registers, commit invalidates (since 0.4.0)

The CRUD overrides **no longer invalidate on their own** — they only register
pending invalidations into `session.info`; the actual invalidation is executed
synchronously after commit by the enhanced `sqlmodel_ext.AsyncSession.commit()`:

```python
async def save(self, session, ...):
    # 1. Register pending invalidation
    self._register_pending_invalidation(session, model_type, instance_id) # [!code focus]

    # 2. super().save() awaits session.commit() internally --
    #    the enhanced AsyncSession sync-invalidates every registered pending after commit # [!code focus]
    result = await super().save(session, ...)

    # 3. Write-through refresh: refill the ID cache of self + every cached ancestor
    serialized = cls._serialize_result(result)
    for refill_cls in [model_type, *model_type._cached_ancestors()]: # [!code focus]
        await refill_cls._cache_set(refill_cls._build_id_cache_key(instance_id), serialized, ...)

    return result
```

With `commit=False` nothing is invalidated (the data is not committed yet);
the pendings wait for the caller's `await session.commit()` — which is exactly
why "several `commit=False` operations + one commit" is naturally correct.

### Invalidation granularity

| Operation | Strategy |
|-----------|----------|
| `save/update` | `DEL id:{cls}:{id}` + `INCR ver:{cls}` |
| `delete(instances)` | per-instance `DEL id:*` + `INCR ver:{cls}` |
| `delete(condition)` | model-level ID cleanup + `INCR ver:{cls}` |
| `add()` | only `INCR ver:{cls}` (new objects have no stale cache) |

### Polymorphic inheritance cascading

When STI subclass data changes, traverses MRO to invalidate all ancestor caches:

```python
async def _invalidate_id_cache(cls, instance_id):
    await cls._cache_delete(f"id:{cls.__name__}:{instance_id}")
    # Traverse ancestors
    for ancestor in cls._cached_ancestors():
        await ancestor._cache_delete(f"id:{ancestor.__name__}:{instance_id}")
```

`_cached_ancestors()` caches all ancestors in the MRO that also inherit `CachedTableBaseMixin`. **Refill mirrors invalidation**: write-through refresh refills exactly the keys invalidation cleared — otherwise callers querying through an ancestor class could observe a stale instance inside the race window.

## Invalidation compensation mechanism

### `session.info` state tracking

```python
session.info['_cache_pending']  # Pending invalidation: dict[type, set[id]]
session.info['_cache_synced']   # Already synced: dict[type, set[id]]
```

### Two paths

1. **Synchronous path** (enhanced `AsyncSession.commit()`): before commit it auto-registers every cached-model mutation in the session (including bare `session.add()` / attribute mutation / `session.delete()`) and snapshots the pendings; after commit it synchronously `await`s invalidation of the snapshot plus cascade children
2. **Async compensation path** (fallback): the `after_commit` event hook triggers a compensation function that invalidates whatever the sync path didn't cover — this covers commits that bypassed the enhanced session (e.g. `begin()` / savepoint paths, plain `sqlmodel` sessions); fire-and-forget semantics with a brief stale window

### Sentinel objects

```python
_QUERY_ONLY_INVALIDATION  # add() scenario: only invalidate query cache
_FULL_MODEL_INVALIDATION  # delete(condition) scenario: full model invalidation
_LOAD_CACHE_MISS          # Multi-ID cache joint query miss
```

## MissingGreenlet avoidance

::: danger Risk
After commit, SQLAlchemy resets object association states. Directly accessing attributes triggers synchronous queries.
:::

Solutions:

- Extract IDs with `getattr()` before commit
- After commit, read from identity map via `sa_inspect()` (no DB query)
- External SQL methods use `_register_pending_invalidation()` + `await session.commit()` (the enhanced `AsyncSession` sync-invalidates registered pendings after commit)

## `check_cache_config()` static check

```python
@classmethod
def check_cache_config(cls) -> None:
```

Validates:
1. Redis client has been set via `configure_redis()`
2. No subclass overrides `_get_client` (would break the shared client contract)
3. All subclasses' `__cache_ttl__` are positive integers
4. AST check: forbids direct calls to `invalidate_by_id()` etc. in non-cache methods (prevents MissingGreenlet)

Side effect: registers SQLAlchemy `after_commit` / `after_rollback` / `persistent_to_deleted` event hooks.

## Graceful degradation

All Redis operations are wrapped in try/except:
- Read failure → return None (degrade to database)
- Write failure → log + continue
- Delete failure → log (TTL provides eventual consistency)
