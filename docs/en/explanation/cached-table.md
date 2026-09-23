# Redis cache mechanism

::: tip Source location
`src/sqlmodel_ext/mixins/cached_table.py` — `CachedTableBaseMixin`; orchestrated by the enhanced `AsyncSession` in `src/sqlmodel_ext/session.py`
:::

`CachedTableBaseMixin` provides a transparent Redis cache layer for `get()` queries, with automatic invalidation on the write paths. This chapter explains **its internal structure**; correctness inside transactions (why it never publishes uncommitted data) is covered separately in [Cache transparency inside transactions](./transactional-cache-transparency); to plug it into your own project, see [Cache queries with Redis](/en/how-to/cache-queries).

## Dual-layer cache architecture

```
1. ID cache (id:{ModelName}:{id_value})
   - For cls.id == value single-row exact queries
   - Row-level invalidation: one multi-key DEL

2. Query cache (query:{ModelName}:v{version}:{md5_hash})
   - For conditional and list queries
   - Model-level invalidation: version bump O(1) (old keys expire via TTL)

3. Version key (ver:{ModelName})
   - Namespace version of the query cache; INCR makes every query key of the old version unreachable
```

Invalidating the query cache takes a single `INCR ver:{ModelName}` instead of `SCAN+DEL` over all keys — the cost drops from O(N keys) to O(1).

### Cache key generation

ID cache keys are directly concatenated: `id:Character:0190...`.

Query cache keys normalize every parameter that can change the result (conditions, pagination, sorting, `load`, `filter`, time filters, the keyset cursor `after_id`) and compute an MD5 hash (first 16 characters). `table_view` is first merged into the explicit parameters using the same rules as `get()`, so semantically identical queries get the same key. Full format: `query:Character:v3:abcdef0123456789`.

## Core class structure

```python
class CachedTableBaseMixin(TableBaseMixin):
    __cache_ttl__: ClassVar[int] = 3600          # override with the cache_ttl= class keyword

    _redis_client: ClassVar[Any] = None          # set by configure_redis()
    on_cache_hit: ClassVar[Callable[[str], None] | None] = None
    on_cache_miss: ClassVar[Callable[[str], None] | None] = None

    @classmethod
    def configure_redis(cls, client: Any) -> None: ...
    @classmethod
    def check_cache_config(cls) -> None: ...
```

`on_cache_hit` / `on_cache_miss` are optional metric hooks that can feed hit ratios into Prometheus / Grafana.

## `get()` override

```python
@classmethod
async def get(cls, session, condition=None, *, no_cache=False, authoritative=False, ...):
    # 1. Explicit or structural skip
    skip_cache = (no_cache or authoritative or options is not None
                  or with_for_update or populate_existing or join is not None)

    # 2. Transaction transparency: a table the query depends on has uncommitted writes in this transaction → skip both read and write
    if not skip_cache and cls._has_uncommitted_writes(
            session, cls._query_dependency_tables(condition, filter, order_by, load)):
        skip_cache = True

    # 3. load + all cacheable MANYTOONE → multi-ID cache joint query (zero SQL if all hit)
    # 4. Pure ID equality query → ID cache key; otherwise → versioned query cache key
    # 5. Hit → deserialize → _adopt_cached_instance merges into the identity map (falls back to the DB if it can't be merged safely)
    # 6. Miss → super().get() queries the database → write to cache
```

### ID query detection

`_extract_id_from_condition()` recognizes conditions of the form `cls.id == value`, and requires no pagination / sorting / `filter` / `table_view` / time filters — in that case the precise ID cache key is used instead of the query hash.

### Multi-ID cache joint query

When every relation in `load` is **many-to-one** and the target model is also a cached model, the primary model and each relation target are read from their own ID caches and assembled; zero SQL if all hit. Chained `load` (`A.b → B.c`) does not take this path. With `fetch_mode='one'`, if the cache records "does not exist", `NoResultFound` is still raised, consistent with the SQL path.

## Serialization scheme

```python
{
    "_t": "none|single|list",   # Result type
    "_data": {...},             # Single item data
    "_items": [{...}, ...],     # List data
    "_c": "ClassName"           # Polymorphic safety: records actual class name
}
```

Serialization: `model_dump()` (columns only) → orjson (falls back to the standard library json when not installed). Deserialization: `json.loads` → `model_validate()` (not `model_validate_json`, which leaves UUID fields of `table=True` models as `str`); the loaded instances carry a valid `_sa_instance_state`. If deserialization fails (e.g. the schema changed), the bad key is deleted and the query falls back to the database.

## Cache invalidation

### CRUD registers, commit invalidates at a single point

The CRUD overrides **do not invalidate on their own** — they only register pending invalidations into `session.info`; the actual invalidation is executed synchronously after commit, in one place, by the enhanced `AsyncSession.commit()`:

```python
async def save(self, session, ...):
    # 1. Register pending invalidation (new objects register the "query cache only" sentinel)
    self._register_pending_invalidation(session, model_type, instance_id)

    # 2. super().save(refresh=False): awaits session.commit() internally,
    #    the enhanced commit synchronously invalidates every registered item after committing
    result = await super().save(session, refresh=False, commit=commit, ...)

    # 3. Read the database bypassing the cache (no_cache=True); when commit=True and there is no load,
    #    refill the latest data into the ID cache of self + every cached ancestor
    ...
```

With `commit=False` nothing is invalidated (the data is not committed yet); the registered items wait for your final `session.commit()`; there is no refill either (those rows may still be rolled back).

### Invalidation granularity

| Operation | Strategy |
|-----------|----------|
| `save` / `update` | `DEL id:{cls}:{id}` (including cached ancestors) + `INCR ver:{cls}` |
| `delete(instances)` | per-instance `DEL id:...` + `INCR ver:{cls}` |
| `delete(condition)` | model-level `SCAN+DEL id:{cls}:*` + `INCR ver:{cls}` |
| `add()` | `INCR ver:{cls}`; instances with an explicitly specified id also have their ID cache invalidated (guards against id reuse) |
| bare `session.add()` / attribute mutation / `session.delete()` + `commit()` | auto-registered before commit: new → query cache only; modified / deleted → row level + query cache |

### Cascade deletes

- `passive_deletes=False`: SQLAlchemy deletes the child objects one by one during flush; the `persistent_to_deleted` event registers each cached child model, which is invalidated synchronously after commit.
- `passive_deletes=True`: the database does the cascading and the ORM never sees the child rows. **Before** issuing the DELETE, `delete()` runs a BFS along the whole `passive_deletes` chain, pre-querying and registering the ids of every cached child model (one level is not enough: when A→B→C all use database CASCADE, neither B nor C is ever loaded by the ORM).

### Polymorphic inheritance cascading

When an STI subclass changes, the keys of all cached ancestor classes are invalidated as well (`_cached_ancestors()` collects them along the MRO):

```python
async def _invalidate_id_cache(cls, instance_id):
    keys = [f"id:{cls.__name__}:{instance_id}"]
    keys += [f"id:{a.__name__}:{instance_id}" for a in cls._cached_ancestors()]
    await client.delete(*keys)          # one command, one round trip
```

The query cache `INCR`s the version keys of itself and all ancestors in one pipeline round trip. **Refill mirrors invalidation**: write-through refill writes exactly as many keys as invalidation cleared — otherwise callers querying through an ancestor class could get a stale instance during the window.

## The two invalidation paths

1. **Synchronous path** (enhanced `AsyncSession.commit()`): snapshots the registered items before commit; after commit it `await`s invalidation of the snapshot items and cascade children.
2. **Compensation path** (`after_commit` event): the event handler is synchronous and cannot `await`, so it schedules a fire-and-forget task that invalidates whatever the synchronous path did not cover (deduplicated by ID against the `synced` record). It catches commits that did not go through the enhanced `commit()`; there is a very short stale window, and TTL provides eventual consistency.

Sentinel objects:

```python
_QUERY_ONLY_INVALIDATION  # add() / new objects: only invalidate query cache
_FULL_MODEL_INVALIDATION  # delete(condition): full model invalidation
_LOAD_CACHE_MISS          # Multi-ID cache joint query miss
```

## Three entry points for manual invalidation

| Method | Timing | Use |
|--------|--------|-----|
| `invalidate_on_commit(session, *ids)` | **after** the next commit | triggers / raw SQL coupled with an ORM transaction; the only manual invalidation allowed inside model methods |
| `invalidate_by_id(*ids)` | immediate | external callers (admin scripts, tests); Redis errors swallowed |
| `invalidate_all(*, strict=False)` | immediate | the whole model; `strict=True` re-raises Redis errors (migration invalidation relies on it to judge success) |

## MissingGreenlet avoidance

::: danger Risk
After commit, objects are expired; directly accessing attributes triggers a synchronous lazy load.
:::

- Extract IDs with `getattr()` before commit; after commit read them from the identity via `sa_inspect()` (no DB query)
- `check_cache_config()` uses AST to inspect subclass method bodies and forbids direct calls to `invalidate_by_id` / `invalidate_all` / internal invalidation methods (they may access expired attributes after commit); inside model methods use `invalidate_on_commit()` + `await session.commit()`

## `check_cache_config()` static check

Call once at startup (after `configure_redis()`):

1. Redis client is configured
2. No subclass overrides `_get_client`
3. All subclasses' `__cache_ttl__` are positive integers
4. AST check: subclass methods (including classmethod / staticmethod / property) do not directly call invalidation methods

Side effects: registers the SQLAlchemy `after_commit` / `after_rollback` / `after_flush` / `persistent_to_deleted` event hooks (idempotent; `configure_redis()` registers them too), and pre-builds the "table name → cached class" index so the hot path of the raw-DML warning does no lazy building.

## Graceful degradation

| Failure | Behavior |
|---------|----------|
| Redis not configured | `RuntimeError` (configuration error, fail fast) |
| Read failure | log, fall back to the database |
| Write failure | log, continue |
| Invalidation failure | synchronous path logs; `invalidate_by_id` / `invalidate_all()` swallow by default, `invalidate_all(strict=True)` re-raises; TTL provides eventual consistency |
