# Cache queries with Redis

**Goal**: add a Redis cache layer to a frequently-read model. CRUD operations invalidate it automatically, with no manual cache clearing; reads inside a transaction always see the correct value.

**Prerequisites**:

- You have a Redis instance (`redis://localhost:6379` works for development)
- Your model inherits `UUIDTableBaseMixin` or `TableBaseMixin`
- Your session factory uses the enhanced session type: `SessionFactory(engine, class_=sqlmodel_ext.AsyncSession)` (or `async_sessionmaker(engine, class_=AsyncSession)`) — cache invalidation is orchestrated by its `commit()`

## 1. Add `CachedTableBaseMixin` to the model

```python
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    CachedTableBaseMixin,
    NonEmptyStrippedStr64, Text10K,
)
from sqlmodel_ext.mixins import CACHE_TTL_WARM

class CharacterBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    system_prompt: Text10K

class Character(
    CachedTableBaseMixin,                     # ← must be first // [!code highlight]
    CharacterBase,
    UUIDTableBaseMixin,
    table=True,
    cache_ttl=CACHE_TTL_WARM,                  # 1800 seconds // [!code highlight]
):
    pass
```

::: warning MRO order
`CachedTableBaseMixin` **must** appear before `UUIDTableBaseMixin` / `TableBaseMixin`; only then do its `get()` / `save()` / `update()` / `delete()` / `add()` overrides take effect.
:::

`cache_ttl` is a class keyword argument that the metaclass converts into `__cache_ttl__` (a non-positive integer raises `ValueError` at class creation). Default is 3600 seconds. Semantic constants: `CACHE_TTL_HOT` (600), `CACHE_TTL_WARM` (1800), `CACHE_TTL_COLD` (3600).

## 2. Configure the Redis client at startup

```python
import redis.asyncio as redis
from sqlmodel_ext import CachedTableBaseMixin

# In application lifespan startup:
redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)
CachedTableBaseMixin.configure_redis(redis_client)
CachedTableBaseMixin.check_cache_config()  # validate every subclass's configuration
```

::: danger decode_responses must be False
Cached values are bytes (orjson output); `decode_responses=True` breaks deserialization.
:::

`check_cache_config()` checks `__cache_ttl__` on every subclass, forbids subclass methods from calling the invalidation methods directly, and registers SQLAlchemy session event hooks.

## 3. Use it directly — no business code changes

```python
# First time: queries DB + writes cache
char = await Character.get_one(session, char_id)

# Second time: cache hit, zero SQL
char = await Character.get_one(session, char_id) # [!code highlight]
```

```python
# UPDATE auto-invalidates
char.name = "new name"
char = await char.save(session)
# Synchronously after commit: DEL id:Character:{id} + INCR ver:Character, then backfill the latest value
```

| Operation | Invalidation strategy |
|-----------|----------------------|
| `save()` / `update()` | `DEL id:Character:{id}` + query cache version `+1` |
| `delete(instance)` | same |
| `delete(condition=...)` | model-wide ID cleanup + version `+1` |
| `add()` | version `+1` (instances with an explicitly specified id also clear their ID cache) |
| Bare `session.add()` / attribute changes / `session.delete()` followed by `session.commit()` | Also invalidated automatically (the enhanced `commit()` registers them before committing) |

## 4. Correct inside transactions too

Inside a transaction that has not committed yet, if a table the query depends on has been written by this transaction (including flushed writes and raw DML writes), that `get()` **neither reads nor writes the cache**: you see your own uncommitted changes, and nobody else ever sees them. Other tables that weren't touched still use the cache as usual. For how this works, see [Cache transparency inside transactions](/en/explanation/transactional-cache-transparency).

## 5. Raw SQL / trigger writes: `invalidate_on_commit`

When you modify data bypassing the ORM, **register** the invalidation in the same transaction; it runs automatically after commit:

```python
from sqlalchemy import update
from sqlmodel import col

Character.invalidate_on_commit(session, char_id)          # register first
await session.execute(
    update(Character).where(col(Character.id) == char_id).values(name='Bob')
)
await session.commit()                                      # invalidated synchronously after commit
```

Running a raw `UPDATE` / `DELETE` on a cached table without registering makes the enhanced session log a warning.

Outside a transaction (admin scripts, tests) you can invalidate immediately:

```python
await Character.invalidate_by_id(char_id)         # invalidate specific IDs (Redis errors are swallowed)
await Character.invalidate_by_id(id1, id2, id3)
await Character.invalidate_all()                  # invalidate every cache for this model
await Character.invalidate_all(strict=True)       # re-raise on Redis errors so the caller knows it failed
```

::: warning Inside model methods, only use `invalidate_on_commit`
`check_cache_config()` rejects direct calls to `invalidate_by_id` / `invalidate_all` inside subclass methods (accessing expired attributes after commit raises MissingGreenlet).
:::

## 6. Bypass the cache

```python
# Explicit bypass for an ordinary query (no_cache only exists on the cached model's get())
char = await Character.get(session, Character.id == char_id, no_cache=True)

# Authoritative read: must see the latest committed row — bypasses Redis and the identity map
char = await Character.get_one(session, char_id, authoritative=True)
```

`get_one()` / `get_exist_one()` have no `no_cache` parameter; to bypass the cache there, use `authoritative=True` or `with_for_update=True`.

**Auto-bypass scenarios**:

- `with_for_update=True` (row lock requires fresh data)
- `populate_existing=True`
- non-empty `options` / `join`
- this transaction has uncommitted writes to a table the query depends on

`session.refresh(obj)` always reads directly from the database.

## 7. Hook into your metrics system (optional)

```python
def on_hit(model_name: str) -> None:
    METRIC_CACHE_HIT.labels(model=model_name).inc()

def on_miss(model_name: str) -> None:
    METRIC_CACHE_MISS.labels(model=model_name).inc()

CachedTableBaseMixin.on_cache_hit = on_hit
CachedTableBaseMixin.on_cache_miss = on_miss
```

## 8. When a migration changes what data means

If a migration rescales a column's values, the old cache still deserializes but is wrong — declare `CACHE_INVALIDATIONS` in the migration file and call `run_pending_migration_cache_invalidations()` at startup. See [More mixins: migration-driven cache invalidation](./extra-mixins#migration-driven-cache-invalidation).

## About ID cache vs query cache

- **ID cache** (`id:Character:{uuid}`) — for `cls.id == value` exact single-row queries; row-level invalidation
- **Query cache** (`query:Character:v3:abcdef0123456789`) — for conditional / list queries. Model-level invalidation uses version bumping (`INCR ver:Character`); old-version keys expire naturally via TTL

All of this is transparent to business code.

## Graceful degradation

| Failure | Behavior |
|---------|----------|
| Read failure | Log + fall back to database query |
| Write failure | Log + continue |
| Invalidation failure | Log (TTL provides eventual consistency) |

The only hard requirement: `configure_redis()` must be called before the first `get()`, otherwise `RuntimeError`.

## Related reference

- [`CachedTableBaseMixin` full API](/en/reference/mixins#cachedtablebasemixin)
- [Redis cache mechanism explanation](/en/explanation/cached-table)
- [Cache transparency inside transactions](/en/explanation/transactional-cache-transparency)
