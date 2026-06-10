# Cache queries with Redis

**Goal**: add a Redis cache layer to a frequently-read model. CRUD operations invalidate it automatically — you never manually clear the cache.

**Prerequisites**:

- You have a Redis instance (`redis://localhost:6379` works for development)
- Your model inherits `UUIDTableBaseMixin` or `TableBaseMixin`
- Your session factory uses the enhanced session type: `async_sessionmaker(engine, class_=sqlmodel_ext.AsyncSession)` — since 0.4.0 cache invalidation is orchestrated by its `commit()`; plain sessions degrade to fire-and-forget compensation (a brief stale window)

## 1. Add `CachedTableBaseMixin` to the model

```python
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    CachedTableBaseMixin,
    NonEmptyStrippedStr64, Text10K,
)

class CharacterBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    system_prompt: Text10K

class Character(
    CachedTableBaseMixin,                     # ← must be first // [!code highlight]
    CharacterBase,
    UUIDTableBaseMixin,
    table=True,
    cache_ttl=1800,                            # 30 minutes // [!code highlight]
):
    pass
```

::: warning MRO order
`CachedTableBaseMixin` **must** appear before `UUIDTableBaseMixin` / `TableBaseMixin`. That's how its `get()` / `save()` / `update()` / `delete()` overrides take effect.
:::

`cache_ttl` is a class kwarg the metaclass converts into `__cache_ttl__: ClassVar[int]`. Default is 3600 seconds (1 hour).

## 2. Configure the Redis client at startup

```python
import redis.asyncio as redis
from sqlmodel_ext import CachedTableBaseMixin

# In application lifespan startup:
redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)
CachedTableBaseMixin.configure_redis(redis_client)
CachedTableBaseMixin.check_cache_config()  # validate every subclass
```

::: danger decode_responses must be False
Cached values are bytes (from `model_dump_json().encode()`); `decode_responses=True` breaks deserialization.
:::

`check_cache_config()` validates `__cache_ttl__` on every subclass and registers SQLAlchemy session event hooks (used by the `commit=False` invalidation compensation path).

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
# Auto: DEL id:Character:{id} + INCR ver:Character
```

| Operation | Invalidation strategy |
|-----------|----------------------|
| `save()` / `update()` | `DEL id:Character:{id}` + query cache version `+1` |
| `delete(instance)` | same |
| `delete(condition=...)` | model-level ID cleanup + version `+1` |
| `add()` | only version `+1` (new objects have no stale cache) |

## 4. Manual invalidation (special cases)

If you bypass the ORM and modify data via raw SQL, you need to notify the cache layer:

```python
await Character.invalidate_by_id(char_id)         # invalidate one ID
await Character.invalidate_by_id(id1, id2, id3)   # invalidate multiple
await Character.invalidate_all()                  # invalidate every cache for this model
```

## 5. Bypass the cache

```python
# Explicit bypass
char = await Character.get_one(session, char_id, no_cache=True)
```

**Auto-bypass scenarios**:

- `with_for_update=True` (row lock requires fresh data)
- `populate_existing=True`
- non-empty `options` / `join` (cannot be hashed stably)
- pending invalidation in the current transaction

## 6. Hook into your metrics system (optional)

```python
def on_hit(model_name: str) -> None:
    METRIC_CACHE_HIT.labels(model=model_name).inc()

def on_miss(model_name: str) -> None:
    METRIC_CACHE_MISS.labels(model=model_name).inc()

CachedTableBaseMixin.on_cache_hit = on_hit
CachedTableBaseMixin.on_cache_miss = on_miss
```

## 7. About ID cache vs query cache

sqlmodel-ext uses a **dual-layer cache**:

- **ID cache** (`id:Character:{uuid}`) — for `cls.id == value` exact single-row queries; row-level invalidation O(1)
- **Query cache** (`query:Character:v3:abcdef0123456789`) — for conditional / list queries. Model-level invalidation uses version bumping (`INCR ver:Character`); old-version keys disappear via TTL, avoiding `SCAN+DEL` overhead

All of this is transparent to business code. You just call `Character.get_one(...)`.

## Graceful degradation

Redis down? It doesn't break the application:

| Failure | Behavior |
|---------|----------|
| Read failure | Log + fall back to database query |
| Write failure | Log + continue |
| Delete failure | Log (TTL provides eventual consistency) |

The only hard requirement: `configure_redis()` must be called before the first `get()`, otherwise `RuntimeError`.

## Related reference

- [`CachedTableBaseMixin` full API](/en/reference/mixins#cachedtablebasemixin)
- [Redis cache mechanism explanation](/en/explanation/cached-table) (the "why")
