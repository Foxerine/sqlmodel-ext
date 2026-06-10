# Mixins

::: tip
This is reference documentation. To learn how to compose these mixins onto your own models, see the [how-to guides](/en/how-to/).
:::

All mixins are composed onto table models via MRO. **MRO order usually matters** — see the "MRO" note under each mixin.

## `CachedTableBaseMixin`

```python
from sqlmodel_ext import CachedTableBaseMixin
```

Inherits from `TableBaseMixin`. Adds a Redis cache layer to the model's `get()` queries.

**MRO**: `CachedTableBaseMixin` must appear **before** `UUIDTableBaseMixin` / `TableBaseMixin`:

```python
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=1800):
    pass
```

**Class variables**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `__cache_ttl__` | `int` | `3600` | Cache TTL (seconds). Set via the `cache_ttl=N` class kwarg |
| `_redis_client` | `Any` | `None` | Redis client; must be set via `configure_redis()` |
| `on_cache_hit` | `Callable[[str], None] \| None` | `None` | Cache-hit callback receiving the model name |
| `on_cache_miss` | `Callable[[str], None] \| None` | `None` | Cache-miss callback receiving the model name |

**Class methods**:

```python
@classmethod
def configure_redis(cls, client: Any) -> None
```

Call once at startup. `client` is a `redis.asyncio.Redis` instance (`decode_responses=False`).

```python
@classmethod
def check_cache_config(cls) -> None
```

Call once at startup (after `configure_redis()`). Validates configuration of all subclasses and registers SQLAlchemy session event hooks.

```python
@classmethod
async def invalidate_by_id(cls, *_ids: Any) -> None
```

Manually invalidate the cache for one or more IDs.

```python
@classmethod
async def invalidate_all(cls) -> None
```

Invalidate all cache entries (ID + query) for this model.

**Additional `get()` parameter**: in addition to the parameters in [CRUD methods](./crud-methods), `get()` also accepts `no_cache: bool = False`.

**Auto-bypass scenarios**: `with_for_update=True`, `populate_existing=True`, non-empty `options`, non-empty `join`, pending invalidation in the transaction, `no_cache=True`.

## `OptimisticLockMixin`

```python
from sqlmodel_ext import OptimisticLockMixin, OptimisticLockError
```

**MRO**: `OptimisticLockMixin` must appear **before** `UUIDTableBaseMixin` / `TableBaseMixin`.

**Fields**:

| Field | Type | Default | Database behavior |
|-------|------|---------|-------------------|
| `version` | `int` | `0` | Auto-incremented on every UPDATE (SQLAlchemy `version_id_col` mechanism) |

**Class marker**:

```python
_has_optimistic_lock: ClassVar[bool] = True
```

Lets `save()` / `update()` know it should apply optimistic-lock logic.

**Trigger condition**: when an UPDATE's `WHERE version = ?` doesn't match (zero rows affected) → `StaleDataError` → converted by `save()` / `update()` into `OptimisticLockError`.

## `OptimisticLockError`

```python
class OptimisticLockError(Exception):
    model_class: str | None
    record_id: str | None
    expected_version: int | None
    original_error: Exception | None
```

Raised by `save()` / `update()` after `optimistic_retry_count` retries are exhausted.

## `PolymorphicBaseMixin`

```python
from sqlmodel_ext import PolymorphicBaseMixin
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `_polymorphic_name` | `Mapped[str]` | Discriminator column (`String`, indexed). Subclasses write it automatically; not part of API serialization |

**Keyword arguments accepted by `__init_subclass__`**:

| Argument | Default | Meaning |
|----------|---------|---------|
| `polymorphic_on` | `'_polymorphic_name'` | Discriminator column name |
| `polymorphic_abstract` | auto-detected | Whether this is an abstract base (auto `True` when class inherits `ABC` and has abstract methods) |

**Class methods**:

```python
@classmethod
def _is_joined_table_inheritance(cls) -> bool

@classmethod
def get_concrete_subclasses(cls) -> list[type]

@classmethod
def get_identity_to_class_map(cls) -> dict[str, type]
```

`get_identity_to_class_map()` returns something like `{'emailnotification': EmailNotification, ...}`.

## `AutoPolymorphicIdentityMixin`

```python
from sqlmodel_ext import AutoPolymorphicIdentityMixin
```

**Keyword arguments accepted by `__init_subclass__`**:

| Argument | Default | Meaning |
|----------|---------|---------|
| `polymorphic_identity` | auto-generated | If explicitly provided, used as-is; otherwise `{parent_identity}.{class_name.lower()}` |

Auto-generated identities use a dot-separated hierarchy, e.g. `'function'` → `'function.codeinterpreter'`.

## `create_subclass_id_mixin()`

```python
from sqlmodel_ext import create_subclass_id_mixin
```

**Signature**:

```python
def create_subclass_id_mixin(parent_table_name: str) -> type
```

Dynamically generates a Mixin providing an `id` column with a foreign key + primary key pointing to `{parent_table_name}.id`. JTI subclasses only.

**MRO requirement**: the returned mixin **must be first in the inheritance list** so its `id` overrides `UUIDTableBaseMixin`'s `id`.

## `register_sti_columns_for_all_subclasses()`

```python
from sqlmodel_ext import (
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)
```

**Signatures**:

```python
def register_sti_columns_for_all_subclasses() -> None
def register_sti_column_properties_for_all_subclasses() -> None
```

STI subclass fields must be registered to the parent table in two phases:

1. `register_sti_columns_for_all_subclasses()` — call **before** `configure_mappers()`
2. `register_sti_column_properties_for_all_subclasses()` — call **after** `configure_mappers()`

## `DeferredIndex` (new in 0.4.0)

```python
from sqlmodel_ext import DeferredIndex
```

**Signature**:

```python
class DeferredIndex(CustomTableArg):
    def __init__(self, name: str, *column_names: str, **kwargs: Any) -> None
```

A **deferred-index marker** for STI base-class `table_args`. STI subclass
columns are registered dynamically during phase 1 and do not exist yet when
the base class's `__table_args__` is evaluated — a plain `Index('name', 'col')`
raises `ConstraintColumnNotFoundError` on the spot. `DeferredIndex` is
intercepted by the metaclass (never handed to SQLAlchemy) and materialized
into a real `Index` at the end of `register_sti_columns_for_all_subclasses()`.

```python
class CanvasNode(
    SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin,
    table=True,
    table_args=(
        DeferredIndex('ix_canvasnode_file_ids_gin', 'file_ids', postgresql_using='gin'),
    ),
):
    ...
```

- `name` — index name (unique per database)
- `*column_names` — column-name strings (columns STI subclasses register onto the parent table)
- `**kwargs` — passed through to SQLAlchemy `Index` (e.g. `unique=True`, `postgresql_using='gin'`)

## `RelationPreloadMixin`

```python
from sqlmodel_ext import RelationPreloadMixin
```

Classes inheriting this mixin can use `@requires_relations` and `@requires_for_update` on their methods.

**`__init_subclass__` behavior**: scans all methods, performs import-time validation on those with `_required_relations` metadata (relation-name typo check).

**Instance methods**:

```python
def _is_relation_loaded(self, rel_name: str) -> bool

async def _ensure_relations_loaded(
    self,
    session: AsyncSession,
    relations: tuple[str | QueryableAttribute, ...],
) -> None

@classmethod
def get_relations_for_method(cls, method_name: str) -> tuple

@classmethod
def get_relations_for_methods(cls, *method_names: str) -> tuple

async def preload_for(self, session: AsyncSession, *method_names: str) -> None
```

You usually don't need to call these directly — `@requires_relations` does it automatically.

## Default `lazy='raise_on_sql'`

Since 0.2.0, all SQLModel `Relationship` fields default to `lazy='raise_on_sql'`: accessing an unloaded relation **raises immediately** instead of triggering an implicit synchronous query. This is the last line of defense against MissingGreenlet.
