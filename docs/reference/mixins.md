# Mixin 类

::: tip
本页是参考文档。要看怎么把这些 Mixin 组合到自己的模型上，去 [操作指南](/how-to/)。
:::

所有 Mixin 通过 MRO 组合到 table 模型上。**MRO 顺序通常重要**——见每个 Mixin 的"MRO"小节。

## `CachedTableBaseMixin`

```python
from sqlmodel_ext import CachedTableBaseMixin
```

继承自 `TableBaseMixin`。为模型的 `get()` 查询添加 Redis 缓存层。

**MRO**：`CachedTableBaseMixin` 必须放在 `UUIDTableBaseMixin` / `TableBaseMixin` **之前**：

```python
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=1800):
    pass
```

**类变量**：

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `__cache_ttl__` | `int` | `3600` | 缓存 TTL（秒）。可用 `cache_ttl=N` 关键字设置 |
| `_redis_client` | `Any` | `None` | Redis 客户端，必须用 `configure_redis()` 设置 |
| `on_cache_hit` | `Callable[[str], None] \| None` | `None` | 缓存命中回调，参数为模型名 |
| `on_cache_miss` | `Callable[[str], None] \| None` | `None` | 缓存未命中回调，参数为模型名 |

**类方法**：

```python
@classmethod
def configure_redis(cls, client: Any) -> None
```

启动时调用一次。`client` 是 `redis.asyncio.Redis` 实例（`decode_responses=False`）。

```python
@classmethod
def check_cache_config(cls) -> None
```

启动时调用一次（`configure_redis()` 之后）。检查所有子类配置正确性，注册 SQLAlchemy session 事件钩子。

```python
@classmethod
async def invalidate_by_id(cls, *_ids: Any) -> None
```

手动失效一个或多个 ID 的缓存。

```python
@classmethod
async def invalidate_all(cls) -> None
```

失效该模型的所有缓存（ID + 查询）。

**`get()` 新参数**：在 [CRUD 方法](./crud-methods) 的 `get()` 之外，多出 `no_cache: bool = False`。

**自动跳过缓存的场景**：`with_for_update=True`、`populate_existing=True`、`options` 非空、`join` 非空、事务内有待失效数据、`no_cache=True`。

## `OptimisticLockMixin`

```python
from sqlmodel_ext import OptimisticLockMixin, OptimisticLockError
```

**MRO**：`OptimisticLockMixin` 必须放在 `UUIDTableBaseMixin` / `TableBaseMixin` **之前**。

**字段**：

| 字段 | 类型 | 默认值 | 数据库行为 |
|------|------|--------|----------|
| `version` | `int` | `0` | 每次 UPDATE 自动 `+1`（SQLAlchemy `version_id_col` 机制） |

**类标记**：

```python
_has_optimistic_lock: ClassVar[bool] = True
```

让 `save()` / `update()` 知道需要处理乐观锁逻辑。

**触发条件**：UPDATE 时 `WHERE version = ?` 不匹配（影响 0 行）→ `StaleDataError` → 由 `save()` / `update()` 转换为 `OptimisticLockError`。

## `OptimisticLockError`

```python
class OptimisticLockError(Exception):
    model_class: str | None
    record_id: str | None
    expected_version: int | None
    original_error: Exception | None
```

`save()` / `update()` 的 `optimistic_retry_count` 重试耗尽后抛出。

## `PolymorphicBaseMixin`

```python
from sqlmodel_ext import PolymorphicBaseMixin
```

**字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `_polymorphic_name` | `Mapped[str]` | 鉴别列（`String`、有索引）。子类自动写入；不参与 API 序列化 |

**`__init_subclass__` 接受的关键字参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `polymorphic_on` | `'_polymorphic_name'` | 鉴别列字段名 |
| `polymorphic_abstract` | 自动检测 | 是否为抽象基类（含 `ABC` + 抽象方法时自动 `True`） |

**类方法**：

```python
@classmethod
def _is_joined_table_inheritance(cls) -> bool

@classmethod
def get_concrete_subclasses(cls) -> list[type]

@classmethod
def get_identity_to_class_map(cls) -> dict[str, type]
```

`get_identity_to_class_map()` 返回如 `{'emailnotification': EmailNotification, ...}`。

## `AutoPolymorphicIdentityMixin`

```python
from sqlmodel_ext import AutoPolymorphicIdentityMixin
```

**`__init_subclass__` 接受的关键字参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `polymorphic_identity` | 自动生成 | 显式指定时直接用，否则用 `{parent_identity}.{class_name.lower()}` |

自动生成的 identity 格式是点分层级，如 `'function'` → `'function.codeinterpreter'`。

## `create_subclass_id_mixin()`

```python
from sqlmodel_ext import create_subclass_id_mixin
```

**签名**：

```python
def create_subclass_id_mixin(parent_table_name: str) -> type
```

动态生成一个 Mixin，提供指向 `{parent_table_name}.id` 的外键 + 主键。仅 JTI 子类需要。

**MRO 要求**：返回的 Mixin **必须放在继承列表第一位**，让其 `id` 字段覆盖 `UUIDTableBaseMixin` 的 `id`。

## `register_sti_columns_for_all_subclasses()`

```python
from sqlmodel_ext import (
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)
```

**签名**：

```python
def register_sti_columns_for_all_subclasses() -> None
def register_sti_column_properties_for_all_subclasses() -> None
```

STI 子类字段需要分两阶段注册到父表：

1. `register_sti_columns_for_all_subclasses()` — 在 `configure_mappers()` **之前**调用
2. `register_sti_column_properties_for_all_subclasses()` — 在 `configure_mappers()` **之后**调用

## `DeferredIndex`（0.4.0 新增）

```python
from sqlmodel_ext import DeferredIndex
```

**签名**：

```python
class DeferredIndex(CustomTableArg):
    def __init__(self, name: str, *column_names: str, **kwargs: Any) -> None
```

STI 基类 `table_args` 中的**延迟索引标记**。STI 子类列由阶段 1 动态注册，基类 `__table_args__` 求值时尚不存在——直接写 `Index('name', 'col')` 会立即抛 `ConstraintColumnNotFoundError`。`DeferredIndex` 被元类拦截（不传给 SQLAlchemy），在 `register_sti_columns_for_all_subclasses()` 末尾物化为真正的 `Index`。

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

- `name` — 索引名（全库唯一）
- `*column_names` — 列名字符串（STI 子类注册到父表的列）
- `**kwargs` — 透传给 SQLAlchemy `Index`（如 `unique=True`、`postgresql_using='gin'`）

## `RelationPreloadMixin`

```python
from sqlmodel_ext import RelationPreloadMixin
```

继承此 Mixin 的类可以使用 `@requires_relations` 和 `@requires_for_update` 装饰方法。

**`__init_subclass__` 行为**：扫描所有方法，对带有 `_required_relations` 元数据的方法做导入时验证（关系名拼写检查）。

**实例方法**：

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

通常你不需要直接调用这些方法——`@requires_relations` 装饰器会自动调用。

## 默认 `lazy='raise_on_sql'`

0.2.0 起，所有 SQLModel `Relationship` 字段的默认 `lazy` 设置为 `'raise_on_sql'`：访问未预加载的关系**立刻抛异常**，而不是触发隐式同步查询。这是 MissingGreenlet 问题的最后一道安全网。
