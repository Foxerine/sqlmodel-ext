# Mixin 类

::: tip
本页是参考文档。要看怎么把这些 Mixin 组合到自己的模型上，去 [操作指南](/how-to/)。
:::

所有 Mixin 通过 MRO 组合到 table 模型上。**MRO 顺序通常重要**——见每个 Mixin 的"MRO"小节。

::: info 导入位置
0.5.0 新增的符号（`GroupSumRow`、keyset 游标异常、`ResourceReferencedError`、新装饰器、`ResourceQuotaMixin` 等）在下文按其定义所在的子模块标注导入路径（`sqlmodel_ext.mixins` / `sqlmodel_ext.session` / `sqlmodel_ext.pagination`），这些路径始终有效。
:::

## `TableBaseMixin` / `UUIDTableBaseMixin`

```python
from sqlmodel_ext import TableBaseMixin, UUIDTableBaseMixin
```

异步 CRUD 的基础 Mixin，方法签名见 [CRUD 方法](./crud-methods)。

| 字段 | `TableBaseMixin` | `UUIDTableBaseMixin` |
|------|------|------|
| `id` | `int \| None`，自增主键 | `uuid.UUID`，**UUIDv7** 主键（`default_factory=uuid7`） |
| `created_at` | `datetime`，时区感知，创建时写入 | 同左 |
| `updated_at` | `datetime`，时区感知，每次有变更的 `save()` / 非空 `update()` 时显式刷新 | 同左 |

**类变量**：

| 名称 | 默认值 | 说明 |
|------|--------|------|
| `__optimistic_retry_default__` | `0` | `save()` / `update()` 在 `optimistic_retry_count=None` 时使用的重试次数。`OptimisticLockMixin` 覆盖为 `3`。`delete()` 永不重试 |

### UUIDv7 主键

UUIDv7（RFC 9562）前 48 位是 Unix 毫秒时间戳，字节序即创建时间序：`ORDER BY id` 近似创建顺序，B-tree 插入集中在右边缘（比 UUIDv4 少得多的页分裂）。由 `sqlmodel_ext.mixins.uuid7` 生成（Python 3.14+ 直接用标准库 `uuid.uuid7`，更早版本用内置的 RFC 9562 实现，同一进程内严格递增）。

已知限制：

- id 暴露创建时间（毫秒精度）。id 是**标识符不是凭证**，不要依赖它不可猜。
- 不要把 id 顺序当作权威时间顺序：时间戳来自应用时钟，跨进程无全局单调性。
- 存量行保留原来的 UUID 版本；v4/v7 混合时字节序仍是一致的全序（可用于锁排序）。
- 需要**确定性派生**主键（如从幂等键算 `uuid5`）时显式赋值——默认工厂只在未提供 id 时生效。

## `CachedTableBaseMixin`

```python
from sqlmodel_ext import CachedTableBaseMixin
```

继承自 `TableBaseMixin`。为模型的 `get()` 查询添加 Redis 缓存层，写路径自动失效，并且**对事务透明**（见 [事务内的缓存透明性](/explanation/transactional-cache-transparency)）。

**MRO**：`CachedTableBaseMixin` 必须放在 `UUIDTableBaseMixin` / `TableBaseMixin` **之前**：

```python
from sqlmodel_ext.mixins import CACHE_TTL_WARM

class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=CACHE_TTL_WARM):
    pass
```

**Session 要求**：session 必须是增强版 `sqlmodel_ext.AsyncSession`（`SessionFactory(engine, class_=AsyncSession)` 或 `async_sessionmaker(engine, class_=AsyncSession)`）。失效由它的 `commit()` 统一编排。

**类变量**：

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `__cache_ttl__` | `int` | `3600` | 缓存 TTL（秒）。用类关键字 `cache_ttl=N` 设置（元类校验为正整数） |
| `on_cache_hit` | `Callable[[str], None] \| None` | `None` | 缓存命中回调，参数为模型名 |
| `on_cache_miss` | `Callable[[str], None] \| None` | `None` | 缓存未命中回调（只在真正查过 Redis 时触发） |

**TTL 语义常量**（`from sqlmodel_ext.mixins import ...`）：

| 常量 | 值 | 用途 |
|------|------|------|
| `CACHE_TTL_HOT` | `600` | 频繁变化的模型 |
| `CACHE_TTL_WARM` | `1800` | 偶尔变化的模型 |
| `CACHE_TTL_COLD` | `3600` | 很少变化的模型（即默认值） |

**类方法**：

| 方法 | 说明 |
|------|------|
| `configure_redis(client)` | 启动时调用一次。`client` 是 `redis.asyncio.Redis`（`decode_responses=False`）。同时安装 session 事件钩子（幂等） |
| `check_cache_config()` | 启动时调用一次（`configure_redis()` 之后）。校验：Redis 已配置；无子类重写 `_get_client`；`__cache_ttl__` 为正整数；子类方法里没有直接调用 `invalidate_by_id` / `invalidate_all` 等（AST 检查）。并预建"表名 → 缓存类"索引 |
| `invalidate_on_commit(session, *ids)` | **登记**行级失效，在该 session 下一次 commit 后执行。用于触发器 / 原生 SQL 与 ORM 事务耦合的场景；`ids` 只接受真实主键（`int` / `UUID`）；rollback / `reset()` 会丢弃登记，重试路径需重新调用 |
| `async invalidate_by_id(*ids)` | **立即**失效给定 ID（面向外部调用方：管理脚本、测试）。Redis 错误记日志后吞掉 |
| `async invalidate_all(*, strict=False)` | 立即失效该模型全部缓存（`id:` + `query:`）。`strict=False` 吞掉 Redis 错误；`strict=True` 记日志并重新抛出，让调用方分辨成败 |
| `register_raw_dml_write(session, statement)` | 原生 DML 的写侧观测点：登记被写的表为"已写未提交"，并在命中缓存表却无登记失效的 `UPDATE` / `DELETE` 时告警。增强 session 的 `execute` / `exec` / `scalar` / `stream` / `stream_scalars` 已自动调用 |

::: warning 模型方法内部不要直接调用 `invalidate_by_id` / `invalidate_all`
`check_cache_config()` 会拒绝这种写法（commit 后访问过期属性会触发 MissingGreenlet）。模型方法内部用 `invalidate_on_commit(session, ...)` 登记，再 `await session.commit()`。
:::

**`get()` 多出的参数**：`no_cache: bool = False`。只有缓存模型的 `get()` 有这个参数；传给非缓存模型会得到 `TypeError`。`get_one()` / `get_exist_one()` 没有 `no_cache`，需要绕过缓存时用 `authoritative=True` 或 `with_for_update=True`。

**跳过缓存（读与写都跳过）的条件**：`no_cache=True`、`authoritative=True`、`with_for_update=True`、`populate_existing=True`、`options` 非空、`join` 非空，以及**本事务对该查询依赖的任一张表有未提交的写**。

## `OptimisticLockMixin`

```python
from sqlmodel_ext import OptimisticLockMixin, OptimisticLockError
```

**MRO**：`OptimisticLockMixin` 必须放在 `UUIDTableBaseMixin` / `TableBaseMixin` **之前**（它覆盖的 `__optimistic_retry_default__` 才能生效）。

**字段**：

| 字段 | 类型 | 数据库 | 说明 |
|------|------|--------|------|
| `oplock_version` | `int`（`0 ≤ v ≤ JS_MAX_SAFE_INTEGER`） | `BIGINT`，`server_default 0` | SQLAlchemy `version_id_col`，每次 UPDATE 自动递增。**`exclude=True`**：不会出现在 `model_dump()` 中 |

列名是 `oplock_version` 而不是 `version`，避免和领域字段 `version` 冲突。`server_default` 让滚动部署期间不认识该列的旧实例的 INSERT 仍然合法。

**类变量**：

| 名称 | 值 | 说明 |
|------|------|------|
| `__optimistic_retry_default__` | `3` | 覆盖基类的 `0`：冲突默认在内部重试 3 次，耗尽后才抛 `OptimisticLockError` |
| `_has_optimistic_lock` | `True` | 元类据此在根表 mapper 上接线 `version_id_col`（STI/JTI 子类经 mapper 继承共享）。可以在一个**中间基类**上覆盖为 `False`：保留版本列但不启用锁行为（两阶段上线的第一步）。元类检查的是**基类**上的标记，所以写在 `table=True` 类自己的类体里不起作用 |

`oplock_version` 这个名字是全局保留的：任何模型在自己的类体里声明它都会在类创建时抛 `TypeError`（领域版本字段请用 `version` / `revision` 等）。

```python
from typing import ClassVar

class ColumnOnlyOptimisticLock(OptimisticLockMixin):
    _has_optimistic_lock: ClassVar[bool] = False   # 第一阶段：只上线列

class Order(ColumnOnlyOptimisticLock, OrderBase, UUIDTableBaseMixin, table=True):
    pass   # 有 oplock_version 列，但不接线 version_id_col
```

**触发条件**：UPDATE / DELETE 的 `WHERE oplock_version = ?` 不匹配（影响 0 行）→ `StaleDataError` → 由 `save()` / `update()` / `delete()` 转换为 `OptimisticLockError`。

## `OptimisticLockError`

```python
class OptimisticLockError(Exception):
    model_class: str | None
    record_id: str | None
    expected_version: int | None
    original_error: StaleDataError | None
```

| 来源 | `model_class` | `record_id` | `expected_version` |
|------|------|------|------|
| `save()` / `update()` 重试耗尽 | 模型名 | 记录 id | 冲突时实例上的版本号 |
| `save()` / `update()` 重试时记录已被删除 | 模型名 | 记录 id | `None` |
| `delete()` | **调用方**模型名 | 恒为 `None` | 恒为 `None` |

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
def is_joined_table_inheritance(cls) -> bool

@classmethod
def get_concrete_subclasses(cls) -> list[type[PolymorphicBaseMixin]]

@classmethod
def get_polymorphic_discriminator(cls) -> str

@classmethod
def get_identity_to_class_map(cls) -> dict[str, type[PolymorphicBaseMixin]]
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

```python
def create_subclass_id_mixin(parent_table_name: str) -> type[SQLModelBase]
```

动态生成一个 Mixin，提供指向 `{parent_table_name}.id` 的外键 + 主键（`default_factory=uuid7`，与 `UUIDTableBaseMixin.id` 保持一致）。仅 JTI 子类需要。

**MRO 要求**：返回的 Mixin **必须放在继承列表第一位**，让其 `id` 字段覆盖 `UUIDTableBaseMixin` 的 `id`。

## `register_sti_columns_for_all_subclasses()`

```python
from sqlmodel_ext import (
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)
```

```python
def register_sti_columns_for_all_subclasses() -> None
def register_sti_column_properties_for_all_subclasses() -> None
```

STI 子类字段需要分两阶段注册到父表：

1. `register_sti_columns_for_all_subclasses()` — 在 `configure_mappers()` **之前**调用
2. `register_sti_column_properties_for_all_subclasses()` — 在 `configure_mappers()` **之后**调用

::: warning STI 共享列在数据库里必须可空
注册到父表的子类列会被剥掉 `default` / `server_default`，于是兄弟子类的每次 INSERT 都会往该列写**显式 NULL**（显式 NULL 会绕过数据库 DEFAULT）。迁移作者必须保证：**这类列永远不加 NOT NULL，也不依赖数据库 DEFAULT 为兄弟子类填值**。只想对拥有该列的子类要求非空，用容忍 NULL 的 CHECK：`discriminator <> 'owner_identity' OR col IS NOT NULL`。`create_all` 建的测试库里该列本就可空，违规只会在迁移过的库上失败。
:::

## `DeferredIndex`

```python
from sqlmodel_ext import DeferredIndex
```

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

继承此 Mixin 的类可以使用 `@requires_relations` 装饰方法（事务契约装饰器见 [装饰器与辅助函数](./decorators)）。

**`__init_subclass__` 行为**：对 SQLModel 类扫描所有带 `_required_relations` 元数据的方法，做导入时验证（字符串关系名拼写检查）。纯 Python mixin 上的声明在被混入具体 SQLModel 类时验证。

**方法**：

| 方法 | 说明 |
|------|------|
| `async ensure_relations_loaded(session, relations)` | **公开入口**：确保给定关系已加载，只增量加载缺失的部分（一次查询）。`@requires_relations` 自动调用它；如果被调方法的签名里没有 session（装饰器找不到 session 会跳过预加载），编排代码必须自己先调用它 |
| `async ensure_relations_loaded_bulk(session, instances, specs_by_class)`（classmethod） | 为一批（可异构的）实例批量预加载关系。查询数 = 0–1 次属主刷新 + 每个目标继承树根 1 次。只批量处理"单列外键指向目标 `id` 的多对一、`primaryjoin` 是纯外键等式"的关系；外键非空但目标行缺失时**刻意不组装**（保持未加载，`raise_on_sql` 访问会响亮失败）。这是查询数优化，不是正确性来源——之后仍应对每个实例调用 `ensure_relations_loaded`（已加载时是空操作） |
| `bulk_preload_unsupported_reason(spec)`（classmethod） | 批量能力边界的唯一真相源：支持时返回 `None`，否则返回人类可读原因。启动检查可用它对无法批量的声明响亮失败 |
| `get_relations_for_method(method_name)`（classmethod） | 某方法声明的关系，`list[QueryableAttribute]` |
| `get_relations_for_methods(*method_names)`（classmethod） | 多个方法声明的关系（按 key 去重） |
| `async preload_for(session, *method_names)` | 手动为指定方法预加载，返回 `self` 以便链式调用 |

## `ResourceQuotaMixin`

```python
from sqlmodel_ext.mixins import (
    ResourceQuotaMixin,
    QuotaExceededError,
    QuotaOwnerNotFoundError,
    CallerDidNotCommitError,
)
```

"每个属主的资源数量有上限"的原子配额槽位获取。不依赖任何应用概念，只依赖 `AsyncSession`；宿主类还需继承提供 `count()` 的表基类（除非覆盖 `_count_owner_resources`）。

**子类契约**（三个 classmethod 必须实现）：

| 方法 | 说明 |
|------|------|
| `async _lock_owner(session, owner_id, *, with_for_update=True)` | 加载属主行并返回（`with_for_update` 时 `SELECT ... FOR UPDATE`），不存在返回 `None`；必须把 `with_for_update` 转发给底层 `get` |
| `_quota_condition(owner_id) -> ColumnElement[bool]` | 选出该属主在本模型中的行 |
| `_quota_max(owner) -> int` | 该属主允许的最大数量 |
| `async _count_owner_resources(session, owner_id, owner) -> int` | 可选覆盖；默认 `cls.count(session, _quota_condition(owner_id))` |

**入口**：

| 方法 | 说明 |
|------|------|
| `async preflight_quota(session, owner_id, count=1)` | **无锁**预检（便宜、尽力而为、存在竞态的软门槛），放在昂贵步骤（如付费外部 API）之前 |
| `acquire_quota_lock(session, owner_id, count=1, defer_commit=False, idempotent_check=None)` | `async with` 上下文：在调用方事务中锁住属主行，校验 `current + count <= max`，与受保护的 INSERT 原子执行。块本身不 commit——调用方必须 commit（锁随 commit/rollback 释放） |

| 参数 | 说明 |
|------|------|
| `count` | 预留的槽位数（`>= 1`） |
| `defer_commit` | `True` 时关闭"块正常退出却没 commit"的守卫——用于 `save(commit=False)` 后由更大的外层事务提交的场景 |
| `idempotent_check` | 可选回调，在拿到锁之后、计数之前执行；返回非 `None` 时跳过配额检查与 INSERT，并把该值 `yield` 给调用方（并发 get-or-create 的竞态守卫）。回调必须读数据库真相（绕过缓存） |

**异常**：

| 异常 | 基类 | `status_code` | 条件 |
|------|------|------|------|
| `QuotaExceededError` | `ValueError` | `400` | `current + count > max_allowed`；带 `model_name` / `max_allowed` / `current_count` |
| `QuotaOwnerNotFoundError` | `LookupError` | `404` | `_lock_owner` 返回 `None`；带 `owner_id` |
| `CallerDidNotCommitError` | `RuntimeError` | — | `defer_commit=False` 且块正常退出时没有发生 commit（编程错误） |
| `ValueError` | — | — | `count < 1` |

用法见 [更多 Mixin](/how-to/extra-mixins#resourcequotamixin)。

## `TrgmSearchableMixin`

```python
from sqlmodel_ext.mixins import TrgmSearchableMixin, TrgmSearchRequest, BIGRAM_FUNCTION_SQL
```

基于 PostgreSQL `pg_trgm` 的名称 / 文本列模糊搜索（**仅 PostgreSQL**）。只产出 `WHERE` 条件，从不改变排序。

| 类变量 | 默认值 | 说明 |
|------|------|------|
| `__trgm_name_column__` | `'name'` | 主名称列：`ILIKE` 子串匹配 **OR** `%` 相似度运算符（可走 GIN 索引） |
| `__trgm_text_columns__` | `()` | 其它文本列：只做 `ILIKE` 子串匹配 |

| 符号 | 说明 |
|------|------|
| `trgm_search_condition(query)`（classmethod） | 构建模糊搜索条件；`query` 需已去空白且非空 |
| `TrgmSearchRequest` | 查询参数 DTO：`query: Str64 \| None = None`；`normalized_query` 属性（去空白后为空则 `None`）；`apply_condition(model, condition=None)` 把模糊条件 AND 进已有条件，无有效搜索词时原样返回 `condition` |
| `BIGRAM_FUNCTION_SQL` | `public.bigrams(text) -> text[]` 的 DDL，需在迁移中执行一次。一旦有索引依赖它，**不要再改函数体** |

宿主契约：数据库已 `CREATE EXTENSION pg_trgm` 并创建 `public.bigrams`；为搜索列建 GIN 索引（否则退化为顺序扫描）；搜索列使用数据库默认排序规则。

## `MixinTableScanMixin`

```python
from sqlmodel_ext.mixins import MixinTableScanMixin
```

给"挂载在多张无关物理表上的同一个字段级 Mixin"提供跨表发现 + 参数化扫描。没有数据字段，不参与 CRUD。

| 方法 | 说明 |
|------|------|
| `_concrete_mixin_subclasses()`（classmethod） | 递归发现所有挂载该 Mixin 的**具体表类**，按 `__table__` 去重（同一 STI 家族只保留多态根） |
| `async _scan_rows_where(session, condition_factory)`（classmethod，异步生成器） | 逐表执行 `condition_factory(table_cls)` 构造的条件，`yield (table_class, row)`。条件用类型安全表达式构造，**不接受字符串字段名** |

## 迁移驱动的缓存失效

```python
from sqlmodel_ext.mixins import (
    run_pending_migration_cache_invalidations,
    collect_migration_invalidation_tasks,
)
```

```python
async def run_pending_migration_cache_invalidations(
    base_class: type,
    *,
    tasks: dict[str, list[str]] | None = None,
    alembic_ini: str = 'alembic.ini',
) -> None

def collect_migration_invalidation_tasks(alembic_ini: str = 'alembic.ini') -> dict[str, list[str]]
```

启动时（迁移完成、`configure_redis()` 之后）调用一次。每个任务 `sentinel -> [模型类名, ...]`：Redis 中不存在哨兵 `cache_invalidation:<sentinel>` 时对每个模型执行 `invalidate_all(strict=True)`，全部成功才写哨兵（失败的任务下次启动重试）。`tasks=None` 时从 Alembic 迁移文件的模块级 `CACHE_INVALIDATIONS` 常量收集（需要可选依赖 `alembic`，未安装时 `collect_migration_invalidation_tasks` 抛 `RuntimeError`）。用法见 [更多 Mixin](/how-to/extra-mixins#迁移驱动的缓存失效)。

## 异常一览（`sqlmodel_ext.mixins`）

| 异常 | 基类 | `status_code` | 抛出方 |
|------|------|------|------|
| `ResourceReferencedError` | `Exception` | `409` | `delete()`：行仍被外键引用。属性 `friendly_message` / `constraint_name` / `original_error` |
| `KeysetCursorError` | `ValueError` | `422` | keyset 游标误用的公共基类；`str(e)` 可安全返回给客户端 |
| `KeysetCursorInvalidError` | `KeysetCursorError` | `422` | `after_id` 锚点不存在或不可见 |
| `KeysetCursorUnsupportedError` | `KeysetCursorError` | `422` | 查询自带 `order_by` 或用了 `join` |
| `OptimisticLockError` | `Exception` | — | 见上 |

`FK_DELETE_RESTRICT_FALLBACK_MESSAGE`：没有为该外键约束注册消息时 `ResourceReferencedError` 使用的通用文案（不含任何表/列/约束名）。

## 默认 `lazy='raise_on_sql'`

所有 SQLModel `Relationship` 字段的默认 `lazy` 是 `'raise_on_sql'`：访问未预加载的关系**立刻抛异常**，而不是触发隐式同步查询。这是 MissingGreenlet 问题的最后一道安全网。
