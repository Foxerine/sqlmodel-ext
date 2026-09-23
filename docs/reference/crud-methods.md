# CRUD 方法

::: tip
本页是参考文档。要看典型用法和常见任务，去 [操作指南](/how-to/) 或 [快速上手](/tutorials/01-getting-started)。
:::

所有方法都定义在 `TableBaseMixin` 上，通过 MRO 暴露给所有继承它的模型类。`UUIDTableBaseMixin` 把 `id` 换成 **UUIDv7** 主键，并重载 `get_one()` / `get_exist_one()` 以只接受 `uuid.UUID`。

通用类型变量：`T = TypeVar('T', bound='TableBaseMixin')`。

::: tip 配合 basedpyright 使用
`get()` 的返回类型由 `fetch_mode` 字面量精确决定，`delete()` 用 `@overload` 强制"`instances` 或 `condition` 二选一"，`get_one()` 按主键类型重载——这些约束都在**类型层**，basedpyright 会在运行前标红误用。
:::

## `add()`

```python
@classmethod
async def add(
    cls: type[T],
    session: AsyncSession,
    instances: T | list[T],
    refresh: bool = True,
    commit: bool = True,
) -> T | list[T]
```

插入一条或多条新记录。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instances` | — | 单个实例或实例列表 |
| `refresh` | `True` | commit 后通过 `cls.get()` 重新获取，绑定数据库生成的字段 |
| `commit` | `True` | `False` 时只 `flush()` 不 `commit()` |

**返回值类型**：与 `instances` 输入类型一致——传入单个实例返回单个，传入列表返回列表。

## `save()`

```python
async def save(
    self: T,
    session: AsyncSession,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    refresh: bool = True,
    commit: bool = True,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    optimistic_retry_count: int | None = None,
) -> T
```

INSERT 或 UPDATE 当前实例。SQLAlchemy 根据实例状态决定。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `load` | `None` | 保存后预加载的关系（单个或列表） |
| `refresh` | `True` | commit 后用 `cls.get()` 重新获取（避免 MissingGreenlet） |
| `commit` | `True` | `False` 时只 flush，适合批量操作 |
| `jti_subclasses` | `None` | JTI 关系预加载选项（需要 `load`）；`'all'` 表示加载所有子类 |
| `optimistic_retry_count` | `None` | 乐观锁冲突的自动重试次数。`None` = 用模型策略 `__optimistic_retry_default__`（`OptimisticLockMixin` 模型为 `3`，其余为 `0`）；显式 `0` = 不重试 |

**行为细节**：

- 持久化实例有列变更时，`save()` **显式**赋值 `updated_at`（不只依赖列级 `onupdate`）——JTI 下只改子表列的 UPDATE 不会触碰父表，`onupdate` 不会触发。
- 重试时重新读取最新行，只把**本实例实际修改过的列**（按 SQLAlchemy 属性历史）重新应用上去，不会用你的旧值覆盖别人已提交的其它列。

**抛出**：`OptimisticLockError`（重试耗尽，或重试时发现记录已被删除）。

::: danger 必须用返回值
`session.commit()` 让所有 session 对象过期。务必写 `user = await user.save(session)`，不能丢弃返回值。
:::

## `update()`

```python
async def update(
    self: T,
    session: AsyncSession,
    other: SQLModelBase,
    extra_data: dict[str, Any] | None = None,
    exclude_unset: bool = True,
    exclude: set[str] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    refresh: bool = True,
    commit: bool = True,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    optimistic_retry_count: int | None = None,
) -> T
```

用 `other` 中的字段局部更新当前实例（PATCH 语义）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `other` | — | 携带新数据的模型实例，通常是 `partial=True` 派生的 `XxxUpdate` DTO |
| `extra_data` | `None` | 额外字段字典，会在 `other` 之上叠加 |
| `exclude_unset` | `True` | 只应用 `other.model_fields_set` 中的字段；显式传入的 `None` 也算"已设置"，会写成 NULL |
| `exclude` | `None` | 排除某些字段不更新 |
| `load`、`refresh`、`commit`、`jti_subclasses` | — | 同 `save()` |
| `optimistic_retry_count` | `None` | 同 `save()`；重试时重新读取最新行，再把 `other` 的改动应用上去 |

::: tip 配合 `partial=True`
`partial=True` 派生的 DTO 字段默认值是 `Unset`，而 `Unset` 字段**永远不会出现在 `model_dump()` 里**——所以"没传"的字段天然不会被写入，不依赖 `exclude_unset` 这个开关。见 [集成 FastAPI](/how-to/integrate-with-fastapi)。
:::

非空更新（有数据或有 `extra_data`）总会显式赋值 `updated_at`；空更新不动它。

**抛出**：`OptimisticLockError`。

## `delete()`

```python
@overload
@classmethod
async def delete(cls: type[T], session: AsyncSession, instances: T | list[T], *, commit: bool = ...) -> int: ...
@overload
@classmethod
async def delete(cls: type[T], session: AsyncSession, *, condition: ColumnElement[bool] | bool, commit: bool = ...) -> int: ...
```

按实例或按条件删除。**两种模式互斥**——两个 `@overload` 让"两者都不传"在类型检查阶段就报"没有匹配的重载"；运行时同样校验。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instances` | `None` | 单个实例或列表（实例模式） |
| `condition` | `None` | WHERE 条件（条件模式，批量删除；STI 子类会自动附加鉴别列过滤，不会误删兄弟子类的行） |
| `commit` | `True` | 是否 commit |

**返回值**：删除的记录数（`int`）。

**抛出**：

| 异常 | 条件 |
|------|------|
| `ValueError` | 同时提供或都不提供 `instances` 和 `condition` |
| `ResourceReferencedError` | 被删的行仍被外键（`RESTRICT` / `NO ACTION`）引用——行**存在**且未被删除。仅当 `IntegrityError` 在本方法内捕获**且**其 `statement` 是 `DELETE` 时才翻译；仅 PostgreSQL（依赖 SQLSTATE `23503`）。见 [处理"仍被引用"的删除](/how-to/handle-referenced-deletes) |
| `OptimisticLockError` | 本次 flush 内出现乐观锁冲突（典型：`OptimisticLockMixin` 模型的带版本 `DELETE` 命中 0 行）。`record_id` 与 `expected_version` 恒为 `None`（冲突在 flush 级无法归属到具体行）；**永不重试**；条件模式不会抛（批量 DELETE 没有逐行版本检查） |

::: warning `commit=False` 的边界
上面两个翻译只覆盖**本方法内部**发出的 SQL。实例模式下 `commit=False` 时，真正的 `DELETE` 由你之后的 flush/commit 发出，发生在本方法之外，不会被翻译。
:::

## `get()`

最复杂的查询方法，通过 `@overload` 提供 `fetch_mode` 字面量到返回类型的精确映射。

```python
@classmethod
async def get(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    offset: int | None = None,
    limit: int | None = None,
    fetch_mode: Literal["one", "first", "all"] = "first",
    join: type[TableBaseMixin] | tuple[type[TableBaseMixin], OnClauseArgument] | None = None,
    options: list[ExecutableOption] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    order_by: list[ColumnElement[Any]] | None = None,
    filter: ColumnElement[bool] | bool | None = None,
    with_for_update: bool = False,
    skip_locked: bool = False,
    table_view: TableViewArgument | None = None,  # TimeFilterRequest | PageWindowRequest
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    populate_existing: bool = False,
    authoritative: bool = False,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> T | list[T] | None
```

`CachedTableBaseMixin` 模型的 `get()` 额外多一个 `no_cache: bool = False`（见 [Mixin 类](./mixins#cachedtablebasemixin)）。

### `fetch_mode` 与返回类型

| `fetch_mode` | 返回类型 | 0 条时 | 多条时 |
|---|---|---|---|
| `"first"`（默认） | `T \| None` | `None` | 返回第一条 |
| `"one"` | `T` | `NoResultFound` | `MultipleResultsFound` |
| `"all"` | `list[T]` | `[]` | 全部返回 |

### 参数

| 参数 | 类型 | 含义 |
|------|------|------|
| `condition` | `ColumnElement[bool]` | 主 WHERE 条件 |
| `offset` / `limit` | `int` | 分页（显式参数优先于 `table_view`） |
| `join` | `type` 或 `(type, on)` 元组 | JOIN 另一张表 |
| `options` | `list[ExecutableOption]` | 自定义 SQLAlchemy options（如 `selectinload`） |
| `load` | `QueryableAttribute` 或 `list` | 预加载关系（自动构建嵌套链；双向关系对按列表顺序断环） |
| `order_by` | `list[ColumnElement]` | 排序表达式 |
| `filter` | `ColumnElement[bool]` | 额外 WHERE 条件 |
| `with_for_update` | `bool` | `SELECT ... FOR UPDATE` 行锁。实例 `id()` 写入 `session.info[SESSION_FOR_UPDATE_KEY]`；**强制** `populate_existing`（无法关闭），保证拿到的是数据库最新值而不是 identity map 里的旧对象 |
| `skip_locked` | `bool` | `FOR UPDATE SKIP LOCKED`：跳过被别的事务锁住的行而不是等待。只在 `with_for_update=True` 时生效。"0 行"也可能意味着"候选都被别人锁了"，**不要**用于存在性判断 |
| `table_view` | `TableViewRequest`（或任意 `TimeFilterRequest` / `PageWindowRequest`，具备哪部分就应用哪部分） | 分页 + 排序 + 时间过滤 + keyset 游标参数包。`order` 总会追加同方向的 `id` 作为决胜列；`after_id` 应用 keyset 游标 |
| `jti_subclasses` | `list[type] \| 'all'` | JTI 多态关系子类加载（需要 `load`） |
| `populate_existing` | `bool` | 无锁地强制用数据库数据覆盖 identity map 中的对象 |
| `authoritative` | `bool` | **授权读取**的唯一开关（结果决定是否允许某件事，必须权威）：本层等价于 `populate_existing=True`；缓存模型额外绕过 Redis。与 `populate_existing` 单调合并（取 `or`） |
| `created_before/after_datetime` | `datetime` | 时间过滤（左闭右开） |
| `updated_before/after_datetime` | `datetime` | 时间过滤（左闭右开） |

**抛出**：

- `ValueError` — `jti_subclasses` 没有配套 `load`；用于嵌套关系链；目标类不是 `PolymorphicBaseMixin`
- `KeysetCursorUnsupportedError` — `after_id` 与显式 `order_by` 或 `join` 同用
- `KeysetCursorInvalidError` — `after_id` 锚点不存在或不在本查询可见范围内（两者刻意不可区分）
- `ValueError` — `after_id` 用在非 UUID 主键的表上（编程错误，改用 offset 分页）

keyset 游标的完整规则见 [Keyset 游标分页](/how-to/keyset-pagination)。

### 多态查询行为

| 场景 | 行为 |
|------|------|
| JTI 模型 | 自动用 `with_polymorphic(cls, '*')` JOIN 所有子表 |
| STI 模型 | 自动加 `WHERE _polymorphic_name IN (...)`（`get()` / `count()` / `delete(condition=)` / keyset 锚点 / 聚合方法共用同一过滤） |
| `with_for_update` + 多态 | 用 `FOR UPDATE OF <主表>`（避免 LEFT JOIN nullable 侧的限制） |

## `get_one()`

```python
@classmethod
async def get_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin 重载为 uuid.UUID
    *,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    with_for_update: bool = False,
    authoritative: bool = False,
) -> T
```

`get(col(cls.id) == id, fetch_mode='one')` 的快捷方式。`authoritative` 语义同 `get()`。

**抛出**：`NoResultFound`（找不到）、`MultipleResultsFound`。

## `get_exist_one()`

```python
@classmethod
async def get_exist_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin 重载为 uuid.UUID
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    *,
    detail: str = "Not found",
    with_for_update: bool = False,
) -> T
```

类似 `get_one()`，但找不到时的异常更友好：

| 环境 | 异常 |
|------|------|
| 已安装 FastAPI | `HTTPException(status_code=404, detail=detail)` |
| 未安装 FastAPI | `RecordNotFoundError` |

- `detail`（keyword-only）自定义 404 文案，例如 `detail="角色不存在"`。
- `with_for_update`（keyword-only）转发给 `get()`：用 `SELECT ... FOR UPDATE` 读行（这本身就绕过 Redis 缓存和 identity map）。典型用途是关闭"存在检查"与"删除"之间的 TOCTOU 窗口——并发的第二个请求会阻塞，等第一个提交后查不到行，得到与串行第二次删除相同的 404。

判定在模块导入时完成：`sqlmodel_ext.mixins.table` 导入时尝试 `from fastapi import HTTPException`，失败则记为 `None`。

## `count()`

```python
@classmethod
async def count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    distinct_column: Mapped[Any] | ColumnElement[Any] | None = None,
    time_filter: TimeFilterRequest | None = None,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> int
```

返回符合条件的记录数，底层 `SELECT COUNT(*)`。传 `distinct_column` 时改为 `COUNT(DISTINCT col)`（如"活跃用户去重数"）。`time_filter` 中的非空字段优先于单独传入的时间参数。

## `distinct_column()`

```python
@classmethod
async def distinct_column(
    cls: type[T],
    session: AsyncSession,
    column: Mapped[V] | ColumnElement[V],
    condition: ColumnElement[bool] | None = None,
    *,
    limit: int | None = None,
) -> list[V]
```

返回某一列的去重值（数据库级 `SELECT DISTINCT`），返回类型随列类型推断（`col(Model.owner_id)` → `list[UUID]`）。STI 过滤与 `get()` / `count()` 一致。

## `group_sum()`

```python
@classmethod
async def group_sum(
    cls: type[T],
    session: AsyncSession,
    sum_columns: Sequence[Mapped[Any] | ColumnElement[Any]],
    *,
    group_by: Mapped[GK] | ColumnElement[GK] | None = None,
    condition: ColumnElement[bool] | None = None,
    order_by: ColumnElement[Any] | None = None,
) -> list[GroupSumRow[GK]]
```

一次查询算出 `COUNT(*)` 与每个 `COALESCE(SUM(col), 0)`。

- 不传 `group_by` → 全表聚合，返回**恰好一个**元素（`key=None`）
- 传 `group_by`（列或表达式，如 `date_trunc` 时间桶）→ 每组一行；默认按 `group_by` 升序，`order_by` 可覆盖

**抛出**：`ValueError`（`sum_columns` 为空——纯计数用 `count()`）。

### `GroupSumRow[GK]`

```python
from sqlmodel_ext.mixins import GroupSumRow
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | `GK` | 分组键（`group_by` 的值）；全表聚合时为 `None` |
| `count` | `NonNegativeBigInt` | 组内行数 |
| `totals` | `list[Decimal]` | 各求和列的结果，**按位置**与 `sum_columns` 对齐 |

条件求和（`SUM ... FILTER (WHERE ...)`）请每个条件调用一次，再按 `key` 在 Python 中合并。用法见 [聚合查询](/how-to/aggregate-queries)。

## `get_with_count()`

```python
@classmethod
async def get_with_count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    join: type[TableBaseMixin] | tuple[type[TableBaseMixin], OnClauseArgument] | None = None,
    options: list[ExecutableOption] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    order_by: list[ColumnElement[Any]] | None = None,
    filter: ColumnElement[bool] | bool | None = None,
    table_view: TableViewRequest | None = None,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
) -> ListResponse[T]
```

先 `get(fetch_mode="all")`（其中完成 keyset 游标的全部校验），再 `count()`，组装成 `ListResponse[T]`。`count` 是**整个过滤集**的大小，不受 `after_id` 影响。典型用于 LIST 端点。

## IntegrityError 友好消息注册表

`TableBaseMixin` 上的静态方法。在声明约束的**同一处**注册"约束名 → 用户可见消息"，查询路径不泄露表名/列名/SQL。先注册者生效（`setdefault`，防重复 import）。

| 方法 | 方向 / 用途 |
|------|------|
| `register_unique_violation_message(constraint_name, friendly_message)` | UNIQUE 违反（23505） |
| `register_foreign_key_violation_message(constraint_name, friendly_message)` | INSERT/UPDATE 子行指向不存在的父行（23503，"引用的资源不存在"，404 语义） |
| `register_check_violation_message(constraint_name, friendly_message)` | ORM 声明的 `CheckConstraint`（23514 且有 `constraint_name`） |
| `register_fk_delete_restrict_message(constraint_name, friendly_message)` | DELETE 父行时仍被引用（409 语义）；只有 `delete()` 查它 |
| `lookup_unique_violation_message` / `lookup_foreign_key_violation_message` / `lookup_check_violation_message` / `lookup_fk_delete_restrict_message(constraint_name)` | 按名字查，未命中或名字为 `None` 返回 `None` |
| `lookup_integrity_violation_message(e)` | 纯查表：命中返回注册的业务消息，否则 `None`；触发器 `RAISE EXCEPTION`（23514 且无 `constraint_name`）视为命中，返回其首行消息。**不查** `fk_delete_restrict` 注册表 |
| `sanitize_integrity_error(e, default_message=...)` | 在 `lookup_integrity_violation_message` 之上加兜底：未命中则记日志并返回 `default_message` |
| `extract_trigger_message(orig)` | 从触发器异常中取首行业务消息（去掉 `ERROR:` 前缀与 `DETAIL:` / `CONTEXT:` 行） |

需要区分"命中/未命中"（例如把用户错误和平台错误分开计数）时用 `lookup_integrity_violation_message`——`sanitize_*` 会把两种情况折叠成一个字符串。SQLSTATE 判定是 PostgreSQL 专有；其它数据库只会得到 `default_message`。

## 方法速查

| 方法 | 类型 | 对应 SQL | 返回值 |
|------|------|---------|--------|
| `add()` | `@classmethod` | `INSERT` | `T` 或 `list[T]` |
| `save()` | 实例方法 | `INSERT` 或 `UPDATE` | 刷新后的 `T` |
| `update()` | 实例方法 | `UPDATE`（PATCH） | 刷新后的 `T` |
| `delete()` | `@classmethod` | `DELETE` | `int`（删除数） |
| `get()` | `@classmethod` | `SELECT ... WHERE ...` | `T \| list[T] \| None` |
| `get_one()` | `@classmethod` | `SELECT WHERE id = ?` | `T` |
| `get_exist_one()` | `@classmethod` | `SELECT WHERE id = ?` + 404 | `T` |
| `count()` | `@classmethod` | `SELECT COUNT(*)` / `COUNT(DISTINCT col)` | `int` |
| `distinct_column()` | `@classmethod` | `SELECT DISTINCT col` | `list[V]` |
| `group_sum()` | `@classmethod` | `SELECT [key,] COUNT(*), SUM(...) [GROUP BY]` | `list[GroupSumRow[GK]]` |
| `get_with_count()` | `@classmethod` | `SELECT` + `COUNT` | `ListResponse[T]` |
