# CRUD 方法

::: tip
本页是参考文档。要看典型用法和常见任务，去 [操作指南](/how-to/) 或 [快速上手](/tutorials/01-getting-started)。
:::

所有方法都定义在 `TableBaseMixin` 上，通过 MRO 暴露给所有继承它的模型类。`UUIDTableBaseMixin` 重载了 `get_one()` / `get_exist_one()` 以接受 `uuid.UUID` 类型的 ID。

通用类型变量：`T = TypeVar('T', bound='TableBaseMixin')`。

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

批量插入新记录。

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
    optimistic_retry_count: int = 0,
) -> T
```

INSERT 或 UPDATE 当前实例。SQLAlchemy 根据实例是否在 session 中决定。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `load` | `None` | 保存后预加载的关系（单个或列表） |
| `refresh` | `True` | commit 后用 `cls.get()` 重新获取（避免 MissingGreenlet） |
| `commit` | `True` | `False` 时只 flush，适合批量操作 |
| `jti_subclasses` | `None` | JTI 关系预加载选项（需要 `load`）；`'all'` 表示加载所有子类 |
| `optimistic_retry_count` | `0` | 乐观锁冲突时的自动重试次数 |

**抛出**：`OptimisticLockError`（重试耗尽后）。

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
    optimistic_retry_count: int = 0,
) -> T
```

用 `other` 中的字段局部更新当前实例（PATCH 语义）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `other` | — | 携带新数据的模型实例（通常是 `XxxUpdateRequest`） |
| `extra_data` | `None` | 额外字段字典，会在 `other` 之上叠加 |
| `exclude_unset` | `True` | 只更新 `other` 中**显式设置**的字段 |
| `exclude` | `None` | 排除某些字段不更新 |
| `load`、`refresh`、`commit`、`jti_subclasses`、`optimistic_retry_count` | — | 同 `save()` |

**抛出**：`OptimisticLockError`。

## `delete()`

```python
@classmethod
async def delete(
    cls: type[T],
    session: AsyncSession,
    instances: T | list[T] | None = None,
    *,
    condition: ColumnElement[bool] | bool | None = None,
    commit: bool = True,
) -> int
```

按实例或按条件删除。**两种模式互斥**——`instances` 和 `condition` 必须且只能提供一个。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instances` | `None` | 单个实例或列表（实例模式） |
| `condition` | `None` | WHERE 条件（条件模式，批量删除） |
| `commit` | `True` | 是否 commit |

**返回值**：删除的记录数（`int`）。

**抛出**：`ValueError`（同时提供或都不提供 `instances` 和 `condition`）。

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
    join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
    options: list[ExecutableOption] | None = None,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    order_by: list[ColumnElement[Any]] | None = None,
    filter: ColumnElement[bool] | bool | None = None,
    with_for_update: bool = False,
    table_view: TableViewRequest | None = None,
    jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    populate_existing: bool = False,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> T | list[T] | None
```

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
| `load` | `QueryableAttribute` 或 `list` | 预加载关系（自动构建嵌套链） |
| `order_by` | `list[ColumnElement]` | 排序表达式 |
| `filter` | `ColumnElement[bool]` | 额外 WHERE 条件 |
| `with_for_update` | `bool` | `SELECT ... FOR UPDATE`（行锁），实例 ID 写入 `session.info[SESSION_FOR_UPDATE_KEY]` |
| `table_view` | `TableViewRequest` | 分页 + 排序 + 时间过滤参数包 |
| `jti_subclasses` | `list[type] \| 'all'` | JTI 多态关系子类加载（需要 `load`） |
| `populate_existing` | `bool` | 强制覆盖 identity map 中的对象 |
| `created_before/after_datetime` | `datetime` | 时间过滤（左闭右开） |
| `updated_before/after_datetime` | `datetime` | 时间过滤（左闭右开） |

**抛出**：

- `ValueError` — `jti_subclasses` 没有配套 `load`
- `ValueError` — `jti_subclasses` 用于嵌套关系链
- `ValueError` — `jti_subclasses` 的目标类不是 `PolymorphicBaseMixin`

### 多态查询行为

| 场景 | 行为 |
|------|------|
| JTI 模型（`is_jti=True`） | 自动用 `with_polymorphic(cls, '*')` JOIN 所有子表 |
| STI 模型（`is_sti=True`） | 自动加 `WHERE _polymorphic_name IN (...)` 过滤 |
| `with_for_update` + JTI | 用 `FOR UPDATE OF <主表>`（避免 LEFT JOIN nullable 侧的限制） |

## `get_one()`

```python
@classmethod
async def get_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin override 为 uuid.UUID
    *,
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    with_for_update: bool = False,
) -> T
```

`get(cls.id == id, fetch_mode='one')` 的快捷方式。

**抛出**：`NoResultFound`（找不到）、`MultipleResultsFound`（多条，理论上 ID 唯一不应发生）。

## `get_exist_one()`

```python
@classmethod
async def get_exist_one(
    cls: type[T],
    session: AsyncSession,
    id: int,                        # UUIDTableBaseMixin override 为 uuid.UUID
    load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
    *,
    detail: str = "Not found",
) -> T
```

类似 `get_one()`，但找不到时的异常更友好：

| 环境 | 异常 |
|------|------|
| 已安装 FastAPI | `HTTPException(status_code=404, detail=detail)` |
| 未安装 FastAPI | `RecordNotFoundError` |

`detail`（keyword-only）自定义 404 错误文案——免去"先 `get` 再手写 null 检查 + raise"的样板代码，例如 `detail="角色不存在"`。

判定在模块导入时完成，缓存为 `_HAS_FASTAPI`。

## `count()`

```python
@classmethod
async def count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    created_before_datetime: datetime | None = None,
    created_after_datetime: datetime | None = None,
    updated_before_datetime: datetime | None = None,
    updated_after_datetime: datetime | None = None,
) -> int
```

返回符合条件的记录数。底层用 `SELECT COUNT(*)`。

## `get_with_count()`

```python
@classmethod
async def get_with_count(
    cls: type[T],
    session: AsyncSession,
    condition: ColumnElement[bool] | bool | None = None,
    *,
    table_view: TableViewRequest | None = None,
    # ... 与 get() 相同的所有参数
) -> ListResponse[T]
```

`count()` + `get(fetch_mode="all")` 的组合，返回 `ListResponse[T]`。典型用于 LIST 端点。

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
| `count()` | `@classmethod` | `SELECT COUNT(*)` | `int` |
| `get_with_count()` | `@classmethod` | `COUNT + SELECT` | `ListResponse[T]` |
