# CRUD 实现

::: tip 源码位置
`src/sqlmodel_ext/mixins/table.py` — `TableBaseMixin` 和 `UUIDTableBaseMixin`
:::

本章解释 `save()` / `get()` / `update()` / `delete()` 等方法的内部如何工作。完整签名见 [CRUD 方法参考](/reference/crud-methods)；典型用法见 [操作指南](/how-to/)。

## `TableBaseMixin` 的基础

```python
class TableBaseMixin(AsyncAttrs):
    _has_table_mixin: ClassVar[bool] = True          # 让元类识别"这是 table 类"
    __optimistic_retry_default__: ClassVar[int] = 0  # 乐观锁重试策略（OptimisticLockMixin 覆盖为 3）

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=now, sa_type=DateTime(timezone=True))
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={'default': now, 'onupdate': now},
        default_factory=now,
    )

class UUIDTableBaseMixin(TableBaseMixin):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
```

- 继承 `AsyncAttrs` 让模型对象支持 `await obj.awaitable_attrs.some_relation`。
- `_has_table_mixin = True` 让元类在 `__new__` 中自动添加 `table=True`。
- 时间戳都是**时区感知**的 UTC。
- UUID 主键用 **UUIDv7**：前 48 位是毫秒时间戳，字节序即创建顺序，B-tree 插入集中在右边缘，页分裂远少于 UUIDv4。Python 3.14+ 直接用标准库 `uuid.uuid7`，更早的版本用内置的 RFC 9562 实现（同一进程内严格递增）。`create_subclass_id_mixin()` 生成的 JTI 子表主键使用同一个工厂——两边不一致会让 JTI 子类悄悄得到不同版本的 UUID。

## `save()` 实现

```python
async def save(self, session, load=None, refresh=True, commit=True,
               jti_subclasses=None, optimistic_retry_count=None):
    retries_remaining = (optimistic_retry_count if optimistic_retry_count is not None
                         else cls.__optimistic_retry_default__)
    while True:
        # flush 前快照 id / oplock_version（冲突回滚后所有属性过期，再读会发 SQL）
        # 需要重试时，按属性历史记录"我实际改过的列"
        session.add(instance)
        if <持久化实例且列有变更>:
            instance.updated_at = now()      # 显式赋值，见下
        try:
            await (session.commit() if commit else session.flush())
            break
        except StaleDataError:
            ...                              # 乐观锁重试，见"乐观锁机制"

    if not refresh:
        return instance
    _insp = inspect(instance)                # commit 后用 identity 安全读 id
    return await cls.get(session, cls.id == _insp.identity[0], load=load, jti_subclasses=jti_subclasses)
```

### 为什么显式赋值 `updated_at`

列级 `onupdate` 只在**该表**被 UPDATE 时触发。JTI 下如果只改了子表的列，父表（`updated_at` 所在的表）根本不会被 UPDATE，`onupdate` 不会触发，时间戳就停在旧值。所以只要持久化实例有列变更（`session.is_modified(instance, include_collections=False)`——纯集合变化不会 UPDATE 本行，不算），`save()` 就显式写入 `updated_at`；`update()` 对任何非空更新同样处理。INSERT 的时间戳来自字段默认值。

### 为什么必须用返回值

::: danger 对象过期
`session.commit()` 让**所有 Session 中的对象过期**。原对象的属性再被访问会触发隐式查询——在异步里就是 `MissingGreenlet`。`save()` 返回经过 `cls.get()` 重新读取的新鲜对象（缓存模型会绕过缓存读取并回填）。
:::

## `update()` 实现

```python
async def update(self, session, other, extra_data=None, exclude_unset=True, exclude=None, ...):
    update_data = other.model_dump(exclude_unset=exclude_unset, exclude=exclude)
    instance.sqlmodel_update(update_data, update=extra_data)
    if update_data or extra_data:
        instance.updated_at = now()
    session.add(instance)
    await session.commit()
```

::: tip PATCH 语义来自类型，不来自开关
`partial=True` 派生的 DTO 字段默认值是 `Unset`，而 `Unset` 字段永远不会出现在 `model_dump()` 里——"客户端没传"的字段在数据层面就不存在，自然不会被写入；客户端显式传的 `null` 是一个真实的值，会写成 NULL。`exclude_unset=True` 仍是默认值，但 PATCH 的正确性不再依赖它。
:::

## `delete()` 实现

```python
@classmethod
async def delete(cls, session, instances=None, *, condition=None, commit=True) -> int:
    try:
        if condition is not None:
            stmt = sql_delete(cls).where(condition)
            if (sti := cls._sti_descendants_condition()) is not None:
                stmt = stmt.where(sti)            # 不误删 STI 兄弟子类的行
            deleted = (await session.execute(stmt)).rowcount
        else:
            for inst in instances_list:
                await session.delete(inst)
        if commit:
            await session.commit()
    except IntegrityError as e:
        if sqlstate == '23503' and e.statement.lstrip().upper().startswith('DELETE'):
            raise ResourceReferencedError(<注册的消息或兜底消息>, constraint, e) from e
        raise
    except StaleDataError as e:
        raise OptimisticLockError(..., record_id=None, expected_version=None) from e
```

两处值得注意：

- **外键违反的方向只能由调用点决定**。"插入子行但父行不存在"与"删除仍被引用的父行"在驱动层是同一个异常；而且 commit 会 flush session 里所有待执行的操作，一条之前排队的坏 INSERT 也可能在这里冒出来。所以翻译需要两个条件同时成立：在 `delete()` 内捕获，**且**失败语句是 `DELETE`（`IntegrityError.statement` 由 SQLAlchemy 生成，与服务器语言无关）。
- **两个 `@overload`** 让"`instances` 和 `condition` 都不传"在类型检查阶段就报"没有匹配的重载"，把运行时不变式提升为编译期约束。

## `get()` 实现

完整签名见 [reference/crud-methods](/reference/crud-methods#get)。处理顺序：

### 1. `table_view` 合并 + 排序决胜列 + keyset 游标

```python
if isinstance(table_view, TimeFilterRequest):   # 时间过滤：显式参数优先
    ...
if isinstance(table_view, PageWindowRequest):   # offset / limit：显式参数优先
    ...
if isinstance(table_view, PaginationRequest):
    if table_view.after_id is not None and (order_by is not None or join is not None):
        raise KeysetCursorUnsupportedError(...)
    if order_by is None:
        order_col = col(getattr(cls, table_view.order or 'created_at'))
        direction = desc if table_view.desc else asc
        order_by = [direction(order_col)]
        if order_field != 'id':
            order_by.append(direction(col(cls.id)))     # id 决胜列
    if table_view.after_id is not None:
        keyset_condition = await cls._build_keyset_condition(session, table_view, condition, filter)
```

**id 决胜列**：`created_at` 不唯一（同一批创建的行共享时间戳），相等的行之间没有确定顺序，offset 与 keyset 分页都会在页边界上跳过或重复行。追加同方向的 `id` 让顺序成为全序。

**keyset 锚点查询**用与主查询相同的 FROM（JTI 基类用 `with_polymorphic`）和相同的可见性（`condition` + `filter` + STI 过滤）：锚点的排序值在服务端查出，客户端只传 id；锚点不可见时与"不存在"报同一个错，避免通过差异探测作用域外的行。

### 2. 多态

```python
if is_jti:
    statement = select(with_polymorphic(cls, '*'))   # 自动 JOIN 所有子表，避免 N+1
else:
    statement = select(cls)
if (sti := cls._sti_descendants_condition()) is not None:
    statement = statement.where(sti)                 # WHERE _polymorphic_name IN (...)
```

SQLAlchemy/SQLModel 不会给 STI 子类查询自动加鉴别列过滤（sqlalchemy#5018、sqlmodel#488）。`_sti_descendants_condition()` 是 `get()` / `count()` / `delete(condition=)` / keyset 锚点 / 聚合方法**共用**的同一个条件，保证它们看到的范围一致。

### 3. 条件、时间过滤、JOIN、options

`condition` → keyset 条件 → 时间过滤（左闭右开）→ `join` → `options`。

### 4. 关系预加载

```python
load_chains = cls._build_load_chains(load_list)
for chain in load_chains:
    loader = selectinload(chain[0])
    for rel in chain[1:]:
        loader = loader.selectinload(rel)
    statement = statement.options(loader)
```

`_build_load_chains` 自动检测依赖：`[rel(User.profile), rel(Profile.avatar)]` → `selectinload(User.profile).selectinload(Profile.avatar)`。双向关系对（`A.b` + `B.a`）会让每个关系都成为别人的后继、不出现在任何根下；这时按列表顺序在第一个成员处断环，而不是静默丢掉请求的加载。

### 5. 排序、分页、行锁

```python
if with_for_update:
    statement = statement.with_for_update(of=cls if polymorphic else None, skip_locked=skip_locked)
if with_for_update or populate_existing or authoritative:
    statement = statement.execution_options(populate_existing=True)
```

- 多态模型用 `FOR UPDATE OF <主表>`（PostgreSQL 不允许锁 LEFT OUTER JOIN 的可空侧）。
- **加锁读强制 `populate_existing`**：数据库返回最新行，但 identity map 默认会交还已加载的旧对象——随后的读-改-写就是一次丢失更新。
- `authoritative` 在本层就是 `populate_existing`（缓存模型额外绕过 Redis），与调用方的 `populate_existing` 取 `or`，永不削弱。

### 6. `fetch_mode` 决定返回值 + 锁跟踪

```python
result = await session.exec(statement)
if fetch_mode == "one":   instance = result.one()
elif fetch_mode == "first": instance = result.first()
else:                     instances = list(result.all())
# with_for_update 时把 id(instance) 记入 session.info[SESSION_FOR_UPDATE_KEY]
```

## FOR UPDATE 追踪

`with_for_update=True` 时锁定实例的 `id()` 写入 `session.info[SESSION_FOR_UPDATE_KEY]`，供 `@requires_for_update` / `@requires_locked_param` 在运行时检查。这个集合的生命周期由模块级的 session 事件监听器维护（与缓存无关，始终生效）：

| 事件 | 处理 |
|------|------|
| 最外层 commit / rollback | 清空（锁随事务释放） |
| savepoint 开始 | 压入当前集合的快照 |
| savepoint 回滚 | 恢复快照（PostgreSQL 释放 savepoint 内取得的锁，保留之前的） |
| savepoint 释放（RELEASE） | 只弹出快照（锁转移给外层事务，保留） |
| savepoint 经 `close()` 结束 | 保守地恢复快照（丢弃内层锁——fail-closed，迫使重新加锁） |

增强 session 的 `reset()` / `close()` 也会清空它。

## `rel()` 和 `cond()`

```python
def rel(relationship: object) -> QueryableAttribute[Any]:
    if not isinstance(relationship, QueryableAttribute):
        raise AttributeError(...)
    return relationship

def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]:
    return cast(ColumnElement[bool], expr)
```

类似 SQLModel 的 `col()`：basedpyright 会把 `User.profile` 推断为 `Profile`、把 `Model.field == value` 推断为 `bool`，这两个函数把它们窄化成可以传给 `load=` / 用 `&` `|` 组合的类型。

## `get_one()` / `get_exist_one()`

```python
@classmethod
async def get_one(cls, session, id, *, load=None, with_for_update=False, authoritative=False):
    return await cls.get(session, col(cls.id) == id, fetch_mode='one',
                         load=load, with_for_update=with_for_update, authoritative=authoritative)

@classmethod
async def get_exist_one(cls, session, id, load=None, *, detail="Not found", with_for_update=False):
    instance = await cls.get(session, col(cls.id) == id, load=load, with_for_update=with_for_update)
    if instance is None:
        if _FastAPIHTTPException is not None:   # 模块导入时 `from fastapi import HTTPException`，失败则为 None
            raise _FastAPIHTTPException(status_code=404, detail=detail)
        raise RecordNotFoundError(detail)
    return instance
```

`UUIDTableBaseMixin` 重载二者，只接受 `uuid.UUID`。FastAPI 是否安装在**模块导入时**检测，避免把 FastAPI 变成硬依赖。

## IntegrityError 友好消息

`sanitize_integrity_error()` / `lookup_integrity_violation_message()` 按 SQLSTATE 分派：

| SQLSTATE | 处理 |
|------|------|
| `23505` 唯一约束 | 查 `register_unique_violation_message` 注册表 |
| `23503` 外键 | 查 `register_foreign_key_violation_message` 注册表（"引用的资源不存在"方向） |
| `23514` 且有 `constraint_name` | 查 `register_check_violation_message` 注册表；绝不返回原始消息（CHECK 表达式可能含列名） |
| `23514` 且无 `constraint_name` | 触发器 `RAISE EXCEPTION`：消息本身就是开发者写给用户的，取首行返回 |

asyncpg 适配器只转发 `sqlstate`、不转发 `constraint_name`，所以约束名回退从 `orig.__cause__`（真正的驱动异常）读取——否则每次查表都会落空。触发器消息同样从 `__cause__` 读，避免把驱动异常类名前缀泄露给用户。

## `count()` / `distinct_column()` / `group_sum()`

都是一条数据库级聚合语句，共用 STI 过滤：

```python
count_expr = func.count(distinct(distinct_column)) if distinct_column is not None else func.count()
statement = select(count_expr).select_from(cls)

statement = select(distinct(column)).select_from(cls)                # distinct_column

sum_exprs = [func.coalesce(func.sum(c), 0) for c in sum_columns]      # group_sum
statement = select(group_by, func.count(), *sum_exprs).select_from(cls).group_by(group_by)
```

`group_sum()` 的求和值统一转成 `Decimal`（PostgreSQL `NUMERIC` 本来就是 `Decimal`；SQLite 可能给 `int` / `float`，经 `str()` 转换避免二进制浮点残差）。`GroupSumRow` 是 `SQLModelBase` 泛型：它只是方法返回值、不进入 OpenAPI，不受 SQLModel 泛型 JSON schema 问题影响（`ListResponse` 受影响，所以它继承 `BaseModel`）。

## `get_with_count()` 实现

```python
items = await cls.get(session, condition, fetch_mode="all", table_view=table_view, ...)
total_count = await cls.count(session, condition, time_filter=time_filter)
return ListResponse(count=total_count, items=items)
```

**先取数据再计数**：`get()` 负责 keyset 游标的全部校验（`after_id` 与 `order_by` / `join`、锚点有效性），先计数会在一个注定失败的请求上浪费一次聚合查询。`count` 不受 `after_id` 影响。
