# CRUD 实现

::: tip 源码位置
`src/sqlmodel_ext/mixins/table.py` — `TableBaseMixin` 和 `UUIDTableBaseMixin`
:::

## `TableBaseMixin` 的基础

```python
class TableBaseMixin(AsyncAttrs):
    _has_table_mixin: ClassVar[bool] = True   # 让元类识别"这是 table 类"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(
        sa_type=DateTime,
        sa_column_kwargs={'default': now, 'onupdate': now},
        default_factory=now,
    )
```

继承 `AsyncAttrs` 让模型对象支持 `await obj.awaitable_attrs.some_relation` 语法，提供额外的异步安全保障。

`_has_table_mixin = True` 是一个标记，让元类在 `__new__` 中自动添加 `table=True`。

## `save()` 实现

`save()` 是最核心的方法，包含乐观锁重试逻辑：

```python
async def save(self, session, ..., optimistic_retry_count=0):
    cls = type(self)
    instance = self
    retries_remaining = optimistic_retry_count
    current_data = None

    while True:
        session.add(instance)
        try:
            await session.commit()
            break                              # 成功，退出
        except StaleDataError as e:            # 版本冲突！
            await session.rollback()

            if retries_remaining <= 0:
                raise OptimisticLockError(
                    message=f"optimistic lock conflict",
                    model_class=cls.__name__,
                    record_id=str(instance.id),
                    expected_version=instance.version,
                    original_error=e,
                ) from e

            retries_remaining -= 1

            # 保存当前修改（排除元数据字段）
            if current_data is None:
                current_data = self.model_dump(
                    exclude={'id', 'version', 'created_at', 'updated_at'}
                )

            # 从数据库获取最新记录
            fresh = await cls.get(session, cls.id == self.id)
            if fresh is None:
                raise OptimisticLockError("record has been deleted") from e

            # 把我的修改重新应用到最新记录上
            for key, value in current_data.items():
                if hasattr(fresh, key):
                    setattr(fresh, key, value)
            instance = fresh

    await session.refresh(instance)
    return instance
```

### `session.add()` 行为

`session.add()` **不执行 SQL**。SQLAlchemy 在 `commit()` 或 `flush()` 时自动决定：
- 对象是新的 → `INSERT`
- 对象已在 Session 中且有变更 → `UPDATE`

### 为什么必须用返回值？

`session.commit()` 让**所有 Session 中的对象过期**。原 `user` 对象属性变成"过期"状态，访问时触发隐式查询。`save()` 返回 `refresh()` 后的新鲜对象。

## `update()` 实现

```python
async def update(self, session, other, extra_data=None,
                 exclude_unset=True, exclude=None, ...):
    update_data = other.model_dump(exclude_unset=exclude_unset, exclude=exclude)
    instance.sqlmodel_update(update_data, update=extra_data)
    session.add(instance)
    await session.commit()
```

核心是 `exclude_unset=True` 实现 **PATCH 语义**：只有显式设置的字段才会被更新。

## `get()` 实现

这是最长的方法（~200行），分层处理多种场景。

### 第一层：基本查询

```python
statement = select(cls)
if condition is not None:
    statement = statement.where(condition)
```

### 第二层：分页 + 排序

```python
if table_view:
    order_column = cls.created_at if table_view.order == "created_at" else cls.updated_at
    order_by = [desc(order_column) if table_view.desc else asc(order_column)]
    statement = statement.order_by(*order_by).offset(table_view.offset).limit(table_view.limit)
```

### 第三层：时间过滤

```python
@classmethod
def _build_time_filters(cls, created_before, created_after, ...):
    filters = []
    if created_after is not None:
        filters.append(cls.created_at >= created_after)
    if created_before is not None:
        filters.append(cls.created_at < created_before)
    ...
    return filters
```

### 第四层：关系预加载

```python
if load:
    load_list = load if isinstance(load, list) else [load]
    load_chains = cls._build_load_chains(load_list)

    for chain in load_chains:
        loader = selectinload(chain[0])
        for rel in chain[1:]:
            loader = loader.selectinload(rel)
        statement = statement.options(loader)
```

`_build_load_chains` 自动检测关系依赖，构建嵌套加载链。比如 `load=[User.profile, Profile.avatar]` → `selectinload(User.profile).selectinload(Profile.avatar)`。

### 第五层：多态查询

```python
if is_jti:
    polymorphic_cls = with_polymorphic(cls, '*')
    statement = select(polymorphic_cls)   # 自动 JOIN 所有子表

if is_sti:
    descendant_identities = [m.polymorphic_identity for m in mapper.self_and_descendants]
    statement = statement.where(poly_on.in_(descendant_identities))
```

JTI 使用 `with_polymorphic` 自动 JOIN 子表。STI 需要手动添加 `WHERE _polymorphic_name IN (...)` 过滤。

### 第六层：`fetch_mode` 决定返回值

```python
result = await session.exec(statement)

if fetch_mode == "first":   return result.first()
elif fetch_mode == "one":   return result.one()
elif fetch_mode == "all":   return list(result.all())
```

## `rel()` 和 `cond()` — 类型安全辅助函数

```python
def rel(relationship: object) -> QueryableAttribute[Any]:
    """Cast Relationship 字段为 QueryableAttribute，解决 basedpyright 推断问题"""
    if not isinstance(relationship, QueryableAttribute):
        raise AttributeError(...)
    return relationship

def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]:
    """Narrow 列比较表达式为 ColumnElement[bool]，解决 & | 运算符类型错误"""
    return cast(ColumnElement[bool], expr)
```

这两个函数类似 SQLModel 的 `col()`，都是在运行时做类型断言/转换，让静态类型检查器（basedpyright）满意。

## `get_one()` 实现

```python
@classmethod
async def get_one(cls, session, id, *, load=None, with_for_update=False):
    return await cls.get(
        session, col(cls.id) == id,
        fetch_mode='one', load=load, with_for_update=with_for_update,
    )
```

本质是 `get(fetch_mode='one')` 的快捷方式。`UUIDTableBaseMixin` 提供了类型更精确的 override（只接受 `uuid.UUID`）。

## `get_exist_one()` 的 FastAPI 集成

```python
@classmethod
async def get_exist_one(cls, session, id, load=None):
    instance = await cls.get(session, col(cls.id) == id, load=load)
    if not instance:
        if _HAS_FASTAPI:
            raise _FastAPIHTTPException(status_code=404, detail="Not found")
        raise RecordNotFoundError("Not found")
    return instance
```

在**模块导入时**检测 FastAPI 是否安装，有则抛 `HTTPException(404)`，无则抛 `RecordNotFoundError`。

## `sanitize_integrity_error()` 实现

```python
@staticmethod
def sanitize_integrity_error(e: IntegrityError, default_message: str = "...") -> str:
    orig = e.orig
    # SQLSTATE 23514 (check_violation): PostgreSQL 触发器的 RAISE EXCEPTION
    if orig is not None and getattr(orig, 'sqlstate', None) == '23514':
        error_msg = str(orig)
        if '\n' in error_msg:
            error_msg = error_msg.split('\n')[0]  # 取第一行
        if error_msg.startswith('ERROR:'):
            error_msg = error_msg[6:].strip()
        return error_msg
    return default_message
```

PostgreSQL 触发器通过 `RAISE EXCEPTION ... USING ERRCODE = 'check_violation'` 可以产生业务语义的错误消息，可以安全地展示给用户。其他约束错误（FK、唯一等）可能泄露表结构信息，返回默认消息。

## FOR UPDATE 追踪

`get()` 方法中 `with_for_update=True` 时，将锁定实例的 `id()` 记录到 `session.info`：

```python
SESSION_FOR_UPDATE_KEY = '_for_update_locked'

# get() 中：
if with_for_update:
    locked: set[int] = session.info.setdefault(SESSION_FOR_UPDATE_KEY, set())
    locked.add(id(instance))
```

供 `@requires_for_update` 装饰器在运行时检查。

## `count()` 实现

```python
@classmethod
async def count(cls, session, condition=None, ...):
    statement = select(func.count()).select_from(cls)
    if condition is not None:
        statement = statement.where(condition)
    result = await session.scalar(statement)
    return result or 0
```

使用数据库级 `COUNT(*)` 而非 Python `len()`。

## `get_with_count()` 实现

```python
@classmethod
async def get_with_count(cls, session, condition=None, *, table_view=None, ...):
    total_count = await cls.count(session, condition, ...)
    items = await cls.get(session, condition, fetch_mode="all", table_view=table_view, ...)
    return ListResponse(count=total_count, items=items)
```

本质是 `count()` + `get(fetch_mode="all")` 的组合。
