# CRUD 操作

继承 `TableBaseMixin`（自增 ID）或 `UUIDTableBaseMixin`（UUID ID）即可获得全套异步 CRUD 方法。

## 工具函数

### `rel()` — 类型安全的关系引用

basedpyright 会把 SQLModel 的 Relationship 字段推断为注解类型（如 `LLM`），而非 `QueryableAttribute`。`rel()` 解决这个类型问题：

```python
from sqlmodel_ext import rel

# 不用 rel：basedpyright 报类型错误
user = await User.get(session, load=User.profile)

# 用 rel：类型正确
user = await User.get(session, load=rel(User.profile))
```

### `cond()` — 类型安全的条件组合

类似问题：basedpyright 把 `Model.field == value` 推断为 `bool`，导致 `&` / `|` 运算符报错。

```python
from sqlmodel_ext import cond

scope = cond(UserFile.user_id == current_user.id)
condition = scope & cond(UserFile.status == FileStatusEnum.uploaded)
users = await UserFile.get(session, condition, fetch_mode="all")
```

### `sanitize_integrity_error()` — 友好化错误消息

从 `IntegrityError` 中提取用户安全的错误消息。特别支持 PostgreSQL 触发器的 SQLSTATE 23514 错误：

```python
from sqlalchemy.exc import IntegrityError

try:
    await order.save(session)
except IntegrityError as e:
    msg = Order.sanitize_integrity_error(e, "操作失败")
    # PostgreSQL 触发器消息会被提取；其他约束错误返回默认消息
```

## 内置字段

继承后自动获得三个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `int` 或 `UUID` | 主键，自动生成 |
| `created_at` | `datetime` | 创建时间，自动设置 |
| `updated_at` | `datetime` | 更新时间，每次 UPDATE 自动刷新 |

## `save()` — 创建或更新

```python
user = User(name="Alice", email="alice@example.com")
user = await user.save(session)
print(user.id)  # 已有值

# 修改后再次 save = UPDATE
user.name = "Bob"
user = await user.save(session)
```

::: warning 必须使用返回值
`save()` 返回刷新后的实例。`commit()` 后原对象属性会过期，直接使用可能触发隐式查询。
:::

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `commit` | `True` | `False` 时只 flush 不 commit，适合批量操作 |
| `load` | `None` | 保存后预加载指定关系 |
| `optimistic_retry_count` | `0` | 乐观锁冲突时的重试次数 |

```python
# 批量操作：前面只 flush，最后一个 commit
await user1.save(session, commit=False)
await user2.save(session, commit=False)
user3 = await user3.save(session)  # 一次性 commit 全部

# 保存后预加载关系
user = await user.save(session, load=User.profile)
```

## `add()` — 批量插入

```python
users = [User(name="Alice"), User(name="Bob")]
users = await User.add(session, users)

# 单条也行
user = await User.add(session, User(name="Eve"))
```

## `update()` — 局部更新（PATCH 语义）

```python
class UserUpdate(SQLModelBase):
    name: str | None = None
    email: str | None = None

data = UserUpdate(name="Bob")  # 只设置了 name
user = await user.update(session, data)
# 只更新 name，email 保持不变（exclude_unset=True）
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exclude_unset` | `True` | 只更新显式设置的字段（PATCH 语义） |
| `exclude` | `None` | 排除某些字段不允许更新 |
| `extra_data` | `None` | 在 update 模型之外追加额外字段 |

```python
# 追加额外字段
user = await user.update(session, data, extra_data={"updated_by": admin.id})

# 排除敏感字段
user = await user.update(session, data, exclude={"role", "is_admin"})
```

## `delete()` — 删除

两种模式，互斥使用：

```python
# 按实例删除
await User.delete(session, user)
await User.delete(session, [user1, user2])

# 按条件批量删除
count = await User.delete(session, condition=User.is_active == False)
```

返回值为删除的记录数。

## `get()` — 查询

万能查询方法，通过不同参数组合覆盖所有查询场景。

### 基本查询

```python
# 按条件查第一条
user = await User.get(session, User.email == "alice@example.com")

# 查所有
users = await User.get(session, fetch_mode="all")

# 精确查一条（0 条或多条都报错）
user = await User.get(session, User.id == some_id, fetch_mode="one")
```

### `fetch_mode` 返回值

| `fetch_mode` | 返回类型 | 0 条时 | 多条时 |
|---|---|---|---|
| `"first"`（默认） | `T \| None` | `None` | 返回第一条 |
| `"one"` | `T` | 抛异常 | 抛异常 |
| `"all"` | `list[T]` | 空列表 | 全部返回 |

### 分页与排序

```python
from sqlmodel_ext import TableViewRequest

tv = TableViewRequest(offset=0, limit=20, desc=True, order="created_at")
users = await User.get(session, fetch_mode="all", table_view=tv)
# → SELECT ... ORDER BY created_at DESC LIMIT 20 OFFSET 0
```

### 时间过滤

```python
users = await User.get(
    session,
    fetch_mode="all",
    created_after=datetime(2024, 1, 1),
    created_before=datetime(2024, 12, 31),
)
```

### 关系预加载

```python
user = await User.get(
    session,
    User.id == user_id,
    load=[User.profile, Profile.avatar],
)
# 自动构建: selectinload(User.profile).selectinload(Profile.avatar)
```

### 其他参数

| 参数 | 作用 |
|------|------|
| `join` | JOIN 另一张表 |
| `options` | 自定义 SQLAlchemy query options |
| `filter` | 额外的 WHERE 条件 |
| `with_for_update` | SELECT ... FOR UPDATE（行锁） |

## `count()` — 计数

```python
total = await User.count(session)
active = await User.count(session, User.is_active == True)
```

## `get_with_count()` — 分页列表

`count()` + `get(fetch_mode="all")` 的组合，返回 `ListResponse`：

```python
result = await User.get_with_count(session, table_view=table_view)
# result.count = 42
# result.items = [User(...), User(...), ...]
```

## `get_one()` — 保证存在的查询

类似 `get(fetch_mode="one")`，但接口更简洁——直接传 ID：

```python
user = await User.get_one(session, user_id)
# 记录不存在 → NoResultFound
# 多条记录 → MultipleResultsFound

# 带锁查询
user = await User.get_one(session, user_id, with_for_update=True)
```

与 `get_exist_one()` 的区别：`get_one()` 抛 SQLAlchemy 异常，`get_exist_one()` 抛 HTTP 404。

## `get_exist_one()` — 查找或 404

```python
user = await User.get_exist_one(session, user_id)
# 找不到自动抛 HTTPException(404)（有 FastAPI 时）
# 或 RecordNotFoundError（无 FastAPI 时）

# 带关系预加载
user = await User.get_exist_one(session, user_id, load=User.profile)
```

## 方法速查表

| 方法 | 类型 | 对应 SQL | 返回值 |
|------|------|---------|--------|
| `add()` | 类方法 | INSERT | 实例或列表 |
| `save()` | 实例方法 | INSERT 或 UPDATE | 刷新后的实例 |
| `update()` | 实例方法 | UPDATE（部分字段） | 刷新后的实例 |
| `delete()` | 类方法 | DELETE | 删除数量 |
| `get()` | 类方法 | SELECT + WHERE + ... | 实例 / 列表 / None |
| `get_one()` | 类方法 | SELECT + 保证存在 | 实例 |
| `count()` | 类方法 | SELECT COUNT(*) | int |
| `get_with_count()` | 类方法 | COUNT + SELECT | ListResponse |
| `get_exist_one()` | 类方法 | SELECT + 404 | 实例 |
