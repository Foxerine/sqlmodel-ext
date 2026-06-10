# 装饰器与辅助函数

::: tip
本页是参考文档。要看怎么用 `@requires_relations` 解决 MissingGreenlet，去 [防止 MissingGreenlet 错误](/how-to/prevent-missing-greenlet)。
:::

## `@requires_relations`

```python
from sqlmodel_ext import requires_relations
```

**签名**：

```python
def requires_relations(
    *relations: str | QueryableAttribute[Any],
) -> Callable[[F], F]
```

**参数**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `*relations` | `str` | 本类直接关系属性名（如 `'profile'`） |
| `*relations` | `QueryableAttribute` | 嵌套关系（如 `Generator.config`） |

**前置条件**：

- 装饰的类必须继承 `RelationPreloadMixin`
- 装饰的方法必须是 `async def`（普通协程）或 `async def ... yield`（异步生成器）
- 方法的某个参数名为 `session`，或某个 kwarg 是 `AsyncSession` 类型

**运行时行为**：

1. 自动从参数中提取 `AsyncSession`
2. 调用 `self._ensure_relations_loaded(session, relations)` 加载缺失的关系
3. 已加载的关系不重复查询（增量加载）
4. 嵌套关系自动解析中间路径
5. 执行原方法

**导入时验证**：`RelationPreloadMixin.__init_subclass__` 在类定义时检查 `relations` 中的字符串名是否存在于类属性或 SQLModel relationships 中；不存在则 `AttributeError`。

**附加属性**：装饰后函数对象上会有 `_required_relations` 元组，存储声明信息。

## `@requires_for_update`

```python
from sqlmodel_ext import requires_for_update
```

**签名**：

```python
def requires_for_update(func: F) -> F
```

**前置条件**：

- 装饰的类必须继承 `RelationPreloadMixin`
- 调用方必须先用 `cls.get(session, ..., with_for_update=True)` 获取实例

**运行时行为**：

1. 从参数中提取 `AsyncSession`
2. 检查 `session.info[SESSION_FOR_UPDATE_KEY]` 是否包含 `id(self)`
3. 不包含 → `RuntimeError`
4. 包含 → 执行原方法

**附加属性**：装饰后函数对象上会有 `_requires_for_update = True`。

## `rel()`

```python
from sqlmodel_ext import rel
```

**签名**：

```python
def rel(relationship: object) -> QueryableAttribute[Any]
```

**用途**：把 SQLModel 的 `Relationship` 字段类型断言为 `QueryableAttribute`，让 basedpyright 不报类型错误。

**运行时行为**：

- 输入是 `QueryableAttribute` → 返回原对象
- 否则 → `AttributeError`

**典型用法**：`load=rel(User.profile)`、`load=[rel(User.profile), rel(Profile.avatar)]`。

## `cond()`

```python
from sqlmodel_ext import cond
```

**签名**：

```python
def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]
```

**用途**：把列比较表达式（basedpyright 推断为 `bool`）窄化为 `ColumnElement[bool]`，让 `&` / `|` 运算符不报类型错误。

**运行时行为**：等价于 `cast(ColumnElement[bool], expr)`，无任何检查。

**典型用法**：

```python
scope = cond(UserFile.user_id == current_user.id)
condition = scope & cond(UserFile.status == FileStatusEnum.uploaded)
```

## `session.reset()`（增强 AsyncSession）{#session-reset}

```python
from sqlmodel_ext import AsyncSession
```

**签名**（增强方法，定义于 `sqlmodel_ext.session.AsyncSession`）：

```python
async def reset(self) -> None
```

> 0.4.0 起替代已删除的 `safe_reset()` 辅助函数。用 `async_sessionmaker(class_=AsyncSession)` 构造 session 后直接 `await session.reset()` 即可。

**用途**：调用上游 `reset()`（释放事务与连接）后，自动清理 `session.info[SESSION_FOR_UPDATE_KEY]` 中跟踪的 FOR UPDATE 锁以及缓存失效跟踪状态——避免跟踪集合泄漏到下一次 session 复用周期。

**典型场景**：HTTP 端点 / Taskiq 任务在中途需要做长时间外部 I/O（S3、ffprobe、第三方 HTTP 轮询等），调用前先 `session.reset()` 释放 DB 连接，避免连接被外部网络阻塞拖死池。详见 [长 I/O 期间释放数据库连接](/how-to/release-connection-during-long-io)。

**调用后对象状态**：

- 所有 ORM 对象进入 **detached** 状态（`sa_inspect(obj).detached == True`）
- 但**已加载的 scalar 字段不会被 expire** —— 仍在 `obj.__dict__` 中，访问安全（不触发 SQL，不抛 `MissingGreenlet`）
- 未预加载的关系字段访问会抛 `InvalidRequestError`（lazy load 在 detached 上失败）
- 写操作（save / update / delete）会失败 —— 需要先用 `Model.get()` 重查拿 attached 实例
- 后续任何 `await Model.get/save` 触发 SQL 时会自动从池 checkout 新连接

## `sanitize_integrity_error()`

```python
TableBaseMixin.sanitize_integrity_error(
    e: IntegrityError,
    default_message: str = "Data integrity constraint violation",
) -> str
```

`TableBaseMixin` 的静态方法。从 `IntegrityError` 中提取用户安全的错误消息。

**行为**：

- SQLSTATE `23514`（`check_violation`）：取错误消息第一行，去除 `ERROR:` 前缀，返回（PostgreSQL 触发器抛出的业务消息可以安全展示给用户）
- 其他约束错误（FK、唯一约束等）：返回 `default_message`（避免泄露表结构信息）

## 常量

```python
from sqlmodel_ext import SESSION_FOR_UPDATE_KEY
```

**`SESSION_FOR_UPDATE_KEY`**：`'_for_update_locked'` 字符串。`get(with_for_update=True)` 用它在 `session.info` 中跟踪锁定的实例 `id()`。`@requires_for_update` 读取这个键做运行时检查。
