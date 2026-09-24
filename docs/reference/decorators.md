# 装饰器与辅助函数

::: tip
本页是参考文档。要看怎么用 `@requires_relations` 解决 MissingGreenlet，去 [防止 MissingGreenlet 错误](/how-to/prevent-missing-greenlet)；要看锁与隔离级别契约怎么组合，去 [强制行锁与隔离级别](/how-to/enforce-locking-and-isolation)。
:::

## `@requires_relations`

```python
from sqlmodel_ext import requires_relations
```

```python
def requires_relations(
    *relations: str | QueryableAttribute[Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]
```

| 参数 | 类型 | 含义 |
|------|------|------|
| `*relations` | `str` | 本类直接关系属性名（如 `'profile'`） |
| `*relations` | `QueryableAttribute` | 嵌套关系（如 `Generator.config`） |

**前置条件**：

- 装饰的类必须继承 `RelationPreloadMixin`
- 装饰的方法必须是 `async def`（普通协程）或 `async def ... yield`（异步生成器）
- 方法的某个参数名为 `session`，或某个 kwarg 是 `AsyncSession` 类型

**运行时行为**：

1. 从参数中提取 `AsyncSession`
2. 调用 `self.ensure_relations_loaded(session, relations)` 增量加载缺失的关系
3. 执行原方法

找不到 session 时**跳过**预加载（不报错）——这种方法的调用方必须自己先调用 `ensure_relations_loaded()`。

**导入时验证**：`RelationPreloadMixin.__init_subclass__` 在类定义时检查字符串关系名；不存在则 `AttributeError`。

**附加属性**：`_required_relations` 元组。

## 事务契约装饰器

下面四个装饰器都是 **fail-closed** 的：只要守卫找不到 session（例如内层装饰器没用 `@wraps`，`inspect.signature` 看不到 `session` 参数），就直接抛错，而不是悄悄放行。

| 装饰器 | 契约 | 检查方式 |
|------|------|------|
| `@requires_for_update` | `self` 必须通过 `get(with_for_update=True)` 获得 | `id(self)` 是否在 `session.info[SESSION_FOR_UPDATE_KEY]` |
| `@requires_locked_param(name)` | 参数 `name` 中的实例都必须已加 FOR UPDATE 锁 | 同上，逐个检查 |
| `@requires_repeatable_read` | 必须运行在**已验证**的 REPEATABLE READ 事务里 | `session.info` 中的 RR 标记（由 `enter_repeatable_read()` 读回确认后写入） |
| `@requires_read_committed` | 必须运行在 READ COMMITTED 事务里 | 每次调用实时 `SHOW transaction_isolation`（仅 PostgreSQL） |

### `@requires_for_update`

```python
from sqlmodel_ext import requires_for_update
```

```python
def requires_for_update(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[Any, Any, R]]
```

用 `ParamSpec` 声明，**保留被装饰方法的签名**——类型检查器继续校验调用参数。

**运行时行为**：从参数提取 session（找不到 → `RuntimeError`）；`id(self)` 不在锁集合 → `RuntimeError`；否则执行原方法。

**附加属性**：`_requires_for_update = True`（供静态分析使用）。

锁集合的生命周期：最外层 commit / rollback 清空（行锁随事务释放）；savepoint 回滚恢复进入 savepoint 时的快照（PostgreSQL 会释放 savepoint 内取得的锁）；savepoint 释放保留（锁转移给外层事务）；增强 session 的 `reset()` / `close()` 也会清空。

### `@requires_locked_param`

```python
from sqlmodel_ext.mixins import requires_locked_param
```

```python
def requires_locked_param(
    param_name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]
```

`@requires_for_update` 的参数化版本，用于 classmethod、`@asynccontextmanager` 工厂等操作**一批**实例的可调用对象。

- 参数值为 `None` → 跳过检查（例如按条件处理的分支）
- 单个实例 → 规范化为单元素序列
- 空序列 → `RuntimeError`（声称"这些实例已加锁"却一个都没传，违反契约）
- session 总是取自被装饰函数的 `session` 参数；没有该参数 → `RuntimeError`
- 支持两种形态：协程函数；同步工厂（如 `@asynccontextmanager` 产物——检查在**创建**上下文管理器时执行，即进入之前）

**附加属性**：`_requires_locked_param = param_name`。

### `validate_locked_instances()`

```python
from sqlmodel_ext.mixins import validate_locked_instances
```

```python
def validate_locked_instances(
    session: AsyncSession,
    instances: Sequence[Any],
    *,
    context: str,
) -> None
```

`@requires_locked_param` 的校验主体，单独导出用于**执行中途**的再校验（装饰器只能在入口检查参数，看不到例如回调返回的实例）。空序列或任一实例未加锁 → `RuntimeError`，消息以 `context` 开头。

### `@requires_repeatable_read`

```python
from sqlmodel_ext.mixins import requires_repeatable_read
```

```python
def requires_repeatable_read(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[Any, Any, R]]
```

用于跨多条语句读取同一批源数据、需要事务级快照的编排（例如深拷贝依次读节点、文件、边）。session 通过 `Signature.bind` 定位，实例方法、classmethod、普通函数都支持。

- 没有 RR 标记 → `RuntimeError`，提示从最外层入口用 `SessionFactory.run_in_repeatable_read(...)` 编排
- **不内置重试**：序列化失败（SQLSTATE `40001`）需要丢弃整个 session，被装饰的方法做不到；重试归最外层入口

**附加属性**：`_requires_repeatable_read = True`。

### `@requires_read_committed`

```python
from sqlmodel_ext.mixins import requires_read_committed
```

```python
def requires_read_committed(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[Any, Any, R]]
```

用于依赖"每条语句一个新快照"来看到并发提交的跨进程协调（例如重新检查交接窗口）。

**刻意不读** `session.info` 标记：没有 RR 标记也可能是 SERIALIZABLE，或在连接 / 引擎上直接设置了隔离级别——信任标记会 fail-open。每次调用都用 `SHOW transaction_isolation` 读回（一次廉价往返），不是 `'read committed'` → `RuntimeError`。**仅 PostgreSQL**。

装饰 classmethod 时把它放在 `@classmethod` 下面（最内层）。

**附加属性**：`_requires_read_committed = True`。

## `rel()`

```python
from sqlmodel_ext import rel
```

```python
def rel(relationship: object) -> QueryableAttribute[Any]
```

把 SQLModel 的 `Relationship` 字段类型断言为 `QueryableAttribute`，让 basedpyright 不报类型错误。输入是 `QueryableAttribute` → 返回原对象；否则 → `AttributeError`（例如误传了实例属性）。

**典型用法**：`load=rel(User.profile)`、`load=[rel(User.profile), rel(Profile.avatar)]`。

## `cond()`

```python
from sqlmodel_ext import cond
```

```python
def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]
```

把列比较表达式（basedpyright 推断为 `bool`）窄化为 `ColumnElement[bool]`，让 `&` / `|` 运算符通过类型检查。运行时等价于 `cast(ColumnElement[bool], expr)`。

```python
scope = cond(UserFile.user_id == current_user.id)
condition = scope & cond(UserFile.status == FileStatusEnum.uploaded)
```

## 增强 `AsyncSession` {#session-reset}

```python
from sqlmodel_ext import AsyncSession
from sqlmodel_ext.session import SessionFactory
```

`sqlmodel_ext.AsyncSession` 是 sqlmodel `AsyncSession` 的子类，也是本库的规范 session 类型。用 `SessionFactory(engine, class_=AsyncSession)`（需要 `run_in_repeatable_read` 时）或 `async_sessionmaker(engine, class_=AsyncSession)` 构造。不涉及缓存模型时所有钩子都退化为上游行为。

| 成员 | 说明 |
|------|------|
| `commit(*, fail_soft_when_observed=False)` | 缓存模型的变更（包括裸 `session.add()` / 改属性 / `session.delete()`）由写出它们的那次 flush 登记——事务中更早的 flush 或本次 commit 的 flush，commit 后**同步**失效，再按顺序执行 post-commit 回调。`fail_soft_when_observed=True`：commit 本身失败照常抛出；commit **之后**的每一步（失效、每个回调）都单独容错（捕获包括取消在内的 `BaseException`，记日志，继续）。默认：失效步骤本身抛出的错误与取消会传播；回调的 `Exception` 记日志后跳过。**两种模式下，失效过程中的 Redis 故障都只记日志、不抛出**——数据库已经提交，受影响的缓存条目在 TTL 到期前会返回提交前的数据；不能依赖缓存的读取请用 `no_cache=True` |
| `commit_count`（属性） | 本 session 成功 commit 的次数。"是否提交了"的唯一真相源：关键写之前记下基线，之后 `commit_count > baseline` 即数据库已提交（充分非必要信号） |
| `add_post_commit_callback(callback)` | 注册一个 `async` 无参回调，**只在下一次真正 commit 之后**执行；`rollback()` / `reset()` / `close()` 丢弃；在 savepoint 内注册 → `RuntimeError`；回调失败不阻塞后续回调，也不改变 commit 结果 |
| `rollback(*, best_effort_budget_seconds=None)` | 回滚 + 丢弃待执行回调。传入预算时进入**有界放弃**模式：在 `asyncio.timeout` 下回滚，失败或超时就尽力 `invalidate()` 连接，**不抛异常**（`CancelledError` 仍传播）。不是硬上限（`invalidate` 可能仍在驱动里等待） |
| `begin()` | `async with session.begin():` 的退出走增强 `commit()` / `rollback()`（原生上下文管理器会直接提交底层事务，跳过失效与回调）；`await session.begin()` 返回 session 本身。`begin_nested()` 不受影响 |
| `reset()` / `close()` | 释放事务与连接，并在 `finally` 中清空 FOR UPDATE 锁跟踪、待执行回调、REPEATABLE READ 标记与缓存跟踪状态 |
| `refresh(instance, attribute_names=None, with_for_update=None)` | 委托原生 `refresh()`：必须读数据库，**永不走缓存** |
| `execute` / `exec` / `scalar` / `stream` / `stream_scalars` | 透传前先调用 `CachedTableBaseMixin.register_raw_dml_write()` 登记原生 DML 写入的表 |
| `set_local_timeouts(*, lock_timeout_ms, statement_timeout_ms)` | 用 `set_config(..., is_local=True)`（等价 `SET LOCAL`）把两个超时钉在**当前事务**上，事务结束自动还原。仅 PostgreSQL |
| `enter_repeatable_read()` | 把本 session 的事务提升到 REPEATABLE READ，**读回验证**后在 `session.info` 记标记。必须是新 session 的第一个数据库动作（否则 SQLAlchemy 只发 `SAWarning` 并静默保持原级别——这里会读回并抛 `RuntimeError`）。不内置重试。仅 PostgreSQL |

### `session.reset()` 之后的对象状态

典型场景：端点 / 任务中途要做长时间外部 I/O，先 `session.reset()` 释放 DB 连接。详见 [长 I/O 期间释放数据库连接](/how-to/release-connection-during-long-io)。

- 所有 ORM 对象进入 **detached** 状态
- **已加载的 scalar 字段不会被 expire**，访问安全（不触发 SQL）；之后的 commit 也无法再让它们过期
- 未预加载的关系字段访问会抛错
- 写操作需要先用 `Model.get()` 重查拿 attached 实例
- 后续任何 `await Model.get/save` 会自动从池 checkout 新连接

## `SessionFactory`

```python
from sqlmodel_ext.session import (
    SessionFactory,
    RepeatableReadSnapshotConflictError,
    SerializationRetryExhaustedError,
    MAX_REPEATABLE_READ_ATTEMPTS,
)
```

`async_sessionmaker[AsyncSession]` 的子类，可传给任何期待 `async_sessionmaker[AsyncSession]` 的地方。

```python
async def run_in_repeatable_read(
    self,
    operation: Callable[[AsyncSession], Awaitable[T]],
    *,
    description: str,
    max_attempts: int = MAX_REPEATABLE_READ_ATTEMPTS,   # 3
) -> T
```

每次尝试：新 session → `enter_repeatable_read()` → `operation(session)` → 检查已提交 → 返回。遇到 SQLSTATE `40001` 或 `RepeatableReadSnapshotConflictError` 时回滚、**丢弃 session**、从头再跑；连续 `max_attempts` 次冲突 → `SerializationRetryExhaustedError`（`status_code = 409`）。仅 PostgreSQL。

`operation` 的契约：

1. **必须自己 commit**——没 commit 就返回会抛 `RuntimeError`，而不是返回一个从未持久化的"成功"结果
2. **必须可重跑**——只捕获不可变输入（id、DTO、标量），不要捕获上一次尝试的 ORM 实例 / session / 生成的 id
3. **授权也放在里面**——重试是在新快照上重跑整个业务动作
4. 返回值不要是绑定到 session 的 ORM 实例（方法返回时 session 已关闭），在 `operation` 内构造 DTO

已经 commit 之后才出现的冲突不会重跑（重跑会重复一个已持久化的业务动作），原样抛出。死锁（`40P01`）刻意**不**重试——一致的锁顺序本应让它不可能发生。

`RepeatableReadSnapshotConflictError`：领域代码在确认"session 处于 REPEATABLE READ + 唯一约束冲突 + 胜者在快照中不可见"后抛出的控制流信号，`run_in_repeatable_read` 对它的处理与 `40001` 完全相同。

## 常量

```python
from sqlmodel_ext import SESSION_FOR_UPDATE_KEY
from sqlmodel_ext.mixins import SESSION_REPEATABLE_READ_KEY
```

| 常量 | 值 | 说明 |
|------|------|------|
| `SESSION_FOR_UPDATE_KEY` | `'_for_update_locked'` | `session.info` 中 FOR UPDATE 锁定实例 `id()` 集合的键 |
| `SESSION_REPEATABLE_READ_KEY` | `'_repeatable_read_verified'` | `session.info` 中"隔离级别已**验证**为 REPEATABLE READ"的标记键 |
