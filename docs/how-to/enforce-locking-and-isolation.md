# 强制行锁与隔离级别

**目标**：把"这个方法必须在持有行锁 / 特定隔离级别下调用"从注释里的约定变成**运行时强制**的契约；把"只有提交成功后才能做的副作用"挂到 commit 之后；给事务里的锁等待和语句加上上限。

**前置条件**：

- session 由增强版 `sqlmodel_ext.AsyncSession` 构造：`SessionFactory(engine, class_=AsyncSession)`
- 需要 `run_in_repeatable_read` / `requires_read_committed` / `set_local_timeouts` 的部分：PostgreSQL

所有契约装饰器都是 **fail-closed** 的：守卫找不到 session 时直接抛错，而不是悄悄放行——"守卫失效"必须和"守卫通过"看起来不一样。

## 1. 读-改-写：`with_for_update`

```python
account = await Account.get(session, Account.id == account_id, with_for_update=True)
account.balance += amount
account = await account.save(session)
```

- 生成 `SELECT ... FOR UPDATE`，行锁持有到事务结束。
- **强制 `populate_existing`**（无法关闭）：数据库返回最新行，但如果该对象已经在 identity map 里，SQLAlchemy 默认会把**旧对象**还给你——接下来的读-改-写就是一次丢失更新。加锁读总是用数据库值覆盖它。
- 缓存模型上加锁读自动绕过 Redis。
- `get_one(..., with_for_update=True)` 与 `get_exist_one(..., with_for_update=True)` 同样可用；后者常用来关闭"存在检查 → 删除"之间的 TOCTOU 窗口。

### 工作队列：`skip_locked`

N 个 worker 各自领取**不同**的候选行，而不是在同一行上排队：

```python
job = await Job.get(
    session,
    Job.status == 'queued',
    order_by=[col(Job.created_at)],
    with_for_update=True,
    skip_locked=True,     # FOR UPDATE SKIP LOCKED
)
if job is None:
    return   # 可能是没有任务，也可能是任务都被别的 worker 锁住了
```

::: warning 不要用 `skip_locked` 做存在性判断
返回 0 行同时意味着"没有"和"都被别人锁了"。`skip_locked` 只在 `with_for_update=True` 时生效。
:::

### 授权读取：`authoritative`

结果要用来决定"是否允许某件事"时，必须读权威值，不能读 identity map 里可能过期的对象，也不能读 Redis 缓存：

```python
member = await Membership.get_one(session, membership_id, authoritative=True)
if member.role != 'admin':
    raise HTTPException(403)
```

`authoritative=True` 在本层等价于 `populate_existing=True`；缓存模型额外绕过 Redis。调用方只传这一个参数，每个模型自己知道要绕过几层。它不加锁——需要锁时用 `with_for_update`。

## 2. 要求 `self` 已加锁：`@requires_for_update`

```python
from decimal import Decimal
from sqlmodel_ext import RelationPreloadMixin, requires_for_update, SignedDecimal38_18
from sqlmodel_ext.session import AsyncSession

class Account(SQLModelBase, UUIDTableBaseMixin, RelationPreloadMixin, table=True):
    balance: SignedDecimal38_18 = Decimal('0')

    @requires_for_update
    async def adjust_balance(self, session: AsyncSession, *, amount: Decimal) -> None:
        self.balance += amount
        await self.save(session, commit=False, refresh=False)
```

```python
account = await Account.get_one(session, account_id)
await account.adjust_balance(session, amount=Decimal('5'))   # RuntimeError: requires a FOR UPDATE locked instance

account = await Account.get_one(session, account_id, with_for_update=True)
await account.adjust_balance(session, amount=Decimal('5'))   # OK
await session.commit()
```

锁跟踪随事务走：最外层 commit / rollback 后清空（锁已释放）；savepoint 回滚恢复进入 savepoint 时的集合（PostgreSQL 会释放 savepoint 内取得的锁）；savepoint 释放保留；`reset()` / `close()` 清空。所以 commit 之后再调用 `adjust_balance` 会再次失败——这正是你想要的。

装饰器用 `ParamSpec` 声明，被装饰方法的签名对类型检查器保持可见。

## 3. 要求参数里的实例已加锁：`@requires_locked_param`

`@requires_for_update` 只能表达"`self` 已加锁"。对操作**一批**实例的 classmethod / 普通函数 / `@asynccontextmanager` 工厂，用参数化版本：

```python
from sqlmodel_ext.mixins import requires_locked_param

class Account(...):
    @classmethod
    @requires_locked_param('accounts')
    async def settle(cls, session: AsyncSession, accounts: list['Account']) -> None:
        for account in accounts:
            account.balance = Decimal('0')
        await session.flush()
```

```python
accounts = await Account.get(session, Account.owner_id == owner_id, fetch_mode='all', with_for_update=True)
await Account.settle(session, accounts)      # OK
```

- 参数为 `None` → 跳过检查；单个实例 → 当作单元素序列；**空序列 → `RuntimeError`**
- session 总是取自被装饰函数的 `session` 参数，没有这个参数 → `RuntimeError`
- 装饰 `@asynccontextmanager` 工厂时，检查在**创建**上下文管理器时执行（进入之前）

执行中途拿到的实例（例如回调返回的）用同一个校验主体再检查一次：

```python
from sqlmodel_ext.mixins import validate_locked_instances

validate_locked_instances(session, returned_accounts, context="batch settle")
```

## 4. 隔离级别契约

### 需要一致快照：REPEATABLE READ

跨多条语句读取同一批源数据（例如深拷贝依次读节点、文件、边）时，READ COMMITTED 下每条语句都是新快照，并发提交会让几次读互相不一致。

```python
from sqlmodel_ext.mixins import requires_repeatable_read

class Board(...):
    @requires_repeatable_read
    async def clone(self, session: AsyncSession) -> 'BoardCloneResult':
        ...  # 所有读取都看到同一个事务快照
```

装饰器只检查 `session.info` 里的 RR 标记，而这个标记**只由** `enter_repeatable_read()` 在从数据库**读回**确认级别之后写入（`reset()` / `close()` 时清除）——"我请求过"永远不会被记成"已经生效"。进入 RR 的正确方式是在最外层入口：

```python
from sqlmodel_ext.session import SessionFactory

session_factory = SessionFactory(engine, class_=AsyncSession)

async def _clone(session: AsyncSession) -> BoardCloneResult:
    board = await Board.get_one(session, board_id)          # 只捕获不可变输入（board_id）
    result = await board.clone(session)
    await session.commit()                                   # 必须自己 commit
    return result                                            # 返回 DTO，不是 ORM 实例

result = await session_factory.run_in_repeatable_read(_clone, description="clone board")
```

`run_in_repeatable_read` 每次尝试都开**新 session**、调用 `enter_repeatable_read()`、执行 `operation`；遇到 SQLSTATE `40001` 或 `RepeatableReadSnapshotConflictError` 时回滚、丢弃 session、从头重跑，连续 `max_attempts`（默认 3）次冲突后抛 `SerializationRetryExhaustedError`（`status_code = 409`）。`operation` 没 commit 就返回 → `RuntimeError`（否则你会拿到一个从未持久化的"成功"结果）。已经 commit 之后才出现的冲突不重跑。死锁 `40P01` 刻意不重试——一致的锁顺序本应让它不可能发生，重试只会把锁顺序 bug 伪装成"偶尔变慢"。

::: tip 唯一约束冲突里的"看不见的胜者"
REPEATABLE READ 下，唯一约束检查的是**当前已提交**状态，而快照只看得到事务开始前的提交。并发事务在你的快照之后提交了同一个键时，你的 INSERT 撞上约束（证明胜者存在），但本事务里任何 SELECT 都看不见它。这时在领域代码里抛 `RepeatableReadSnapshotConflictError`，`run_in_repeatable_read` 会像对待 `40001` 一样换一个新快照重跑。
:::

直接调用 `enter_repeatable_read()` 时它必须是新 session 的**第一个**数据库动作；否则 SQLAlchemy 只发 `SAWarning` 并静默保持原级别——这里会读回并抛 `RuntimeError`，把静默的正确性损失变成响亮的失败。

### 需要看见并发提交：READ COMMITTED

依赖"每条语句一个新快照"来看到其它进程刚提交的行（例如重新检查交接窗口）：

```python
from sqlmodel_ext.mixins import requires_read_committed

class Handoff(...):
    @classmethod
    @requires_read_committed          # 放在 @classmethod 下面（最内层）
    async def load_ready(cls, session: AsyncSession) -> list['Handoff']:
        ...
```

它**每次调用都实时** `SHOW transaction_isolation`，而不看 `session.info` 标记——"没有 RR 标记"也可能是 SERIALIZABLE，或在连接 / 引擎上直接改过隔离级别，信任标记会 fail-open。仅 PostgreSQL。

## 5. 只在提交后执行的副作用：post-commit 回调

删除数据库行之后要删除对象存储里的文件——如果内联 `await` 删除文件，而事务之后回滚了，文件就没了、行却还在。把不可逆的外部副作用挂到 commit 之后：

```python
async def delete_blob() -> None:
    await storage.delete(blob_key)          # 捕获不可变值，不捕获 ORM 实例

session.add_post_commit_callback(delete_blob)
await Attachment.delete(session, attachment)    # commit 成功后才执行 delete_blob
```

- 下一次**真正的** commit 之后按注册顺序执行，然后清空队列
- `rollback()` / `reset()` / `close()` 丢弃未执行的回调
- 在 savepoint 里注册 → `RuntimeError`（savepoint 回滚无法精确撤销它）
- 回调失败只记日志，不阻塞后续回调，也不改变 commit 结果
- `async with session.begin():` 的退出同样走增强 `commit()`，回调照常执行

需要"数据库一定已经提交"的信号时看 `commit_count`，不要看 `in_transaction()`（commit 后的重新查询会打开一个新的读事务）：

```python
baseline = session.commit_count
await do_critical_write(session)
if session.commit_count > baseline:
    ...   # 数据库已提交
```

最后一段副作用必须"尽力而为、一个都不能丢"时，用 `await session.commit(fail_soft_when_observed=True)`：commit 本身失败照常抛出（结果不确定，调用方按"可能已提交"处理）；commit **之后**的缓存失效与每个回调都单独容错，连取消也不会让剩下已弹出的回调丢失。

## 6. 给事务设上限

### 锁等待与语句超时

```python
await session.set_local_timeouts(lock_timeout_ms=2_000, statement_timeout_ms=10_000)
```

用 `set_config(..., is_local=True)`（等价 `SET LOCAL`）钉在**当前事务**上，事务结束自动还原。在事务开始处调用一次（rollback 开启了新事务后要再调用）。仅 PostgreSQL。

### 有界回滚

清理路径（例如取消处理、关停）里的回滚本身可能卡住。给它一个预算：

```python
await session.rollback(best_effort_budget_seconds=5.0)
```

在 `asyncio.timeout` 下回滚；失败或超时就尽力 `invalidate()` 连接，**不抛异常**（`CancelledError` 仍传播）。这不是硬上限——`invalidate()` 仍可能在驱动里等待；残余风险是行锁一直持有到连接最终断开。默认（不传）是普通回滚，错误照常传播。

## 相关参考

- [装饰器与增强 session 参考](/reference/decorators)
- [`get()` 的 `with_for_update` / `skip_locked` / `authoritative`](/reference/crud-methods#get)
- [处理并发更新（乐观锁）](./handle-concurrent-updates)
