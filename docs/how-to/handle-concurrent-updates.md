# 处理并发更新

**目标**：防止两个并发操作互相覆盖修改（"丢失更新"问题），让冲突可识别、可自动重试。

**前置条件**：

- 你的模型记录会被多个用户/进程同时修改
- 你能接受少量重试开销（高频写入的场景不适用——见底部）

## 1. 给模型加 `OptimisticLockMixin`

```python
from enum import StrEnum

from sqlmodel_ext import OptimisticLockMixin, SQLModelBase, UUIDTableBaseMixin, NonNegativeDecimal38_18

class OrderStatusEnum(StrEnum):
    pending = 'pending'
    paid = 'paid'

class OrderBase(SQLModelBase):
    status: OrderStatusEnum = OrderStatusEnum.pending
    amount: NonNegativeDecimal38_18

class Order(OptimisticLockMixin, OrderBase, UUIDTableBaseMixin, table=True): # [!code highlight]
    pass
```

::: warning MRO 顺序
`OptimisticLockMixin` **必须**放在 `UUIDTableBaseMixin` / `TableBaseMixin` 之前。
:::

混入后自动获得 `oplock_version` 列（`BIGINT`，每次 UPDATE 自动递增）。它是 ORM 内部状态，**不会出现在 `model_dump()` 里**，也不能被模型自己重新声明。领域上的"版本"字段请另起名字（`version`、`revision`）。

## 2. 什么都不用写：冲突默认自动重试 3 次

```python
order = await order.save(session)
# 冲突时自动：rollback → 重读最新行 → 只重放你改过的列 → 再 commit，最多 3 次

order = await order.update(session, patch)
# 同样默认重试 3 次：重读最新行，再应用 patch 中的改动
```

重试次数是模型策略 `__optimistic_retry_default__`（`OptimisticLockMixin` 上为 `3`），调用点不用记着传参数。

**重试时发生了什么**（以 `save()` 为例）：

1. commit → `StaleDataError`（`WHERE oplock_version = ?` 命中 0 行）
2. rollback
3. 按 SQLAlchemy 属性历史收集**你实际修改过的列**（排除 `id` / `oplock_version` / `created_at` / `updated_at`）
4. `cls.get(session, cls.id == ...)` 读最新记录
5. 把你的修改逐列 `setattr` 到最新记录上——别人刚提交的其它列保持不动
6. 再次 commit → 成功（或继续重试）

效果：两个人同时编辑同一订单的不同字段，两边的修改都会保留，客户端完全无感。

```python
# 两个 session 同时读到同一行
a = await Order.get_one(s1, order_id)
b = await Order.get_one(s2, order_id)

a.status = OrderStatusEnum.paid
a = await a.save(s1)             # oplock_version +1

b.amount = Decimal('200')
b = await b.save(s2)             # 冲突 → 自动重试 → 成功
assert b.status == OrderStatusEnum.paid and b.amount == Decimal('200')
```

## 3. 需要自己处理冲突时：显式传 `0`

```python
from sqlmodel_ext import OptimisticLockError

try:
    order = await order.save(session, optimistic_retry_count=0)   # 显式要求不重试
except OptimisticLockError as e:
    logger.warning(
        f"乐观锁冲突: model={e.model_class} id={e.record_id} "
        f"version={e.expected_version}"
    )
    raise HTTPException(status_code=409, detail="数据已被其他人修改，请刷新后重试")
```

同样的 `except` 也用来处理**重试耗尽**的情况（默认策略下连续 3 次冲突后才抛出），以及重试时发现记录已被删除的情况。

## 4. 删除时的冲突

带版本的模型删除时也会检查版本：

```python
try:
    await Order.delete(session, order)
except OptimisticLockError:
    # 有人在你读取之后改过（或删过）这一行——要不要继续删，由你决定
    await session.rollback()
    ...
```

`delete()` **从不重试**（"它刚被改过，我还要删吗？"是业务决定），并且 `record_id` / `expected_version` 恒为 `None`：冲突在 flush 级无法可靠归属到某一行。

## 选择 `optimistic_retry_count` 的值

| 值 | 适用场景 |
|---|---------|
| 不传（`None`） | 绝大多数情况。使用模型策略：乐观锁模型重试 3 次，其余模型 0 次 |
| `0` | 你想自己处理冲突（捕获 `OptimisticLockError`，返回 409 让用户刷新） |
| `> 5` | 不推荐。重试次数过多说明该资源争用太严重，应考虑行锁（见 [强制行锁与隔离级别](./enforce-locking-and-isolation)）、消息队列或 CRDT |

## 不适用的场景

| 场景 | 为什么 | 应该用什么 |
|------|--------|----------|
| 日志/审计表 | 只插入不更新 | 直接 INSERT |
| 简单计数器 | 高频争用 | `UPDATE table SET count = count + 1` 原子操作 |
| 高频写入（每秒上千次） | 冲突太多，重试成本高 | `get(with_for_update=True)` 行锁 + 队列、或 CRDT 数据结构 |

## 相关参考

- [`OptimisticLockMixin` 完整字段](/reference/mixins#optimisticlockmixin)
- [`OptimisticLockError` 异常字段](/reference/mixins#optimisticlockerror)
- [乐观锁机制讲解](/explanation/optimistic-lock)（讲为什么这么设计）
