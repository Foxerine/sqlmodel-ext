# 处理并发更新

**目标**：防止两个并发操作互相覆盖修改（"丢失更新"问题），让冲突可识别可重试。

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

混入后自动获得 `version: int` 字段，每次 UPDATE 时自动 `+1`。

## 2. 让 `save()` / `update()` 自动重试

```python
order = await order.save(session, optimistic_retry_count=3)
# 冲突时最多重试 3 次：自动从 DB 读最新版本，重新应用你的修改，再 commit

# update() 同样支持
order = await order.update(session, update_data, optimistic_retry_count=3)
```

**重试时发生了什么**：

1. 第一次 commit → `StaleDataError`（`WHERE version = ?` 不匹配，影响 0 行）
2. rollback
3. 用 `model_dump(exclude={'id', 'version', 'created_at', 'updated_at'})` 保存你的修改
4. 用 `cls.get(session, cls.id == self.id)` 读最新记录
5. 把你的修改逐字段 `setattr` 到最新记录上
6. 再次 commit → 成功（或继续重试）

**业务代码完全感知不到这个过程**——这就是自动重试的价值。

## 3. 处理重试耗尽的情况

```python
from sqlmodel_ext import OptimisticLockError

try:
    order = await order.save(session, optimistic_retry_count=3)
except OptimisticLockError as e:
    # 异常携带丰富的上下文
    logger.warning(
        f"乐观锁冲突: model={e.model_class} id={e.record_id} "
        f"version={e.expected_version}"
    )
    # 通常的处理：返回 409 Conflict 给前端，提示用户刷新页面重试
    raise HTTPException(status_code=409, detail="数据已被其他人修改，请刷新后重试")
```

## 选择 `optimistic_retry_count` 的值

| 值 | 适用场景 |
|---|---------|
| `0`（默认） | 你想自己处理冲突（捕获 `OptimisticLockError`） |
| `1` ~ `3` | 大多数 web 端点。冲突不频繁时几乎总能在第一次重试成功 |
| `> 5` | 不推荐。重试次数过多说明该资源争用太严重，应考虑其他方案（行锁、消息队列、CRDT） |

## 不适用的场景

| 场景 | 为什么 | 应该用什么 |
|------|--------|----------|
| 日志/审计表 | 只插入不更新 | 直接 INSERT |
| 简单计数器 | 高频争用 | `UPDATE table SET count = count + 1` 原子操作 |
| 高频写入（每秒上千次） | 冲突太多，重试成本高 | 行锁 + 队列、或 CRDT 数据结构 |

## 相关参考

- [`OptimisticLockMixin` 完整字段](/reference/mixins#optimisticlockmixin)
- [`OptimisticLockError` 异常字段](/reference/mixins#optimisticlockerror)
- [乐观锁机制讲解](/explanation/optimistic-lock)（讲为什么这么设计）
