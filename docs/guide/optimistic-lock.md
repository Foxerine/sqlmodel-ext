# 乐观锁

乐观锁是一种并发控制机制，通过版本号检测冲突，防止多个操作互相覆盖修改。

## 问题：丢失更新

两个管理员同时修改同一个订单：

```
t1  管理员 A 读取订单（status="待发货", amount=100）
t2  管理员 B 读取订单（status="待发货", amount=100）
t3  管理员 A 改为 status="已发货" → 写回数据库
t4  管理员 B 改为 amount=200     → 写回数据库
```

结果：B 的写入覆盖了 A 的修改，`status` 变回 "待发货"。

## 解决方案

给记录加 `version` 字段，每次更新时检查版本号：

```sql
-- A 更新：version=0 → 1
UPDATE "order" SET status='已发货', version=1 WHERE id=1 AND version=0;
-- 影响 1 行

-- B 更新：version 已经是 1，不再是 0
UPDATE "order" SET amount=200, version=1 WHERE id=1 AND version=0;
-- 影响 0 行 → 检测到冲突
```

## 使用方式

### 基本用法

```python
from sqlmodel_ext import OptimisticLockMixin, UUIDTableBaseMixin, SQLModelBase

class Order(OptimisticLockMixin, UUIDTableBaseMixin, SQLModelBase, table=True):
    status: str
    amount: int
```

::: warning MRO 顺序
`OptimisticLockMixin` 必须放在 `UUIDTableBaseMixin` **之前**。
:::

混入后自动获得 `version: int` 字段，每次 UPDATE 时自动递增。

### 手动处理冲突

```python
from sqlmodel_ext import OptimisticLockError

try:
    order = await order.save(session)
except OptimisticLockError as e:
    print(f"冲突: {e.model_class} id={e.record_id}")
    print(f"期望版本: {e.expected_version}")
    # 重新查询，提示用户刷新页面...
```

### 自动重试（推荐）

```python
order = await order.save(session, optimistic_retry_count=3)
# 冲突时最多重试 3 次，自动合并修改

order = await order.update(session, data, optimistic_retry_count=3)
# update() 也支持
```

重试过程：
1. 第 1 次尝试 commit → 版本冲突
2. 自动从数据库重新读取最新记录
3. 把你的修改重新应用到最新记录上
4. 第 2 次尝试 commit → 成功

## `OptimisticLockError` 上下文

异常携带丰富的调试信息：

| 属性 | 说明 |
|------|------|
| `model_class` | 模型类名（如 "Order"） |
| `record_id` | 记录 ID |
| `expected_version` | 期望的版本号 |
| `original_error` | 原始的 `StaleDataError` |

## 适用与不适用场景

| 场景 | 适用？ | 原因 |
|------|--------|------|
| 订单状态转换 | 适用 | 并发修改同一记录 |
| 库存扣减 | 适用 | 并发数值变更 |
| 用户资料编辑 | 适用 | 多端同时编辑 |
| 日志/审计表 | **不适用** | 只插入不更新 |
| 简单计数器 | **不适用** | `SET count = count + 1` 原子操作就够了 |
| 高频写入（每秒上千次） | **不适用** | 冲突太多，重试成本高 |
