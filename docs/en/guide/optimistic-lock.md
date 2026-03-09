# Optimistic Locking

Optimistic locking is a concurrency control mechanism that detects conflicts through version numbers, preventing multiple operations from overwriting each other's changes.

## Problem: Lost Updates

Two admins modify the same order simultaneously:

```
t1  Admin A reads order (status="pending", amount=100)
t2  Admin B reads order (status="pending", amount=100)
t3  Admin A changes status="shipped" → writes to DB ✓
t4  Admin B changes amount=200      → writes to DB ✓ (overwrites A!)
```

::: danger Danger
B's write overwrites A's modification — `status` reverts to "pending" and A's change is lost.
:::

## Solution

Add a `version` field to records, checking the version on each update:

```sql
-- A's update: version=0 → 1
UPDATE "order" SET status='shipped', version=1
  WHERE id=1 AND version=0;  -- 1 row affected ✓ -- [!code highlight]

-- B's update: version is already 1, no longer 0
UPDATE "order" SET amount=200, version=1
  WHERE id=1 AND version=0;  -- 0 rows affected → conflict detected! -- [!code error]
```

## Usage

### Basic Usage

```python
from sqlmodel_ext import OptimisticLockMixin, UUIDTableBaseMixin, SQLModelBase

class Order(OptimisticLockMixin, UUIDTableBaseMixin, SQLModelBase, table=True):
    status: str
    amount: int
```

::: warning Warning
`OptimisticLockMixin` must be placed **before** `UUIDTableBaseMixin` in the inheritance list.
:::

After mixing in, a `version: int` field is automatically added, incrementing on every UPDATE.

### Manual Conflict Handling

```python
from sqlmodel_ext import OptimisticLockError

try:
    order = await order.save(session)
except OptimisticLockError as e: # [!code error]
    print(f"Conflict: {e.model_class} id={e.record_id}")
    print(f"Expected version: {e.expected_version}")
    # Re-query, prompt user to refresh the page...
```

### Automatic Retry (Recommended)

```python
order = await order.save(session, optimistic_retry_count=3) # [!code highlight]
# Retries up to 3 times on conflict, automatically merging changes

order = await order.update(session, data, optimistic_retry_count=3) # [!code highlight]
# update() supports it too
```

::: details Details
1. 1st attempt to commit → version conflict
2. Automatically re-reads the latest record from the database
3. Re-applies your changes to the latest record
4. 2nd attempt to commit → success
:::

## `OptimisticLockError` Context

The exception carries rich debugging information:

| Property | Description |
|----------|-------------|
| `model_class` | Model class name (e.g., "Order") |
| `record_id` | Record ID |
| `expected_version` | Expected version number |
| `original_error` | Original `StaleDataError` |

## Applicable & Non-applicable Scenarios

| Scenario | Applicable? | Reason |
|----------|------------|--------|
| Order status transitions | Yes | Concurrent modifications to the same record |
| Inventory deduction | Yes | Concurrent numeric changes |
| User profile editing | Yes | Multi-device simultaneous editing |
| Log/audit tables | **No** | Insert-only, no updates |
| Simple counters | **No** | Atomic `SET count = count + 1` is sufficient |
| High-frequency writes (thousands/sec) | **No** | Too many conflicts, retry cost is high |
