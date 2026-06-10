# Handle concurrent updates

**Goal**: prevent two concurrent operations from overwriting each other's changes (the "lost update" problem) — make conflicts detectable and retryable.

**Prerequisites**:

- Records on this model are modified by multiple users / processes concurrently
- You can tolerate a small retry overhead (high-frequency-write scenarios are not a fit — see the bottom)

## 1. Add `OptimisticLockMixin` to the model

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

::: warning MRO order
`OptimisticLockMixin` **must** appear before `UUIDTableBaseMixin` / `TableBaseMixin`.
:::

After mixing in, the model gains a `version: int` field that is auto-incremented on every UPDATE.

## 2. Let `save()` / `update()` retry automatically

```python
order = await order.save(session, optimistic_retry_count=3)
# On conflict, retries up to 3 times: re-reads latest version from DB,
# re-applies your changes, then commits again.

# update() supports it too
order = await order.update(session, update_data, optimistic_retry_count=3)
```

**What happens during a retry**:

1. First commit → `StaleDataError` (the `WHERE version = ?` doesn't match, 0 rows affected)
2. rollback
3. Save your changes via `model_dump(exclude={'id', 'version', 'created_at', 'updated_at'})`
4. Read the latest record via `cls.get(session, cls.id == self.id)`
5. Re-apply your changes field-by-field via `setattr` onto the latest record
6. Commit again → success (or keep retrying)

**Business code is completely unaware** that retries happened — that's the value of automatic retry.

## 3. Handling exhausted retries

```python
from sqlmodel_ext import OptimisticLockError

try:
    order = await order.save(session, optimistic_retry_count=3)
except OptimisticLockError as e:
    # The exception carries rich context
    logger.warning(
        f"Optimistic lock conflict: model={e.model_class} id={e.record_id} "
        f"version={e.expected_version}"
    )
    # Typical handling: return 409 Conflict, ask the user to refresh and retry
    raise HTTPException(status_code=409, detail="Record was modified by someone else. Please refresh and retry.")
```

## Choosing `optimistic_retry_count`

| Value | When to use |
|-------|-------------|
| `0` (default) | You want to handle conflicts yourself (catch `OptimisticLockError`) |
| `1`–`3` | Most web endpoints. Conflicts are rare and the first retry almost always succeeds |
| `> 5` | Not recommended. High retry counts indicate severe contention — consider other approaches (row locking, message queues, CRDTs) |

## When this isn't a fit

| Scenario | Why | Use instead |
|----------|-----|-------------|
| Log / audit tables | Insert-only | Direct INSERT |
| Simple counters | High contention | `UPDATE table SET count = count + 1` atomic operation |
| High-frequency writes (thousands per second) | Too many conflicts, retry cost is high | Row locking + queues, or CRDT data structures |

## Related reference

- [`OptimisticLockMixin` field details](/en/reference/mixins#optimisticlockmixin)
- [`OptimisticLockError` exception fields](/en/reference/mixins#optimisticlockerror)
- [Optimistic lock mechanism explanation](/en/explanation/optimistic-lock) (the "why")
