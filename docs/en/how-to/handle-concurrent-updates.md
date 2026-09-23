# Handle concurrent updates

**Goal**: prevent two concurrent operations from overwriting each other's changes (the "lost update" problem) — make conflicts detectable and automatically retryable.

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

After mixing in, the model automatically gains an `oplock_version` column (`BIGINT`, auto-incremented on every UPDATE). It is internal ORM state: it **does not appear in `model_dump()`**, and the model cannot redeclare it itself. For a domain-level "version" field, pick a different name (`version`, `revision`).

## 2. Write nothing: conflicts are retried 3 times by default

```python
order = await order.save(session)
# On conflict, automatically: rollback → re-read the latest row → replay only the columns you changed → commit again, up to 3 times

order = await order.update(session, patch)
# Also retries 3 times by default: re-reads the latest row, then applies the changes in patch
```

The retry count is the model policy `__optimistic_retry_default__` (`3` on `OptimisticLockMixin`), so call sites don't need to remember to pass a parameter.

**What happens during a retry** (using `save()` as the example):

1. commit → `StaleDataError` (`WHERE oplock_version = ?` matches 0 rows)
2. rollback
3. Collect **the columns you actually changed** from SQLAlchemy attribute history (excluding `id` / `oplock_version` / `created_at` / `updated_at`)
4. `cls.get(session, cls.id == ...)` reads the latest record
5. `setattr` your changes column by column onto the latest record — other columns someone else just committed stay untouched
6. Commit again → success (or keep retrying)

Effect: when two people edit different fields of the same order at the same time, both sets of changes are kept, and the client notices nothing.

```python
# Two sessions read the same row at the same time
a = await Order.get_one(s1, order_id)
b = await Order.get_one(s2, order_id)

a.status = OrderStatusEnum.paid
a = await a.save(s1)             # oplock_version +1

b.amount = Decimal('200')
b = await b.save(s2)             # conflict → automatic retry → success
assert b.status == OrderStatusEnum.paid and b.amount == Decimal('200')
```

## 3. When you need to handle conflicts yourself: pass `0` explicitly

```python
from sqlmodel_ext import OptimisticLockError

try:
    order = await order.save(session, optimistic_retry_count=0)   # explicitly ask for no retries
except OptimisticLockError as e:
    logger.warning(
        f"Optimistic lock conflict: model={e.model_class} id={e.record_id} "
        f"version={e.expected_version}"
    )
    raise HTTPException(status_code=409, detail="Record was modified by someone else. Please refresh and retry.")
```

The same `except` also handles **exhausted retries** (under the default policy it is raised only after 3 consecutive conflicts), as well as the case where a retry finds the record has been deleted.

## 4. Conflicts on delete

Deleting a versioned model also checks the version:

```python
try:
    await Order.delete(session, order)
except OptimisticLockError:
    # Someone modified (or deleted) this row after you read it — whether to still delete is your call
    await session.rollback()
    ...
```

`delete()` **never retries** ("it was just modified — do I still want to delete it?" is a business decision), and `record_id` / `expected_version` are always `None`: at the flush level the conflict cannot be reliably attributed to a specific row.

## Choosing `optimistic_retry_count`

| Value | When to use |
|-------|-------------|
| Not passed (`None`) | The vast majority of cases. Uses the model policy: optimistic-lock models retry 3 times, other models 0 times |
| `0` | You want to handle conflicts yourself (catch `OptimisticLockError`, return 409 so the user refreshes) |
| `> 5` | Not recommended. Too many retries means the resource is too heavily contended — consider row locks (see [Enforce row locks and isolation levels](./enforce-locking-and-isolation)), message queues, or CRDTs |

## When this isn't a fit

| Scenario | Why | Use instead |
|----------|-----|-------------|
| Log / audit tables | Insert-only, never updated | Direct INSERT |
| Simple counters | High contention | `UPDATE table SET count = count + 1` atomic operation |
| High-frequency writes (thousands per second) | Too many conflicts, retry cost is high | `get(with_for_update=True)` row lock + queues, or CRDT data structures |

## Related reference

- [`OptimisticLockMixin` full fields](/en/reference/mixins#optimisticlockmixin)
- [`OptimisticLockError` exception fields](/en/reference/mixins#optimisticlockerror)
- [Optimistic lock mechanism explanation](/en/explanation/optimistic-lock) (explains why it's designed this way)
