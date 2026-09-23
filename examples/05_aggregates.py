"""
05 -- Aggregates: ``count(distinct_column=)``, ``distinct_column()``, ``group_sum()``.

Run::

    python examples/05_aggregates.py

Aggregations are parameters of the model's own query methods, not hand-written
``select(func.sum(...))`` blocks scattered through endpoints:

* ``count(session, condition, distinct_column=...)`` -> ``COUNT(DISTINCT col)``
* ``distinct_column(session, column, condition, limit=...)`` -> distinct values
* ``group_sum(session, [columns], group_by=..., condition=...)`` -> one
  ``GroupSumRow`` per group with ``count`` and ``totals`` (``Decimal``, aligned
  by position with the summed columns). Without ``group_by`` exactly one row is
  returned, even for an empty table.

Money is written through ``NonNegativeWriteDecimal38_18`` (35 digits, leaving
SUM() headroom inside the NUMERIC(38, 18) column) and totals are exposed through
``SignedSumDecimal38_18`` (the full 38 digits), so a sum of valid rows is always
a valid response value.
"""
import asyncio
from decimal import Decimal

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, col

from sqlmodel_ext import (
    AsyncSession,
    NonNegativeWriteDecimal38_18,
    SignedSumDecimal38_18,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
)


class Order(SQLModelBase, UUIDTableBaseMixin, table=True):
    customer: Str64
    """Customer name."""

    amount: NonNegativeWriteDecimal38_18
    """Order total."""


class CustomerTotal(SQLModelBase):
    """Response DTO for one aggregated customer."""
    customer: Str64
    orders: int
    total: SignedSumDecimal38_18


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        empty = await Order.group_sum(session, [col(Order.amount)])
        assert empty[0].count == 0 and empty[0].totals == [Decimal(0)], "whole-table sum of nothing is 0"

        for customer, amount in (("ann", "10.50"), ("ann", "4.50"), ("bob", "7"), ("cat", "0")):
            _ = await Order(customer=customer, amount=Decimal(amount)).save(session, refresh=False)

        assert await Order.count(session) == 4
        assert await Order.count(session, distinct_column=col(Order.customer)) == 3
        paying = await Order.distinct_column(session, col(Order.customer), col(Order.amount) > 0)
        assert sorted(paying) == ["ann", "bob"]

        whole = await Order.group_sum(session, [col(Order.amount)])
        assert whole[0].key is None and whole[0].totals == [Decimal("22")]

        per_customer = await Order.group_sum(
            session, [col(Order.amount)], group_by=col(Order.customer),
        )
        report = [
            CustomerTotal(customer=row.key, orders=row.count, total=row.totals[0])
            for row in per_customer
        ]
        assert [(r.customer, r.orders, r.total) for r in report] == [
            ("ann", 2, Decimal("15")), ("bob", 1, Decimal("7")), ("cat", 1, Decimal("0")),
        ]
        print([r.model_dump(mode='json') for r in report])

    await engine.dispose()
    print("[OK] 05_aggregates: count / distinct / group_sum")


if __name__ == "__main__":
    asyncio.run(main())
