"""
Optimistic Locking Mixin

Provides optimistic lock support based on SQLAlchemy's version_id_col mechanism.

Suitable for:
- Tables with state transitions (e.g. pending -> paid)
- Tables with numeric changes (e.g. balance, inventory)

Not suitable for:
- Log tables, insert-only tables, low-value statistics
- Simple counters solvable with ``UPDATE table SET col = col + 1``

Usage::

    class Order(OptimisticLockMixin, UUIDTableBaseMixin, table=True):
        status: OrderStatusEnum
        amount: Decimal

    try:
        order = await order.save(session)
    except OptimisticLockError as e:
        logger.warning(f"Optimistic lock conflict: {e}")
"""
from typing import ClassVar

from sqlalchemy.orm.exc import StaleDataError


class OptimisticLockError(Exception):
    """
    Optimistic lock conflict exception.

    Raised when save/update detects a version mismatch, meaning another
    transaction has modified the record between read and write.

    Attributes:
        model_class: Name of the model class where the conflict occurred
        record_id: Record ID (if available)
        expected_version: Expected version number (if available)
        original_error: The original StaleDataError
    """

    def __init__(
            self,
            message: str,
            model_class: str | None = None,
            record_id: str | None = None,
            expected_version: int | None = None,
            original_error: StaleDataError | None = None,
    ):
        super().__init__(message)
        self.model_class = model_class
        self.record_id = record_id
        self.expected_version = expected_version
        self.original_error = original_error


class OptimisticLockMixin:
    """
    Optimistic Locking Mixin using SQLAlchemy's version_id_col mechanism.

    Each UPDATE automatically checks and increments the version number.
    If the version doesn't match (another transaction modified the record),
    ``session.commit()`` raises ``StaleDataError``, which is caught by
    save/update methods and converted to ``OptimisticLockError``.

    Principle:
    1. Each record has a ``version`` field, starting at 0
    2. Each UPDATE generates SQL like:
       ``UPDATE table SET ..., version = version + 1 WHERE id = ? AND version = ?``
    3. If WHERE doesn't match (version changed by another transaction),
       UPDATE affects 0 rows, SQLAlchemy raises StaleDataError

    Inheritance order:
        OptimisticLockMixin must come before TableBaseMixin/UUIDTableBaseMixin::

            class Order(OptimisticLockMixin, UUIDTableBaseMixin, table=True):
                ...
    """
    _has_optimistic_lock: ClassVar[bool] = True
    """Internal flag indicating optimistic locking is enabled."""

    version: int = 0
    """Optimistic lock version number, auto-incremented on each update."""
