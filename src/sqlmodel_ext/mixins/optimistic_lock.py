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
from typing import Annotated, ClassVar

from sqlalchemy import BigInteger, text
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import Field

from sqlmodel_ext.constants import (
    OPTIMISTIC_LOCK_ENABLED_FLAG as OPTIMISTIC_LOCK_ENABLED_FLAG,
    OPTIMISTIC_LOCK_VERSION_COLUMN as OPTIMISTIC_LOCK_VERSION_COLUMN,
)
from sqlmodel_ext.field_types import JS_MAX_SAFE_INTEGER

OPLOCK_INITIAL_VERSION: int = 0
"""Initial version value -- single source for the Python default and the DB ``server_default``."""


class OptimisticLockError(Exception):
    """
    Optimistic lock conflict exception.

    Raised when save/update detects a version mismatch, meaning another
    transaction has modified the record between read and write, and by
    ``delete()`` when its flush hits an optimistic-lock conflict.

    Attributes:
        model_class: Name of the model class where the conflict occurred
            (for ``delete()``: the calling model, see its docstring)
        record_id: Record ID (if available; always ``None`` for ``delete()``)
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
        self.model_class: str | None = model_class
        self.record_id: str | None = record_id
        self.expected_version: int | None = expected_version
        self.original_error: StaleDataError | None = original_error


class OptimisticLockMixin:
    """
    Optimistic Locking Mixin using SQLAlchemy's version_id_col mechanism.

    Each UPDATE automatically checks and increments the version number.
    If the version doesn't match (another transaction modified the record),
    ``session.commit()`` raises ``StaleDataError``, which is caught by
    save/update/delete and converted to ``OptimisticLockError``.

    Principle:
    1. Each record has an ``oplock_version`` field, starting at 0
    2. Each UPDATE generates SQL like:
       ``UPDATE table SET ..., oplock_version = oplock_version + 1 WHERE id = ? AND oplock_version = ?``
    3. If WHERE doesn't match (version changed by another transaction),
       UPDATE affects 0 rows, SQLAlchemy raises StaleDataError

    The metaclass wires ``version_id_col`` on the root table mapper
    automatically; STI/JTI children share it through mapper inheritance.

    The version column is named ``oplock_version`` (not ``version``) so it
    never collides with a domain ``version`` field. It is excluded from
    ``model_dump()`` (pure ORM-internal state, never part of a model's public
    representation).

    Inheritance order:
        OptimisticLockMixin must come before TableBaseMixin/UUIDTableBaseMixin::

            class Order(OptimisticLockMixin, UUIDTableBaseMixin, table=True):
                ...

    Retries:
        ``save()`` / ``update()`` retry conflicts internally
        ``__optimistic_retry_default__`` times (3) unless an explicit
        ``optimistic_retry_count`` is passed; only when retries are
        exhausted is ``OptimisticLockError`` raised.
    """
    __optimistic_retry_default__: ClassVar[int] = 3
    """Overrides ``TableBaseMixin``'s 0: conflicts are retried internally by default.

    An ``update()`` retry re-reads the latest row and re-applies only the
    caller's delta -- "apply my change on top of the other writer's result",
    which is what concurrent edits should do, transparently to clients. Only
    when retries are exhausted is ``OptimisticLockError`` raised.

    3 rather than more: each retry is a full rollback + re-read + re-write;
    persistent conflicts indicate real contention, where more retries would
    only hold connections longer."""

    _has_optimistic_lock: ClassVar[bool] = True
    """Marks the class as optimistic-lock enabled (the metaclass wires ``version_id_col`` from it).

    A subclass may override it with ``False`` to keep the version column but
    not wire ``version_id_col`` -- e.g. as the first step of a two-phase
    rollout where the column ships before the locking behavior."""

    # Field shape (all three parts are required):
    # 1. The default lives inside the Annotated ``Field``: this mixin is a
    #    plain class (no ``model_fields``), so a default given only by ``=``
    #    assignment would be lost when the metaclass recovers Annotated fields
    #    on subclasses; the annotation metadata carries it reliably.
    # 2. The ``= OPLOCK_INITIAL_VERSION`` assignment is kept as well so type
    #    checkers see an initialized attribute.
    # 3. ``server_default``: during a rolling deployment, old application
    #    instances do not know the column and omit it on INSERT -- a DB-side
    #    default keeps those INSERTs valid.
    oplock_version: Annotated[
        int,
        Field(
            default=OPLOCK_INITIAL_VERSION,
            ge=0,
            le=JS_MAX_SAFE_INTEGER,
            sa_type=BigInteger,
            sa_column_kwargs={'server_default': text(str(OPLOCK_INITIAL_VERSION))},
            # exclude=True is the encapsulation boundary of this mixin: the
            # column is ORM-internal state maintained by version_id_col, so it
            # must not leak into ``model_dump()`` (e.g. into response DTOs with
            # extra='forbid').
            exclude=True,
        ),
    ] = OPLOCK_INITIAL_VERSION
    """Optimistic lock version number, auto-incremented on each update.

    BIGINT with an upper bound of ``JS_MAX_SAFE_INTEGER``: removes any
    realistic risk of INTEGER overflow on frequently updated rows."""
