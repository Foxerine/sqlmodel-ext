"""
Table Base Mixins -- async CRUD operations.

Provides TableBaseMixin and UUIDTableBaseMixin with full async CRUD,
pagination (offset and ``after_id`` keyset), polymorphic query support,
relationship preloading, FOR UPDATE tracking, aggregation helpers, and
type-safe helper functions.
"""
import logging
import uuid
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import TypeVar, Literal, override, overload, Any, ClassVar, Generic, cast

from sqlalchemy import DateTime, ColumnElement, desc, asc, event, func, distinct, delete as sql_delete, inspect
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    InstanceState,
    Mapped,
    Mapper,
    QueryableAttribute,
    RelationshipProperty,
    Session as _SyncSession,
    SessionTransaction,
    selectinload,
    with_polymorphic,
)
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql.base import ExecutableOption
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import Field, select, col
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.sql._typing import _OnClauseArgument
from sqlalchemy.ext.asyncio import AsyncAttrs

from sqlmodel_ext._utils import now, now_date
from sqlmodel_ext._exceptions import RecordNotFoundError
from sqlmodel_ext.field_types import NonNegativeBigInt
from sqlmodel_ext.mixins._uuid import uuid7
from sqlmodel_ext.mixins.exceptions import (
    FK_DELETE_RESTRICT_FALLBACK_MESSAGE,
    KeysetCursorInvalidError,
    KeysetCursorUnsupportedError,
    ResourceReferencedError,
)
from sqlmodel_ext.mixins.optimistic_lock import OPTIMISTIC_LOCK_VERSION_COLUMN, OptimisticLockError
from sqlmodel_ext.mixins.polymorphic import PolymorphicBaseMixin
from sqlmodel_ext.base import SQLModelBase
from sqlmodel_ext.pagination import (
    ListResponse,
    TimeFilterRequest,
    PageWindowRequest,
    PaginationRequest,
    TableViewRequest,
)

# Conditional FastAPI import
try:
    from fastapi import HTTPException as _FastAPIHTTPException
except ImportError:
    _FastAPIHTTPException = None

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="TableBaseMixin")
M = TypeVar("M", bound="SQLModelBase")
GK = TypeVar("GK")
"""Group key type of ``group_sum`` (str / datetime / ... depending on ``group_by``)."""
V = TypeVar("V")
"""Column value type of ``distinct_column`` (UUID / str / ... depending on ``column``)."""

# FOR UPDATE tracking: get(with_for_update=True) records id(instance) to session.info,
# for runtime checking by the @requires_for_update decorator.
SESSION_FOR_UPDATE_KEY = '_for_update_locked'
"""Key in session.info storing the set of id() values for FOR UPDATE locked instances.

Lifecycle (maintained by the session event listeners below, independent of
caching): the outermost commit / rollback clears it (row locks are released
with the transaction); a savepoint rollback restores the snapshot taken when
the savepoint began (PostgreSQL releases locks acquired inside it); a
savepoint release keeps it (PostgreSQL keeps those locks). The enhanced
``AsyncSession.reset()`` / ``close()`` clear it as well."""

SESSION_REPEATABLE_READ_KEY = '_repeatable_read_verified'
"""Key in session.info marking "this session's isolation level has been *verified* to be REPEATABLE READ".

Written by ``AsyncSession.enter_repeatable_read()`` only **after** reading the
level back from the database: if ``session.connection(execution_options=
{'isolation_level': ...})`` is called after the session already executed SQL,
SQLAlchemy only emits a ``SAWarning`` and silently keeps the old level -- so
"I requested it" must never be recorded as "it is in effect". Consumed by
``@requires_repeatable_read``; cleared by ``AsyncSession.reset()`` / ``close()``
(the connection returns to the pool and its isolation level is reset)."""

# Savepoint-level FOR UPDATE lock snapshot stack: every nested (savepoint)
# transaction pushes a copy of SESSION_FOR_UPDATE_KEY at its start; a nested
# rollback restores the top snapshot (drops locks taken inside the savepoint,
# keeps the ones held before it -- PostgreSQL's ROLLBACK TO SAVEPOINT
# semantics); a nested commit (RELEASE) only pops (PostgreSQL keeps the locks).
_SESSION_LOCK_SNAPSHOT_STACK = '_for_update_snapshot_stack'
# Marks "this nested end was already handled by after_commit/after_rollback",
# so after_transaction_end can tell a normal savepoint commit/rollback apart
# from a savepoint closed via close() (which only fires the end event).
_SESSION_NESTED_LOCK_HANDLED = '_nested_savepoint_lock_handled'


def _on_lock_tracking_commit(session: _SyncSession) -> None:
    """``after_commit``: release (outermost) or keep (savepoint RELEASE) the tracked FOR UPDATE locks."""
    if session.in_nested_transaction():
        # RELEASE SAVEPOINT: PostgreSQL keeps the savepoint's row locks (they
        # move to the parent transaction) -- only pop the snapshot stack.
        stack: list[set[int]] = session.info.get(_SESSION_LOCK_SNAPSHOT_STACK, [])
        if stack:
            stack.pop()
        session.info[_SESSION_NESTED_LOCK_HANDLED] = True
        return
    session.info.pop(SESSION_FOR_UPDATE_KEY, None)
    session.info.pop(_SESSION_LOCK_SNAPSHOT_STACK, None)
    session.info.pop(_SESSION_NESTED_LOCK_HANDLED, None)


def _on_lock_tracking_rollback(session: _SyncSession) -> None:
    """``after_rollback``: restore the savepoint snapshot (nested) or clear everything (outermost)."""
    if session.in_nested_transaction():
        # ROLLBACK TO SAVEPOINT releases locks acquired inside the savepoint
        # and keeps earlier ones: restore the snapshot instead of clearing all
        # (clearing would forget locks the caller took before the savepoint).
        stack: list[set[int]] = session.info.get(_SESSION_LOCK_SNAPSHOT_STACK, [])
        if stack:
            session.info[SESSION_FOR_UPDATE_KEY] = stack.pop()
        else:
            session.info.pop(SESSION_FOR_UPDATE_KEY, None)
        session.info[_SESSION_NESTED_LOCK_HANDLED] = True
        return
    session.info.pop(SESSION_FOR_UPDATE_KEY, None)
    session.info.pop(_SESSION_LOCK_SNAPSHOT_STACK, None)
    session.info.pop(_SESSION_NESTED_LOCK_HANDLED, None)


def _on_lock_tracking_transaction_create(session: _SyncSession, transaction: SessionTransaction) -> None:
    """``after_transaction_create``: snapshot the lock set when a savepoint begins."""
    if transaction.nested:
        stack: list[set[int]] = session.info.setdefault(_SESSION_LOCK_SNAPSHOT_STACK, [])
        stack.append(set(session.info.get(SESSION_FOR_UPDATE_KEY, set())))


def _on_lock_tracking_transaction_end(session: _SyncSession, transaction: SessionTransaction) -> None:
    """``after_transaction_end``: fallback for a savepoint ended via ``close()``.

    Normal savepoint commit/rollback were handled by the handlers above (they
    set ``_SESSION_NESTED_LOCK_HANDLED``). A savepoint ended by ``close()``
    fires only this event: restore the snapshot conservatively (drop inner
    locks -- fail-closed, forces re-locking) and keep the stack balanced.
    """
    if not transaction.nested:
        return
    if session.info.pop(_SESSION_NESTED_LOCK_HANDLED, False):
        return
    stack: list[set[int]] = session.info.get(_SESSION_LOCK_SNAPSHOT_STACK, [])
    if stack:
        session.info[SESSION_FOR_UPDATE_KEY] = stack.pop()
    else:
        session.info.pop(SESSION_FOR_UPDATE_KEY, None)


event.listen(_SyncSession, "after_commit", _on_lock_tracking_commit)
event.listen(_SyncSession, "after_rollback", _on_lock_tracking_rollback)
event.listen(_SyncSession, "after_transaction_create", _on_lock_tracking_transaction_create)
event.listen(_SyncSession, "after_transaction_end", _on_lock_tracking_transaction_end)


# NOTE(SQLModel typing): load parameter uses QueryableAttribute[Any] (InstrumentedAttribute at runtime).
# basedpyright infers SQLModel Relationship fields as the annotated type (e.g. LLM), not QueryableAttribute.
# Callers should use rel(Model.relation) to pass load args (see rel() below).
# Ref: https://github.com/fastapi/sqlmodel/discussions/1391


def rel(relationship: object) -> QueryableAttribute[Any]:
    """Cast a SQLModel Relationship field to QueryableAttribute for ``load`` parameter.

    Similar to ``sqlmodel.col()``, this resolves basedpyright inferring
    SQLModel Relationship fields as their annotated type rather than
    ``QueryableAttribute``.

    Example::

        from sqlmodel_ext.mixins.table import rel

        character = await Character.get(session, load=rel(Character.llm))
    """
    if not isinstance(relationship, QueryableAttribute):
        raise AttributeError(
            f"Expected a Relationship field, got {type(relationship).__name__}. "
            f"Pass a class attribute (e.g. Character.llm), not an instance attribute."
        )
    return relationship


def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]:
    """Narrow a SQLModel column comparison to ``ColumnElement[bool]``.

    Similar to ``sqlmodel.col()`` and ``rel()``, this resolves basedpyright
    inferring ``Model.field == value`` as ``bool``. At runtime the expression
    is actually a ``ColumnElement[bool]``; this function narrows the type via
    ``cast`` so subsequent ``&`` / ``|`` operators pass type checking.

    Example::

        from sqlmodel_ext.mixins.table import cond

        scope = cond(UserFile.user_id == current_user.id)
        condition = scope & cond(UserFile.status == FileStatusEnum.uploaded)
    """
    return cast(ColumnElement[bool], expr)


class GroupSumRow(SQLModelBase, Generic[GK]):
    """One aggregated group returned by ``TableBaseMixin.group_sum`` (one row per GROUP BY group).

    ``totals`` is a homogeneous ``Decimal`` list aligned **by position** with
    the ``sum_columns`` argument of ``group_sum`` (``totals[0]`` is the sum of
    ``sum_columns[0]`` and so on) -- a variable number of sum columns cannot
    be named in advance, and positional alignment avoids ``.label()`` strings
    while staying typed.

    Note:
        Inherits ``SQLModelBase`` (unlike ``ListResponse``): it is only a
        method return value and never enters OpenAPI, so SQLModel's generic
        JSON-schema limitation does not apply.
    """
    key: GK
    """Group key value (value of the ``group_by`` column / expression); ``None`` for whole-table aggregation."""

    count: NonNegativeBigInt
    """Number of rows in the group (``COUNT(*)``)."""

    totals: list[Decimal]
    """``COALESCE(SUM(col), 0)`` of each sum column, in ``sum_columns`` order."""


class TableBaseMixin(AsyncAttrs):
    """
    Async CRUD operations base mixin for SQLModel models.

    Must be used together with SQLModelBase.

    Provides ``add()``, ``save()``, ``update()``, ``delete()``, ``get()``,
    ``get_one()``, ``get_exist_one()``, ``count()``, ``get_with_count()``,
    ``distinct_column()`` and ``group_sum()`` methods.

    Attributes:
        id: Integer primary key, auto-increment.
        created_at: Record creation timestamp, auto-set.
        updated_at: Record update timestamp, auto-updated.
    """
    _has_table_mixin: ClassVar[bool] = True
    """Internal flag marking TableBaseMixin inheritance."""

    __optimistic_retry_default__: ClassVar[int] = 0
    """Retry count used by ``save()`` / ``update()`` when ``optimistic_retry_count`` is not passed (``None``).

    The base class uses **0**: on a model without optimistic locking a
    ``StaleDataError`` can only mean "the UPDATE matched 0 rows = the row no
    longer exists", and retrying would only re-raise "record deleted".

    ``OptimisticLockMixin`` overrides it with a non-zero value -- the policy
    lives where the capability is declared instead of relying on every call
    site to remember the argument. MRO requirement: ``OptimisticLockMixin``
    must come **before** ``TableBaseMixin`` / ``UUIDTableBaseMixin``.

    Only ``save()`` / ``update()`` use this policy. ``delete()`` intentionally
    never retries ("someone just changed it -- do I still want to delete it?"
    is the caller's decision); it only normalizes a conflict into
    ``OptimisticLockError``."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Accept and forward keyword arguments from subclass definitions."""
        super().__init_subclass__(**kwargs)

    id: int | None = Field(default=None, primary_key=True)

    created_at: datetime = Field(default_factory=now, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]  # older sqlmodel (e.g. 0.0.38) annotates sa_type as type[Any]; a TypeEngine instance is accepted at runtime
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # pyright: ignore[reportArgumentType]  # older sqlmodel (e.g. 0.0.38) annotates sa_type as type[Any]; a TypeEngine instance is accepted at runtime
        sa_column_kwargs={'default': now, 'onupdate': now},
        default_factory=now
    )

    # ==================== IntegrityError friendly-message registry ====================
    #
    # Each application module declares its own constraint-name → user-facing message
    # at the same site as the ``UniqueConstraint`` / ``ForeignKey`` / ``CheckConstraint``
    # declaration, by calling ``TableBaseMixin.register_*_violation_message(...)``.
    # Both ``sanitize_integrity_error`` and any global FastAPI integrity-error handler
    # look up the registry; on hit they return the registered message, otherwise they
    # fall through to a generic fallback. No code path leaks table/column names or SQL.
    #
    # Design:
    # - The registry hangs off ``TableBaseMixin`` (the root mixin for all table models),
    #   so it is shared across subclasses.
    # - ``setdefault`` semantics (first registration wins) protect against duplicate
    #   registration from re-imports.
    # - The ``CheckConstraint`` registry only serves ORM-declared CHECKs. Trigger
    #   ``RAISE EXCEPTION`` errors typically don't carry ``constraint_name``; those
    #   are surfaced via ``extract_trigger_message`` instead.

    _UNIQUE_VIOLATION_MESSAGES: ClassVar[dict[str, str]] = {}
    """UNIQUE constraint name -> user-facing message"""

    _FOREIGN_KEY_VIOLATION_MESSAGES: ClassVar[dict[str, str]] = {}
    """FK constraint name -> user-facing message ("the referenced resource does not exist" direction)"""

    _CHECK_VIOLATION_MESSAGES: ClassVar[dict[str, str]] = {}
    """Declared CHECK constraint name -> user-facing message (trigger-raised check_violation does not go through here)"""

    _FK_DELETE_RESTRICT_MESSAGES: ClassVar[dict[str, str]] = {}
    """FK constraint name -> user-facing message when **deleting** this row is rejected because it is still referenced.

    Not the same as ``_FOREIGN_KEY_VIOLATION_MESSAGES``: both are keyed by the
    same constraint name but serve **opposite directions**:

    ======================  ==========================================  =========
    Direction               Trigger                                     Semantics
    ======================  ==========================================  =========
    points at missing row   INSERT/UPDATE a child whose FK target is    404
                            gone
    still referenced        DELETE a parent still referenced by a       409
                            child (RESTRICT / NO ACTION)
    ======================  ==========================================  =========

    The driver exception is identical in both directions, so the direction
    is decided by the **call site**: only a violation caught by
    :meth:`TableBaseMixin.delete` (and caused by a DELETE statement) consults
    this registry.
    """

    @staticmethod
    def register_unique_violation_message(constraint_name: str, friendly_message: str) -> None:
        """
        Declare the user-facing message shown when a UNIQUE constraint is violated.

        Call this at module top-level next to the matching ``UniqueConstraint(..., name='xxx')``
        declaration. SQLModel modules are import-eager (loaded during FastAPI startup),
        so registration completes before any request runs.

        :param constraint_name: Constraint name, must exactly match the ``name=`` kwarg
            of the ``UniqueConstraint`` declaration.
        :param friendly_message: Message returned to the user when the constraint fires.
            Should not contain table/column names.
        """
        _ = TableBaseMixin._UNIQUE_VIOLATION_MESSAGES.setdefault(constraint_name, friendly_message)

    @staticmethod
    def register_foreign_key_violation_message(constraint_name: str, friendly_message: str) -> None:
        """
        Declare the user-facing message shown when a FK constraint is violated
        (typically for "referenced resource missing" semantics).

        :param constraint_name: FK constraint name (PostgreSQL default is ``<table>_<col>_fkey``)
        :param friendly_message: User-facing message
        """
        _ = TableBaseMixin._FOREIGN_KEY_VIOLATION_MESSAGES.setdefault(constraint_name, friendly_message)

    @staticmethod
    def register_check_violation_message(constraint_name: str, friendly_message: str) -> None:
        """
        Declare the user-facing message shown when a declared ``CheckConstraint`` is violated.

        Only applies to ORM-level ``CheckConstraint(..., name='ck_xxx')``. Database trigger
        ``RAISE EXCEPTION`` that produces a check_violation typically lacks a
        ``constraint_name``; those are surfaced by ``extract_trigger_message`` directly
        and bypass this registry.

        :param constraint_name: CHECK constraint name
        :param friendly_message: User-facing message
        """
        _ = TableBaseMixin._CHECK_VIOLATION_MESSAGES.setdefault(constraint_name, friendly_message)

    @staticmethod
    def register_fk_delete_restrict_message(constraint_name: str, friendly_message: str) -> None:
        """Declare the message returned when deleting a row is rejected because it is still referenced (409 semantics).

        Call it next to the **referenced parent** model (same convention as
        ``register_unique_violation_message``). Unregistered constraints fall
        back to :data:`FK_DELETE_RESTRICT_FALLBACK_MESSAGE`; the registry only
        decides *what to say*, never the direction.

        :param constraint_name: FK constraint name; must match the database
            name **exactly** -- a typo silently falls back to the generic
            message (a lookup miss is not an error). Verify the real name from
            the running database (e.g. the ``constraint=`` field of the
            warning logged by ``delete()``), not from a schema created with
            ``create_all``: migrations may have named constraints differently.
        :param friendly_message: User-facing message; should state the next
            actionable step and must not contain table/column names.
        """
        _ = TableBaseMixin._FK_DELETE_RESTRICT_MESSAGES.setdefault(constraint_name, friendly_message)

    @staticmethod
    def lookup_unique_violation_message(constraint_name: str | None) -> str | None:
        """Look up the friendly message for a UNIQUE constraint; returns None if missing/unregistered."""
        if not constraint_name:
            return None
        return TableBaseMixin._UNIQUE_VIOLATION_MESSAGES.get(constraint_name)

    @staticmethod
    def lookup_foreign_key_violation_message(constraint_name: str | None) -> str | None:
        """Look up the friendly message for a FK constraint; returns None if missing/unregistered."""
        if not constraint_name:
            return None
        return TableBaseMixin._FOREIGN_KEY_VIOLATION_MESSAGES.get(constraint_name)

    @staticmethod
    def lookup_check_violation_message(constraint_name: str | None) -> str | None:
        """Look up the friendly message for a declared CHECK constraint; returns None if missing/unregistered."""
        if not constraint_name:
            return None
        return TableBaseMixin._CHECK_VIOLATION_MESSAGES.get(constraint_name)

    @staticmethod
    def lookup_fk_delete_restrict_message(constraint_name: str | None) -> str | None:
        """Look up the "delete rejected, still referenced" message; returns None if missing/unregistered.

        Only :meth:`delete` should call it -- only there is the direction
        known to be "still referenced" (see ``_FK_DELETE_RESTRICT_MESSAGES``).
        """
        if not constraint_name:
            return None
        return TableBaseMixin._FK_DELETE_RESTRICT_MESSAGES.get(constraint_name)

    @staticmethod
    def extract_trigger_message(orig: BaseException) -> str:
        """
        Extract the first-line business message from an asyncpg ``CheckViolationError``
        raised by a trigger ``RAISE EXCEPTION`` (strips ``ERROR:`` prefix and ``DETAIL:`` /
        ``CONTEXT:`` trailing lines).

        Shared between ``sanitize_integrity_error`` and any global integrity-error
        handler (hence exposed). Other callers should not consume raw trigger
        messages directly.
        """
        message = str(orig)
        if '\n' in message:
            message = message.split('\n')[0]
        if message.startswith('ERROR:'):
            message = message[6:].strip()
        return message

    @staticmethod
    def sanitize_integrity_error(e: IntegrityError, default_message: str = "Data integrity constraint violation") -> str:
        """
        Extract a safe, user-friendly error message from an IntegrityError.

        Priority:

        1. ``UniqueViolationError`` (SQLSTATE 23505) / ``ForeignKeyViolationError``
           (23503) / declared ``CheckConstraint`` (23514 *with* ``constraint_name``):
           look up the registry, return the registered message on hit, else
           ``default_message``.
        2. Trigger ``RAISE EXCEPTION`` check_violation (23514 *without* ``constraint_name``):
           the message itself is a developer-authored user-facing string; surface
           it directly via ``extract_trigger_message``.
        3. Fallback: log the raw error and return ``default_message``.

        The registry lookup itself is :meth:`lookup_integrity_violation_message`.

        Note: SQLSTATE values are PostgreSQL-specific. For other databases, only
        ``default_message`` will be returned for non-trigger constraint errors.

        :param e: SQLAlchemy ``IntegrityError``
        :param default_message: Fallback message when registry misses and the error
            is not a trigger-raised check_violation
        :returns: A user-safe error description
        """
        friendly = TableBaseMixin.lookup_integrity_violation_message(e)
        if friendly is not None:
            return friendly

        sqlstate, constraint = TableBaseMixin._extract_violation_identity(e)
        logger.warning(f"Data integrity constraint error: constraint={constraint}, sqlstate={sqlstate}, orig={e}")
        return default_message

    @staticmethod
    def _extract_violation_identity(e: IntegrityError) -> tuple[str | None, str | None]:
        """Return ``(sqlstate, constraint_name)`` of an ``IntegrityError``.

        SQLAlchemy's asyncpg adapter keeps the real asyncpg exception in
        ``orig.__cause__`` (the adapter wrapper only forwards ``sqlstate``,
        not ``constraint_name``), so the constraint name falls back to
        ``orig.__cause__`` -- otherwise every registry lookup would miss.

        Either value may be ``None`` (no ``orig`` / non-PostgreSQL error /
        trigger-raised error without a constraint name).
        """
        orig = e.orig
        if orig is None:
            return None, None
        sqlstate = getattr(orig, 'sqlstate', None)
        constraint = getattr(orig, 'constraint_name', None)
        if constraint is None:
            constraint = getattr(orig.__cause__, 'constraint_name', None)
        return sqlstate, constraint

    @staticmethod
    def lookup_integrity_violation_message(e: IntegrityError) -> str | None:
        """Look up the registries: return the **registered business message** on hit, ``None`` otherwise.

        Division of labor with ``sanitize_integrity_error``: this method
        answers "is this constraint a business case the developer registered
        in advance?", ``sanitize_*`` adds a fallback message on top. Callers
        that must branch on hit/miss (e.g. classifying user errors vs.
        platform errors) must use this method -- ``sanitize_*`` collapses both
        cases into a string.

        Pure lookup: no side effects, no logging (log in the caller's context).

        Hit rules match ``sanitize_integrity_error``: ``UniqueViolation``
        (23505) / ``ForeignKeyViolation`` (23503) / declared
        ``CheckConstraint`` (23514 *with* ``constraint_name``) consult their
        registries; a trigger ``RAISE EXCEPTION`` (23514 *without*
        ``constraint_name``) carries a developer-written user-facing message
        and counts as a hit.
        """
        sqlstate, constraint = TableBaseMixin._extract_violation_identity(e)
        if sqlstate is None:
            return None

        if sqlstate == '23505':  # UniqueViolation
            return TableBaseMixin.lookup_unique_violation_message(constraint)
        if sqlstate == '23503':  # ForeignKeyViolation
            return TableBaseMixin.lookup_foreign_key_violation_message(constraint)
        if sqlstate == '23514':  # CheckViolation
            if constraint:
                # Declared CheckConstraint: only return a registered friendly
                # message; never surface the raw message (CheckConstraint
                # expressions may contain column names).
                return TableBaseMixin.lookup_check_violation_message(constraint)
            # Trigger RAISE EXCEPTION: the message is already user-facing.
            # Read it from the real driver exception (``__cause__``) when
            # present -- ``str(adapter_error)`` is prefixed with the driver
            # exception class name, which must not leak to users.
            orig = e.orig
            if orig is None:
                return None
            source = orig.__cause__ if orig.__cause__ is not None else orig
            trigger_msg = TableBaseMixin.extract_trigger_message(source)
            return trigger_msg if trigger_msg else None
        return None

    @classmethod
    async def add(
            cls: type[T],
            session: AsyncSession,
            instances: T | list[T],
            refresh: bool = True,
            commit: bool = True,
    ) -> T | list[T]:
        """
        Add one or more new records to the database.

        :param session: Async database session
        :param instances: Single instance or list of instances to add
        :param refresh: If True, refresh instances after commit to sync DB-generated values
        :param commit: If True, commit the transaction; otherwise only flush
        :returns: The added (and optionally refreshed) instance(s)
        """
        if isinstance(instances, list):
            session.add_all(instances)
        else:
            session.add(instances)

        if commit:
            await session.commit()
        else:
            await session.flush()

        if refresh:
            if isinstance(instances, list):
                for i, instance in enumerate(instances):
                    # After commit objects expire; use sa_inspect to safely read id
                    _insp = cast(InstanceState[Any], inspect(instance))
                    _inst_id = _insp.identity[0] if _insp.identity else None
                    if _inst_id is None:
                        raise RuntimeError(f"{cls.__name__} id is None after add")
                    result = await cls.get(session, cls.id == _inst_id)
                    if result is None:
                        raise RuntimeError(f"{cls.__name__} record not found (id={_inst_id})")
                    instances[i] = result
            else:
                _insp = cast(InstanceState[Any], inspect(instances))
                _inst_id = _insp.identity[0] if _insp.identity else None
                if _inst_id is None:
                    raise RuntimeError(f"{cls.__name__} id is None after add")
                result = await cls.get(session, cls.id == _inst_id)
                if result is None:
                    raise RuntimeError(f"{cls.__name__} record not found (id={_inst_id})")
                instances = result

        return instances

    async def save(
            self: T,
            session: AsyncSession,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            refresh: bool = True,
            commit: bool = True,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            optimistic_retry_count: int | None = None,
    ) -> T:
        """
        Save (insert or update) this instance to the database.

        **Important**: After calling this method, all session objects expire.
        Always use the return value::

            client = await client.save(session)
            return client

        ``updated_at`` is assigned explicitly whenever a persistent instance
        has column changes (not only via the column-level ``onupdate``): under
        joined-table inheritance an update that touches only subclass-table
        columns would otherwise never UPDATE the parent table that holds
        ``updated_at``.

        :param session: Async database session
        :param load: Relationship(s) to eagerly load after save
        :param refresh: Whether to refresh the object after save (default True)
        :param commit: Whether to commit (default True). Set False for batch operations.
        :param jti_subclasses: Polymorphic subclass loading option (requires load)
        :param optimistic_retry_count: Auto-retry count for optimistic lock
            conflicts. ``None`` (default) uses the model policy
            ``__optimistic_retry_default__`` (3 for ``OptimisticLockMixin``
            models, 0 otherwise); an explicit ``0`` demands no retry. A retry
            re-reads the row and re-applies only the columns this instance
            actually modified.
        :returns: The refreshed instance (if refresh=True), otherwise self
        :raises OptimisticLockError: Version mismatch after retries exhausted
        """
        cls = type(self)
        instance = self
        # None = not specified -> model policy. An explicit 0 still means "no
        # retry", so the default cannot be written as 0.
        retries_remaining = (
            optimistic_retry_count if optimistic_retry_count is not None
            else cls.__optimistic_retry_default__
        )
        current_data: dict[str, Any] | None = None

        while True:
            # Snapshot scalar state BEFORE attempting the flush: a failed
            # versioned UPDATE rolls back the inner transaction, which expires
            # every attribute -- inside the except block any attribute read
            # would emit SQL and raise PendingRollbackError/MissingGreenlet.
            # Read via identity/__dict__ only: the instance itself may already
            # be expired by an earlier commit, so plain getattr could emit SQL.
            _pre_insp = cast(InstanceState[Any], inspect(instance))
            instance_id = _pre_insp.identity[0] if _pre_insp.identity else instance.__dict__.get('id')
            instance_version = instance.__dict__.get(OPTIMISTIC_LOCK_VERSION_COLUMN)
            if retries_remaining > 0 and current_data is None:
                # Capture only the columns the caller actually modified (via
                # SQLAlchemy attribute history). Re-applying a full model_dump
                # on retry would overwrite the other transaction's committed
                # values with our stale ones -- the exact lost update the
                # optimistic lock exists to prevent. For a transient instance
                # every set attribute has history, so this also covers INSERTs.
                _hist_insp = cast(InstanceState[Any], inspect(instance))
                current_data = {}
                for _col_attr in _hist_insp.mapper.column_attrs:
                    if _col_attr.key in ('id', OPTIMISTIC_LOCK_VERSION_COLUMN, 'created_at', 'updated_at'):
                        continue
                    if _hist_insp.attrs[_col_attr.key].history.has_changes():
                        current_data[_col_attr.key] = _hist_insp.attrs[_col_attr.key].value

            session.add(instance)
            # Explicit assignment instead of relying on the column-level
            # onupdate: under JTI an update touching only subclass columns
            # never UPDATEs the parent table, so its onupdate would not fire.
            # Only for persistent instances (INSERT timestamps come from the
            # field defaults) whose column attributes actually changed
            # (include_collections=False ignores pure collection changes that
            # would not UPDATE this row).
            _save_state = cast(InstanceState[Any], inspect(instance))
            if _save_state.persistent and session.is_modified(instance, include_collections=False):
                instance.updated_at = now()
            try:
                if commit:
                    await session.commit()
                else:
                    await session.flush()
                break
            except StaleDataError as e:
                await session.rollback()
                if retries_remaining <= 0:
                    raise OptimisticLockError(
                        message=f"{cls.__name__} optimistic lock conflict: record modified by another transaction",
                        model_class=cls.__name__,
                        record_id=str(instance_id) if instance_id is not None else None,
                        expected_version=instance_version,
                        original_error=e,
                    ) from e

                retries_remaining -= 1
                fresh = await cls.get(session, cls.id == instance_id) if instance_id is not None else None
                if fresh is None:
                    raise OptimisticLockError(
                        message=f"{cls.__name__} retry failed: record has been deleted",
                        model_class=cls.__name__,
                        record_id=str(instance_id) if instance_id is not None else None,
                        original_error=e,
                    ) from e

                for key, value in (current_data or {}).items():
                    if hasattr(fresh, key):
                        setattr(fresh, key, value)
                instance = fresh

        if not refresh:
            return instance

        # After commit objects expire; use sa_inspect to safely read id from identity map
        _insp = cast(InstanceState[Any], inspect(instance))
        _instance_id = _insp.identity[0] if _insp.identity else None
        if _instance_id is None:
            raise RuntimeError(f"{cls.__name__} id is None after save")
        result = await cls.get(session, cls.id == _instance_id, load=load, jti_subclasses=jti_subclasses)
        if result is None:
            raise RuntimeError(f"{cls.__name__} record not found (id={_instance_id})")
        return result

    async def update(
            self: T,
            session: AsyncSession,
            other: SQLModelBase,
            extra_data: dict[str, Any] | None = None,
            exclude_unset: bool = True,
            exclude: set[str] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            refresh: bool = True,
            commit: bool = True,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            optimistic_retry_count: int | None = None,
    ) -> T:
        """
        Update this instance using data from another model instance.

        **Important**: After calling this method, all session objects expire.
        Always use the return value.

        ``updated_at`` is assigned explicitly for every non-empty update (see
        ``save()`` for the joined-table-inheritance rationale); an empty update
        (no data and no ``extra_data``) leaves it untouched.

        :param session: Async database session
        :param other: Model instance whose data will be merged into self
        :param extra_data: Additional dict of fields to update
        :param exclude_unset: If True, only fields explicitly set on ``other``
            (``model_fields_set``) are applied (default True). An explicitly
            passed ``None`` counts as set and writes NULL -- build ``other``
            with only the fields to change if ``None`` must not overwrite.
        :param exclude: Field names to exclude from the update
        :param load: Relationship(s) to eagerly load after update
        :param refresh: Whether to refresh after update (default True)
        :param commit: Whether to commit (default True)
        :param jti_subclasses: Polymorphic subclass loading option (requires load)
        :param optimistic_retry_count: Auto-retry count for optimistic lock
            conflicts. ``None`` (default) uses the model policy
            ``__optimistic_retry_default__`` (3 for ``OptimisticLockMixin``
            models, 0 otherwise); an explicit ``0`` demands no retry. A retry
            re-reads the row and re-applies ``other``'s changes to it.
        :returns: The refreshed instance
        :raises OptimisticLockError: Version mismatch after retries exhausted
        """
        cls = type(self)
        update_data = other.model_dump(exclude_unset=exclude_unset, exclude=exclude)
        instance = self
        retries_remaining = (
            optimistic_retry_count if optimistic_retry_count is not None
            else cls.__optimistic_retry_default__
        )

        while True:
            # Snapshot scalar state BEFORE attempting the flush: a failed
            # versioned UPDATE rolls back the inner transaction, which expires
            # every attribute -- inside the except block any attribute read
            # would emit SQL and raise PendingRollbackError/MissingGreenlet.
            # Read via identity/__dict__ only: the instance itself may already
            # be expired by an earlier commit, so plain getattr could emit SQL.
            _pre_insp = cast(InstanceState[Any], inspect(instance))
            instance_id = _pre_insp.identity[0] if _pre_insp.identity else instance.__dict__.get('id')
            instance_version = instance.__dict__.get(OPTIMISTIC_LOCK_VERSION_COLUMN)

            # TableBaseMixin is always used with SQLModelBase; sqlmodel_update provided by SQLModel
            # (the mixin is not a SQLModelBase subclass statically, hence the cast through object)
            _ = cast(SQLModelBase, cast(object, instance)).sqlmodel_update(update_data, update=extra_data)
            if update_data or extra_data:
                # Explicit assignment (see save()): JTI updates touching only
                # subclass columns would not fire the parent's onupdate.
                instance.updated_at = now()
            session.add(instance)

            try:
                if commit:
                    await session.commit()
                else:
                    await session.flush()
                break
            except StaleDataError as e:
                await session.rollback()
                if retries_remaining <= 0:
                    raise OptimisticLockError(
                        message=f"{cls.__name__} optimistic lock conflict: record modified by another transaction",
                        model_class=cls.__name__,
                        record_id=str(instance_id) if instance_id is not None else None,
                        expected_version=instance_version,
                        original_error=e,
                    ) from e

                retries_remaining -= 1
                fresh = await cls.get(session, cls.id == instance_id) if instance_id is not None else None
                if fresh is None:
                    raise OptimisticLockError(
                        message=f"{cls.__name__} retry failed: record has been deleted",
                        model_class=cls.__name__,
                        record_id=str(instance_id) if instance_id is not None else None,
                        original_error=e,
                    ) from e
                instance = fresh

        if not refresh:
            return instance

        # After commit objects expire; use sa_inspect to safely read id from identity map
        _insp = cast(InstanceState[Any], inspect(instance))
        _instance_id = _insp.identity[0] if _insp.identity else None
        if _instance_id is None:
            raise RuntimeError(f"{cls.__name__} id is None after update")
        result = await cls.get(session, cls.id == _instance_id, load=load, jti_subclasses=jti_subclasses)
        if result is None:
            raise RuntimeError(f"{cls.__name__} record not found (id={_instance_id})")
        return result

    # The @overload stubs make the type checker report "no matching overload"
    # when a caller passes neither instances nor condition -- promoting the
    # "must provide instances or condition" runtime business invariant into a
    # compile-time constraint, eliminating ``await obj.delete(session)``
    # (missing-argument) bugs entirely.
    @overload
    @classmethod
    async def delete(
            cls: type[T],
            session: AsyncSession,
            instances: T | list[T],
            *,
            commit: bool = ...,
    ) -> int:
        """Instance-deletion overload: instances is required."""
        ...

    @overload
    @classmethod
    async def delete(
            cls: type[T],
            session: AsyncSession,
            *,
            condition: ColumnElement[bool] | bool,
            commit: bool = ...,
    ) -> int:
        """Condition-deletion overload: the condition kwarg is required."""
        ...

    @classmethod
    async def delete(
            cls: type[T],
            session: AsyncSession,
            instances: T | list[T] | None = None,
            *,
            condition: ColumnElement[bool] | bool | None = None,
            commit: bool = True,
    ) -> int:
        """
        Delete records from the database. Supports instance and condition modes.

        :param session: Async database session
        :param instances: Instance(s) to delete (instance mode)
        :param condition: WHERE condition for bulk delete (condition mode)
        :param commit: Whether to commit after delete (default True)
        :returns: Number of deleted records
        :raises ValueError: If both or neither of instances/condition are provided
        :raises ResourceReferencedError: A target row is still referenced
            through a foreign key (``RESTRICT`` / ``NO ACTION``) -- the row
            **exists** and was not deleted. Raised only when both hold: the
            ``IntegrityError`` was caught here **and** its ``statement`` is a
            ``DELETE`` (a commit flushes every pending operation of the
            session, so an unrelated bad ``INSERT`` surfacing here is
            re-raised untouched). Only SQL issued *inside* this method is
            covered: with ``commit=False`` in instance mode the real
            ``DELETE`` is emitted by the caller's later flush/commit, outside
            this method. PostgreSQL only (relies on SQLSTATE 23503).
        :raises OptimisticLockError: an optimistic-lock conflict occurred
            **within this flush** -- typically the versioned ``DELETE`` of an
            ``OptimisticLockMixin`` model matched 0 rows. The attribution
            reach is the whole flush, not the delete target (a co-flushed
            versioned UPDATE of another object can be the cause), so
            ``model_class`` is the *calling* model, ``record_id`` is always
            ``None`` and ``expected_version`` is ``None``. Never retried (see
            ``__optimistic_retry_default__``). Same ``commit=False`` boundary
            as above; condition mode never raises it (bulk DELETE has no
            per-row version check).
        """
        if instances is not None and condition is not None:
            raise ValueError("Cannot provide both instances and condition")
        if instances is None and condition is None:
            raise ValueError("Must provide either instances or condition")

        deleted_count = 0

        try:
            if condition is not None:
                # cast to ColumnElement[bool]: at runtime condition is always a column expression
                stmt = sql_delete(cls).where(cast(ColumnElement[bool], condition))
                # STI auto-filter: a Core DELETE built from an STI subclass targets the
                # shared table with no discriminator criteria, so without this filter a
                # subclass-level conditional delete would also remove sibling-subclass
                # rows. Same discriminator filter as get()/count().
                sti_condition = cls._sti_descendants_condition()
                if sti_condition is not None:
                    stmt = stmt.where(sti_condition)
                result = cast(CursorResult[Any], await session.execute(stmt))
                deleted_count = result.rowcount
            else:
                if isinstance(instances, list):
                    for instance in instances:
                        await session.delete(instance)
                    deleted_count = len(instances)
                else:
                    await session.delete(instances)
                    deleted_count = 1

            if commit:
                await session.commit()
        except IntegrityError as e:
            sqlstate, constraint = TableBaseMixin._extract_violation_identity(e)
            # "Caught inside delete()" does not prove the direction: the flush
            # also writes every other pending operation of the session. The
            # statement that failed does -- ``IntegrityError.statement`` is
            # generated by SQLAlchemy (independent of server message
            # language). A missing statement is conservatively not translated.
            is_delete_stmt = (
                e.statement is not None
                and e.statement.lstrip().upper().startswith('DELETE')
            )
            if sqlstate == '23503' and is_delete_stmt:  # ForeignKeyViolation caused by a DELETE
                registered = TableBaseMixin.lookup_fk_delete_restrict_message(constraint)
                friendly = registered if registered is not None else FK_DELETE_RESTRICT_FALLBACK_MESSAGE
                logger.warning(
                    f"Delete rejected by foreign key constraint: model={cls.__name__}, "
                    f"constraint={constraint}, registered={registered is not None}"
                )
                raise ResourceReferencedError(friendly, constraint, e) from e
            raise
        except StaleDataError as e:
            # Optimistic-lock conflict (or concurrent delete) -> OptimisticLockError.
            # No direction check is needed (unlike IntegrityError): both
            # possible sources -- this DELETE's version mismatch, or a
            # co-flushed object's UPDATE matching 0 rows -- mean the same to
            # the caller: "re-read and decide again".
            # record_id stays None on purpose: the flush covers the whole
            # session, so naming the delete target would misattribute a
            # conflict that may belong to another row.
            # No rollback: the transaction belongs to the caller.
            logger.warning(
                f"Optimistic lock conflict during delete: calling model={cls.__name__}, "
                f"batch={isinstance(instances, list)} (conflicting row cannot be attributed at flush level)"
            )
            raise OptimisticLockError(
                message=(
                    f"{cls.__name__}.delete detected a concurrent modification: a record in this "
                    f"transaction was modified or deleted by another transaction"
                ),
                model_class=cls.__name__,
                record_id=None,
                expected_version=None,
                original_error=e,
            ) from e

        return deleted_count

    @classmethod
    def _build_time_filters(
            cls: type[T],
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> list[ColumnElement[bool]]:
        """Build time filter conditions using col() for proper column expression types."""
        filters: list[ColumnElement[bool]] = []
        if created_after_datetime is not None:
            filters.append(col(cls.created_at) >= created_after_datetime)
        if created_before_datetime is not None:
            filters.append(col(cls.created_at) < created_before_datetime)
        if updated_after_datetime is not None:
            filters.append(col(cls.updated_at) >= updated_after_datetime)
        if updated_before_datetime is not None:
            filters.append(col(cls.updated_at) < updated_before_datetime)
        return filters

    @classmethod
    def _sti_descendants_condition(cls: type[T]) -> ColumnElement[Any] | None:
        """The "this class and its subclasses" discriminator filter under STI; ``None`` for non-STI models.

        SQLAlchemy does not add ``WHERE discriminator IN (...)`` to STI
        subclass queries built this way (see
        https://github.com/sqlalchemy/sqlalchemy/issues/5018 and
        https://github.com/fastapi/sqlmodel/issues/488). ``get()``,
        ``count()``, ``delete(condition=...)``, the keyset anchor lookup and
        the aggregation helpers share this condition so their visible ranges
        agree.
        """
        if not issubclass(cls, PolymorphicBaseMixin) or cls._is_joined_table_inheritance():
            return None
        mapper = cast(Mapper[Any], inspect(cls))
        poly_on = mapper.polymorphic_on
        if poly_on is None:
            return None
        descendant_identities = [
            m.polymorphic_identity
            for m in mapper.self_and_descendants
            if m.polymorphic_identity is not None
        ]
        if not descendant_identities:
            return None
        return poly_on.in_(descendant_identities)

    @classmethod
    async def _build_keyset_condition(
            cls: type[T],
            session: AsyncSession,
            table_view: PaginationRequest,
            condition: ColumnElement[bool] | bool | None,
            filter_condition: ColumnElement[bool] | bool | None,
    ) -> ColumnElement[bool]:
        """Build the row-value comparison for the ``after_id`` keyset cursor (composite order ``(order column, id)``).

        The anchor's sort value is looked up server-side by primary key --
        the client only passes ``after_id``, avoiding precision loss when
        timestamps round-trip (database microseconds vs. transport
        milliseconds). With ``order='id'`` the id itself is the anchor value
        and no lookup happens.

        Anchor visibility = ``condition`` + ``filter`` + STI filter (the same
        WHERE as the main query). Otherwise anyone holding the UUID of a row
        outside their scope could probe its existence and creation time
        through the difference between a page and an error (a UUID is an
        identifier, not a capability). Time filters are not applied to the
        anchor (a time window is a page boundary, not a visibility boundary).
        Trade-off: once the anchor is deleted or leaves the filter, the
        cursor is invalid and the client restarts from the first page.

        Precondition: ``table_view.after_id`` is not None and ``order`` was
        validated to be immutable (``created_at`` / ``id``).

        :raises KeysetCursorInvalidError: the anchor does not exist **or** is
            outside the visible range (deliberately indistinguishable).
        :raises ValueError: the model's primary key is not a UUID (programming error).
        """
        after_id = table_view.after_id
        if after_id is None:
            raise ValueError("_build_keyset_condition requires table_view.after_id to be set")

        # after_id is a UUID cursor; comparing an int primary key with a UUID
        # is a type error -- reject explicitly instead of an obscure SQL error.
        id_column_type = cast(Mapper[Any], inspect(cls)).columns['id'].type.python_type
        if id_column_type is not uuid.UUID:
            raise ValueError(
                f"the after_id keyset cursor only supports UUID primary keys; {cls.__name__}'s "
                f"primary key is {id_column_type.__name__} -- use offset pagination"
            )

        id_col = col(cls.id)
        order_field = table_view.order if table_view.order is not None else 'created_at'
        if order_field == 'id':
            return (id_col < after_id) if table_view.desc else (id_col > after_id)

        order_col = col(getattr(cls, order_field))
        # The anchor query must use the same FROM as the main query, otherwise
        # its visible range differs:
        # - JTI base: the main query uses with_polymorphic('*'); a condition
        #   may reference a subclass column, which with a bare FROM would add
        #   an unjoined subclass table (cartesian product).
        # - JTI leaf: the mapper selectable is parent JOIN child.
        # - STI / plain models: a single table, select_from(cls) is a no-op.
        anchor_from: type[T] | AliasedClass[T]
        if issubclass(cls, PolymorphicBaseMixin) and cls._is_joined_table_inheritance():
            anchor_from = with_polymorphic(cls, '*')
        else:
            anchor_from = cls
        anchor_stmt = select(order_col).select_from(anchor_from).where(id_col == after_id)
        if condition is not None:
            anchor_stmt = anchor_stmt.where(condition)
        if filter_condition is not None:
            anchor_stmt = anchor_stmt.where(filter_condition)
        sti_condition = cls._sti_descendants_condition()
        if sti_condition is not None:
            anchor_stmt = anchor_stmt.where(sti_condition)
        anchor_value = await session.scalar(anchor_stmt)
        if anchor_value is None:
            raise KeysetCursorInvalidError(
                f"the record after_id={after_id} does not exist or is outside the visible "
                "range of this query; the keyset cursor is invalid, restart from the first page"
            )
        if table_view.desc:
            return (order_col < anchor_value) | ((order_col == anchor_value) & (id_col < after_id))
        return (order_col > anchor_value) | ((order_col == anchor_value) & (id_col > after_id))

    @overload
    @classmethod
    async def get(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["all"],
            join: type['TableBaseMixin'] | tuple[type['TableBaseMixin'], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            skip_locked: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            authoritative: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> list[T]: ...

    @overload
    @classmethod
    async def get(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["one"],
            join: type['TableBaseMixin'] | tuple[type['TableBaseMixin'], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            skip_locked: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            authoritative: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> T: ...

    @overload
    @classmethod
    async def get(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["first"] = ...,
            join: type['TableBaseMixin'] | tuple[type['TableBaseMixin'], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            skip_locked: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            authoritative: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> T | None: ...

    @classmethod
    async def get(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["one", "first", "all"] = "first",
            join: type['TableBaseMixin'] | tuple[type['TableBaseMixin'], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            skip_locked: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            authoritative: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> T | list[T] | None:
        """
        Fetch one or more records from the database with filtering, sorting,
        pagination, joins, and relationship preloading.

        :param session: Async database session
        :param condition: Main query filter (e.g. ``User.id == 1``).
            Type includes ``bool`` because SQLAlchemy ``where(True/False)`` is valid,
            and basedpyright infers SQLModel column expressions as ``bool``.
        :param offset: Pagination offset
        :param limit: Max records to return
        :param fetch_mode: "one", "first", or "all"
        :param join: Model class or (model, ON clause) tuple to JOIN
        :param options: SQLAlchemy query options (e.g. selectinload)
        :param load: Relationship(s) to eagerly load via selectinload.
            Supports nested chains: ``[Parent.children, Child.toys]`` auto-builds
            ``selectinload(children).selectinload(toys)``.
        :param order_by: Sort expressions
        :param filter: Additional filter condition
        :param with_for_update: Use FOR UPDATE row locking. Locked instances are
            tracked in ``session.info[SESSION_FOR_UPDATE_KEY]`` for
            ``@requires_for_update`` decorator verification. A locking read
            also **forces** ``populate_existing``: the database returns the
            latest row, but SQLAlchemy's identity map would otherwise hand
            back an already-loaded object with stale attributes -- a lost
            update in the subsequent read-modify-write. No opt-out.
        :param skip_locked: ``FOR UPDATE SKIP LOCKED`` -- skip rows locked by
            other transactions instead of waiting. Only meaningful with
            ``with_for_update=True`` (ignored otherwise). Intended for work
            queues where N workers each claim a *different* candidate row;
            without it they queue up on the same row. Note that "0 rows"
            then also means "all candidates are locked by others", so never
            use it for existence checks.
        :param table_view: TableViewRequest for pagination + sorting + time
            filtering (explicit arguments take precedence). ``order`` is
            always completed with ``id`` as a same-direction tie-break;
            ``after_id`` applies the keyset cursor (mutually exclusive with an
            explicit ``order_by`` or ``join`` -> ``KeysetCursorUnsupportedError``).
        :param jti_subclasses: Polymorphic subclass loading (requires load param)
        :param populate_existing: Force overwrite identity map objects with DB
            data (lock-free bulk refresh). Not needed with ``with_for_update``.
        :param authoritative: The single switch for **authorization reads**
            ("the result decides whether to allow something, it must be
            authoritative"). At this layer it equals ``populate_existing=True``
            (bypass possibly stale identity-map objects); cached models
            additionally bypass Redis. Merged monotonically: the effective
            value is ``populate_existing or authoritative``.
        :param created_before_datetime: Filter created_at < datetime
        :param created_after_datetime: Filter created_at >= datetime
        :param updated_before_datetime: Filter updated_at < datetime
        :param updated_after_datetime: Filter updated_at >= datetime
        :returns: Single instance, list, or None depending on fetch_mode
        :raises ValueError: Invalid fetch_mode or jti_subclasses without load
        :raises KeysetCursorInvalidError: ``after_id`` anchor missing / not visible
        :raises KeysetCursorUnsupportedError: ``after_id`` combined with ``order_by`` or ``join``
        """
        if jti_subclasses is not None and load is None:
            raise ValueError(
                "jti_subclasses requires the load parameter -- "
                "specify which relationship to load"
            )

        # keyset cursor condition (after_id), appended after condition
        keyset_condition: ColumnElement[bool] | None = None

        # Apply table_view defaults
        if table_view:
            if isinstance(table_view, TimeFilterRequest):
                if created_after_datetime is None and table_view.created_after_datetime is not None:
                    created_after_datetime = table_view.created_after_datetime
                if created_before_datetime is None and table_view.created_before_datetime is not None:
                    created_before_datetime = table_view.created_before_datetime
                if updated_after_datetime is None and table_view.updated_after_datetime is not None:
                    updated_after_datetime = table_view.updated_after_datetime
                if updated_before_datetime is None and table_view.updated_before_datetime is not None:
                    updated_before_datetime = table_view.updated_before_datetime
            if isinstance(table_view, PageWindowRequest):
                if offset is None:
                    offset = table_view.offset
                if limit is None:
                    limit = table_view.limit
            if isinstance(table_view, PaginationRequest):
                # The keyset cursor's no-gap/no-duplicate guarantee depends on
                # the fixed (order column, id) ordering; a custom order_by or
                # a join (the anchor lookup has no join) breaks it. The
                # user-facing message does not reveal which one.
                if table_view.after_id is not None and order_by is not None:
                    logger.info(f"keyset cursor rejected: {cls.__name__} query fixes order_by")
                    raise KeysetCursorUnsupportedError(
                        "this query does not support the after_id keyset cursor; use offset pagination"
                    )
                if table_view.after_id is not None and join is not None:
                    logger.info(f"keyset cursor rejected: {cls.__name__} query uses join")
                    raise KeysetCursorUnsupportedError(
                        "this query does not support the after_id keyset cursor; use offset pagination"
                    )
                if order_by is None:
                    # Resolve the column by name: the order value is constrained
                    # by the request class's Literal (subclasses may add domain
                    # sort columns), so it is always a real column.
                    order_field = table_view.order if table_view.order is not None else 'created_at'
                    order_col = col(getattr(cls, order_field))
                    direction = desc if table_view.desc else asc
                    order_by = [direction(order_col)]
                    # id tie-break: non-unique sort columns (e.g. rows created
                    # in one batch share created_at) have no defined order
                    # among ties, so offset and keyset pages would skip or
                    # repeat rows at page boundaries. Composite-PK tables
                    # without an id column are skipped.
                    if order_field != 'id' and hasattr(cls, 'id'):
                        order_by.append(direction(col(getattr(cls, 'id'))))
                if table_view.after_id is not None:
                    keyset_condition = await cls._build_keyset_condition(session, table_view, condition, filter)

        # Polymorphic base class handling
        polymorphic_cls = None
        is_polymorphic = issubclass(cls, PolymorphicBaseMixin)
        is_jti = is_polymorphic and cls._is_joined_table_inheritance()

        # JTI: always use with_polymorphic (avoids N+1 queries)
        # STI: don't use with_polymorphic
        if is_jti:
            polymorphic_cls = with_polymorphic(cls, '*')
            statement = select(polymorphic_cls)
        else:
            statement = select(cls)

        # STI auto-filter: SQLAlchemy/SQLModel does NOT auto-add WHERE discriminator
        # filter for STI sub-class queries (shared with count()/delete()/keyset).
        sti_condition = cls._sti_descendants_condition()
        if sti_condition is not None:
            statement = statement.where(sti_condition)

        if condition is not None:
            statement = statement.where(condition)

        if keyset_condition is not None:
            statement = statement.where(keyset_condition)

        # Time filters
        for time_filter in cls._build_time_filters(
            created_before_datetime, created_after_datetime,
            updated_before_datetime, updated_after_datetime
        ):
            statement = statement.where(time_filter)

        if join is not None:
            if isinstance(join, tuple):
                statement = statement.join(*join)
            else:
                statement = statement.join(join)

        if options:
            statement = statement.options(*options)

        if load is not None:
            load_list: list[QueryableAttribute[Any]] = load if isinstance(load, list) else [load]
            load_chains = cls._build_load_chains(load_list)

            if jti_subclasses is not None:
                if len(load_chains) > 1 or len(load_chains[0]) > 1:
                    raise ValueError(
                        "jti_subclasses only supports a single relationship (no nested chains)"
                    )
                single_load = load_chains[0][0]
                single_load_rel = cast(RelationshipProperty[Any], single_load.property)
                target_class = single_load_rel.mapper.class_

                if not issubclass(target_class, PolymorphicBaseMixin):
                    raise ValueError(
                        f"Target class {target_class.__name__} is not polymorphic. "
                        f"Ensure it inherits PolymorphicBaseMixin."
                    )

                if jti_subclasses == 'all':
                    subclasses_to_load = await cls._resolve_polymorphic_subclasses(
                        session, condition, single_load, target_class
                    )
                else:
                    subclasses_to_load = jti_subclasses

                if subclasses_to_load:
                    statement = statement.options(
                        selectinload(single_load).selectin_polymorphic(subclasses_to_load)
                    )
                else:
                    statement = statement.options(selectinload(single_load))
            else:
                for chain in load_chains:
                    first_rel = chain[0]
                    first_rel_parent = cast(RelationshipProperty[Any], first_rel.property).parent.class_

                    if (
                        polymorphic_cls is not None
                        and first_rel_parent is not cls
                        and issubclass(first_rel_parent, cls)
                    ):
                        subclass_alias = getattr(polymorphic_cls, first_rel_parent.__name__)
                        rel_name = first_rel.key
                        first_rel_via_poly = getattr(subclass_alias, rel_name)
                        loader = selectinload(first_rel_via_poly)
                    else:
                        loader = selectinload(first_rel)

                    for r in chain[1:]:
                        loader = loader.selectinload(r)
                    statement = statement.options(loader)

        if order_by is not None:
            statement = statement.order_by(*order_by)

        if offset:
            statement = statement.offset(offset)

        if limit:
            statement = statement.limit(limit)

        if filter is not None:
            statement = statement.filter(cast(ColumnElement[bool], filter))

        if with_for_update:
            # For JTI polymorphic models, use FOR UPDATE OF <main_table> to avoid
            # PostgreSQL's restriction on FOR UPDATE with LEFT OUTER JOIN nullable side
            if issubclass(cls, PolymorphicBaseMixin):
                statement = statement.with_for_update(of=cls, skip_locked=skip_locked)
            else:
                statement = statement.with_for_update(skip_locked=skip_locked)

        # A locking read always refreshes the identity map (see the
        # with_for_update parameter doc); populate_existing is the explicit
        # lock-free refresh; authoritative implies it. Merged with ``or`` --
        # never weakening a caller's populate_existing=True.
        if with_for_update or populate_existing or authoritative:
            statement = statement.execution_options(populate_existing=True)

        result = await session.exec(statement)

        if fetch_mode == "one":
            instance = result.one()
            if with_for_update:
                locked: set[int] = session.info.setdefault(SESSION_FOR_UPDATE_KEY, set())
                locked.add(id(instance))
            return instance
        elif fetch_mode == "first":
            instance = result.first()
            if with_for_update and instance is not None:
                locked = session.info.setdefault(SESSION_FOR_UPDATE_KEY, set())
                locked.add(id(instance))
            return instance
        else:
            instances = list(result.all())
            if with_for_update and instances:
                locked = session.info.setdefault(SESSION_FOR_UPDATE_KEY, set())
                for inst in instances:
                    locked.add(id(inst))
            return instances

    @staticmethod
    def _build_load_chains(load_list: list[QueryableAttribute[Any]]) -> list[list[QueryableAttribute[Any]]]:
        """
        Build chained selectinload structures from a flat relationship list.

        Auto-detects dependencies between relationships and builds nested chains.
        For example: ``[Parent.children, Child.toys]`` becomes ``[[children, toys]]``.

        :param load_list: Flat list of relationship attributes
        :returns: List of chains, where each chain is a list of relationships
        """
        if not load_list:
            return []

        rel_info: dict[QueryableAttribute[Any], tuple[type, type]] = {}
        for r in load_list:
            prop = cast(RelationshipProperty[Any], r.property)
            parent_class = prop.parent.class_
            target_class = prop.mapper.class_
            rel_info[r] = (parent_class, target_class)

        predecessors: dict[QueryableAttribute[Any], QueryableAttribute[Any] | None] = {r: None for r in load_list}
        for rel_b in load_list:
            parent_b, _ = rel_info[rel_b]
            for rel_a in load_list:
                if rel_a is rel_b:
                    continue
                _, target_a = rel_info[rel_a]
                if parent_b is target_a:
                    predecessors[rel_b] = rel_a
                    break

        roots = [r for r, pred in predecessors.items() if pred is None]

        chains: list[list[QueryableAttribute[Any]]] = []
        used: set[QueryableAttribute[Any]] = set()

        def _walk_chain(root: QueryableAttribute[Any]) -> None:
            chain = [root]
            used.add(root)
            current = root
            while True:
                _, current_target = rel_info[current]
                next_rel = None
                for r, (parent, _) in rel_info.items():
                    if r not in used and parent is current_target:
                        next_rel = r
                        break
                if next_rel is None:
                    break
                chain.append(next_rel)
                used.add(next_rel)
                current = next_rel
            chains.append(chain)

        for root in roots:
            _walk_chain(root)

        # Cycle guard: a bidirectional pair (A.b + B.a) makes every
        # relationship someone's successor, so it never appears under any
        # root and the requested loads would be silently dropped. Break each
        # remaining cycle at its first-listed member -- list order expresses
        # the caller's intended chain direction.
        for r in load_list:
            if r not in used:
                _walk_chain(r)

        return chains

    @classmethod
    async def _resolve_polymorphic_subclasses(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None,
            load: QueryableAttribute[Any],
            target_class: type[PolymorphicBaseMixin]
    ) -> list[type[PolymorphicBaseMixin]]:
        """
        Query actual polymorphic subclass types in use.

        Avoids loading all possible subclass tables for large hierarchies.
        """
        discriminator = target_class.get_polymorphic_discriminator()
        poly_name_col = getattr(target_class, discriminator)

        relationship_property = cast(RelationshipProperty[Any], load.property)

        if relationship_property.secondary is not None:
            secondary = relationship_property.secondary
            local_cols = list(relationship_property.local_columns)

            type_query = (
                select(distinct(poly_name_col))
                .select_from(target_class)
                .join(secondary)
                .where(secondary.c[local_cols[0].name].in_(
                    select(cls.id).where(condition) if condition is not None else select(cls.id)
                ))
            )
        else:
            local_remote_pairs = relationship_property.local_remote_pairs
            assert local_remote_pairs, f"Relationship {load.key} missing local_remote_pairs"
            local_fk_col = local_remote_pairs[0][0]
            remote_pk_col = local_remote_pairs[0][1]
            type_query = (
                select(distinct(poly_name_col))
                .where(remote_pk_col.in_(
                    select(local_fk_col).where(condition) if condition is not None else select(local_fk_col)
                ))
            )

        type_result = await session.exec(type_query)
        poly_names = list(type_result.all())

        if not poly_names:
            return []

        identity_map = target_class.get_identity_to_class_map()
        return [identity_map[name] for name in poly_names if name in identity_map]

    @classmethod
    async def distinct_column(
            cls: type[T],
            session: AsyncSession,
            column: Mapped[V] | ColumnElement[V],
            condition: ColumnElement[bool] | None = None,
            *,
            limit: int | None = None,
    ) -> list[V]:
        """
        Return the DISTINCT values of one column (optional condition + limit).

        ``get()`` returns whole rows and ``count()`` counts them; this fills
        the "distinct values of a column" gap (e.g. scanning distinct foreign
        keys) with a database-level ``SELECT DISTINCT <column>`` instead of
        loading rows and de-duplicating in Python. The STI subclass filter
        matches ``get()`` / ``count()``.

        :param session: Async database session
        :param column: Column to take distinct values of, e.g. ``col(Model.owner_id)``
        :param condition: Optional WHERE condition
        :param limit: Optional maximum number of values
        :returns: The distinct values
        """
        statement = select(distinct(column)).select_from(cls)

        sti_condition = cls._sti_descendants_condition()
        if sti_condition is not None:
            statement = statement.where(sti_condition)

        if condition is not None:
            statement = statement.where(condition)
        if limit is not None:
            statement = statement.limit(limit)

        result = await session.scalars(statement)
        return list(result.all())

    @classmethod
    async def group_sum(
            cls: type[T],
            session: AsyncSession,
            sum_columns: Sequence[Mapped[Any] | ColumnElement[Any]],
            *,
            group_by: Mapped[GK] | ColumnElement[GK] | None = None,
            condition: ColumnElement[bool] | None = None,
            order_by: ColumnElement[Any] | None = None,
    ) -> list[GroupSumRow[GK]]:
        """
        Aggregate ``SUM`` of each of ``sum_columns`` plus the row count, optionally grouped by ``group_by``.

        Grouping is just a parameter:

        - ``group_by`` omitted (``None``) -> **whole-table aggregation**,
          returns a **single-element** list (``key=None``);
        - ``group_by`` given (a column or expression) -> one row per group
          (e.g. per category, or per ``date_trunc`` time bucket).

        One query computes ``COUNT(*)`` and every ``COALESCE(SUM(col), 0)``;
        the STI subclass filter matches the other aggregates.
        ``GroupSumRow.totals`` is aligned by position with ``sum_columns``.

        For a *conditional* sum (``SUM ... FILTER (WHERE ...)``) call this
        once per condition and merge by ``key`` in Python.

        :param session: Async database session
        :param sum_columns: Numeric columns to sum (each ``COALESCE(SUM(col), 0)``);
            their order is the index order of ``GroupSumRow.totals``
        :param group_by: Group key column / expression; ``None`` for whole-table aggregation
        :param condition: Optional WHERE condition
        :param order_by: Optional ordering (defaults to ``group_by`` ascending when grouping; ignored otherwise)
        :returns: One ``GroupSumRow`` per group; exactly one (``key=None``) without ``group_by``
        :raises ValueError: ``sum_columns`` is empty (use :meth:`count` for row counts)
        """
        n_sums = len(sum_columns)
        if n_sums == 0:
            raise ValueError("group_sum() requires at least one sum column; use count() for plain row counts")
        sum_exprs = [func.coalesce(func.sum(c), 0) for c in sum_columns]

        if group_by is None:
            statement = select(func.count(), *sum_exprs).select_from(cls)
        else:
            statement = select(group_by, func.count(), *sum_exprs).select_from(cls).group_by(group_by)

        sti_condition = cls._sti_descendants_condition()
        if sti_condition is not None:
            statement = statement.where(sti_condition)
        if condition is not None:
            statement = statement.where(condition)

        if group_by is not None:
            statement = statement.order_by(order_by if order_by is not None else group_by)

        rows = (await session.exec(statement)).all()

        def _to_decimal(value: Any) -> Decimal:
            # PostgreSQL NUMERIC already yields Decimal; SQLite may yield int
            # or float (str() avoids binary float artifacts).
            return value if isinstance(value, Decimal) else Decimal(str(value))

        if group_by is None:
            # Without GROUP BY there is always exactly one row:
            # [0] = COUNT(*), [1:] = sums.
            row = rows[0]
            return [GroupSumRow(
                key=None,
                count=row[0],
                totals=[_to_decimal(row[1 + i]) for i in range(n_sums)],
            )]
        # Grouped: [0] = group key, [1] = COUNT(*), [2:] = sums (never NULL thanks to COALESCE).
        return [
            GroupSumRow[GK](
                key=cast(GK, row[0]),
                count=row[1],
                totals=[_to_decimal(row[2 + i]) for i in range(n_sums)],
            )
            for row in rows
        ]

    @classmethod
    async def count(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            distinct_column: Mapped[Any] | ColumnElement[Any] | None = None,
            time_filter: TimeFilterRequest | None = None,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> int:
        """
        Count records matching conditions (supports time filtering and distinct counting).

        Uses database-level COUNT() for efficiency.

        :param session: Async database session
        :param condition: Query condition
        :param distinct_column: When given, count the **distinct values** of
            this column (``COUNT(DISTINCT col)``, e.g. distinct active users);
            otherwise a plain ``COUNT(*)``
        :param time_filter: TimeFilterRequest (takes priority over individual params)
        :param created_before_datetime: Filter created_at < datetime
        :param created_after_datetime: Filter created_at >= datetime
        :param updated_before_datetime: Filter updated_at < datetime
        :param updated_after_datetime: Filter updated_at >= datetime
        :returns: Number of matching records

        Example::

            count = await User.count(
                session,
                created_after_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
                created_before_datetime=datetime(2025, 2, 1, tzinfo=timezone.utc),
            )
        """
        if isinstance(time_filter, TimeFilterRequest):
            if time_filter.created_after_datetime is not None:
                created_after_datetime = time_filter.created_after_datetime
            if time_filter.created_before_datetime is not None:
                created_before_datetime = time_filter.created_before_datetime
            if time_filter.updated_after_datetime is not None:
                updated_after_datetime = time_filter.updated_after_datetime
            if time_filter.updated_before_datetime is not None:
                updated_before_datetime = time_filter.updated_before_datetime

        count_expr = func.count(distinct(distinct_column)) if distinct_column is not None else func.count()
        statement = select(count_expr).select_from(cls)

        # STI sub-class filter (consistent with get())
        sti_condition = cls._sti_descendants_condition()
        if sti_condition is not None:
            statement = statement.where(sti_condition)

        if condition is not None:
            statement = statement.where(condition)

        for time_condition in cls._build_time_filters(
            created_before_datetime, created_after_datetime,
            updated_before_datetime, updated_after_datetime
        ):
            statement = statement.where(time_condition)

        result = await session.scalar(statement)
        # COUNT without GROUP BY always returns one row (0, not NULL); this
        # only narrows scalar()'s ``int | None`` declaration.
        return result if result is not None else 0

    @classmethod
    async def get_with_count(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            join: type['TableBaseMixin'] | tuple[type['TableBaseMixin'], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
    ) -> 'ListResponse[T]':
        """
        Get paginated list with total count, returns ListResponse.

        ``count`` is the size of the whole filtered set; the keyset cursor
        (``after_id``) does not affect it.

        :param session: Async database session
        :param condition: Query condition
        :param join: JOIN target
        :param options: SQLAlchemy query options
        :param load: Relationships to eagerly load
        :param order_by: Sort expressions
        :param filter: Additional filter
        :param table_view: Pagination + sorting + time filtering
        :param jti_subclasses: Polymorphic subclass loading
        :returns: ListResponse with count and items
        """
        time_filter: TimeFilterRequest | None = None
        if table_view is not None:
            time_filter = TimeFilterRequest(
                created_after_datetime=table_view.created_after_datetime,
                created_before_datetime=table_view.created_before_datetime,
                updated_after_datetime=table_view.updated_after_datetime,
                updated_before_datetime=table_view.updated_before_datetime,
            )

        # Items first, then count: get() performs the keyset cursor checks
        # (after_id + order_by/join, anchor validity); counting first would
        # waste an aggregate query on a request that is bound to fail.
        items = await cls.get(
            session,
            condition,
            fetch_mode="all",
            join=join,
            options=options,
            load=load,
            order_by=order_by,
            filter=filter,
            table_view=table_view,
            jti_subclasses=jti_subclasses,
        )

        total_count = await cls.count(session, condition, time_filter=time_filter)

        return ListResponse(count=total_count, items=items)

    @overload
    @classmethod
    async def get_one(
            cls: type[T],
            session: AsyncSession,
            id: int,
            *,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            with_for_update: bool = False,
            authoritative: bool = False,
    ) -> T: ...

    @overload
    @classmethod
    async def get_one(
            cls: type[T],
            session: AsyncSession,
            id: uuid.UUID,
            *,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            with_for_update: bool = False,
            authoritative: bool = False,
    ) -> T: ...

    @classmethod
    async def get_one(
            cls: type[T],
            session: AsyncSession,
            id: int | uuid.UUID,
            *,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            with_for_update: bool = False,
            authoritative: bool = False,
    ) -> T:
        """
        Get a single record by primary key ID (guaranteed to exist).

        Equivalent to ``cls.get(session, col(cls.id) == id, fetch_mode='one', ...)``.

        :param session: Async database session
        :param id: Primary key ID (int or UUID depending on subclass)
        :param load: Relationship(s) to eagerly load
        :param with_for_update: Whether to acquire a row lock
        :param authoritative: The single switch for authorization reads (see
            :meth:`get`); without it "fetch one by id, authoritatively" would
            need the longer ``get(..., fetch_mode='one')`` form, and this more
            natural entry point would silently return a possibly stale object.
        :returns: The model instance
        :raises NoResultFound: Record does not exist
        :raises MultipleResultsFound: Multiple records found
        """
        return await cls.get(
            session, col(cls.id) == id,
            fetch_mode='one', load=load, with_for_update=with_for_update,
            authoritative=authoritative,
        )

    @overload
    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: int, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None, *, detail: str = "Not found", with_for_update: bool = False) -> T: ...

    @overload
    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: uuid.UUID, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None, *, detail: str = "Not found", with_for_update: bool = False) -> T: ...

    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: int | uuid.UUID, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None, *, detail: str = "Not found", with_for_update: bool = False) -> T:
        """
        Get a record by primary key ID, raising 404 if not found.

        If FastAPI is installed, raises ``HTTPException(404)``.
        Otherwise, raises ``RecordNotFoundError``.

        :param session: Async database session
        :param id: Primary key ID
        :param load: Relationship(s) to eagerly load
        :param detail: 404 response detail text (default ``"Not found"``).
            Callers may supply a localized / context-specific message
            (e.g. ``"Bundle not found"``) without falling back to manual
            ``get(...) + null check + raise`` boilerplate.
        :param with_for_update: Forwarded to :meth:`get` -- read the row with
            ``SELECT ... FOR UPDATE`` (which by itself bypasses the Redis
            cache and the identity map). Typical use: close the TOCTOU window
            between "exists" and "delete" -- a concurrent second request
            blocks, then finds no row after the first commits and gets the
            same 404 as a serial second delete.
        :returns: The found instance
        :raises HTTPException: (FastAPI) If not found
        :raises RecordNotFoundError: (no FastAPI) If not found
        """
        instance = await cls.get(session, col(cls.id) == id, load=load, with_for_update=with_for_update)
        if instance is None:
            if _FastAPIHTTPException is not None:
                raise _FastAPIHTTPException(status_code=404, detail=detail)
            raise RecordNotFoundError(detail)
        return instance


class UUIDTableBaseMixin(TableBaseMixin):
    """
    UUID-based async CRUD mixin.

    Inherits all CRUD methods from TableBaseMixin, with the ``id`` field
    overridden to a **UUIDv7** primary key generated on creation.

    UUIDv7 (RFC 9562) starts with a 48-bit Unix millisecond timestamp, so the
    byte order is the creation-time order: ``ORDER BY id`` approximates
    creation order and B-tree inserts concentrate on the right edge (far fewer
    page splits and random I/O than UUIDv4). Generated by
    :func:`sqlmodel_ext.mixins.uuid7` (``uuid.uuid7`` on Python 3.14+).

    Known limitations:

    - The id reveals its creation time (millisecond precision). IDs are
      identifiers, not capabilities; do not rely on them being unguessable.
    - Do not treat id order as authoritative time order: the timestamp comes
      from the application clock, with no global monotonicity across
      processes. Use a database-clock column for authoritative ordering.
    - Existing rows keep whatever UUID version they were created with; mixed
      v4/v7 values still have a well-defined byte order, so ``ORDER BY id``
      remains a consistent total order (e.g. for lock ordering).
    - If a primary key must be *derived* deterministically (e.g. ``uuid5``
      from an idempotency key), assign it explicitly -- the default factory
      only applies when no id is given.

    Attributes:
        id: UUIDv7 primary key, auto-generated.
    """
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    """UUIDv7 primary key (time-ordered), auto-generated."""

    @override
    @classmethod
    async def get_one(
            cls: type[T],
            session: AsyncSession,
            id: uuid.UUID,
            *,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            with_for_update: bool = False,
            authoritative: bool = False,
    ) -> T:
        """
        Get a single record by UUID primary key (guaranteed to exist).

        :param session: Async database session
        :param id: UUID primary key
        :param load: Relationship(s) to eagerly load
        :param with_for_update: Whether to acquire a row lock
        :param authoritative: The single switch for authorization reads (see :meth:`TableBaseMixin.get_one`)
        :returns: The model instance
        """
        return await super().get_one(
            session, id, load=load, with_for_update=with_for_update, authoritative=authoritative,
        )

    @override
    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: uuid.UUID, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None, *, detail: str = "Not found", with_for_update: bool = False) -> T:
        """
        Get a record by UUID primary key, raising 404 if not found.

        :param session: Async database session
        :param id: UUID primary key
        :param load: Relationship(s) to eagerly load
        :param detail: 404 response detail text (default ``"Not found"``)
        :param with_for_update: Read with a row lock (see :meth:`TableBaseMixin.get_exist_one`)
        :returns: The found instance
        :raises HTTPException: (FastAPI) If not found
        :raises RecordNotFoundError: (no FastAPI) If not found
        """
        return await super().get_exist_one(session, id, load, detail=detail, with_for_update=with_for_update)
