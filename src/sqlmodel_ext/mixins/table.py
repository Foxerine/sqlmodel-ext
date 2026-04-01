"""
Table Base Mixins -- async CRUD operations.

Provides TableBaseMixin and UUIDTableBaseMixin with full async CRUD,
pagination, polymorphic query support, relationship preloading,
FOR UPDATE tracking, and type-safe helper functions.
"""
import logging
import uuid
from datetime import datetime
from typing import TypeVar, Literal, override, overload, Any, ClassVar, cast

from sqlalchemy import DateTime, ColumnElement, desc, asc, func, distinct, delete as sql_delete, inspect
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import InstanceState, selectinload, with_polymorphic, QueryableAttribute, Mapper, RelationshipProperty
from sqlalchemy.sql.base import ExecutableOption
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import Field, select, col
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.sql._typing import _OnClauseArgument
from sqlalchemy.ext.asyncio import AsyncAttrs

from sqlmodel_ext._utils import now, now_date
from sqlmodel_ext._exceptions import RecordNotFoundError
from sqlmodel_ext.mixins.optimistic_lock import OptimisticLockError
from sqlmodel_ext.mixins.polymorphic import PolymorphicBaseMixin
from sqlmodel_ext.base import SQLModelBase
from sqlmodel_ext.pagination import (
    ListResponse,
    TimeFilterRequest,
    PaginationRequest,
    TableViewRequest,
)

# Conditional FastAPI import
try:
    from fastapi import HTTPException as _FastAPIHTTPException
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="TableBaseMixin")
M = TypeVar("M", bound="SQLModelBase")

# FOR UPDATE tracking: get(with_for_update=True) records id(instance) to session.info,
# for runtime checking by the @requires_for_update decorator.
SESSION_FOR_UPDATE_KEY = '_for_update_locked'
"""Key in session.info storing the set of id() values for FOR UPDATE locked instances."""


async def safe_reset(session: AsyncSession) -> None:
    """
    session.reset() + clear FOR UPDATE lock tracking.

    reset() releases the transaction and DB connection but does not clear session.info.
    Long-lived sessions (e.g. Taskiq tasks with multiple reset+reuse cycles) accumulate
    stale locked id() values. If Python reuses an object id, @requires_for_update may
    incorrectly treat an unlocked instance as locked.

    :param session: The async session to reset
    """
    await session.reset()
    session.info.pop(SESSION_FOR_UPDATE_KEY, None)


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


class TableBaseMixin(AsyncAttrs):
    """
    Async CRUD operations base mixin for SQLModel models.

    Must be used together with SQLModelBase.

    Provides ``add()``, ``save()``, ``update()``, ``delete()``, ``get()``,
    ``get_one()``, ``get_exist_one()``, ``count()``, and ``get_with_count()`` methods.

    Attributes:
        id: Integer primary key, auto-increment.
        created_at: Record creation timestamp, auto-set.
        updated_at: Record update timestamp, auto-updated.
    """
    _has_table_mixin: ClassVar[bool] = True
    """Internal flag marking TableBaseMixin inheritance."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Accept and forward keyword arguments from subclass definitions."""
        super().__init_subclass__(**kwargs)

    id: int | None = Field(default=None, primary_key=True)

    created_at: datetime = Field(default_factory=now, sa_type=DateTime(timezone=True))
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={'default': now, 'onupdate': now},
        default_factory=now
    )

    @staticmethod
    def sanitize_integrity_error(e: IntegrityError, default_message: str = "Data integrity constraint violation") -> str:
        """
        Extract a safe, user-friendly error message from an IntegrityError.

        PostgreSQL triggers (``RAISE EXCEPTION ... USING ERRCODE = 'check_violation'``)
        produce SQLSTATE 23514 errors with business-semantic messages that can be
        shown directly to users. Other constraint errors (FK, unique, etc.) may leak
        table structure information and need sanitizing.

        Note: This method is PostgreSQL-specific. For other databases, only the
        default_message will be returned for non-trigger constraint errors.

        :param e: SQLAlchemy IntegrityError
        :param default_message: Fallback message for non-trigger constraint errors
        :returns: A user-safe error description
        """
        orig = e.orig
        # check_violation (SQLSTATE 23514): produced by trigger RAISE EXCEPTION
        if orig is not None and getattr(orig, 'sqlstate', None) == '23514':
            error_msg = str(orig)
            # PostgreSQL format: "ERROR: message\nDETAIL: ...\nCONTEXT: ..."
            if '\n' in error_msg:
                error_msg = error_msg.split('\n')[0]
            if error_msg.startswith('ERROR:'):
                error_msg = error_msg[6:].strip()
            return error_msg
        logger.warning(f"Data integrity constraint error: {e}")
        return default_message

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
                    assert _inst_id is not None, f"{cls.__name__} id is None after add"
                    result = await cls.get(session, cls.id == _inst_id)
                    assert result is not None, f"{cls.__name__} record not found (id={_inst_id})"
                    instances[i] = result
            else:
                _insp = cast(InstanceState[Any], inspect(instances))
                _inst_id = _insp.identity[0] if _insp.identity else None
                assert _inst_id is not None, f"{cls.__name__} id is None after add"
                result = await cls.get(session, cls.id == _inst_id)
                assert result is not None, f"{cls.__name__} record not found (id={_inst_id})"
                instances = result

        return instances

    async def save(
            self: T,
            session: AsyncSession,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            refresh: bool = True,
            commit: bool = True,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            optimistic_retry_count: int = 0,
    ) -> T:
        """
        Save (insert or update) this instance to the database.

        **Important**: After calling this method, all session objects expire.
        Always use the return value::

            client = await client.save(session)
            return client

        :param session: Async database session
        :param load: Relationship(s) to eagerly load after save
        :param refresh: Whether to refresh the object after save (default True)
        :param commit: Whether to commit (default True). Set False for batch operations.
        :param jti_subclasses: Polymorphic subclass loading option (requires load)
        :param optimistic_retry_count: Auto-retry count for optimistic lock conflicts (default 0)
        :returns: The refreshed instance (if refresh=True), otherwise self
        :raises OptimisticLockError: Version mismatch after retries exhausted
        """
        cls = type(self)
        instance = self
        retries_remaining = optimistic_retry_count
        current_data: dict[str, Any] | None = None

        while True:
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
                        record_id=str(getattr(instance, 'id', None)),
                        expected_version=getattr(instance, 'version', None),
                        original_error=e,
                    ) from e

                retries_remaining -= 1
                if current_data is None:
                    # TableBaseMixin is always used with SQLModelBase; model_dump provided by Pydantic
                    current_data = cast(SQLModelBase, self).model_dump(exclude={'id', 'version', 'created_at', 'updated_at'})

                fresh = await cls.get(session, cls.id == self.id)
                if fresh is None:
                    raise OptimisticLockError(
                        message=f"{cls.__name__} retry failed: record has been deleted",
                        model_class=cls.__name__,
                        record_id=str(getattr(self, 'id', None)),
                        original_error=e,
                    ) from e

                for key, value in current_data.items():
                    if hasattr(fresh, key):
                        setattr(fresh, key, value)
                instance = fresh

        if not refresh:
            return instance

        # After commit objects expire; use sa_inspect to safely read id from identity map
        _insp = cast(InstanceState[Any], inspect(instance))
        _instance_id = _insp.identity[0] if _insp.identity else None
        assert _instance_id is not None, f"{cls.__name__} id is None after save"
        result = await cls.get(session, cls.id == _instance_id, load=load, jti_subclasses=jti_subclasses)
        assert result is not None, f"{cls.__name__} record not found (id={_instance_id})"
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
            optimistic_retry_count: int = 0,
    ) -> T:
        """
        Update this instance using data from another model instance.

        **Important**: After calling this method, all session objects expire.
        Always use the return value.

        :param session: Async database session
        :param other: Model instance whose data will be merged into self
        :param extra_data: Additional dict of fields to update
        :param exclude_unset: If True, skip unset fields from other (default True)
        :param exclude: Field names to exclude from the update
        :param load: Relationship(s) to eagerly load after update
        :param refresh: Whether to refresh after update (default True)
        :param commit: Whether to commit (default True)
        :param jti_subclasses: Polymorphic subclass loading option (requires load)
        :param optimistic_retry_count: Auto-retry count for optimistic lock conflicts (default 0)
        :returns: The refreshed instance
        :raises OptimisticLockError: Version mismatch after retries exhausted
        """
        cls = type(self)
        update_data = other.model_dump(exclude_unset=exclude_unset, exclude=exclude)
        instance = self
        retries_remaining = optimistic_retry_count

        while True:
            # TableBaseMixin is always used with SQLModelBase; sqlmodel_update provided by SQLModel
            _ = cast(SQLModelBase, instance).sqlmodel_update(update_data, update=extra_data)
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
                        record_id=str(getattr(instance, 'id', None)),
                        expected_version=getattr(instance, 'version', None),
                        original_error=e,
                    ) from e

                retries_remaining -= 1
                fresh = await cls.get(session, cls.id == self.id)
                if fresh is None:
                    raise OptimisticLockError(
                        message=f"{cls.__name__} retry failed: record has been deleted",
                        model_class=cls.__name__,
                        record_id=str(getattr(self, 'id', None)),
                        original_error=e,
                    ) from e
                instance = fresh

        if not refresh:
            return instance

        # After commit objects expire; use sa_inspect to safely read id from identity map
        _insp = cast(InstanceState[Any], inspect(instance))
        _instance_id = _insp.identity[0] if _insp.identity else None
        assert _instance_id is not None, f"{cls.__name__} id is None after update"
        result = await cls.get(session, cls.id == _instance_id, load=load, jti_subclasses=jti_subclasses)
        assert result is not None, f"{cls.__name__} record not found (id={_instance_id})"
        return result

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
        """
        if instances is not None and condition is not None:
            raise ValueError("Cannot provide both instances and condition")
        if instances is None and condition is None:
            raise ValueError("Must provide either instances or condition")

        deleted_count = 0

        if condition is not None:
            # cast to ColumnElement[bool]: at runtime condition is always a column expression
            stmt = sql_delete(cls).where(cast(ColumnElement[bool], condition))
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
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
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
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
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
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
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
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
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
            ``@requires_for_update`` decorator verification.
        :param table_view: TableViewRequest for pagination + sorting + time filtering
        :param jti_subclasses: Polymorphic subclass loading (requires load param)
        :param populate_existing: Force overwrite identity map objects with DB data
        :param created_before_datetime: Filter created_at < datetime
        :param created_after_datetime: Filter created_at >= datetime
        :param updated_before_datetime: Filter updated_at < datetime
        :param updated_after_datetime: Filter updated_at >= datetime
        :returns: Single instance, list, or None depending on fetch_mode
        :raises ValueError: Invalid fetch_mode or jti_subclasses without load
        """
        if jti_subclasses is not None and load is None:
            raise ValueError(
                "jti_subclasses requires the load parameter -- "
                "specify which relationship to load"
            )

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
            if isinstance(table_view, PaginationRequest):
                if offset is None:
                    offset = table_view.offset
                if limit is None:
                    limit = table_view.limit
                if order_by is None:
                    order_col = col(cls.created_at) if table_view.order == "created_at" else col(cls.updated_at)
                    order_clause: ColumnElement[Any] = desc(order_col) if table_view.desc else asc(order_col)
                    order_by = [order_clause]

        # Polymorphic base class handling
        polymorphic_cls = None
        is_polymorphic = issubclass(cls, PolymorphicBaseMixin)
        is_jti = is_polymorphic and cls._is_joined_table_inheritance()
        is_sti = is_polymorphic and not cls._is_joined_table_inheritance()

        # JTI: always use with_polymorphic (avoids N+1 queries)
        # STI: don't use with_polymorphic
        if is_jti:
            polymorphic_cls = with_polymorphic(cls, '*')
            statement = select(polymorphic_cls)
        else:
            statement = select(cls)

        # STI auto-filter: SQLAlchemy/SQLModel does NOT auto-add WHERE discriminator
        # filter for STI sub-class queries. We manually add WHERE _polymorphic_name IN (...)
        # using mapper.self_and_descendants to include the class and all its children.
        if is_sti:
            mapper = cast(Mapper[Any], inspect(cls))
            poly_on = mapper.polymorphic_on
            if poly_on is not None:
                descendant_identities = [
                    m.polymorphic_identity
                    for m in mapper.self_and_descendants
                    if m.polymorphic_identity is not None
                ]
                if descendant_identities:
                    statement = statement.where(poly_on.in_(descendant_identities))

        if condition is not None:
            statement = statement.where(condition)

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
                statement = statement.with_for_update(of=cls)
            else:
                statement = statement.with_for_update()

        if populate_existing:
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

        for root in roots:
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
    async def count(
            cls: type[T],
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            time_filter: TimeFilterRequest | None = None,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> int:
        """
        Count records matching conditions (supports time filtering).

        Uses database-level COUNT() for efficiency.

        :param session: Async database session
        :param condition: Query condition
        :param time_filter: TimeFilterRequest (takes priority over individual params)
        :param created_before_datetime: Filter created_at < datetime
        :param created_after_datetime: Filter created_at >= datetime
        :param updated_before_datetime: Filter updated_at < datetime
        :param updated_after_datetime: Filter updated_at >= datetime
        :returns: Number of matching records
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

        statement = select(func.count()).select_from(cls)

        # STI sub-class filter (consistent with get())
        is_polymorphic = issubclass(cls, PolymorphicBaseMixin)
        is_sti = is_polymorphic and not cls._is_joined_table_inheritance()
        if is_sti:
            mapper = cast(Mapper[Any], inspect(cls))
            poly_on = mapper.polymorphic_on
            if poly_on is not None:
                descendant_identities = [
                    m.polymorphic_identity
                    for m in mapper.self_and_descendants
                    if m.polymorphic_identity is not None
                ]
                if descendant_identities:
                    statement = statement.where(poly_on.in_(descendant_identities))

        if condition is not None:
            statement = statement.where(condition)

        for time_condition in cls._build_time_filters(
            created_before_datetime, created_after_datetime,
            updated_before_datetime, updated_after_datetime
        ):
            statement = statement.where(time_condition)

        result = await session.scalar(statement)
        return result or 0

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

        total_count = await cls.count(session, condition, time_filter=time_filter)

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
    ) -> T: ...

    @classmethod
    async def get_one(
            cls: type[T],
            session: AsyncSession,
            id: int | uuid.UUID,
            *,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            with_for_update: bool = False,
    ) -> T:
        """
        Get a single record by primary key ID (guaranteed to exist).

        Equivalent to ``cls.get(session, col(cls.id) == id, fetch_mode='one', ...)``.

        :param session: Async database session
        :param id: Primary key ID (int or UUID depending on subclass)
        :param load: Relationship(s) to eagerly load
        :param with_for_update: Whether to acquire a row lock
        :returns: The model instance
        :raises NoResultFound: Record does not exist
        :raises MultipleResultsFound: Multiple records found
        """
        return await cls.get(
            session, col(cls.id) == id,
            fetch_mode='one', load=load, with_for_update=with_for_update,
        )

    @overload
    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: int, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None) -> T: ...

    @overload
    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: uuid.UUID, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None) -> T: ...

    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: int | uuid.UUID, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None) -> T:
        """
        Get a record by primary key ID, raising 404 if not found.

        If FastAPI is installed, raises ``HTTPException(404)``.
        Otherwise, raises ``RecordNotFoundError``.

        :param session: Async database session
        :param id: Primary key ID
        :param load: Relationship(s) to eagerly load
        :returns: The found instance
        :raises HTTPException: (FastAPI) If not found
        :raises RecordNotFoundError: (no FastAPI) If not found
        """
        instance = await cls.get(session, col(cls.id) == id, load=load)
        if instance is None:
            if _HAS_FASTAPI:
                raise _FastAPIHTTPException(status_code=404, detail="Not found")
            raise RecordNotFoundError("Not found")
        return instance


class UUIDTableBaseMixin(TableBaseMixin):
    """
    UUID-based async CRUD mixin.

    Inherits all CRUD methods from TableBaseMixin, with the ``id`` field
    overridden to use UUID with auto-generation.

    Attributes:
        id: UUID primary key, auto-generated.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    """UUID primary key, auto-generated."""

    @override
    @classmethod
    async def get_one(
            cls: type[T],
            session: AsyncSession,
            id: uuid.UUID,
            *,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            with_for_update: bool = False,
    ) -> T:
        """
        Get a single record by UUID primary key (guaranteed to exist).

        :param session: Async database session
        :param id: UUID primary key
        :param load: Relationship(s) to eagerly load
        :param with_for_update: Whether to acquire a row lock
        :returns: The model instance
        """
        return await super().get_one(session, id, load=load, with_for_update=with_for_update)

    @override
    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: uuid.UUID, load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None) -> T:
        """
        Get a record by UUID primary key, raising 404 if not found.

        :param session: Async database session
        :param id: UUID primary key
        :param load: Relationship(s) to eagerly load
        :returns: The found instance
        :raises HTTPException: (FastAPI) If not found
        :raises RecordNotFoundError: (no FastAPI) If not found
        """
        return await super().get_exist_one(session, id, load)
