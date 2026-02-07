"""
Table Base Mixins -- async CRUD operations.

Provides TableBaseMixin and UUIDTableBaseMixin with full async CRUD,
pagination, polymorphic query support, and relationship preloading.
"""
import uuid
from datetime import datetime
from typing import TypeVar, Literal, override, Any, ClassVar

from sqlalchemy import DateTime, BinaryExpression, ClauseElement, desc, asc, func, distinct, delete as sql_delete, inspect
from sqlalchemy.orm import selectinload, Relationship, with_polymorphic
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import Field, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.sql._typing import _OnClauseArgument
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlmodel.main import RelationshipInfo

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

T = TypeVar("T", bound="TableBaseMixin")
M = TypeVar("M", bound="SQLModelBase")


class TableBaseMixin(AsyncAttrs):
    """
    Async CRUD operations base mixin for SQLModel models.

    Must be used together with SQLModelBase.

    Provides ``add()``, ``save()``, ``update()``, ``delete()``, ``get()``,
    ``count()``, ``get_with_count()``, and ``get_exist_one()`` methods.

    Attributes:
        id: Integer primary key, auto-increment.
        created_at: Record creation timestamp, auto-set.
        updated_at: Record update timestamp, auto-updated.
    """
    _has_table_mixin: ClassVar[bool] = True
    """Internal flag marking TableBaseMixin inheritance."""

    def __init_subclass__(cls, **kwargs):
        """Accept and forward keyword arguments from subclass definitions."""
        super().__init_subclass__(**kwargs)

    id: int | None = Field(default=None, primary_key=True)

    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(
        sa_type=DateTime,
        sa_column_kwargs={'default': now, 'onupdate': now},
        default_factory=now
    )

    @classmethod
    async def add(cls: type[T], session: AsyncSession, instances: T | list[T], refresh: bool = True) -> T | list[T]:
        """
        Add one or more new records to the database.

        :param session: Async database session
        :param instances: Single instance or list of instances to add
        :param refresh: If True, refresh instances after commit to sync DB-generated values
        :returns: The added (and optionally refreshed) instance(s)
        """
        is_list = False
        if isinstance(instances, list):
            is_list = True
            session.add_all(instances)
        else:
            session.add(instances)

        await session.commit()

        if refresh:
            if is_list:
                for instance in instances:
                    await session.refresh(instance)
            else:
                await session.refresh(instances)

        return instances

    async def save(
            self: T,
            session: AsyncSession,
            load: RelationshipInfo | list[RelationshipInfo] | None = None,
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
                    current_data = self.model_dump(exclude={'id', 'version', 'created_at', 'updated_at'})

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

        if load is not None:
            await session.refresh(instance)
            return await cls.get(session, cls.id == instance.id, load=load, jti_subclasses=jti_subclasses)
        else:
            await session.refresh(instance)
            return instance

    async def update(
            self: T,
            session: AsyncSession,
            other: M,
            extra_data: dict[str, Any] | None = None,
            exclude_unset: bool = True,
            exclude: set[str] | None = None,
            load: RelationshipInfo | list[RelationshipInfo] | None = None,
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
            instance.sqlmodel_update(update_data, update=extra_data)
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

        if load is not None:
            await session.refresh(instance)
            return await cls.get(session, cls.id == instance.id, load=load, jti_subclasses=jti_subclasses)
        else:
            await session.refresh(instance)
            return instance

    @classmethod
    async def delete(
            cls: type[T],
            session: AsyncSession,
            instances: T | list[T] | None = None,
            *,
            condition: BinaryExpression | ClauseElement | None = None,
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
            stmt = sql_delete(cls).where(condition)
            result = await session.execute(stmt)
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
    ) -> list[BinaryExpression]:
        """Build time filter conditions."""
        filters: list[BinaryExpression] = []
        if created_after_datetime is not None:
            filters.append(cls.created_at >= created_after_datetime)
        if created_before_datetime is not None:
            filters.append(cls.created_at < created_before_datetime)
        if updated_after_datetime is not None:
            filters.append(cls.updated_at >= updated_after_datetime)
        if updated_before_datetime is not None:
            filters.append(cls.updated_at < updated_before_datetime)
        return filters

    @classmethod
    async def get(
            cls: type[T],
            session: AsyncSession,
            condition: BinaryExpression | ClauseElement | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["one", "first", "all"] = "first",
            join: type[T] | tuple[type[T], _OnClauseArgument] | None = None,
            options: list | None = None,
            load: RelationshipInfo | list[RelationshipInfo] | None = None,
            order_by: list[ClauseElement] | None = None,
            filter: BinaryExpression | ClauseElement | None = None,
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
        :param condition: Main query filter (e.g. ``User.id == 1``)
        :param offset: Pagination offset
        :param limit: Max records to return
        :param fetch_mode: "one", "first", or "all"
        :param join: Model class or (model, ON clause) tuple to JOIN
        :param options: SQLAlchemy query options (e.g. selectinload)
        :param load: Relationship(s) to eagerly load via selectinload
        :param order_by: Sort expressions
        :param filter: Additional filter condition
        :param with_for_update: Use FOR UPDATE row locking
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
                    order_column = cls.created_at if table_view.order == "created_at" else cls.updated_at
                    order_by = [desc(order_column) if table_view.desc else asc(order_column)]

        # Polymorphic base class handling
        polymorphic_cls = None
        is_polymorphic = issubclass(cls, PolymorphicBaseMixin)
        is_jti = is_polymorphic and cls._is_joined_table_inheritance()
        is_sti = is_polymorphic and not cls._is_joined_table_inheritance()

        if is_jti:
            polymorphic_cls = with_polymorphic(cls, '*')
            statement = select(polymorphic_cls)
        else:
            statement = select(cls)

        # STI auto-filter
        if issubclass(cls, PolymorphicBaseMixin) and not cls._is_joined_table_inheritance():
            mapper = inspect(cls)
            if mapper.polymorphic_identity is not None and not mapper.polymorphic_abstract:
                poly_on = mapper.polymorphic_on
                if poly_on is not None:
                    statement = statement.where(poly_on == mapper.polymorphic_identity)

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

        if load:
            load_list = load if isinstance(load, list) else [load]
            load_chains = cls._build_load_chains(load_list)

            if jti_subclasses is not None:
                if len(load_chains) > 1 or len(load_chains[0]) > 1:
                    raise ValueError(
                        "jti_subclasses only supports a single relationship (no nested chains)"
                    )
                single_load = load_chains[0][0]
                target_class = single_load.property.mapper.class_

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
                    first_rel_parent = first_rel.property.parent.class_

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

                    for rel in chain[1:]:
                        loader = loader.selectinload(rel)
                    statement = statement.options(loader)

        if order_by is not None:
            statement = statement.order_by(*order_by)

        if offset:
            statement = statement.offset(offset)

        if limit:
            statement = statement.limit(limit)

        if filter:
            statement = statement.filter(filter)

        if with_for_update:
            if issubclass(cls, PolymorphicBaseMixin):
                statement = statement.with_for_update(of=cls)
            else:
                statement = statement.with_for_update()

        if populate_existing:
            statement = statement.execution_options(populate_existing=True)

        result = await session.exec(statement)

        if fetch_mode == "one":
            return result.one()
        elif fetch_mode == "first":
            return result.first()
        elif fetch_mode == "all":
            return list(result.all())
        else:
            raise ValueError(f"Invalid fetch_mode: {fetch_mode}")

    @staticmethod
    def _build_load_chains(load_list: list[RelationshipInfo]) -> list[list[RelationshipInfo]]:
        """
        Build chained selectinload structures from a flat relationship list.

        Auto-detects dependencies between relationships and builds nested chains.
        For example: ``[Parent.children, Child.toys]`` becomes ``[[children, toys]]``.

        :param load_list: Flat list of relationship attributes
        :returns: List of chains, where each chain is a list of relationships
        """
        if not load_list:
            return []

        rel_info: dict[RelationshipInfo, tuple[type, type]] = {}
        for rel in load_list:
            parent_class = rel.property.parent.class_
            target_class = rel.property.mapper.class_
            rel_info[rel] = (parent_class, target_class)

        predecessors: dict[RelationshipInfo, RelationshipInfo | None] = {rel: None for rel in load_list}
        for rel_b in load_list:
            parent_b, _ = rel_info[rel_b]
            for rel_a in load_list:
                if rel_a is rel_b:
                    continue
                _, target_a = rel_info[rel_a]
                if parent_b is target_a:
                    predecessors[rel_b] = rel_a
                    break

        roots = [rel for rel, pred in predecessors.items() if pred is None]

        chains: list[list[RelationshipInfo]] = []
        used: set[RelationshipInfo] = set()

        for root in roots:
            chain = [root]
            used.add(root)
            current = root
            while True:
                _, current_target = rel_info[current]
                next_rel = None
                for rel, (parent, _) in rel_info.items():
                    if rel not in used and parent is current_target:
                        next_rel = rel
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
            condition: BinaryExpression | ClauseElement | None,
            load: RelationshipInfo,
            target_class: type[PolymorphicBaseMixin]
    ) -> list[type[PolymorphicBaseMixin]]:
        """
        Query actual polymorphic subclass types in use.

        Avoids loading all possible subclass tables for large hierarchies.
        """
        discriminator = target_class.get_polymorphic_discriminator()
        poly_name_col = getattr(target_class, discriminator)

        relationship_property = load.property

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
            local_fk_col = relationship_property.local_remote_pairs[0][0]
            remote_pk_col = relationship_property.local_remote_pairs[0][1]
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
            condition: BinaryExpression | ClauseElement | None = None,
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
            condition: BinaryExpression | ClauseElement | None = None,
            *,
            join: type[T] | tuple[type[T], _OnClauseArgument] | None = None,
            options: list | None = None,
            load: RelationshipInfo | list[RelationshipInfo] | None = None,
            order_by: list[ClauseElement] | None = None,
            filter: BinaryExpression | ClauseElement | None = None,
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

    @classmethod
    async def get_exist_one(cls: type[T], session: AsyncSession, id: int, load: RelationshipInfo | list[RelationshipInfo] | None = None) -> T:
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
        instance = await cls.get(session, cls.id == id, load=load)
        if not instance:
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
    async def get_exist_one(cls: type[T], session: AsyncSession, id: uuid.UUID, load: Relationship | None = None) -> T:
        """
        Get a record by UUID primary key, raising 404 if not found.

        :param session: Async database session
        :param id: UUID primary key
        :param load: Relationship(s) to eagerly load
        :returns: The found instance
        :raises HTTPException: (FastAPI) If not found
        :raises RecordNotFoundError: (no FastAPI) If not found
        """
        return await super().get_exist_one(session, id, load)  # type: ignore
