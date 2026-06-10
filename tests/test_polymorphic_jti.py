"""
Joined Table Inheritance (JTI): parent and child each own a distinct table.

Declaration pattern (per ``mixins/polymorphic.py`` module docstring):

    1. Root class: ``UUIDTableBaseMixin + PolymorphicBaseMixin`` -- owns the
       parent table and the ``_polymorphic_name`` discriminator column.
    2. ``create_subclass_id_mixin('<parent_table>')`` -- generates a mixin whose
       ``id`` field is simultaneously PK and FK to the parent table. The
       presence of this PK+FK is exactly what the metaclass uses to classify
       the subclass as JTI (instead of STI), giving it its own table.
    3. Concrete subclass: ``(IdMixin, Root, AutoPolymorphicIdentityMixin, table=True)``.

Covered behaviors:
    - child table is distinct; subclass fields live on the child table only
    - child PK is an FK to the parent PK
    - polymorphic identity auto-naming for JTI subclasses
    - insert subclass -> rows land in BOTH tables
    - query through the root returns correctly-typed subclass instances
      (``get()`` uses ``with_polymorphic(cls, '*')`` for JTI roots)
    - querying a subclass returns only that subclass
    - CRUD (save / get_one / save-update / delete) through the mixin API
    - ``create_subclass_id_mixin`` unit behavior (naming, field def, validation)

Models are module-level so the conftest session fixture picks them up during
collection for the two-phase STI/JTI registration (JTI classes are queued too;
the registration phases detect and skip them).
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import class_mapper
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    AutoPolymorphicIdentityMixin,
    PolymorphicBaseMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
    create_subclass_id_mixin,
)


# ==================== module-level JTI hierarchy ====================

class PolyJtiVehicle(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    """JTI root: owns the ``polyjtivehicle`` table + discriminator."""
    name: str


PolyJtiVehicleIdMixin = create_subclass_id_mixin('polyjtivehicle')


class PolyJtiCar(PolyJtiVehicleIdMixin, PolyJtiVehicle, AutoPolymorphicIdentityMixin, table=True):
    """JTI child with its own ``polyjticar`` table."""
    num_doors: int = 4


class PolyJtiBike(PolyJtiVehicleIdMixin, PolyJtiVehicle, AutoPolymorphicIdentityMixin, table=True):
    """Second JTI child -- proves sibling tables stay independent."""
    has_basket: bool = False


# ==================== helpers ====================

async def _count_rows(session: AsyncSession, table) -> int:
    result = await session.execute(select(func.count()).select_from(table))
    return result.scalar_one()


# ==================== schema-level assertions (no DB) ====================

class TestJtiSchema:
    def test_each_class_has_its_own_table(self) -> None:
        assert PolyJtiVehicle.__table__.name == 'polyjtivehicle'  # pyright: ignore[reportAttributeAccessIssue]
        assert PolyJtiCar.__table__.name == 'polyjticar'  # pyright: ignore[reportAttributeAccessIssue]
        assert PolyJtiBike.__table__.name == 'polyjtibike'  # pyright: ignore[reportAttributeAccessIssue]

    def test_subclass_fields_live_on_child_table_only(self) -> None:
        """JTI: subclass-declared columns must NOT leak onto the parent table."""
        assert 'num_doors' in PolyJtiCar.__table__.c  # pyright: ignore[reportAttributeAccessIssue]
        assert 'num_doors' not in PolyJtiVehicle.__table__.c  # pyright: ignore[reportAttributeAccessIssue]
        assert 'has_basket' in PolyJtiBike.__table__.c  # pyright: ignore[reportAttributeAccessIssue]
        assert 'has_basket' not in PolyJtiVehicle.__table__.c  # pyright: ignore[reportAttributeAccessIssue]
        # siblings don't see each other's columns either
        assert 'num_doors' not in PolyJtiBike.__table__.c  # pyright: ignore[reportAttributeAccessIssue]
        assert 'has_basket' not in PolyJtiCar.__table__.c  # pyright: ignore[reportAttributeAccessIssue]

    def test_parent_columns_not_duplicated_on_child_table(self) -> None:
        """The metaclass strips inherited parent columns from the child table."""
        assert 'name' not in PolyJtiCar.__table__.c  # pyright: ignore[reportAttributeAccessIssue]
        assert '_polymorphic_name' not in PolyJtiCar.__table__.c  # pyright: ignore[reportAttributeAccessIssue]

    def test_child_pk_is_fk_to_parent_pk(self) -> None:
        id_col = PolyJtiCar.__table__.c['id']  # pyright: ignore[reportAttributeAccessIssue]
        assert id_col.primary_key is True
        fks = list(id_col.foreign_keys)
        assert len(fks) == 1
        assert fks[0].target_fullname == 'polyjtivehicle.id'

    def test_root_detects_joined_table_inheritance(self) -> None:
        assert PolyJtiVehicle._is_joined_table_inheritance() is True

    def test_auto_polymorphic_identity_is_lowercased_classname(self) -> None:
        """Root has no identity -> subclass identity = classname.lower()."""
        assert class_mapper(PolyJtiCar).polymorphic_identity == 'polyjticar'
        assert class_mapper(PolyJtiBike).polymorphic_identity == 'polyjtibike'

    def test_identity_to_class_map(self) -> None:
        mapping = PolyJtiVehicle.get_identity_to_class_map()
        assert mapping['polyjticar'] is PolyJtiCar
        assert mapping['polyjtibike'] is PolyJtiBike

    def test_get_concrete_subclasses(self) -> None:
        assert set(PolyJtiVehicle.get_concrete_subclasses()) == {PolyJtiCar, PolyJtiBike}

    def test_discriminator_name(self) -> None:
        assert PolyJtiVehicle.get_polymorphic_discriminator() == '_polymorphic_name'


# ==================== create_subclass_id_mixin unit behavior ====================

class TestCreateSubclassIdMixin:
    def test_empty_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            create_subclass_id_mixin('')

    def test_generated_class_name_camelcases_snake_case(self) -> None:
        # never attached to a table -- only the generated class itself is inspected
        mixin = create_subclass_id_mixin('poly_jti_widget')
        assert mixin.__name__ == 'PolyJtiWidgetSubclassIdMixin'
        assert mixin.__qualname__ == 'PolyJtiWidgetSubclassIdMixin'

    def test_id_field_is_pk_fk_with_uuid_factory(self) -> None:
        field_info = PolyJtiVehicleIdMixin.model_fields['id']
        assert getattr(field_info, 'foreign_key', None) == 'polyjtivehicle.id'
        assert getattr(field_info, 'primary_key', None) is True
        assert field_info.default_factory is uuid.uuid4


# ==================== DB-level behavior ====================

@pytest.mark.asyncio
class TestJtiInsertAndQuery:
    async def test_save_returns_concrete_subclass(self, session: AsyncSession) -> None:
        car = await PolyJtiCar(name='mini', num_doors=2).save(session)
        assert isinstance(car, PolyJtiCar)
        assert car.name == 'mini'
        assert car.num_doors == 2
        assert car.id is not None

    async def test_insert_writes_rows_to_both_tables(self, session: AsyncSession) -> None:
        car = await PolyJtiCar(name='sedan', num_doors=4).save(session)

        # parent table row carries shared fields + discriminator
        parent_t = PolyJtiVehicle.__table__  # pyright: ignore[reportAttributeAccessIssue]
        row = (await session.execute(
            select(parent_t.c.name, parent_t.c['_polymorphic_name'])
            .where(parent_t.c.id == car.id)
        )).one()
        assert row.name == 'sedan'
        assert row._polymorphic_name == 'polyjticar'

        # child table row carries the subclass field, keyed by the same id
        child_t = PolyJtiCar.__table__  # pyright: ignore[reportAttributeAccessIssue]
        child_row = (await session.execute(
            select(child_t.c.num_doors).where(child_t.c.id == car.id)
        )).one()
        assert child_row.num_doors == 4

    async def test_parent_query_returns_correct_subclass_types(
        self, session: AsyncSession,
    ) -> None:
        """``get()`` on a JTI root uses with_polymorphic -- one query, typed results."""
        await PolyJtiCar(name='c1', num_doors=2).save(session)
        await PolyJtiBike(name='b1', has_basket=True).save(session)

        vehicles = await PolyJtiVehicle.get(session, fetch_mode='all')
        assert len(vehicles) == 2
        assert {type(v) for v in vehicles} == {PolyJtiCar, PolyJtiBike}

        # subclass fields are eagerly loaded by the polymorphic JOIN
        car = next(v for v in vehicles if isinstance(v, PolyJtiCar))
        bike = next(v for v in vehicles if isinstance(v, PolyJtiBike))
        assert car.num_doors == 2
        assert bike.has_basket is True

    async def test_get_one_via_parent_returns_subclass_instance(
        self, session: AsyncSession,
    ) -> None:
        car = await PolyJtiCar(name='uno', num_doors=3).save(session)

        fetched = await PolyJtiVehicle.get_one(session, car.id)
        assert isinstance(fetched, PolyJtiCar)
        assert fetched.num_doors == 3

    async def test_subclass_query_returns_only_that_subclass(
        self, session: AsyncSession,
    ) -> None:
        await PolyJtiCar(name='c1').save(session)
        await PolyJtiCar(name='c2').save(session)
        await PolyJtiBike(name='b1').save(session)

        cars = await PolyJtiCar.get(session, fetch_mode='all')
        assert len(cars) == 2
        assert all(isinstance(c, PolyJtiCar) for c in cars)

        bikes = await PolyJtiBike.get(session, fetch_mode='all')
        assert len(bikes) == 1
        assert isinstance(bikes[0], PolyJtiBike)


@pytest.mark.asyncio
class TestJtiCrud:
    async def test_modify_and_save_persists_subclass_field(
        self, session: AsyncSession,
    ) -> None:
        car = await PolyJtiCar(name='upgrade-me', num_doors=2).save(session)
        car.num_doors = 5
        car = await car.save(session)

        fetched = await PolyJtiCar.get_one(session, car.id)
        assert fetched.num_doors == 5

        # the child table row itself was updated
        child_t = PolyJtiCar.__table__  # pyright: ignore[reportAttributeAccessIssue]
        stored = (await session.execute(
            select(child_t.c.num_doors).where(child_t.c.id == car.id)
        )).scalar_one()
        assert stored == 5

    async def test_modify_and_save_persists_parent_field(
        self, session: AsyncSession,
    ) -> None:
        bike = await PolyJtiBike(name='old-name').save(session)
        bike.name = 'new-name'
        bike = await bike.save(session)

        parent_t = PolyJtiVehicle.__table__  # pyright: ignore[reportAttributeAccessIssue]
        stored = (await session.execute(
            select(parent_t.c.name).where(parent_t.c.id == bike.id)
        )).scalar_one()
        assert stored == 'new-name'

    async def test_delete_removes_rows_from_both_tables(
        self, session: AsyncSession,
    ) -> None:
        car = await PolyJtiCar(name='doomed').save(session)
        parent_t = PolyJtiVehicle.__table__  # pyright: ignore[reportAttributeAccessIssue]
        child_t = PolyJtiCar.__table__  # pyright: ignore[reportAttributeAccessIssue]
        assert await _count_rows(session, parent_t) == 1
        assert await _count_rows(session, child_t) == 1

        deleted = await PolyJtiCar.delete(session, car)
        assert deleted == 1
        assert await _count_rows(session, parent_t) == 0
        assert await _count_rows(session, child_t) == 0

    async def test_delete_one_sibling_leaves_other_intact(
        self, session: AsyncSession,
    ) -> None:
        car = await PolyJtiCar(name='stays').save(session)
        bike = await PolyJtiBike(name='goes').save(session)

        await PolyJtiBike.delete(session, bike)

        remaining = await PolyJtiVehicle.get(session, fetch_mode='all')
        assert len(remaining) == 1
        assert isinstance(remaining[0], PolyJtiCar)
        assert remaining[0].id == car.id
