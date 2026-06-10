"""
STI polymorphic queries, identity auto-naming, subclass CRUD, and edge cases.

Hierarchy (all sharing the ``polystianimal`` table):

    PolyStiAnimal (root, PolymorphicBaseMixin)
    +-- PolyStiDog                       identity 'polystidog'
    |   +-- PolyStiPuppy (3rd level)     identity 'polystidog.polystipuppy'
    +-- PolyStiCat (StrEnum field)       identity 'polysticat'
    +-- PolyStiBird (explicit identity)  identity 'poly-bird'

Plus a second hierarchy for same-named-column handling:

    PolyConflictRoot
    +-- PolyConflictText      shared_val: str  -> registers String column
    +-- PolyConflictNumber    shared_val: int  -> Integer vs String = conflict (TypeError)
    +-- PolyConflictTextTwin  shared_val: str  -> compatible, shares the column

Covered behaviors:
    - root `get()` returns the correct concrete subclass per row (manual
      discriminator WHERE filter added by ``TableBaseMixin.get``)
    - subclass `get()` only returns that subclass branch (incl. descendants)
    - AutoPolymorphicIdentityMixin naming: '{parent_identity}.{classname}' /
      classname.lower() at the first level / explicit override via class kwarg
    - 3rd-level STI subclass: column registration + typed loading via root query
    - StrEnum columns are downcast to String on the shared table but coerced
      back to the StrEnum type on load and on __init__
    - CRUD integration on STI subclasses (save / get / update / delete / count)
    - same-named column conflict: incompatible types raise TypeError,
      compatible types share one column

Models are module-level so the conftest session-scoped ``_register_sti``
fixture registers them (two-phase) before any test runs.
"""
from __future__ import annotations

from enum import StrEnum

import pytest
from sqlalchemy import Enum as SAEnum, Integer, String, select
from sqlalchemy.orm import class_mapper
from sqlmodel import col
from sqlmodel.sql.sqltypes import AutoString
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    AutoPolymorphicIdentityMixin,
    PolymorphicBaseMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
)


# ==================== module-level STI hierarchy ====================

class PolyStiMood(StrEnum):
    happy = 'happy'
    grumpy = 'grumpy'


class PolyStiAnimal(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    """STI root owning the shared table."""
    name: str


class PolyStiDog(PolyStiAnimal, AutoPolymorphicIdentityMixin, table=True):
    breed: str | None = None


class PolyStiPuppy(PolyStiDog, table=True):
    """Third level -- AutoPolymorphicIdentityMixin is inherited through PolyStiDog."""
    toy: str | None = None


class PolyStiCat(PolyStiAnimal, AutoPolymorphicIdentityMixin, table=True):
    mood: PolyStiMood = PolyStiMood.happy


class PolyStiBird(
    PolyStiAnimal,
    AutoPolymorphicIdentityMixin,
    table=True,
    polymorphic_identity='poly-bird',
):
    """Explicit identity via class kwarg overrides the auto-naming rule."""
    wing_span: float | None = None


class PolyStiDogPatch(SQLModelBase):
    """Plain (non-table) DTO used for ``update()``."""
    breed: str | None = None


# ==================== same-named-column hierarchy ====================

class PolyConflictRoot(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    name: str


class PolyConflictText(PolyConflictRoot, AutoPolymorphicIdentityMixin, table=True):
    """First to register ``shared_val`` -> the shared column becomes String."""
    shared_val: str | None = None


class PolyConflictNumber(PolyConflictRoot, AutoPolymorphicIdentityMixin, table=True):
    """Integer vs existing String column: documented as an incompatible conflict."""
    shared_val: int = 0


class PolyConflictTextTwin(PolyConflictRoot, AutoPolymorphicIdentityMixin, table=True):
    """Same name, same (String-compatible) type -> silently shares the column."""
    shared_val: str | None = None


# ==================== identity naming / schema assertions ====================

class TestPolymorphicIdentityNaming:
    def test_first_level_identity_is_lowercased_classname(self) -> None:
        """Root has no identity -> first-level subclass uses classname.lower()."""
        assert class_mapper(PolyStiDog).polymorphic_identity == 'polystidog'
        assert class_mapper(PolyStiCat).polymorphic_identity == 'polysticat'

    def test_nested_identity_is_dot_joined_with_parent(self) -> None:
        """3rd level: '{parent_identity}.{classname_lowercase}'."""
        assert (
            class_mapper(PolyStiPuppy).polymorphic_identity
            == 'polystidog.polystipuppy'
        )

    def test_explicit_identity_overrides_auto_naming(self) -> None:
        assert class_mapper(PolyStiBird).polymorphic_identity == 'poly-bird'

    def test_identity_to_class_map_covers_all_levels(self) -> None:
        mapping = PolyStiAnimal.get_identity_to_class_map()
        assert mapping == {
            'polystidog': PolyStiDog,
            'polystidog.polystipuppy': PolyStiPuppy,
            'polysticat': PolyStiCat,
            'poly-bird': PolyStiBird,
        }

    def test_get_concrete_subclasses_recurses(self) -> None:
        assert set(PolyStiAnimal.get_concrete_subclasses()) == {
            PolyStiDog, PolyStiPuppy, PolyStiCat, PolyStiBird,
        }


class TestStiSchema:
    def test_sti_not_detected_as_jti(self) -> None:
        assert PolyStiAnimal._is_joined_table_inheritance() is False

    def test_all_subclasses_share_the_root_table(self) -> None:
        root_table = PolyStiAnimal.__table__  # pyright: ignore[reportAttributeAccessIssue]
        for sub in (PolyStiDog, PolyStiPuppy, PolyStiCat, PolyStiBird):
            assert sub.__table__ is root_table  # pyright: ignore[reportAttributeAccessIssue]

    def test_subclass_columns_registered_on_shared_table(self) -> None:
        cols = PolyStiAnimal.__table__.c  # pyright: ignore[reportAttributeAccessIssue]
        for name in ('breed', 'toy', 'mood', 'wing_span'):
            assert name in cols, f"STI subclass column {name!r} missing from shared table"
            # other subclasses' rows have no value -> must be nullable
            assert cols[name].nullable is True

    def test_strenum_column_downcast_to_string(self) -> None:
        """Shared STI columns never use native SAEnum (sibling enum types may differ)."""
        mood_col = PolyStiAnimal.__table__.c['mood']  # pyright: ignore[reportAttributeAccessIssue]
        assert not isinstance(mood_col.type, SAEnum)
        assert isinstance(mood_col.type, String)


# ==================== polymorphic query behavior ====================

@pytest.mark.asyncio
class TestStiPolymorphicQuery:
    async def test_root_query_returns_concrete_subclass_instances(
        self, session: AsyncSession,
    ) -> None:
        await PolyStiDog(name='rex', breed='corgi').save(session)
        await PolyStiCat(name='tom').save(session)
        await PolyStiPuppy(name='pup', breed='corgi', toy='ball').save(session)

        animals = await PolyStiAnimal.get(session, fetch_mode='all')
        assert len(animals) == 3
        by_type = {type(a): a for a in animals}
        assert set(by_type) == {PolyStiDog, PolyStiCat, PolyStiPuppy}

    async def test_subclass_query_returns_only_its_branch(
        self, session: AsyncSession,
    ) -> None:
        """Dog query includes Puppy (descendant); Cat query excludes both."""
        await PolyStiDog(name='rex').save(session)
        await PolyStiPuppy(name='pup').save(session)
        await PolyStiCat(name='tom').save(session)

        dogs = await PolyStiDog.get(session, fetch_mode='all')
        assert {type(d) for d in dogs} == {PolyStiDog, PolyStiPuppy}

        puppies = await PolyStiPuppy.get(session, fetch_mode='all')
        assert len(puppies) == 1
        assert isinstance(puppies[0], PolyStiPuppy)

        cats = await PolyStiCat.get(session, fetch_mode='all')
        assert len(cats) == 1
        assert isinstance(cats[0], PolyStiCat)

    async def test_discriminator_value_stored_in_row(
        self, session: AsyncSession,
    ) -> None:
        dog = await PolyStiDog(name='rex').save(session)
        table = PolyStiAnimal.__table__  # pyright: ignore[reportAttributeAccessIssue]
        stored = (await session.execute(
            select(table.c['_polymorphic_name']).where(table.c.id == dog.id)
        )).scalar_one()
        assert stored == 'polystidog'

    async def test_get_via_root_returns_subclass_for_id(
        self, session: AsyncSession,
    ) -> None:
        cat = await PolyStiCat(name='tom', mood=PolyStiMood.grumpy).save(session)
        fetched = await PolyStiAnimal.get_one(session, cat.id)
        assert isinstance(fetched, PolyStiCat)

    async def test_root_query_filter_on_subclass_column(
        self, session: AsyncSession,
    ) -> None:
        """Phase 2 registers subclass columns on ancestor mappers, so the root
        class can filter on a subclass-declared column."""
        await PolyStiDog(name='rex', breed='corgi').save(session)
        await PolyStiDog(name='odi', breed='dachshund').save(session)

        breed_attr = getattr(PolyStiAnimal, 'breed')
        matches = await PolyStiAnimal.get(session, breed_attr == 'corgi', fetch_mode='all')
        assert len(matches) == 1
        assert matches[0].name == 'rex'

    async def test_third_level_fields_loaded_via_root_query(
        self, session: AsyncSession,
    ) -> None:
        """Puppy row fetched through the root carries both the mid-level field
        (breed) and its own field (toy) without deferred-load surprises."""
        await PolyStiPuppy(name='pup', breed='corgi', toy='ball').save(session)

        animals = await PolyStiAnimal.get(session, fetch_mode='all')
        puppy = animals[0]
        assert isinstance(puppy, PolyStiPuppy)
        assert puppy.breed == 'corgi'
        assert puppy.toy == 'ball'

    async def test_count_respects_sti_branch_filter(
        self, session: AsyncSession,
    ) -> None:
        await PolyStiDog(name='d1').save(session)
        await PolyStiDog(name='d2').save(session)
        await PolyStiPuppy(name='p1').save(session)
        await PolyStiCat(name='c1').save(session)

        assert await PolyStiAnimal.count(session) == 4
        assert await PolyStiDog.count(session) == 3  # dogs + puppy descendant
        assert await PolyStiPuppy.count(session) == 1
        assert await PolyStiCat.count(session) == 1
        assert await PolyStiBird.count(session) == 0


# ==================== StrEnum coercion ====================

@pytest.mark.asyncio
class TestStrEnumCoercion:
    async def test_strenum_roundtrip_returns_enum_instance(
        self, session: AsyncSession,
    ) -> None:
        cat = await PolyStiCat(name='tom', mood=PolyStiMood.grumpy).save(session)
        assert isinstance(cat.mood, PolyStiMood)
        assert cat.mood is PolyStiMood.grumpy

        fetched = await PolyStiCat.get_one(session, cat.id)
        assert isinstance(fetched.mood, PolyStiMood)
        assert fetched.mood is PolyStiMood.grumpy

    async def test_default_enum_value_persisted(self, session: AsyncSession) -> None:
        cat = await PolyStiCat(name='smiley').save(session)
        assert cat.mood is PolyStiMood.happy
        # raw storage is the plain string (String column, not native enum)
        table = PolyStiAnimal.__table__  # pyright: ignore[reportAttributeAccessIssue]
        raw = (await session.execute(
            select(table.c.mood).where(table.c.id == cat.id)
        )).scalar_one()
        assert raw == 'happy'

    def test_init_coerces_raw_string_to_enum(self) -> None:
        """SQLModel table __init__ bypasses validation; the wrapped __init__
        must coerce raw strings immediately."""
        cat = PolyStiCat(name='tom', mood='grumpy')  # type: ignore[arg-type]
        assert isinstance(cat.mood, PolyStiMood)
        assert cat.mood is PolyStiMood.grumpy


# ==================== CRUD integration on STI subclasses ====================

@pytest.mark.asyncio
class TestStiSubclassCrud:
    async def test_save_and_get_one(self, session: AsyncSession) -> None:
        dog = await PolyStiDog(name='rex', breed='corgi').save(session)
        assert isinstance(dog, PolyStiDog)
        assert dog.id is not None

        fetched = await PolyStiDog.get_one(session, dog.id)
        assert isinstance(fetched, PolyStiDog)
        assert fetched.name == 'rex'
        assert fetched.breed == 'corgi'

    async def test_update_with_dto_merges_set_fields_only(
        self, session: AsyncSession,
    ) -> None:
        dog = await PolyStiDog(name='rex', breed='corgi').save(session)
        dog = await dog.update(session, PolyStiDogPatch(breed='labrador'))
        assert dog.breed == 'labrador'
        assert dog.name == 'rex'  # untouched (exclude_unset=True)

        fetched = await PolyStiDog.get_one(session, dog.id)
        assert fetched.breed == 'labrador'

    async def test_modify_and_save_persists(self, session: AsyncSession) -> None:
        dog = await PolyStiDog(name='rex').save(session)
        dog.breed = 'poodle'
        dog = await dog.save(session)

        fetched = await PolyStiDog.get_one(session, dog.id)
        assert fetched.breed == 'poodle'

    async def test_delete_instance(self, session: AsyncSession) -> None:
        dog = await PolyStiDog(name='doomed').save(session)
        cat = await PolyStiCat(name='survivor').save(session)

        deleted = await PolyStiDog.delete(session, dog)
        assert deleted == 1

        remaining = await PolyStiAnimal.get(session, fetch_mode='all')
        assert len(remaining) == 1
        assert isinstance(remaining[0], PolyStiCat)
        assert remaining[0].id == cat.id

    async def test_delete_by_condition_only_hits_own_branch(
        self, session: AsyncSession,
    ) -> None:
        """ORM-enabled DELETE against an STI subclass should include the
        single-table inheritance criteria -- the same-named cat survives."""
        await PolyStiDog(name='dup').save(session)
        await PolyStiCat(name='dup').save(session)

        deleted = await PolyStiDog.delete(
            session, condition=col(PolyStiDog.name) == 'dup',
        )
        assert deleted == 1

        remaining = await PolyStiAnimal.get(session, fetch_mode='all')
        assert len(remaining) == 1
        assert isinstance(remaining[0], PolyStiCat)

    async def test_get_exist_one_raises_after_delete(
        self, session: AsyncSession,
    ) -> None:
        from sqlmodel_ext import RecordNotFoundError

        not_found_excs: tuple[type[BaseException], ...]
        try:
            from fastapi import HTTPException
            not_found_excs = (RecordNotFoundError, HTTPException)
        except ImportError:
            not_found_excs = (RecordNotFoundError,)

        dog = await PolyStiDog(name='ghost').save(session)
        await PolyStiDog.delete(session, dog)

        with pytest.raises(not_found_excs):
            await PolyStiDog.get_exist_one(session, dog.id)


# ==================== same-named column handling ====================

class TestSameNamedColumnHandling:
    def test_first_registrant_defines_shared_column_type(self) -> None:
        shared = PolyConflictRoot.__table__.c['shared_val']  # pyright: ignore[reportAttributeAccessIssue]
        # PolyConflictText registered first -> string-typed column
        # (sqlmodel maps plain ``str`` to AutoString, a String TypeDecorator)
        assert isinstance(shared.type, (String, AutoString))
        assert not isinstance(shared.type, Integer)

    def test_incompatible_type_conflict_raises_type_error(self) -> None:
        """Integer vs non-Integer on the same shared column name is documented
        as incompatible: ``_register_sti_columns`` raises TypeError telling the
        user to pick a different field name.

        Note: during the bulk ``register_sti_columns_for_all_subclasses()``
        pass this TypeError is caught and only logged as a warning -- here we
        invoke the per-class registration directly to assert the contract.
        """
        with pytest.raises(TypeError, match='STI column type conflict'):
            PolyConflictNumber._register_sti_columns()


@pytest.mark.asyncio
class TestCompatibleSharedColumn:
    async def test_compatible_siblings_share_one_column_with_isolated_rows(
        self, session: AsyncSession,
    ) -> None:
        a = await PolyConflictText(name='a', shared_val='alpha').save(session)
        a_id = a.id  # capture before the next commit expires the instance
        b = await PolyConflictTextTwin(name='b', shared_val='beta').save(session)
        b_id = b.id

        fetched_a = await PolyConflictText.get_one(session, a_id)
        fetched_b = await PolyConflictTextTwin.get_one(session, b_id)
        assert fetched_a.shared_val == 'alpha'
        assert fetched_b.shared_val == 'beta'

        # exactly one physical column backs both subclasses
        table = PolyConflictRoot.__table__  # pyright: ignore[reportAttributeAccessIssue]
        assert len([c for c in table.columns if c.name == 'shared_val']) == 1
