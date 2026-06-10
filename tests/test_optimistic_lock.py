"""
OptimisticLockMixin behavior tests.

Covers:
- ``version`` field default and SQLAlchemy ``version_id_col`` auto-increment
- concurrent-modification conflict detection with two independent sessions
- the ``optimistic_retry_count`` auto-retry knob on ``save()``
- ``OptimisticLockError`` payload attributes

Wiring: the metaclass consumes ``_has_optimistic_lock`` and registers the
mixin's ``version`` column as SQLAlchemy's ``version_id_col`` automatically
(``LockPlainDoc`` below). An explicit ``mapper_args={'version_id_col': ...}``
override still works and takes precedence (``LockGadget`` below).
"""
from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    OptimisticLockError,
    OptimisticLockMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
)

# Explicit version column so the mapper-level optimistic lock actually engages.
_lock_version_col = Column('version', Integer, nullable=False, default=0)


class LockGadget(
    OptimisticLockMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
    table=True,
    mapper_args={'version_id_col': _lock_version_col},
):
    """Optimistic-lock model with version_id_col wired explicitly."""
    name: str
    quantity: int = 0
    version: int = Field(default=0, sa_column=_lock_version_col)


class LockPlainDoc(OptimisticLockMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    """Mixin-only model, exactly as the OptimisticLockMixin docstring shows."""
    name: str
    quantity: int = 0


@pytest.mark.asyncio
class TestVersionColumn:
    async def test_version_id_col_is_wired(self, session: AsyncSession) -> None:
        mapper = sa_inspect(LockGadget)
        assert mapper.version_id_col is not None
        assert mapper.version_id_col.name == "version"

    async def test_version_increments_on_insert_and_update(self, session: AsyncSession) -> None:
        gadget = LockGadget(name="g")
        assert gadget.version == 0  # pre-persist default

        gadget = await gadget.save(session)
        v_after_insert = gadget.version
        assert v_after_insert == 1  # SQLAlchemy version_id_col starts counting at 1

        gadget.quantity = 5
        gadget = await gadget.save(session)
        assert gadget.version == v_after_insert + 1

        # Another update keeps incrementing monotonically.
        gadget.quantity = 6
        gadget = await gadget.save(session)
        assert gadget.version == v_after_insert + 2


@pytest.mark.asyncio
class TestConcurrentConflict:
    async def test_conflict_is_detected_and_loser_does_not_overwrite(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        """Two sessions race on the same row: the stale writer must fail.

        Only asserts "some exception" -- the precise OptimisticLockError
        contract is covered by the next test. The key invariant here is that
        the stale write MUST NOT reach the database.
        """
        gadget = await LockGadget(name="g", quantity=0).save(session)
        gid = gadget.id

        async with AsyncSession(engine) as s2:
            a = await LockGadget.get(session, LockGadget.id == gid)
            b = await LockGadget.get(s2, LockGadget.id == gid)
            assert a is not None and b is not None

            b.quantity = 99
            b = await b.save(s2)  # winner commits first

            a.quantity = 50  # stale writer
            with pytest.raises(Exception):
                await a.save(session)

        # The stale write never landed.
        async with AsyncSession(engine) as s3:
            fresh = await LockGadget.get(s3, LockGadget.id == gid)
            assert fresh is not None
            assert fresh.quantity == 99

    async def test_conflict_raises_optimistic_lock_error(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        gadget = await LockGadget(name="g", quantity=0).save(session)
        gid = gadget.id

        async with AsyncSession(engine) as s2:
            a = await LockGadget.get(session, LockGadget.id == gid)
            b = await LockGadget.get(s2, LockGadget.id == gid)

            b.quantity = 99
            await b.save(s2)

            a.quantity = 50
            with pytest.raises(OptimisticLockError) as exc_info:
                await a.save(session)
            assert exc_info.value.model_class == "LockGadget"

    async def test_auto_retry_merges_and_succeeds(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        gadget = await LockGadget(name="orig", quantity=0).save(session)
        gid = gadget.id

        async with AsyncSession(engine) as s2:
            a = await LockGadget.get(session, LockGadget.id == gid)
            b = await LockGadget.get(s2, LockGadget.id == gid)

            b.quantity = 7
            await b.save(s2)

            a.name = "renamed"
            a = await a.save(session, optimistic_retry_count=1)

            # Retry re-fetches the fresh row and re-applies our changes on top.
            assert a.name == "renamed"
            assert a.quantity == 7


@pytest.mark.asyncio
class TestRetryKnobWithoutConflict:
    async def test_save_with_retry_count_and_no_conflict(self, session: AsyncSession) -> None:
        """optimistic_retry_count > 0 must be a no-op when there is no conflict."""
        gadget = await LockGadget(name="g").save(session, optimistic_retry_count=3)
        gadget.quantity = 11
        gadget = await gadget.save(session, optimistic_retry_count=3)
        assert gadget.quantity == 11
        assert gadget.version == 2


@pytest.mark.asyncio
class TestMixinOnlyUsage:
    async def test_mixin_alone_increments_version(self, session: AsyncSession) -> None:
        doc = await LockPlainDoc(name="d").save(session)
        doc.quantity = 1
        doc = await doc.save(session)
        assert doc.version >= 1


class TestOptimisticLockErrorPayload:
    def test_attributes_and_message(self) -> None:
        err = OptimisticLockError(
            message="conflict on Order",
            model_class="Order",
            record_id="42",
            expected_version=3,
            original_error=None,
        )
        assert str(err) == "conflict on Order"
        assert err.model_class == "Order"
        assert err.record_id == "42"
        assert err.expected_version == 3
        assert err.original_error is None

    def test_defaults_are_none(self) -> None:
        err = OptimisticLockError("boom")
        assert err.model_class is None
        assert err.record_id is None
        assert err.expected_version is None
        assert err.original_error is None
