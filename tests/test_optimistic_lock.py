"""
OptimisticLockMixin behavior tests.

Covers:
- ``oplock_version`` column shape (BIGINT, server default, excluded from
  ``model_dump``) and SQLAlchemy ``version_id_col`` auto-increment
- concurrent-modification conflict detection with two independent sessions
- the retry policy: ``__optimistic_retry_default__`` (3 for lock-enabled
  models, 0 otherwise) and the explicit ``optimistic_retry_count`` knob
- ``delete()`` normalizing a version conflict into ``OptimisticLockError``
- ``OptimisticLockError`` payload attributes

Wiring: the metaclass consumes ``_has_optimistic_lock`` and registers the
mixin's ``oplock_version`` column as SQLAlchemy's ``version_id_col``
automatically (``LockPlainDoc`` below). An explicit
``mapper_args={'version_id_col': ...}`` override still works and takes
precedence (``LockGadget`` below).
"""
from __future__ import annotations

import pytest
from sqlalchemy import BigInteger, Column, Integer
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    OptimisticLockError,
    OptimisticLockMixin,
    SQLModelBase,
    TableBaseMixin,
    UUIDTableBaseMixin,
)

# Explicit version column so the mapper-level optimistic lock uses it.
_lock_version_col = Column('version', Integer, nullable=False, default=0)


class LockGadget(
    OptimisticLockMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
    table=True,
    mapper_args={'version_id_col': _lock_version_col},
):
    """Optimistic-lock model with version_id_col wired explicitly (a domain ``version`` column is allowed)."""
    name: str
    quantity: int = 0
    version: int = Field(default=0, sa_column=_lock_version_col)


class LockPlainDoc(OptimisticLockMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    """Mixin-only model, exactly as the OptimisticLockMixin docstring shows."""
    name: str
    quantity: int = 0


class LockPatch(SQLModelBase):
    """Non-table DTO driving ``update()``."""
    quantity: int


class NoLockDoc(SQLModelBase, UUIDTableBaseMixin, table=True):
    """Model without optimistic locking (retry default 0)."""
    name: str


async def _race(engine: AsyncEngine, session: AsyncSession, model: type, gid: object) -> tuple[object, object]:
    """Load the same row in ``session`` and a second session; the second one wins with quantity=99."""
    s2 = AsyncSession(engine)
    a = await model.get(session, model.id == gid)
    b = await model.get(s2, model.id == gid)
    assert a is not None and b is not None
    b.quantity = 99
    await b.save(s2)
    await s2.close()
    return a, b


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

        gadget.quantity = 6
        gadget = await gadget.save(session)
        assert gadget.version == v_after_insert + 2

    async def test_mixin_column_shape(self) -> None:
        mapper = sa_inspect(LockPlainDoc)
        assert mapper.version_id_col is not None
        assert mapper.version_id_col.name == "oplock_version"
        column = LockPlainDoc.__table__.c.oplock_version  # type: ignore[attr-defined]
        assert isinstance(column.type, BigInteger)
        assert column.server_default is not None
        assert str(column.server_default.arg) == "0"

    async def test_oplock_version_excluded_from_model_dump(self, session: AsyncSession) -> None:
        doc = await LockPlainDoc(name="d").save(session)
        assert doc.oplock_version == 1
        assert "oplock_version" not in doc.model_dump()
        assert "oplock_version" not in doc.model_dump_json()

    async def test_mixin_alone_increments_version(self, session: AsyncSession) -> None:
        doc = await LockPlainDoc(name="d").save(session)
        doc.quantity = 1
        doc = await doc.save(session)
        assert doc.oplock_version == 2


@pytest.mark.asyncio
class TestConcurrentConflict:
    async def test_conflict_is_detected_and_loser_does_not_overwrite(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        """Two sessions race on the same row: with retries disabled the stale writer must fail."""
        gid = (await LockGadget(name="g", quantity=0).save(session)).id
        a, _ = await _race(engine, session, LockGadget, gid)
        a.quantity = 50  # stale writer
        with pytest.raises(OptimisticLockError):
            await a.save(session, optimistic_retry_count=0)

        async with AsyncSession(engine) as s3:
            fresh = await LockGadget.get(s3, LockGadget.id == gid)
            assert fresh is not None
            assert fresh.quantity == 99

    async def test_conflict_raises_optimistic_lock_error_with_record_id(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        gid = (await LockPlainDoc(name="g", quantity=0).save(session)).id
        a, _ = await _race(engine, session, LockPlainDoc, gid)
        a.quantity = 50
        with pytest.raises(OptimisticLockError) as exc_info:
            await a.save(session, optimistic_retry_count=0)
        assert exc_info.value.model_class == "LockPlainDoc"
        assert exc_info.value.record_id == str(gid)

    async def test_default_policy_retries_lock_enabled_models(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        # No optimistic_retry_count -> OptimisticLockMixin's default (3): the
        # conflict is retried transparently and only our change is re-applied.
        gid = (await LockPlainDoc(name="orig", quantity=0).save(session)).id
        a, _ = await _race(engine, session, LockPlainDoc, gid)
        a.name = "renamed"
        a = await a.save(session)
        assert a.name == "renamed"
        assert a.quantity == 99

    async def test_update_default_policy_reapplies_delta(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        gid = (await LockPlainDoc(name="orig", quantity=0).save(session)).id
        a, _ = await _race(engine, session, LockPlainDoc, gid)
        a = await a.update(session, LockPatch(quantity=7))
        assert a.quantity == 7

    async def test_explicit_retry_count(self, engine: AsyncEngine, session: AsyncSession) -> None:
        gid = (await LockGadget(name="orig", quantity=0).save(session)).id
        a, _ = await _race(engine, session, LockGadget, gid)
        a.name = "renamed"
        a = await a.save(session, optimistic_retry_count=1)
        assert a.name == "renamed"
        assert a.quantity == 99

    async def test_retry_defaults(self) -> None:
        assert TableBaseMixin.__optimistic_retry_default__ == 0
        assert NoLockDoc.__optimistic_retry_default__ == 0
        assert OptimisticLockMixin.__optimistic_retry_default__ == 3
        assert LockPlainDoc.__optimistic_retry_default__ == 3


@pytest.mark.asyncio
class TestDeleteConflict:
    async def test_stale_delete_raises_optimistic_lock_error(
        self, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        gid = (await LockPlainDoc(name="g", quantity=0).save(session)).id
        a, _ = await _race(engine, session, LockPlainDoc, gid)
        with pytest.raises(OptimisticLockError) as exc_info:
            await LockPlainDoc.delete(session, a)
        err = exc_info.value
        assert err.model_class == "LockPlainDoc"
        # Flush-level conflict: never attributed to a specific row.
        assert err.record_id is None
        assert err.expected_version is None
        await session.rollback()
        async with AsyncSession(engine) as s3:
            assert await LockPlainDoc.get(s3, LockPlainDoc.id == gid) is not None

    async def test_fresh_delete_succeeds(self, session: AsyncSession) -> None:
        doc = await LockPlainDoc(name="g").save(session)
        assert await LockPlainDoc.delete(session, doc) == 1


@pytest.mark.asyncio
class TestRetryKnobWithoutConflict:
    async def test_save_with_retry_count_and_no_conflict(self, session: AsyncSession) -> None:
        """optimistic_retry_count > 0 must be a no-op when there is no conflict."""
        gadget = await LockGadget(name="g").save(session, optimistic_retry_count=3)
        gadget.quantity = 11
        gadget = await gadget.save(session, optimistic_retry_count=3)
        assert gadget.quantity == 11
        assert gadget.version == 2


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
