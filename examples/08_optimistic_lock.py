"""
08 -- Optimistic locking with ``OptimisticLockMixin`` (``oplock_version``).

Run::

    python examples/08_optimistic_lock.py

* Put ``OptimisticLockMixin`` **before** ``UUIDTableBaseMixin`` / ``TableBaseMixin``.
  The metaclass wires the ``oplock_version`` column (BIGINT, excluded from
  ``model_dump()``) as SQLAlchemy's ``version_id_col``: the INSERT stores 1,
  every UPDATE increments it and becomes ``... WHERE id = ? AND oplock_version = ?``.
* ``save()`` / ``update()`` retry a conflict up to
  ``__optimistic_retry_default__`` times (3 on lock-enabled models): the row is
  re-read and **only the columns this instance changed** are re-applied, so the
  other writer's changes survive. Pass ``optimistic_retry_count=0`` to get
  ``OptimisticLockError`` immediately.
* ``delete()`` never retries; a conflict becomes ``OptimisticLockError``.
* The name ``oplock_version`` is reserved; a domain field called ``version``
  is free to exist.
"""
import asyncio
from uuid import UUID

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, col

from sqlmodel_ext import (
    AsyncSession,
    NonNegativeInt,
    OptimisticLockError,
    OptimisticLockMixin,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
)


class Document(OptimisticLockMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    """Document title."""

    stock: NonNegativeInt
    """Units in stock."""

    version: Str64 = "draft"
    """A *domain* version label -- no clash with the lock column ``oplock_version``."""


class DocumentStockPatch(SQLModelBase):
    stock: NonNegativeInt


async def other_writer_sets_stock(engine_session: AsyncSession, doc_id: UUID, stock: int) -> None:
    """Simulates a concurrent request that commits between our read and our write."""
    other = await Document.get(engine_session, col(Document.id) == doc_id, fetch_mode='one')
    other.stock = stock
    _ = await other.save(engine_session)


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session, AsyncSession(engine) as concurrent:
        doc = await Document(title="Manual", stock=10).save(session)
        doc_id = doc.id
        assert doc.oplock_version == 1, "SQLAlchemy stamps version 1 on INSERT"
        assert 'oplock_version' not in doc.model_dump(), "internal state never leaks into dumps"

        # --- 1. conflict + default retry: both changes survive ---------------
        stale = await Document.get(session, col(Document.id) == doc_id, fetch_mode='one')
        await other_writer_sets_stock(concurrent, doc_id, stock=7)
        stale.title = "Manual, 2nd edition"
        merged = await stale.save(session)
        assert merged.title == "Manual, 2nd edition", "our change applied"
        assert merged.stock == 7, "their change preserved (no lost update)"
        assert merged.oplock_version == 3, "insert, their update, our retried update"

        # --- 2. opt out of retries: the conflict surfaces --------------------
        stale = await Document.get(session, col(Document.id) == doc_id, fetch_mode='one')
        await other_writer_sets_stock(concurrent, doc_id, stock=3)
        try:
            _ = await stale.update(session, DocumentStockPatch(stock=0), optimistic_retry_count=0)
        except OptimisticLockError as exc:
            assert exc.model_class == "Document" and exc.record_id == str(doc_id)
        else:
            raise AssertionError("retry disabled: the conflict must raise")

        current = await Document.get(session, col(Document.id) == doc_id, fetch_mode='one')
        assert current.stock == 3, "the losing write did not overwrite the winner"

    await engine.dispose()
    print("[OK] 08_optimistic_lock: conflicts retried or reported, never lost")


if __name__ == "__main__":
    asyncio.run(main())
