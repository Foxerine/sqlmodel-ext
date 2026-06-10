"""
DeferredIndex: STI base-class indexes over subclass-registered columns.

A plain ``Index('name', 'col')`` in the STI base class's ``table_args`` would
raise ``ConstraintColumnNotFoundError`` at class-creation time, because the
referenced column is only registered onto the shared table later (STI phase 1).
``DeferredIndex`` is intercepted by the metaclass and materialized at the end
of ``register_sti_columns_for_all_subclasses()``.

Models live in ``tests/_models.py`` (``Node`` base + ``TextNode`` subclass);
the conftest session fixture drives the two-phase STI registration, so by the
time these tests run the deferred index must be a real SQLAlchemy ``Index``.
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncEngine

from tests._models import Node, TextNode


def test_deferred_index_materialized_on_table() -> None:
    """After STI phase 1 the marker has become a real Index on the shared table."""
    table = Node.__table__  # pyright: ignore[reportAttributeAccessIssue]
    index_names = {ix.name for ix in table.indexes}
    assert 'ix_node_payload' in index_names, (
        f"DeferredIndex was not materialized; table indexes: {index_names}"
    )


def test_deferred_index_targets_subclass_column() -> None:
    table = Node.__table__  # pyright: ignore[reportAttributeAccessIssue]
    index = next(ix for ix in table.indexes if ix.name == 'ix_node_payload')
    assert [c.name for c in index.columns] == ['payload']
    # The column itself was registered by the TextNode subclass.
    assert 'payload' in table.c
    assert 'payload' in TextNode.model_fields


@pytest.mark.asyncio
async def test_deferred_index_created_in_database(engine: AsyncEngine) -> None:
    """``metadata.create_all()`` (run by the engine fixture) emits the index."""
    async with engine.connect() as conn:
        def _get_indexes(sync_conn):  # type: ignore[no-untyped-def]
            return sa_inspect(sync_conn).get_indexes('node')
        indexes = await conn.run_sync(_get_indexes)
    assert any(ix['name'] == 'ix_node_payload' for ix in indexes), (
        f"index missing in database; got: {indexes}"
    )
