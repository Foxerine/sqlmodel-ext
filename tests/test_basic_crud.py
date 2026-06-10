"""
Basic non-polymorphic CRUD sanity check.

Ensures ``TableBaseMixin`` / ``UUIDTableBaseMixin`` can save, retrieve,
update and delete without Redis or FastAPI installed. Exercises the
public API the way a first-time consumer would.
"""
from __future__ import annotations

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin


class Widget(SQLModelBase, UUIDTableBaseMixin, table=True):
    """Non-STI test model. Exists at module scope so SA picks up the table."""
    name: str
    quantity: int = 0


@pytest.mark.asyncio
class TestBasicCRUD:
    async def test_save_and_get_one(self, session: AsyncSession) -> None:
        widget = Widget(name="hammer", quantity=3)
        widget = await widget.save(session)

        assert widget.id is not None
        assert widget.name == "hammer"

        fetched = await Widget.get_one(session, widget.id)
        assert fetched.id == widget.id
        assert fetched.name == "hammer"
        assert fetched.quantity == 3

    async def test_update_persists(self, session: AsyncSession) -> None:
        widget = await Widget(name="nail", quantity=10).save(session)
        widget.quantity = 99
        widget = await widget.save(session)

        fetched = await Widget.get_one(session, widget.id)
        assert fetched.quantity == 99

    async def test_delete_removes(self, session: AsyncSession) -> None:
        from sqlmodel_ext import RecordNotFoundError

        # get_exist_one raises HTTPException(404) when FastAPI is installed,
        # RecordNotFoundError otherwise -- accept both so the test is
        # environment-independent.
        not_found_excs: tuple[type[BaseException], ...]
        try:
            from fastapi import HTTPException
            not_found_excs = (RecordNotFoundError, HTTPException)
        except ImportError:
            not_found_excs = (RecordNotFoundError,)

        widget = await Widget(name="temp").save(session)
        deleted_count = await Widget.delete(session, widget)
        assert deleted_count == 1

        with pytest.raises(not_found_excs):
            await Widget.get_exist_one(session, widget.id)

    async def test_fetch_all_returns_list(self, session: AsyncSession) -> None:
        for name in ("a", "b", "c"):
            await Widget(name=name).save(session)

        results = await Widget.get(session, fetch_mode="all")
        assert isinstance(results, list)
        assert len(results) == 3
        assert {w.name for w in results} == {"a", "b", "c"}
