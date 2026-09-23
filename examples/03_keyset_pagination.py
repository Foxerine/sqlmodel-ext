"""
03 -- Offset and keyset (``after_id``) pagination with ``TableViewRequest``.

Run::

    python examples/03_keyset_pagination.py

* ``get(..., table_view=...)`` applies ``offset`` / ``limit`` / ``order`` /
  ``desc`` and always appends ``id`` as a tie-break, so rows sharing a
  timestamp are never skipped or repeated at page boundaries.
* ``after_id`` is a keyset cursor: pass the ``id`` of the last row of the
  previous page. The anchor's sort value is looked up server-side, so clients
  never round-trip timestamps. Only UUID primary keys, and only immutable sort
  columns (``created_at`` / ``id``).
* Illegal combinations fail at construction (``after_id`` + ``offset``) or at
  query time (``KeysetCursorInvalidError`` when the anchor disappeared) instead
  of silently returning a wrong page.
"""
import asyncio
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, col

from sqlmodel_ext import (
    AsyncSession,
    KeysetCursorInvalidError,
    ListResponse,
    SQLModelBase,
    Str64,
    TableViewRequest,
    UUIDTableBaseMixin,
)


class Event(SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    """Event title."""


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        for i in range(7):
            _ = await Event(title=f"event-{i}").save(session, refresh=False)
        all_ids = {event.id for event in await Event.get(session, fetch_mode='all')}

        # --- offset pagination + total count --------------------------------
        first: ListResponse[Event] = await Event.get_with_count(
            session, table_view=TableViewRequest(offset=0, limit=3),
        )
        assert first.count == 7 and len(first.items) == 3

        # --- keyset pagination: walk the whole table ------------------------
        seen: list[UUID] = []
        cursor: UUID | None = None
        while True:
            page = await Event.get(
                session, fetch_mode='all', table_view=TableViewRequest(limit=3, after_id=cursor),
            )
            if not page:
                break
            seen.extend(event.id for event in page)
            cursor = page[-1].id
        assert len(seen) == len(set(seen)) == 7, "no gaps, no duplicates"
        assert set(seen) == all_ids

        # --- misuse is rejected, not silently mis-paged ----------------------
        try:
            _ = TableViewRequest(after_id=seen[0], offset=3)
        except ValidationError:
            pass
        else:
            raise AssertionError("after_id + offset would skip rows; must be rejected")

        anchor = await Event.get(session, col(Event.id) == seen[2], fetch_mode='one')
        _ = await Event.delete(session, anchor)
        try:
            _ = await Event.get(
                session, fetch_mode='all', table_view=TableViewRequest(limit=3, after_id=seen[2]),
            )
        except KeysetCursorInvalidError as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError("a deleted anchor must invalidate the cursor")

    await engine.dispose()
    print("[OK] 03_keyset_pagination: 7 rows paged without gaps or duplicates")


if __name__ == "__main__":
    asyncio.run(main())
