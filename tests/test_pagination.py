"""
Behavioral tests for ``sqlmodel_ext.pagination``.

Covers ListResponse (generic + docstring descriptions), PaginationRequest
(defaults + ge/le bounds + literal order + extra='forbid'),
TimeFilterRequest (range validation rules), TableViewRequest (combined
fields), and integration with ``TableBaseMixin.get_with_count`` /
``get`` (sort field, time filtering, None limit).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlmodel import col
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    ListResponse,
    PaginationRequest,
    SQLModelBase,
    TableBaseMixin,
    TableViewRequest,
    TimeFilterRequest,
    UUIDTableBaseMixin,
)
from sqlmodel_ext.mixins import (
    KeysetCursorError,
    KeysetCursorInvalidError,
    KeysetCursorUnsupportedError,
)
from sqlmodel_ext.pagination import (
    MAX_TABLE_VIEW_OFFSET,
    PageWindowRequest,
)


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


class PagNote(SQLModelBase, TableBaseMixin, table=True):
    """Table model used for pagination integration tests."""
    body: str


class PagKeysetItem(SQLModelBase, UUIDTableBaseMixin, table=True):
    """UUID-keyed table for after_id keyset cursor tests."""
    body: str
    group: int = 0


class _Seeded:
    """Plain snapshot of a seeded row (ORM instances expire on every later commit)."""

    def __init__(self, id: uuid.UUID, body: str, group: int) -> None:
        self.id = id
        self.body = body
        self.group = group


async def _seed_keyset_items(session: AsyncSession, n: int = 6, *, same_ts: bool = False) -> list[_Seeded]:
    items: list[_Seeded] = []
    for i in range(1, n + 1):
        created = _utc(2024, 1, 1) if same_ts else _utc(2024, 1, i)
        # With a shared timestamp, ids are assigned in *descending* insertion
        # order so that "insertion order" and "id order" differ: only a real
        # id tie-break produces the id order.
        explicit_id = {'id': uuid.UUID(int=1000 - i)} if same_ts else {}
        item = await PagKeysetItem(
            body=f"k{i}", group=i % 2, created_at=created, updated_at=created, **explicit_id,
        ).save(session)
        items.append(_Seeded(item.id, item.body, item.group))
    return items


async def _seed_notes(session: AsyncSession, n: int = 5) -> list[PagNote]:
    notes = []
    for i in range(1, n + 1):
        note = PagNote(
            body=f"note{i}",
            created_at=_utc(2024, 1, i),
            updated_at=_utc(2024, 6, n + 1 - i),  # inverse of created order
        )
        notes.append(await note.save(session))
    return notes


# --------------------------------------------------------------------------
# PaginationRequest
# --------------------------------------------------------------------------

class TestPaginationRequest:
    def test_defaults(self) -> None:
        req = PaginationRequest()
        assert req.offset == 0
        assert req.limit == 50
        assert req.desc is True
        assert req.order == "created_at"

    def test_boundary_values_accepted(self) -> None:
        assert PaginationRequest(limit=1).limit == 1
        assert PaginationRequest(limit=100).limit == 100
        assert PaginationRequest(offset=0).offset == 0
        assert PaginationRequest(offset=10**9).offset == 10**9

    def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaginationRequest(offset=-1)

    def test_limit_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaginationRequest(limit=0)

    def test_limit_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaginationRequest(limit=101)
        with pytest.raises(ValidationError):
            PaginationRequest(limit=10_000)

    def test_invalid_order_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaginationRequest(order="name")

    def test_order_id_accepted(self) -> None:
        assert PaginationRequest(order="id").order == "id"

    def test_offset_upper_bound(self) -> None:
        assert PaginationRequest(offset=MAX_TABLE_VIEW_OFFSET).offset == MAX_TABLE_VIEW_OFFSET
        with pytest.raises(ValidationError):
            PaginationRequest(offset=MAX_TABLE_VIEW_OFFSET + 1)
        with pytest.raises(ValidationError):
            PaginationRequest(offset=10**100)

    def test_after_id_default_none(self) -> None:
        assert PaginationRequest().after_id is None

    def test_after_id_with_mutable_order_rejected(self) -> None:
        with pytest.raises(ValidationError, match="does not support order=updated_at"):
            PaginationRequest(after_id=uuid.uuid4(), order="updated_at")

    def test_after_id_with_immutable_orders_accepted(self) -> None:
        anchor = uuid.uuid4()
        assert PaginationRequest(after_id=anchor, order="created_at").after_id == anchor
        assert PaginationRequest(after_id=anchor, order="id").after_id == anchor

    def test_after_id_with_nonzero_offset_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be combined"):
            PaginationRequest(after_id=uuid.uuid4(), offset=2)
        # offset=0 (the default) is fine.
        assert PaginationRequest(after_id=uuid.uuid4(), offset=0).offset == 0


class TestPageWindowRequest:
    def test_fields_are_window_only(self) -> None:
        assert set(PageWindowRequest.model_fields) == {"offset", "limit", "desc"}

    def test_pagination_request_extends_window(self) -> None:
        assert issubclass(PaginationRequest, PageWindowRequest)
        assert set(PaginationRequest.model_fields) == {"offset", "limit", "desc", "order", "after_id"}

    def test_unknown_field_forbidden(self) -> None:
        # There is no "page" concept -- extra='forbid' must reject it.
        with pytest.raises(ValidationError):
            PaginationRequest(page=1)

    def test_explicit_none_limit_and_offset_allowed(self) -> None:
        req = PaginationRequest(limit=None, offset=None)
        assert req.limit is None
        assert req.offset is None


# --------------------------------------------------------------------------
# TimeFilterRequest
# --------------------------------------------------------------------------

class TestTimeFilterRequest:
    def test_defaults_all_none(self) -> None:
        tf = TimeFilterRequest()
        assert tf.created_after_datetime is None
        assert tf.created_before_datetime is None
        assert tf.updated_after_datetime is None
        assert tf.updated_before_datetime is None

    def test_valid_range_accepted(self) -> None:
        tf = TimeFilterRequest(
            created_after_datetime=_utc(2024, 1, 1),
            created_before_datetime=_utc(2024, 2, 1),
            updated_after_datetime=_utc(2024, 1, 1),
            updated_before_datetime=_utc(2024, 2, 1),
        )
        assert tf.created_after_datetime == _utc(2024, 1, 1)

    def test_created_range_inverted_rejected(self) -> None:
        with pytest.raises(ValueError, match="created_after_datetime must be less than"):
            TimeFilterRequest(
                created_after_datetime=_utc(2024, 2, 1),
                created_before_datetime=_utc(2024, 1, 1),
            )

    def test_created_range_equal_rejected(self) -> None:
        # The rule is strict: after must be < before, equality is invalid.
        with pytest.raises(ValueError, match="created_after_datetime must be less than"):
            TimeFilterRequest(
                created_after_datetime=_utc(2024, 1, 1),
                created_before_datetime=_utc(2024, 1, 1),
            )

    def test_updated_range_inverted_rejected(self) -> None:
        with pytest.raises(ValueError, match="updated_after_datetime must be less than"):
            TimeFilterRequest(
                updated_after_datetime=_utc(2024, 2, 1),
                updated_before_datetime=_utc(2024, 1, 1),
            )

    def test_cross_type_created_after_ge_updated_before_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be >= updated_before_datetime"):
            TimeFilterRequest(
                created_after_datetime=_utc(2024, 3, 1),
                updated_before_datetime=_utc(2024, 2, 1),
            )

    def test_cross_type_valid_combination_accepted(self) -> None:
        tf = TimeFilterRequest(
            created_after_datetime=_utc(2024, 1, 1),
            updated_before_datetime=_utc(2024, 2, 1),
        )
        assert tf.updated_before_datetime == _utc(2024, 2, 1)

    def test_naive_datetime_rejected(self) -> None:
        # AwareDatetime: a naive value cannot be compared with timezone-aware
        # columns, so it is rejected at validation time.
        with pytest.raises(ValidationError):
            TimeFilterRequest(created_after_datetime=datetime(2024, 1, 1))
        with pytest.raises(ValidationError):
            TableViewRequest(updated_before_datetime=datetime(2024, 1, 1))


# --------------------------------------------------------------------------
# TableViewRequest
# --------------------------------------------------------------------------

class TestTableViewRequest:
    def test_combines_both_parents(self) -> None:
        tv = TableViewRequest()
        # pagination side
        assert tv.offset == 0
        assert tv.limit == 50
        assert tv.desc is True
        assert tv.order == "created_at"
        # time-filter side
        assert tv.created_after_datetime is None
        assert tv.updated_before_datetime is None

    def test_field_set_is_union_of_parents(self) -> None:
        assert set(TableViewRequest.model_fields) == {
            "offset", "limit", "desc", "order", "after_id",
            "created_after_datetime", "created_before_datetime",
            "updated_after_datetime", "updated_before_datetime",
        }

    def test_inherits_pagination_validation(self) -> None:
        with pytest.raises(ValidationError):
            TableViewRequest(limit=101)

    def test_inherits_time_filter_validation(self) -> None:
        with pytest.raises(ValueError, match="must be less than"):
            TableViewRequest(
                created_after_datetime=_utc(2024, 2, 1),
                created_before_datetime=_utc(2024, 1, 1),
            )


# --------------------------------------------------------------------------
# ListResponse
# --------------------------------------------------------------------------

class TestListResponse:
    def test_generic_construction(self) -> None:
        resp = ListResponse[int](count=2, items=[1, 2])
        assert resp.count == 2
        assert resp.items == [1, 2]

    def test_item_type_validated(self) -> None:
        with pytest.raises(ValidationError):
            ListResponse[int](count=1, items=["not-an-int"])

    def test_count_required(self) -> None:
        with pytest.raises(ValidationError):
            ListResponse[int](items=[])

    def test_attribute_docstrings_become_descriptions(self) -> None:
        assert ListResponse.model_fields["count"].description is not None
        assert "Total number" in ListResponse.model_fields["count"].description
        assert ListResponse.model_fields["items"].description is not None


# --------------------------------------------------------------------------
# Integration with get / get_with_count
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPaginationIntegration:
    async def test_default_table_view_newest_first_all_items(
        self, session: AsyncSession
    ) -> None:
        await _seed_notes(session)
        resp = await PagNote.get_with_count(session, table_view=TableViewRequest())
        assert resp.count == 5
        assert [n.body for n in resp.items] == ["note5", "note4", "note3", "note2", "note1"]

    async def test_page_slicing_keeps_total_count(self, session: AsyncSession) -> None:
        await _seed_notes(session)
        tv = TableViewRequest(limit=2, offset=2, desc=False)
        resp = await PagNote.get_with_count(session, table_view=tv)
        assert resp.count == 5
        assert [n.body for n in resp.items] == ["note3", "note4"]

    async def test_offset_beyond_total_yields_empty_page(self, session: AsyncSession) -> None:
        await _seed_notes(session)
        tv = TableViewRequest(offset=100, desc=False)
        resp = await PagNote.get_with_count(session, table_view=tv)
        assert resp.count == 5
        assert resp.items == []

    async def test_sort_by_updated_at(self, session: AsyncSession) -> None:
        await _seed_notes(session)  # updated_at order is inverse of created_at
        tv = TableViewRequest(order="updated_at", desc=True)
        resp = await PagNote.get_with_count(session, table_view=tv)
        assert [n.body for n in resp.items] == ["note1", "note2", "note3", "note4", "note5"]

    async def test_time_filter_narrows_count_and_items(self, session: AsyncSession) -> None:
        await _seed_notes(session)  # created Jan 1..5
        tv = TableViewRequest(
            created_after_datetime=_utc(2024, 1, 2),
            created_before_datetime=_utc(2024, 1, 5),
            desc=False,
        )
        resp = await PagNote.get_with_count(session, table_view=tv)
        # created_at >= Jan 2 and < Jan 5 -> notes 2, 3, 4
        assert resp.count == 3
        assert [n.body for n in resp.items] == ["note2", "note3", "note4"]

    async def test_get_respects_table_view_limit(self, session: AsyncSession) -> None:
        await _seed_notes(session)
        tv = TableViewRequest(limit=3, desc=False)
        items = await PagNote.get(session, fetch_mode="all", table_view=tv)
        assert [n.body for n in items] == ["note1", "note2", "note3"]


# --------------------------------------------------------------------------
# after_id keyset cursor + (order, id) tie-break
# --------------------------------------------------------------------------

async def _walk_keyset(session: AsyncSession, *, limit: int, desc: bool, order: str = "created_at") -> list[str]:
    """Page through PagKeysetItem with after_id and return all bodies in page order."""
    seen: list[str] = []
    after: uuid.UUID | None = None
    while True:
        tv = TableViewRequest(limit=limit, desc=desc, order=order, after_id=after)
        page = await PagKeysetItem.get(session, fetch_mode="all", table_view=tv)
        if not page:
            return seen
        seen.extend(item.body for item in page)
        after = page[-1].id


@pytest.mark.asyncio
class TestKeysetCursor:
    async def test_keyset_walk_ascending_covers_all_without_duplicates(self, session: AsyncSession) -> None:
        await _seed_keyset_items(session)
        assert await _walk_keyset(session, limit=4, desc=False) == [f"k{i}" for i in range(1, 7)]

    async def test_keyset_walk_descending(self, session: AsyncSession) -> None:
        await _seed_keyset_items(session)
        assert await _walk_keyset(session, limit=4, desc=True) == [f"k{i}" for i in range(6, 0, -1)]

    async def test_keyset_same_timestamp_uses_id_tie_break(self, session: AsyncSession) -> None:
        # All rows share created_at: without the id tie-break the page boundary
        # would skip or repeat rows.
        items = await _seed_keyset_items(session, same_ts=True)
        expected = [item.body for item in sorted(items, key=lambda x: x.id)]
        assert await _walk_keyset(session, limit=4, desc=False) == expected
        assert await _walk_keyset(session, limit=4, desc=True) == list(reversed(expected))

    async def test_keyset_order_by_id(self, session: AsyncSession) -> None:
        items = await _seed_keyset_items(session)
        expected = [item.body for item in sorted(items, key=lambda x: x.id)]
        assert await _walk_keyset(session, limit=4, desc=False, order="id") == expected

    async def test_keyset_prefix_deletion_does_not_skip(self, session: AsyncSession) -> None:
        await _seed_keyset_items(session)
        first = await PagKeysetItem.get(
            session, fetch_mode="all", table_view=TableViewRequest(limit=2, desc=False),
        )
        # A concurrent delete of an already-read row would shift an offset page;
        # the keyset cursor anchored on the last row is unaffected.
        await PagKeysetItem.delete(session, first[0])
        anchor = await PagKeysetItem.get(session, col(PagKeysetItem.body) == "k2")
        assert anchor is not None
        nxt = await PagKeysetItem.get(
            session, fetch_mode="all",
            table_view=TableViewRequest(limit=2, desc=False, after_id=anchor.id),
        )
        assert [i.body for i in nxt] == ["k3", "k4"]

    async def test_deleted_anchor_raises_invalid(self, session: AsyncSession) -> None:
        items = await _seed_keyset_items(session)
        anchor_id = items[1].id
        await PagKeysetItem.delete(session, condition=col(PagKeysetItem.id) == anchor_id)
        with pytest.raises(KeysetCursorInvalidError):
            await PagKeysetItem.get(
                session, fetch_mode="all",
                table_view=TableViewRequest(desc=False, after_id=anchor_id),
            )

    async def test_anchor_outside_condition_raises_invalid(self, session: AsyncSession) -> None:
        # The anchor must be visible to the same condition as the main query
        # (no existence oracle for rows outside the caller's scope).
        items = await _seed_keyset_items(session)
        odd_anchor = next(i for i in items if i.group == 1)
        with pytest.raises(KeysetCursorInvalidError):
            await PagKeysetItem.get(
                session, col(PagKeysetItem.group) == 0, fetch_mode="all",
                table_view=TableViewRequest(desc=False, after_id=odd_anchor.id),
            )

    async def test_keyset_with_explicit_order_by_unsupported(self, session: AsyncSession) -> None:
        items = await _seed_keyset_items(session)
        with pytest.raises(KeysetCursorUnsupportedError):
            await PagKeysetItem.get(
                session, fetch_mode="all", order_by=[col(PagKeysetItem.body)],
                table_view=TableViewRequest(after_id=items[0].id),
            )

    async def test_keyset_errors_share_base_class(self) -> None:
        assert issubclass(KeysetCursorInvalidError, KeysetCursorError)
        assert issubclass(KeysetCursorUnsupportedError, KeysetCursorError)
        assert issubclass(KeysetCursorError, ValueError)

    async def test_keyset_on_int_pk_table_rejected(self, session: AsyncSession) -> None:
        await _seed_notes(session)
        with pytest.raises(ValueError, match="only supports UUID primary keys"):
            await PagNote.get(
                session, fetch_mode="all",
                table_view=TableViewRequest(after_id=uuid.uuid4()),
            )

    async def test_get_with_count_count_ignores_cursor(self, session: AsyncSession) -> None:
        items = await _seed_keyset_items(session)
        resp = await PagKeysetItem.get_with_count(
            session, table_view=TableViewRequest(limit=2, desc=False, after_id=items[1].id),
        )
        assert resp.count == 6
        assert [i.body for i in resp.items] == ["k3", "k4"]

    async def test_offset_pages_tie_break_on_same_timestamp(self, session: AsyncSession) -> None:
        items = await _seed_keyset_items(session, same_ts=True)
        expected = [item.body for item in sorted(items, key=lambda x: x.id)]
        seen: list[str] = []
        for offset in range(0, 6, 4):
            page = await PagKeysetItem.get(
                session, fetch_mode="all",
                table_view=TableViewRequest(limit=4, offset=offset, desc=False),
            )
            seen.extend(i.body for i in page)
        assert seen == expected
