"""
Behavioral tests for ``sqlmodel_ext.pagination``.

Covers ListResponse (generic + docstring descriptions), PaginationRequest
(defaults + ge/le bounds + literal order + extra='forbid'),
TimeFilterRequest (range validation rules), TableViewRequest (combined
fields), and integration with ``TableBaseMixin.get_with_count`` /
``get`` (sort field, time filtering, None limit).
"""
from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    ListResponse,
    PaginationRequest,
    SQLModelBase,
    TableBaseMixin,
    TableViewRequest,
    TimeFilterRequest,
)


class PagNote(SQLModelBase, TableBaseMixin, table=True):
    """Table model used for pagination integration tests."""
    body: str


async def _seed_notes(session: AsyncSession, n: int = 5) -> list[PagNote]:
    notes = []
    for i in range(1, n + 1):
        note = PagNote(
            body=f"note{i}",
            created_at=datetime(2024, 1, i),
            updated_at=datetime(2024, 6, n + 1 - i),  # inverse of created order
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
            PaginationRequest(order="id")

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
            created_after_datetime=datetime(2024, 1, 1),
            created_before_datetime=datetime(2024, 2, 1),
            updated_after_datetime=datetime(2024, 1, 1),
            updated_before_datetime=datetime(2024, 2, 1),
        )
        assert tf.created_after_datetime == datetime(2024, 1, 1)

    def test_created_range_inverted_rejected(self) -> None:
        with pytest.raises(ValueError, match="created_after_datetime must be less than"):
            TimeFilterRequest(
                created_after_datetime=datetime(2024, 2, 1),
                created_before_datetime=datetime(2024, 1, 1),
            )

    def test_created_range_equal_rejected(self) -> None:
        # The rule is strict: after must be < before, equality is invalid.
        with pytest.raises(ValueError, match="created_after_datetime must be less than"):
            TimeFilterRequest(
                created_after_datetime=datetime(2024, 1, 1),
                created_before_datetime=datetime(2024, 1, 1),
            )

    def test_updated_range_inverted_rejected(self) -> None:
        with pytest.raises(ValueError, match="updated_after_datetime must be less than"):
            TimeFilterRequest(
                updated_after_datetime=datetime(2024, 2, 1),
                updated_before_datetime=datetime(2024, 1, 1),
            )

    def test_cross_type_created_after_ge_updated_before_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be >= updated_before_datetime"):
            TimeFilterRequest(
                created_after_datetime=datetime(2024, 3, 1),
                updated_before_datetime=datetime(2024, 2, 1),
            )

    def test_cross_type_valid_combination_accepted(self) -> None:
        tf = TimeFilterRequest(
            created_after_datetime=datetime(2024, 1, 1),
            updated_before_datetime=datetime(2024, 2, 1),
        )
        assert tf.updated_before_datetime == datetime(2024, 2, 1)


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
            "offset", "limit", "desc", "order",
            "created_after_datetime", "created_before_datetime",
            "updated_after_datetime", "updated_before_datetime",
        }

    def test_inherits_pagination_validation(self) -> None:
        with pytest.raises(ValidationError):
            TableViewRequest(limit=101)

    def test_inherits_time_filter_validation(self) -> None:
        with pytest.raises(ValueError, match="must be less than"):
            TableViewRequest(
                created_after_datetime=datetime(2024, 2, 1),
                created_before_datetime=datetime(2024, 1, 1),
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
            created_after_datetime=datetime(2024, 1, 2),
            created_before_datetime=datetime(2024, 1, 5),
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
