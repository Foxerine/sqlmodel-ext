"""
Pagination and time filtering request models.

These DTO classes carry query parameters for list endpoints.
SQL clause construction is handled by TableBaseMixin.

Class hierarchy::

    PageWindowRequest            offset / limit / desc
      └── PaginationRequest      + order / after_id (keyset cursor)
    TimeFilterRequest            created_* / updated_* time bounds
    TableViewRequest             TimeFilterRequest + PaginationRequest
"""
import uuid
from datetime import datetime
from typing import TypeVar, Literal, Generic

# Generic container choice:
# - A generic container used as a FastAPI **response_model** (ListResponse)
#   must inherit pydantic ``BaseModel``: SQLModel + Generic still produces a
#   broken JSON schema for the parametrized field (``{"items": {}}`` instead
#   of a ``$ref``). See https://github.com/fastapi/sqlmodel/discussions/1002
#   and https://github.com/fastapi/sqlmodel/pull/1275.
# - A generic container that is only a method return value and never enters
#   OpenAPI (``GroupSumRow``) is unaffected by that bug and inherits
#   ``SQLModelBase`` like every other data carrier.
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationInfo, field_validator
from sqlmodel import Field

from sqlmodel_ext.base import SQLModelBase
from sqlmodel_ext.field_types import JS_MAX_SAFE_INTEGER

ItemT = TypeVar("ItemT")

DEFAULT_PAGE_SIZE: int = 50
"""Default ``limit`` of ``PageWindowRequest``."""

MAX_PAGE_SIZE: int = 100
"""Upper bound of ``PageWindowRequest.limit``."""

MAX_SHARED_PAGE_WINDOW: int = 1000
"""The largest page window (``limit``) any ``PageWindowRequest`` subclass may allow.

Subclasses may widen ``limit`` beyond ``MAX_PAGE_SIZE``; ``MAX_TABLE_VIEW_OFFSET``
reserves this much headroom so that ``offset + limit`` never exceeds
``JS_MAX_SAFE_INTEGER``. A subclass that widens ``limit`` above this value must
also lower its own ``offset`` bound."""

MAX_TABLE_VIEW_OFFSET: int = JS_MAX_SAFE_INTEGER - MAX_SHARED_PAGE_WINDOW
"""Upper bound of ``PageWindowRequest.offset``.

Why a bound at all: Python ``int`` has arbitrary precision; with only ``ge=0``
an ``offset=10**100`` passes validation and fails later at the database driver
(``bigint out of range``) -- at query time instead of validation time.

Why ``2**53 - 1`` rather than the int8 maximum: the bound is published as the
OpenAPI ``maximum`` and becomes part of the API contract, while most JS/TS
clients map ``integer`` to an IEEE-754 double whose exact integer range ends
at ``Number.MAX_SAFE_INTEGER``.

Why one window less: callers routinely compute the next page as
``offset + limit``; reserving ``MAX_SHARED_PAGE_WINDOW`` keeps that addition in
range without per-call overflow checks."""

if DEFAULT_PAGE_SIZE > MAX_PAGE_SIZE:
    # Import-time fail-fast: Pydantic does not validate defaults, so a default
    # outside its own declared range would make "omitted" valid while passing
    # the same value explicitly is rejected.
    raise RuntimeError(
        f"pagination invariant broken: DEFAULT_PAGE_SIZE({DEFAULT_PAGE_SIZE}) > MAX_PAGE_SIZE({MAX_PAGE_SIZE})"
    )


class ListResponse(BaseModel, Generic[ItemT]):
    """
    Generic paginated response.

    Standard response format for all LIST endpoints, containing
    total count and item list. Use with ``TableBaseMixin.get_with_count()``.

    Example::

        @router.get("", response_model=ListResponse[CharacterInfoResponse])
        async def list_characters(...) -> ListResponse[Character]:
            return await Character.get_with_count(session, table_view=table_view)

    Note:
        Inherits ``BaseModel`` instead of ``SQLModelBase`` because this class
        is used as a **response_model** and SQLModel + Generic still generates
        a broken JSON schema for the parametrized field. See the module-level
        comment.
    """
    model_config = ConfigDict(use_attribute_docstrings=True)

    count: int
    """Total number of records matching the query conditions."""

    items: list[ItemT]
    """List of records for the current page."""


class TimeFilterRequest(SQLModelBase):
    """
    Time filtering request parameters.

    Used for scenarios that only need time-based filtering (e.g. ``count()``).
    Pure data class -- only carries parameters; SQL clause building is
    handled by TableBaseMixin.

    All bounds are ``AwareDatetime``: a naive datetime cannot be compared
    with timezone-aware database values (a naive value would silently be
    interpreted in the database's timezone), so it is rejected at validation.

    :raises ValueError: Invalid time range
    """
    created_after_datetime: AwareDatetime | None = None
    """Filter created_at >= datetime (None means no limit). Must carry a timezone."""

    created_before_datetime: AwareDatetime | None = None
    """Filter created_at < datetime (None means no limit). Must carry a timezone."""

    updated_after_datetime: AwareDatetime | None = None
    """Filter updated_at >= datetime (None means no limit). Must carry a timezone."""

    updated_before_datetime: AwareDatetime | None = None
    """Filter updated_at < datetime (None means no limit). Must carry a timezone."""

    # Time range consistency. Rules:
    # 1. Same-type: after must be less than before
    # 2. Cross-type: created_after cannot be >= updated_before
    # Each rule is a field validator on its ``*_before_datetime`` field (declared
    # after the ``*_after_datetime`` fields it reads from ``info.data``), so the
    # error location names that field -- ``query_dependency`` reports it as
    # ``('query', '<field>')``. A key missing from ``info.data`` means that
    # field already failed its own validation, which is reported on its own.

    @field_validator('created_before_datetime')
    @classmethod
    def _validate_created_range(cls, before: datetime | None, info: ValidationInfo) -> datetime | None:
        """``created_after_datetime < created_before_datetime``."""
        after = info.data.get('created_after_datetime')
        if before is not None and after is not None and after >= before:
            raise ValueError("created_after_datetime must be less than created_before_datetime")
        return before

    @field_validator('updated_before_datetime')
    @classmethod
    def _validate_updated_range(cls, before: datetime | None, info: ValidationInfo) -> datetime | None:
        """``updated_after_datetime < updated_before_datetime`` and ``created_after_datetime < updated_before_datetime``."""
        if before is None:
            return before
        updated_after = info.data.get('updated_after_datetime')
        if updated_after is not None and updated_after >= before:
            raise ValueError("updated_after_datetime must be less than updated_before_datetime")
        created_after = info.data.get('created_after_datetime')
        if created_after is not None and created_after >= before:
            raise ValueError(
                "created_after_datetime cannot be >= updated_before_datetime "
                "(a record's update time cannot be earlier than its creation time)"
            )
        return before


class PageWindowRequest(SQLModelBase):
    """
    Page window request parameters (the lowest layer: offset / limit / desc).

    For consumers whose sort order is fixed by their own domain semantics
    (they choose the order column and only need a window + direction).
    It deliberately carries no ``order`` / ``after_id``: inheriting the full
    ``PaginationRequest`` would expose fields that nothing consumes in the
    public schema, only to be silently dropped. Pure data class.
    """
    offset: int | None = Field(default=0, ge=0, le=MAX_TABLE_VIEW_OFFSET)
    """Offset (skip first N records), non-negative and at most ``MAX_TABLE_VIEW_OFFSET``."""

    limit: int | None = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    """Page size (return at most N records), min 1, default ``DEFAULT_PAGE_SIZE`` (50), max ``MAX_PAGE_SIZE`` (100)"""

    desc: bool | None = True
    """Sort descending (True: descending, False: ascending)"""


class PaginationRequest(PageWindowRequest):
    """
    Pagination and sorting request parameters (window + order column + keyset cursor).

    For consumers that honor both ``order`` and ``after_id``.
    Pure data class -- SQL clause building is handled by TableBaseMixin.
    """
    order: Literal["created_at", "updated_at", "id"] | None = "created_at"
    """Sort field (created_at, updated_at or id).

    Subclasses may override the ``Literal`` to add domain sort columns --
    ``get()`` resolves the column by name, so every value must be a real
    column of the model.
    """

    after_id: uuid.UUID | None = None
    """Keyset cursor: only return records sorted **after** this record (pass the id of the last item of the previous page).

    Unlike ``offset``, a keyset cursor anchors on the last record read, so
    concurrent inserts/deletes in the already-read prefix do not shift the
    next page (except for the anchor itself -- see below). Use it for
    sequential traversal; ``offset`` remains suitable for random access.

    Ordering is always the composite ``(order column, id)`` (``id`` breaks
    ties, so rows sharing a timestamp are split across pages without gaps or
    duplicates). The anchor's sort value is looked up server-side from
    ``after_id``, so clients never round-trip timestamps.

    Constraints: ``order`` must be ``created_at`` (default) or ``id`` -- both
    immutable; a mutable column (``updated_at`` or domain columns) would move
    the anchor after an update and break the no-gap/no-duplicate guarantee,
    so it is rejected at validation. The anchor must still be visible to the
    query (condition + filter + STI filter); if it was deleted or no longer
    matches, ``get()`` raises ``KeysetCursorInvalidError`` instead of returning
    an empty page that looks like the end. ``after_id`` is only supported on
    UUID primary-key tables.
    """

    # The two cross-field rules below are field validators on ``after_id`` (not
    # ``model_validator``s) so that the error location is ``('after_id',)``
    # instead of the model root: ``query_dependency`` turns it into
    # ``('query', 'after_id')``. They can read ``offset`` / ``order`` from
    # ``info.data`` because both are declared before ``after_id`` (a subclass
    # re-declaring ``order`` keeps its position). A key missing from
    # ``info.data`` means that field already failed its own validation, which
    # is reported on its own.

    @field_validator('after_id')
    @classmethod
    def _validate_keyset_anchor_column(cls, after_id: uuid.UUID | None, info: ValidationInfo) -> uuid.UUID | None:
        """``after_id`` may only anchor an immutable sort column (see ``after_id``)."""
        if after_id is None or 'order' not in info.data:
            return after_id
        order = info.data['order']
        if order not in ('created_at', 'id', None):
            raise ValueError(
                f"after_id keyset cursor does not support order={order}: a mutable sort "
                "column moves rows after updates and breaks the no-gap/no-duplicate "
                "guarantee; use an immutable sort column (e.g. order=created_at)"
            )
        return after_id

    @field_validator('after_id')
    @classmethod
    def _reject_keyset_with_offset(cls, after_id: uuid.UUID | None, info: ValidationInfo) -> uuid.UUID | None:
        """``after_id`` and a non-zero ``offset`` are **mutually exclusive** -- they would stack, not alternate.

        ``get()`` adds the keyset condition to ``WHERE`` ("after the anchor")
        and still applies ``OFFSET``, meaning "skip N more records after the
        anchor". With records A B C D E F, anchor B and a leftover
        ``offset=2``, the query returns E F -- C and D are never shown and
        become unreachable, silently. ``offset`` defaults to 0, so "forgot to
        reset it" is the most natural misuse; the combination is therefore
        made unrepresentable at construction.
        """
        if after_id is None or 'offset' not in info.data:
            return after_id
        offset = info.data['offset']
        if offset:
            raise ValueError(
                "after_id and offset cannot be combined: the keyset cursor already means "
                f"'continue after the anchor'; adding offset={offset} would additionally "
                f"skip {offset} records after the anchor (they would become unreachable). "
                "Pass only after_id and omit offset when paging."
            )
        return after_id


class TableViewRequest(TimeFilterRequest, PaginationRequest):
    """
    Table view request parameters (pagination + sorting + time filtering).

    Combines TimeFilterRequest and PaginationRequest for endpoints needing
    full query parameters. Pure data class.

    Example::

        TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]

        @router.get("/list")
        async def list_items(
            session: SessionDep,
            table_view: TableViewDep,
        ):
            items = await Item.get(session, fetch_mode="all", table_view=table_view)
            return items
    """
    pass
