"""
Pagination and time filtering request models.

These DTO classes carry query parameters for list endpoints.
SQL clause construction is handled by TableBaseMixin.
"""
from datetime import datetime
from typing import TypeVar, Literal, Generic, Any

# ListResponse uses BaseModel due to SQLModel Generic[T] schema generation bug
# See: https://github.com/fastapi/sqlmodel/discussions/1002
from pydantic import BaseModel, ConfigDict
from sqlmodel import Field

from sqlmodel_ext.base import SQLModelBase

ItemT = TypeVar("ItemT")


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
        Inherits BaseModel instead of SQLModelBase because SQLModel's metaclass
        conflicts with Generic. See module docstring for details.
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

    :raises ValueError: Invalid time range
    """
    created_after_datetime: datetime | None = None
    """Filter created_at >= datetime (None means no limit)"""

    created_before_datetime: datetime | None = None
    """Filter created_at < datetime (None means no limit)"""

    updated_after_datetime: datetime | None = None
    """Filter updated_at >= datetime (None means no limit)"""

    updated_before_datetime: datetime | None = None
    """Filter updated_at < datetime (None means no limit)"""

    def model_post_init(self, __context: Any) -> None:
        """
        Validate time range consistency.

        Rules:
        1. Same-type: after must be less than before
        2. Cross-type: created_after cannot be greater than updated_before
        """
        if self.created_after_datetime and self.created_before_datetime:
            if self.created_after_datetime >= self.created_before_datetime:
                raise ValueError("created_after_datetime must be less than created_before_datetime")
        if self.updated_after_datetime and self.updated_before_datetime:
            if self.updated_after_datetime >= self.updated_before_datetime:
                raise ValueError("updated_after_datetime must be less than updated_before_datetime")

        if self.created_after_datetime and self.updated_before_datetime:
            if self.created_after_datetime >= self.updated_before_datetime:
                raise ValueError(
                    "created_after_datetime cannot be >= updated_before_datetime "
                    "(a record's update time cannot be earlier than its creation time)"
                )


class PaginationRequest(SQLModelBase):
    """
    Pagination and sorting request parameters.

    Pure data class -- SQL clause building is handled by TableBaseMixin.
    """
    offset: int | None = Field(default=0, ge=0)
    """Offset (skip first N records), must be non-negative"""

    limit: int | None = Field(default=50, le=100)
    """Page size (return at most N records), default 50, max 100"""

    desc: bool | None = True
    """Sort descending (True: descending, False: ascending)"""

    order: Literal["created_at", "updated_at"] | None = "created_at"
    """Sort field (created_at or updated_at)"""


class TableViewRequest(TimeFilterRequest, PaginationRequest):
    """
    Table view request parameters (pagination + sorting + time filtering).

    Combines TimeFilterRequest and PaginationRequest for endpoints needing
    full query parameters. Pure data class.

    Example::

        @router.get("/list")
        async def list_items(
            session: SessionDep,
            table_view: TableViewRequestDep
        ):
            items = await Item.get(session, fetch_mode="all", table_view=table_view)
            return items
    """
    pass
