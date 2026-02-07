"""
InfoResponse DTO Mixins.

Provides mixins for response DTOs, defining id/created_at/updated_at fields.

Design:
- These mixins are for **response DTOs**, not database tables
- When returned from the database, these fields are always populated,
  so they are defined as required (non-optional)
- TableBase's id=None and default_factory=now are correct (None before DB insert)
- These mixins express "these fields always have values in API responses"
"""
from datetime import datetime
from uuid import UUID

from sqlmodel_ext.base import SQLModelBase


class IntIdInfoMixin(SQLModelBase):
    """Integer ID response mixin for InfoResponse DTOs."""
    id: int
    """Record ID"""


class UUIDIdInfoMixin(SQLModelBase):
    """UUID ID response mixin for InfoResponse DTOs."""
    id: UUID
    """Record ID"""


class DatetimeInfoMixin(SQLModelBase):
    """Timestamp response mixin for InfoResponse DTOs."""
    created_at: datetime
    """Creation timestamp"""

    updated_at: datetime
    """Last update timestamp"""


class IntIdDatetimeInfoMixin(IntIdInfoMixin, DatetimeInfoMixin):
    """Integer ID + timestamps response mixin."""
    pass


class UUIDIdDatetimeInfoMixin(UUIDIdInfoMixin, DatetimeInfoMixin):
    """UUID ID + timestamps response mixin."""
    pass
