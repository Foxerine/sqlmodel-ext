"""Adversarial release-review tests for mixed 0.4.x/0.5.0 cache payloads."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from sqlmodel_ext import (
    CachedTableBaseMixin,
    OptimisticLockMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
)


class ReviewOrderWithoutDomainVersion(
    SQLModelBase,
    OptimisticLockMixin,
    CachedTableBaseMixin,
    UUIDTableBaseMixin,
    table=True,
):
    status: str


class ReviewOrderWithDomainVersion(
    SQLModelBase,
    OptimisticLockMixin,
    CachedTableBaseMixin,
    UUIDTableBaseMixin,
    table=True,
):
    version: int
    status: str


def _old_042_payload(class_name: str) -> dict[str, object]:
    return {
        'id': str(uuid4()),
        'version': 17,
        'status': 'paid',
        '_c': class_name,
    }


def test_old_optimistic_cache_payload_is_rejected_without_reused_version_name() -> None:
    with pytest.raises(ValidationError):
        ReviewOrderWithoutDomainVersion._deserialize_item(
            _old_042_payload(ReviewOrderWithoutDomainVersion.__name__)
        )


def test_old_payload_can_validate_with_different_semantics_when_version_is_reused() -> None:
    """Pins why the documented 0.4→0.5 switch must clear shared Redis keys."""
    item = ReviewOrderWithDomainVersion._deserialize_item(
        _old_042_payload(ReviewOrderWithDomainVersion.__name__)
    )

    assert item.version == 17
    assert item.oplock_version == 0
