"""
Regression: ``all_fields_optional=True`` must preserve constraints *nested*.

``_make_annotation_optional`` historically produced
``Annotated[T | None, Ge(0), ...]`` -- constraint markers on the outer layer.
Pydantic then applied the constraints to ``None`` during JSON parse and raised
``TypeError: Unable to apply constraint 'ge' to supplied value None``.

The fix splits Annotated metadata by None-safety:

- constraint markers (Ge/Le/MultipleOf/max_digits/StringConstraints...) ->
  inner ``Annotated[T, *constraints]``
- schema/orm markers (PlainSerializer/sa_type/default...) -> outer

so the derived UpdateRequest accepts ``null``, still enforces constraints on
non-None values, and keeps the JSON-string serializer working.
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from sqlmodel_ext import (
    NonNegativeDecimal38_18,
    SQLModelBase,
    Str64,
)


class ItemBase(SQLModelBase):
    """Parent DTO with constrained fields."""
    name: Str64
    amount: NonNegativeDecimal38_18


class ItemUpdate(ItemBase, all_fields_optional=True):
    """Auto-derived PATCH DTO: every field optional, constraints intact."""


def test_all_fields_default_to_none() -> None:
    update = ItemUpdate()
    assert update.name is None
    assert update.amount is None


def test_json_null_parses_without_typeerror() -> None:
    """The original crash scenario: explicit nulls in a PATCH body."""
    update = ItemUpdate.model_validate_json('{"name": null, "amount": null}')
    assert update.name is None
    assert update.amount is None


def test_constraints_still_enforced_on_values() -> None:
    # Ge(0) survives on the inner Annotated layer
    with pytest.raises(ValidationError):
        ItemUpdate.model_validate_json('{"amount": "-1"}')
    # max_length=64 survives for the string alias
    with pytest.raises(ValidationError):
        ItemUpdate(name='x' * 65)


def test_valid_values_accepted_and_serializer_kept() -> None:
    update = ItemUpdate.model_validate_json('{"amount": "0.92"}')
    assert update.amount == Decimal('0.92')
    # The PlainSerializer stays on the outer layer: json mode emits a string.
    parsed = json.loads(update.model_dump_json())
    assert parsed['amount'] == '0.92'


def test_float_rejection_survives_optionalization() -> None:
    """The reject-float BeforeValidator (outer, None-safe) keeps working."""
    with pytest.raises(ValidationError):
        ItemUpdate.model_validate_json('{"amount": 0.5}')
