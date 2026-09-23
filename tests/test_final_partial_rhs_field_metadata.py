"""Final-review regressions for ``partial=True`` fields declared with RHS ``Field`` metadata."""
from pydantic import ValidationError
from sqlmodel import Field

from sqlmodel_ext import SQLModelBase, Unset


class _Base(SQLModelBase):
    score: int = Field(
        default_factory=lambda: 7,
        alias='externalScore',
        exclude=True,
        ge=1,
    )


class _Patch(_Base, partial=True):
    pass


def test_partial_preserves_rhs_field_alias_and_exclude() -> None:
    omitted = _Patch.model_validate({})
    assert omitted.score is Unset
    assert omitted.model_dump() == {}

    supplied = _Patch.model_validate({'externalScore': 3})
    assert supplied.score == 3
    assert supplied.model_dump() == {}


def test_partial_rhs_field_constraint_still_rejects_invalid_value() -> None:
    try:
        _Patch.model_validate({'externalScore': 0})
    except ValidationError:
        return
    raise AssertionError('the inherited ge=1 constraint must remain effective')
