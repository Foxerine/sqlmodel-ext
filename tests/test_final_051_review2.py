"""Independent final-review probes for the 0.5.1 inherited-field repair."""
from typing import Annotated

from pydantic import AfterValidator
from sqlmodel import Field, SQLModel

from sqlmodel_ext import SQLModelBase, TableBaseMixin


FinalReviewAlias = Annotated[str, Field(alias='final_review_legacy_name', max_length=16)]


def _record_final_review_call(value: str) -> str:
    _FINAL_REVIEW_CALLS.append(value)
    return value


_FINAL_REVIEW_CALLS: list[str] = []
FinalReviewValidatedAlias = Annotated[
    str,
    Field(alias='final_review_legacy_validated', max_length=16),
    AfterValidator(_record_final_review_call),
]


class FinalReviewPureBase(SQLModel):
    """Plain SQLModel control: the right-hand Field explicitly clears the alias."""

    name: FinalReviewAlias = Field(alias=None, index=True)


class FinalReviewPureRow(FinalReviewPureBase, table=True):
    id: int | None = Field(default=None, primary_key=True)


class FinalReviewExtBase(SQLModelBase):
    """sqlmodel-ext base with the same declaration as the plain control."""

    name: FinalReviewAlias = Field(alias=None, index=True)


class FinalReviewExtRow(FinalReviewExtBase, TableBaseMixin, table=True):
    pass


class FinalReviewValidatedBase(SQLModelBase):
    value: FinalReviewValidatedAlias = Field(alias=None, index=True)


class FinalReviewValidatedRow(FinalReviewValidatedBase, TableBaseMixin, table=True):
    pass


def test_final_review_inherited_alias_clear_matches_plain_sqlmodel() -> None:
    """The inherited resolved FieldInfo preserves an explicit ``alias=None``."""

    assert FinalReviewPureBase.model_fields['name'].alias is None
    assert FinalReviewPureRow.model_fields['name'].alias is None
    assert FinalReviewExtBase.model_fields['name'].alias is None
    assert FinalReviewExtRow.model_fields['name'].alias is None
    assert FinalReviewExtRow.model_json_schema()['properties']['name']['maxLength'] == 16
    assert 'final_review_legacy_name' not in FinalReviewExtRow.model_json_schema()['properties']
    assert FinalReviewExtRow.__table__.c.name.index is True  # pyright: ignore[reportAttributeAccessIssue]


def test_final_review_inherited_annotation_validator_runs_once() -> None:
    """Rebuilding the inherited annotation must not duplicate non-Field metadata."""

    _FINAL_REVIEW_CALLS.clear()
    row = FinalReviewValidatedRow.model_validate({'value': 'value'})
    assert row.value == 'value'
    assert _FINAL_REVIEW_CALLS == ['value']
