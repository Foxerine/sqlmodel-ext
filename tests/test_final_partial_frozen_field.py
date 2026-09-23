"""Final-review regression for inherited Pydantic ``frozen`` metadata."""
from typing import Annotated

from pydantic import Field as PydanticField

from sqlmodel_ext import SQLModelBase, Unset


class _Base(SQLModelBase):
    value: Annotated[int, PydanticField(frozen=True)]


class _Patch(_Base, partial=True):
    pass


def test_partial_preserves_frozen_field_without_class_creation_failure() -> None:
    omitted = _Patch()
    assert omitted.value is Unset
    supplied = _Patch(value=3)
    assert supplied.value == 3
