"""Final-review regression: hoisted field attributes must not remain on a union member."""
from typing import Annotated
import warnings

from pydantic import Field as PydanticField
from pydantic.warnings import UnsupportedFieldAttributeWarning

from sqlmodel_ext import SQLModelBase


def test_partial_annotated_field_metadata_emits_no_unsupported_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')

        class Base(SQLModelBase):
            value: Annotated[int, PydanticField(alias='externalValue', frozen=True)]

        class Patch(Base, partial=True):
            pass

    unsupported = [
        warning
        for warning in caught
        if isinstance(warning.message, UnsupportedFieldAttributeWarning)
    ]
    assert unsupported == []
    assert Patch.model_fields['value'].alias == 'externalValue'
    assert Patch.model_fields['value'].frozen is True
