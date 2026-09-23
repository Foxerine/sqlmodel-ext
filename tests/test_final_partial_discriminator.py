"""Final-review regression for discriminated unions inherited by partial DTOs."""
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from sqlmodel_ext import SQLModelBase, Unset


class _Cat(SQLModelBase):
    kind: Literal['cat']
    lives: int


class _Dog(SQLModelBase):
    kind: Literal['dog']
    bark: str


class _Base(SQLModelBase):
    pet: Annotated[_Cat | _Dog, Field(discriminator='kind')]


class _Patch(_Base, partial=True):
    pass


def test_partial_discriminated_union_supports_omission_and_tag_dispatch() -> None:
    omitted = _Patch()
    assert omitted.pet is Unset

    supplied = _Patch.model_validate({'pet': {'kind': 'dog', 'bark': 'woof'}})
    assert isinstance(supplied.pet, _Dog)
    assert supplied.model_dump() == {'pet': {'kind': 'dog', 'bark': 'woof'}}


def test_partial_discriminated_union_rejects_unknown_tag_via_discriminator() -> None:
    try:
        _Patch.model_validate({'pet': {'kind': 'bird', 'bark': 'chirp'}})
    except ValidationError as error:
        assert any(item['type'] == 'union_tag_invalid' for item in error.errors())
        return
    raise AssertionError('unknown discriminator tag must be rejected')
