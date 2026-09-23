"""Final-review regression for alias generators on ``partial=True`` subclasses."""
from pydantic import ConfigDict

from sqlmodel_ext import SQLModelBase


class _Base(SQLModelBase):
    model_config = ConfigDict(alias_generator=lambda name: f'base_{name}')

    some_field: int


class _Patch(_Base, partial=True):
    model_config = ConfigDict(alias_generator=lambda name: f'patch_{name}')


def test_partial_subclass_alias_generator_controls_inherited_field() -> None:
    assert _Patch.model_fields['some_field'].alias == 'patch_some_field'
    assert set(_Patch.model_json_schema()['properties']) == {'patch_some_field'}
    assert _Patch.model_validate({'patch_some_field': 3}).some_field == 3
