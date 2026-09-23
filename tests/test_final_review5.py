"""Independent final-review probes for release 0.5.0."""

from typing import Annotated, Generic, Literal, TypeVar

import pytest
from fastapi import Query
from fastapi._compat.v2 import ModelField
from pydantic import AliasChoices, AliasPath, ConfigDict, Field

from sqlmodel_ext import OMITTED_SENTINEL, SQLModelBase, SQLModelExtConfig, Unset


def _camel(name: str) -> str:
    return f'c_{name}'


def _replacement(name: str) -> str:
    return f'r_{name}'


class _AliasBase(SQLModelBase):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True)

    explicit_validation: Annotated[str, Field(validation_alias='inputName')]
    explicit_serialization: str = Field(serialization_alias='outputName')
    generated: str


class _AliasPartial(_AliasBase, partial=True):
    model_config = SQLModelExtConfig(
        alias_generator=_replacement,
        populate_by_name=True,
        omitted_sentinel=True,
    )


def test_partial_alias_generator_matches_plain_inheritance() -> None:
    class Plain(_AliasBase):
        model_config = ConfigDict(alias_generator=_replacement, populate_by_name=True)

    for name in _AliasBase.model_fields:
        partial = _AliasPartial.model_fields[name]
        plain = Plain.model_fields[name]
        assert (
            partial.alias,
            partial.validation_alias,
            partial.serialization_alias,
            partial.alias_priority,
        ) == (
            plain.alias,
            plain.validation_alias,
            plain.serialization_alias,
            plain.alias_priority,
        )

    for mode in ('validation', 'serialization'):
        partial_keys = set(_AliasPartial.model_json_schema(mode=mode)['properties'])
        plain_keys = set(Plain.model_json_schema(mode=mode)['properties'])
        assert partial_keys == plain_keys


def test_sentinel_schema_alias_choices_and_alias_paths_match_pydantic_keys() -> None:
    class AliasForms(SQLModelBase):
        model_config = SQLModelExtConfig(omitted_sentinel=True)

        path_then_fallback: Unset | str = Field(
            default=Unset,
            validation_alias=AliasChoices(AliasPath('outer', 'inner'), 'fallback'),
        )
        single_path: Unset | str = Field(
            default=Unset,
            validation_alias=AliasChoices(AliasPath('single')),
        )
        bare_path: Unset | str = Field(
            default=Unset,
            validation_alias=AliasPath('outer', 'bare'),
        )

    schema = AliasForms.model_json_schema(mode='validation')
    assert set(schema['properties']) == {'fallback', 'single', 'bare_path'}
    for value in schema['properties'].values():
        assert {'const': OMITTED_SENTINEL, 'type': 'string'} in value['anyOf']


def test_fastapi_query_default_copy_keeps_unset_identity() -> None:
    field = ModelField(
        field_info=Query(default=Unset),
        name='value',
    )

    assert field.get_default() is Unset


def test_sentinel_schema_reaches_generic_list_and_union_nested_models() -> None:
    value_type = TypeVar('value_type')

    class Box(SQLModelBase, Generic[value_type]):
        value: Unset | value_type = Unset

    class Leaf(SQLModelBase):
        label: Unset | str = Unset

    class Host(SQLModelBase):
        model_config = SQLModelExtConfig(omitted_sentinel=True)

        boxes: list[Box[int]]
        choice: Leaf | Box[str]

    schema = Host.model_json_schema()
    sentinel_count = str(schema).count(OMITTED_SENTINEL)
    assert sentinel_count >= 3


def test_required_literal_in_base_stays_required_in_partial() -> None:
    class Base(SQLModelBase):
        kind: Literal['a', 'b']
        value: str

    class Patch(Base, partial=True):
        pass

    with pytest.raises(ValueError):
        Patch.model_validate({})
    assert Patch.model_validate({'kind': 'a'}).value is Unset


def test_explicit_null_stays_distinct_from_omission() -> None:
    class Base(SQLModelBase):
        nullable: str | None = None
        required: str

    class Patch(Base, partial=True):
        pass

    omitted = Patch.model_validate({})
    assert omitted.nullable is Unset
    assert omitted.required is Unset
    assert omitted.model_dump() == {}
    assert omitted.model_dump_json() == '{}'

    explicit_null = Patch.model_validate({'nullable': None})
    assert explicit_null.nullable is None
    assert explicit_null.model_dump() == {'nullable': None}
    with pytest.raises(ValueError):
        Patch.model_validate({'required': None})
