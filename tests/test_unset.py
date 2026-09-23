"""
Tri-state semantics (``Unset`` / ``null`` / value) and the opt-in ``omitted_sentinel`` wire value.

The matrix section pins every combination of
switch (off/on) x nullability x inbound form x output, so that none of them can
drift silently.
"""
import copy
import json
import pickle

import pytest
from pydantic import ValidationError
from typing_extensions import Sentinel

from sqlmodel_ext import OMITTED_SENTINEL, SQLModelBase, SQLModelExtConfig, Unset


# ==================== Unset itself ====================

class TestUnsetSentinel:
    def test_unset_is_pydantic_missing(self) -> None:
        from pydantic.experimental.missing_sentinel import MISSING

        assert Unset is MISSING

    def test_deepcopy_keeps_identity(self) -> None:
        assert copy.deepcopy(Unset) is Unset
        assert copy.copy(Unset) is Unset
        assert copy.deepcopy({'a': [Unset]})['a'][0] is Unset

    def test_other_sentinels_still_refuse_deepcopy(self) -> None:
        other = Sentinel('OtherSentinel')
        with pytest.raises(TypeError):
            copy.deepcopy(other)

    def test_pickle_still_refused(self) -> None:
        with pytest.raises(Exception):
            pickle.dumps(Unset)

    def test_model_with_unset_default_deepcopies(self) -> None:
        class _M(SQLModelBase):
            x: Unset | int = Unset

        clone = copy.deepcopy(_M())
        assert clone.x is Unset


# ==================== omitted_sentinel: nesting & isolation ====================

class _NestedItem(SQLModelBase):
    """Nested model with the switch off (default) -- the host propagates it."""

    required: str
    omissible: Unset | int = Unset


class _NestedHost(SQLModelBase):
    model_config = SQLModelExtConfig(omitted_sentinel=True)

    items: list[_NestedItem] = []


class _NestedHostOff(SQLModelBase):
    items: list[_NestedItem] = []


class _SelfRef(SQLModelBase):
    model_config = SQLModelExtConfig(omitted_sentinel=True)

    name: Unset | str = Unset
    children: list['_SelfRef'] = []


class TestOmittedSentinelScope:
    def test_nested_payload_is_normalised_and_schema_injected(self) -> None:
        parsed = _NestedHost.model_validate({
            'items': [{'required': 'r', 'omissible': OMITTED_SENTINEL}],
        })
        assert parsed.items[0].omissible is Unset
        assert parsed.model_dump() == {'items': [{'required': 'r'}]}

        nested_props = _NestedHost.model_json_schema()['$defs']['_NestedItem']['properties']
        assert OMITTED_SENTINEL in str(nested_props['omissible'])
        assert OMITTED_SENTINEL not in str(nested_props['required'])

        # No leak into the nested model's own schema.
        assert OMITTED_SENTINEL not in str(_NestedItem.model_json_schema())

    def test_switch_off_host_gets_no_sentinel(self) -> None:
        assert OMITTED_SENTINEL not in str(_NestedHostOff.model_json_schema())

    def test_self_referential_model_terminates(self) -> None:
        schema = _SelfRef.model_json_schema()
        assert OMITTED_SENTINEL in str(schema)

    def test_switch_defaults_off(self) -> None:
        class _DefaultModel(SQLModelBase):
            value: str

        parsed = _DefaultModel(value=OMITTED_SENTINEL)
        assert parsed.value == OMITTED_SENTINEL
        assert OMITTED_SENTINEL not in str(_DefaultModel.model_json_schema())

    def test_explicit_false_equals_default(self) -> None:
        class _Disabled(SQLModelBase):
            model_config = SQLModelExtConfig(omitted_sentinel=False)

            value: str

        assert _Disabled(value=OMITTED_SENTINEL).value == OMITTED_SENTINEL

    def test_switch_inherited_and_merged_with_base_config(self) -> None:
        class _Child(_NestedHost):
            extra_field: Unset | str = Unset

        assert _Child.model_validate({'extra_field': OMITTED_SENTINEL}).extra_field is Unset
        # Base config keys (extra='forbid') still apply.
        with pytest.raises(ValidationError):
            _Child.model_validate({'bogus': 1})

    def test_non_dict_input_passes_through(self) -> None:
        item = _NestedItem(required='r')
        host = _NestedHost.model_validate(_NestedHost(items=[item]))
        assert host.items[0].required == 'r'

    def test_required_field_given_sentinel_fails_loud(self) -> None:
        # A non-omissible field cannot accept Unset -- the sentinel is not a value.
        with pytest.raises(ValidationError):
            _NestedHost.model_validate({'items': [{'required': OMITTED_SENTINEL}]})


# ==================== Matrix ====================

class _SwitchOff(SQLModelBase):
    not_nullable: Unset | str = Unset
    nullable: Unset | str | None = Unset


class _SwitchOn(SQLModelBase):
    model_config = SQLModelExtConfig(omitted_sentinel=True)

    not_nullable: Unset | str = Unset
    nullable: Unset | str | None = Unset


_MATRIX_MODELS = (_SwitchOff, _SwitchOn)
_MATRIX_FIELDS = ('not_nullable', 'nullable')


@pytest.mark.parametrize('model', _MATRIX_MODELS, ids=lambda m: m.__name__)
@pytest.mark.parametrize('field', _MATRIX_FIELDS)
def test_matrix_omitted_key_yields_unset(model: type[SQLModelBase], field: str) -> None:
    parsed = model.model_validate({})
    assert getattr(parsed, field) is Unset
    assert field not in parsed.model_fields_set
    assert model.field_is_omissible(field)


@pytest.mark.parametrize('model', _MATRIX_MODELS, ids=lambda m: m.__name__)
@pytest.mark.parametrize('field', _MATRIX_FIELDS)
def test_matrix_real_value_passes_through(model: type[SQLModelBase], field: str) -> None:
    parsed = model.model_validate({field: 'real'})
    assert getattr(parsed, field) == 'real'
    assert field in parsed.model_fields_set


@pytest.mark.parametrize('model', _MATRIX_MODELS, ids=lambda m: m.__name__)
def test_matrix_null_depends_only_on_type(model: type[_SwitchOff] | type[_SwitchOn]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({'not_nullable': None})

    parsed = model.model_validate({'nullable': None})
    assert parsed.nullable is None
    assert 'nullable' in parsed.model_fields_set


@pytest.mark.parametrize('field', _MATRIX_FIELDS)
def test_matrix_sentinel_normalised_only_when_switch_on(field: str) -> None:
    assert getattr(_SwitchOn.model_validate({field: OMITTED_SENTINEL}), field) is Unset
    assert getattr(_SwitchOff.model_validate({field: OMITTED_SENTINEL}), field) == OMITTED_SENTINEL


@pytest.mark.parametrize('model', _MATRIX_MODELS, ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    ('payload', 'expected_keys'),
    [
        ({}, set()),
        ({'not_nullable': 'v'}, {'not_nullable'}),
        ({'nullable': None}, {'nullable'}),
        ({'not_nullable': 'v', 'nullable': None}, {'not_nullable', 'nullable'}),
    ],
    ids=['omitted', 'value', 'explicit-null', 'both'],
)
def test_matrix_serialisation_excludes_unset_without_flags(
        model: type[SQLModelBase], payload: dict[str, object], expected_keys: set[str],
) -> None:
    parsed = model.model_validate(payload)
    assert set(parsed.model_dump()) == expected_keys
    assert set(json.loads(parsed.model_dump_json())) == expected_keys


@pytest.mark.parametrize('field', _MATRIX_FIELDS)
def test_matrix_schema_sentinel_branch_only_when_switch_on(field: str) -> None:
    on_schema = _SwitchOn.model_json_schema()['properties'][field]
    assert {'const': OMITTED_SENTINEL, 'type': 'string'} in on_schema['anyOf']
    assert on_schema['default'] == OMITTED_SENTINEL
    assert OMITTED_SENTINEL not in str(_SwitchOff.model_json_schema()['properties'][field])


def test_matrix_schema_stays_flat_and_keeps_annotations() -> None:
    nullable_schema = _SwitchOn.model_json_schema()['properties']['nullable']
    assert 'title' in nullable_schema
    branches = nullable_schema['anyOf']
    assert all('anyOf' not in b and 'title' not in b for b in branches), branches
    assert {'type': 'string'} in branches
    assert {'type': 'null'} in branches
    assert {'const': OMITTED_SENTINEL, 'type': 'string'} in branches


def test_schema_uses_alias_key_when_present() -> None:
    from pydantic import Field as PydanticField

    class _Aliased(SQLModelBase):
        model_config = SQLModelExtConfig(omitted_sentinel=True)

        display: Unset | str = PydanticField(default=Unset, alias='displayName')

    props = _Aliased.model_json_schema()['properties']
    assert OMITTED_SENTINEL in str(props['displayName'])
    assert _Aliased.model_validate({'displayName': OMITTED_SENTINEL}).display is Unset


def test_annotation_is_omissible_sees_through_annotated() -> None:
    from typing import Annotated

    from sqlmodel import Field

    assert SQLModelBase.annotation_is_omissible(Annotated[Unset | int | None, Field(ge=0)])
    assert SQLModelBase.annotation_is_omissible(Unset | int)
    assert not SQLModelBase.annotation_is_omissible(int | None)
