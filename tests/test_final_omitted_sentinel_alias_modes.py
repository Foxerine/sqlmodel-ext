"""Final-review regression for mode-specific aliases in sentinel-aware schemas."""
from pydantic import Field

from sqlmodel_ext import OMITTED_SENTINEL, SQLModelBase, SQLModelExtConfig, Unset


class _AliasedModel(SQLModelBase):
    model_config = SQLModelExtConfig(omitted_sentinel=True)

    validation_only: Unset | str = Field(
        default=Unset,
        validation_alias='validationOnly',
    )
    serialization_only: Unset | str = Field(
        default=Unset,
        serialization_alias='serializationOnly',
    )


def _assert_sentinel_branch(field_schema: dict[str, object]) -> None:
    branches = field_schema['anyOf']
    assert isinstance(branches, list)
    assert {'const': OMITTED_SENTINEL, 'type': 'string'} in branches
    assert field_schema['default'] == OMITTED_SENTINEL


def test_validation_schema_injects_validation_alias_property() -> None:
    properties = _AliasedModel.model_json_schema(mode='validation')['properties']

    _assert_sentinel_branch(properties['validationOnly'])
    _assert_sentinel_branch(properties['serialization_only'])


def test_serialization_schema_injects_serialization_alias_property() -> None:
    properties = _AliasedModel.model_json_schema(mode='serialization')['properties']

    _assert_sentinel_branch(properties['validation_only'])
    _assert_sentinel_branch(properties['serializationOnly'])
