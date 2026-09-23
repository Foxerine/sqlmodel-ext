"""Final-review regression for sentinel injection into disambiguated nested schemas."""
from sqlmodel_ext import OMITTED_SENTINEL, SQLModelBase, SQLModelExtConfig, Unset


def _make_integer_item() -> type[SQLModelBase]:
    class Item(SQLModelBase):
        value: Unset | int = Unset

    return Item


def _make_string_item() -> type[SQLModelBase]:
    class Item(SQLModelBase):
        value: Unset | str = Unset

    return Item


IntegerItem = _make_integer_item()
StringItem = _make_string_item()


class _Host(SQLModelBase):
    model_config = SQLModelExtConfig(omitted_sentinel=True)

    integer_item: IntegerItem
    string_item: StringItem


def test_omitted_sentinel_reaches_nested_models_with_duplicate_class_names() -> None:
    schema = _Host.model_json_schema()
    definitions = schema['$defs']

    assert len(definitions) == 2
    for definition in definitions.values():
        value_schema = definition['properties']['value']
        assert {'const': OMITTED_SENTINEL, 'type': 'string'} in value_schema['anyOf']
        assert value_schema['default'] == OMITTED_SENTINEL
