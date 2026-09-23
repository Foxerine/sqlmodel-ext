"""
``partial=True`` derives PATCH DTOs whose inherited fields become ``Unset | T = Unset``.

Covers:

- inherited fields become omissible; omitted -> ``Unset`` (not ``None``)
- the three states (omitted / explicit null / value) stay distinguishable in dumps
- explicit ``null`` is rejected at the *field type* level unless the base field
  allows ``None``
- constraints (Annotated and right-hand ``Field(...)``) survive; custom
  serializers / before-validators survive
- ``default_factory`` fields are ``Unset`` when omitted (factory not called)
- fields the class declares itself and ``Literal`` discriminators are untouched
- field-level attributes (``exclude`` / ``alias``) are hoisted out of the union
- multiple inheritance
- ``partial`` + ``table`` and the removed ``all_fields_optional`` keyword fail loud
- ``optional_dto_registry``: every registered class builds from ``{}`` and dumps ``{}``
"""
import json
from decimal import Decimal
from typing import Annotated, Literal, get_args

import pytest
from pydantic import ValidationError
from sqlmodel import Field

from sqlmodel_ext import (
    NonNegativeDecimal38_18,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
    Unset,
)
from sqlmodel_ext.base import optional_dto_registry


# ==================== Models ====================

class PartialBaseWithFactory(SQLModelBase):
    items: list[str] = Field(default_factory=list, max_length=10)
    """List field with default_factory."""

    tags: Annotated[list[str], Field(default_factory=lambda: ['default_tag'], max_length=20)]
    """Annotated-style default_factory."""

    name: str
    """Plain required field."""

    count: int = 0
    """Plain defaulted field."""

    score: Annotated[float, Field(ge=0.0, le=100.0)] = 50.0
    """Annotated constraint field."""

    note: str | None = None
    """Already nullable in the base -- becomes ``Unset | str | None``."""

    kind: Literal['base'] = 'base'
    """Literal discriminator -- skipped by ``partial``."""


class PartialBaseWithoutFactory(SQLModelBase):
    title: str
    enabled: bool = True
    value: Annotated[int, Field(ge=1, le=500)] = 100
    ratio: float = Field(default=0.5, gt=0, le=1)
    """Non-Annotated constraint form: constraints live on the right-hand ``Field``."""


class PartialWithFactory(PartialBaseWithFactory, partial=True):
    pass


class PartialNoFactory(PartialBaseWithoutFactory, partial=True):
    pass


class PartialMixed(PartialBaseWithFactory, PartialBaseWithoutFactory, partial=True):
    pass


class PartialWithOwnOverride(PartialBaseWithFactory, partial=True):
    count: int = 42
    """Declared by the author: ``partial`` must not touch it."""


class PartialItemBase(SQLModelBase):
    name: Str64
    amount: NonNegativeDecimal38_18


class PartialItemUpdate(PartialItemBase, partial=True):
    pass


class PartialFieldAttrsBase(SQLModelBase):
    temp_id: Annotated[str, Field(exclude=True)] = 'x'
    """Must stay excluded from dumps in the partial class."""

    display: Annotated[str, Field(alias='displayName')] = 'd'
    """Alias must keep working in the partial class."""


class PartialFieldAttrs(PartialFieldAttrsBase, partial=True):
    pass


def _annotation_allows_none(annotation: object) -> bool:
    pending: list[object] = [annotation]
    while pending:
        node = pending.pop()
        if node is type(None):
            return True
        pending.extend(get_args(node))
    return False


# ==================== Structure ====================

class TestPartialShape:
    """Structural guard: reverting to the old ``T | None`` shape turns these red."""

    def test_required_origin_fields_exclude_none(self) -> None:
        for field_name in ('name', 'count', 'score', 'items', 'tags'):
            annotation = PartialWithFactory.model_fields[field_name].annotation
            assert SQLModelBase.annotation_is_omissible(annotation), field_name
            assert not _annotation_allows_none(annotation), field_name

    def test_nullable_origin_field_keeps_none(self) -> None:
        annotation = PartialWithFactory.model_fields['note'].annotation
        assert SQLModelBase.annotation_is_omissible(annotation)
        assert _annotation_allows_none(annotation)

    def test_own_override_not_transformed(self) -> None:
        field = PartialWithOwnOverride.model_fields['count']
        assert not SQLModelBase.annotation_is_omissible(field.annotation)
        assert field.default == 42
        assert PartialWithOwnOverride().count == 42

    def test_literal_discriminator_not_transformed(self) -> None:
        assert not PartialWithFactory.field_is_omissible('kind')

    def test_field_is_omissible_unknown_name_is_false(self) -> None:
        assert PartialWithFactory.field_is_omissible('nope') is False

    def test_parent_unchanged(self) -> None:
        with pytest.raises(ValidationError):
            PartialBaseWithFactory()
        base = PartialBaseWithFactory(name='test')
        assert base.items == []
        assert base.tags == ['default_tag']
        assert base.count == 0
        assert not PartialBaseWithFactory.field_is_omissible('name')


# ==================== Three states ====================

class TestPartialThreeStates:
    def test_omitted_yields_unset(self) -> None:
        u = PartialWithFactory()
        for field_name in ('name', 'count', 'score', 'note', 'items', 'tags'):
            assert getattr(u, field_name) is Unset, field_name
        assert u.kind == 'base'

    def test_omitted_excluded_from_dump_without_any_flag(self) -> None:
        u = PartialWithFactory(name='x')
        assert u.model_dump() == {'name': 'x', 'kind': 'base'}
        assert u.model_dump(exclude_unset=True) == {'name': 'x'}
        assert json.loads(u.model_dump_json()) == {'name': 'x', 'kind': 'base'}

    def test_explicit_null_on_nullable_field_is_a_real_value(self) -> None:
        u = PartialWithFactory(note=None)
        assert u.note is None
        assert 'note' in u.model_fields_set
        assert u.model_dump()['note'] is None

    def test_explicit_null_on_required_origin_field_rejected_at_field_level(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PartialWithFactory.model_validate({'count': None})
        assert any(
            err.get('type') == 'missing_sentinel_error' and tuple(err['loc'])[:1] == ('count',)
            for err in exc_info.value.errors()
        ), exc_info.value.errors()

    def test_real_value_passes_through(self) -> None:
        u = PartialWithFactory(name='hello', items=['a', 'b'])
        assert u.name == 'hello'
        assert u.items == ['a', 'b']
        assert u.note is Unset
        assert u.model_fields_set == {'name', 'items'}

    def test_explicit_false_is_not_unset(self) -> None:
        u = PartialNoFactory(enabled=False, value=1)
        assert u.enabled is False
        assert u.value == 1

    def test_extra_forbid_inherited(self) -> None:
        with pytest.raises(ValidationError):
            PartialWithFactory(bogus=1)


# ==================== Constraints, factories, serializers ====================

class TestPartialConstraints:
    def test_annotated_constraints_preserved(self) -> None:
        with pytest.raises(ValidationError):
            PartialWithFactory(score=200.0)
        with pytest.raises(ValidationError):
            PartialNoFactory(value=0)
        assert PartialWithFactory(score=99.9).score == 99.9

    def test_right_hand_field_constraints_preserved(self) -> None:
        with pytest.raises(ValidationError):
            PartialNoFactory(ratio=0)
        with pytest.raises(ValidationError):
            PartialNoFactory(ratio=1.5)
        assert PartialNoFactory(ratio=0.25).ratio == 0.25

    def test_factory_not_invoked_when_omitted(self) -> None:
        u = PartialWithFactory()
        assert u.items is Unset
        assert u.tags is Unset

    def test_alias_types_keep_constraints(self) -> None:
        with pytest.raises(ValidationError):
            PartialItemUpdate.model_validate_json('{"amount": "-1"}')
        with pytest.raises(ValidationError):
            PartialItemUpdate(name='x' * 65)

    def test_alias_types_reject_null(self) -> None:
        with pytest.raises(ValidationError):
            PartialItemUpdate.model_validate_json('{"name": null}')

    def test_serializer_and_before_validator_kept(self) -> None:
        update = PartialItemUpdate.model_validate_json('{"amount": "0.92"}')
        assert update.amount == Decimal('0.92')
        assert json.loads(update.model_dump_json()) == {'amount': '0.92'}
        with pytest.raises(ValidationError):
            PartialItemUpdate.model_validate_json('{"amount": 0.5}')


class TestPartialFieldAttributesHoisted:
    """``exclude`` / ``alias`` on a union member would be silently dropped by Pydantic."""

    def test_exclude_survives(self) -> None:
        u = PartialFieldAttrs(temp_id='t')
        assert u.temp_id == 't'
        assert 'temp_id' not in u.model_dump()

    def test_alias_survives(self) -> None:
        u = PartialFieldAttrs.model_validate({'displayName': 'via-alias'})
        assert u.display == 'via-alias'
        assert u.model_dump(by_alias=True) == {'displayName': 'via-alias'}


# ==================== Inheritance ====================

class TestPartialMixedInheritance:
    def test_all_fields_unset_when_omitted(self) -> None:
        u = PartialMixed()
        assert u.items is Unset
        assert u.title is Unset

    def test_assign_from_both_bases(self) -> None:
        u = PartialMixed(name='a', title='b', items=['c'])
        assert u.model_dump() == {'name': 'a', 'title': 'b', 'items': ['c'], 'kind': 'base'}


class TestPartialDescriptions:
    def test_descriptions_inherited(self) -> None:
        assert PartialWithFactory.model_fields['name'].description == 'Plain required field.'
        schema = PartialWithFactory.model_json_schema()
        assert schema['properties']['name']['description'] == 'Plain required field.'
        assert 'required' not in schema or 'name' not in schema['required']


# ==================== Fail-loud guards ====================

class TestPartialGuards:
    def test_partial_and_table_are_mutually_exclusive(self) -> None:
        class _TableBase(SQLModelBase):
            label: str = 'x'

        with pytest.raises(TypeError, match='partial=True'):
            class _PartialTable(_TableBase, UUIDTableBaseMixin, table=True, partial=True):
                pass

    def test_removed_all_fields_optional_keyword_fails_loud(self) -> None:
        with pytest.raises(TypeError, match='all_fields_optional') as exc_info:
            class _Legacy(PartialBaseWithoutFactory, all_fields_optional=True):
                pass
        message = str(exc_info.value)
        assert 'partial=True' in message
        assert 'Unset' in message

    def test_removed_keyword_fails_even_when_false(self) -> None:
        with pytest.raises(TypeError, match='all_fields_optional'):
            class _Legacy(PartialBaseWithoutFactory, all_fields_optional=False):
                pass


# ==================== Registry ====================

class TestOptionalDtoRegistry:
    def test_partial_classes_registered(self) -> None:
        for cls in (PartialWithFactory, PartialNoFactory, PartialMixed, PartialItemUpdate):
            assert cls in optional_dto_registry
        assert PartialBaseWithFactory not in optional_dto_registry

    def test_every_registered_class_is_inert_when_empty(self) -> None:
        """An empty PATCH must construct and dump to only non-omissible fields."""
        assert optional_dto_registry, 'registry empty -- the check below would be vacuous'
        failures: list[str] = []
        for cls in optional_dto_registry:
            try:
                inst = cls.model_validate({})
            except ValidationError as e:
                # Only "missing" is tolerated (fields the author declared as required).
                non_missing = [err for err in e.errors() if err['type'] != 'missing']
                if non_missing:
                    failures.append(f'{cls.__qualname__}: {non_missing}')
                continue
            leaked = [key for key in inst.model_dump() if cls.field_is_omissible(key)]
            if leaked:
                failures.append(f'{cls.__qualname__}: omissible fields in dump {leaked}')
        assert not failures, failures
