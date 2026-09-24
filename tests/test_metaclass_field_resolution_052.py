"""
Table-class fields resolve like the same declaration in a non-table class (0.5.2).

Two metaclass defects, both in ``_recover_annotated_sqlmodel_fields``:

A. Several SQLModel ``Field(...)`` in one ``Annotated`` and no right-hand side
   (``value: Legacy`` with ``Legacy = Annotated[str, Field(alias=...), Field(unique=True)]``):
   a table class declaring it directly kept ``alias`` / ``title`` /
   ``description`` / ``default_factory`` / ... of the earlier ``Field`` that
   the later one sets to ``None`` (sqlmodel's ``Field()`` passes those
   explicitly). A table class inheriting the declaration, a non-table class
   and plain SQLModel all follow Pydantic's merge, where the later ``Field``
   wins, ``None`` included. The same path took the Pydantic attributes of a
   ``Field`` inside a union member (``OptionalNonNegativeDecimal38_18 | None``),
   which Pydantic does not merge at field level.

B. A plain right-hand value on a type-alias field (``x: Str64 = 'a'``): the
   default was set on the field after construction, so Pydantic did not
   record it as set and built the core schema without it -- the field was in
   the JSON schema's ``required`` with no ``default`` and ``model_validate({})``
   rejected it, while ``model_fields`` (restored afterwards) showed the default.
   A metaclass rebuild (inherited description, relationship removal) hid it.

Criterion: for every declaration below, the Pydantic facts (field attributes,
JSON schema property, ``required``) are identical whether the declaration is
made in a table class, in a non-table base the table class inherits it from,
or in a plain SQLModel / non-table class; and every table class's JSON schema
agrees with its ``model_fields``. Column attributes keep the library's fold of
all ``Field``s (the right-hand one first, then the annotation's from the last
to the first).

Outside this fix, not asserted here: a field an STI / JTI child inherits from
its **table** parent (its ``model_fields`` default is the parent's column
attribute until the STI registration phases run, and a required field stays
optional), and the ``title`` / ``description`` of a ``Field`` inside a union
member (see ``_field_level_schema``).
"""
from decimal import Decimal
from typing import Annotated, Any, get_args

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from sqlalchemy import JSON, Column, Numeric
from sqlmodel import Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.main import FieldInfoMetadata

from sqlmodel_ext import (
    AutoPolymorphicIdentityMixin,
    FilePathType,
    NonNegativeDecimal38_18,
    OptionalNonNegativeDecimal38_18,
    PolymorphicBaseMixin,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
    Unset,
    create_subclass_id_mixin,
)

_PYDANTIC_ATTRS = (
    'default', 'default_factory', 'alias', 'validation_alias', 'serialization_alias',
    'title', 'description', 'examples', 'exclude', 'discriminator', 'repr', 'frozen',
    'json_schema_extra',
)


def _field_level_schema(prop: dict[str, Any]) -> dict[str, Any]:
    """
    The field-level part of a JSON schema property: its union members (``anyOf``) are left out.

    A ``Field(...)`` inside a union member (``Annotated[str, Field(title='U')] | None``)
    is rendered by Pydantic inside that member's schema (its ``title`` /
    ``description`` / ``default``, a nested ``anyOf``); the table class strips the
    SQLModel ``Field`` from the class annotation, so its member schemas lack them.
    Every table path (direct, inherited, right-hand ``Field``) has done so since
    before 0.5.2; it is outside this fix.
    """
    return {k: v for k, v in prop.items() if k != 'anyOf'}


def _facts(model: type[SQLModel], name: str) -> dict[str, Any]:
    """Pydantic-level facts of one field: attributes, JSON schema property, ``required``."""
    field = model.model_fields[name]
    schema = model.model_json_schema()
    key = field.alias if field.alias is not None else name
    facts: dict[str, Any] = {attr: getattr(field, attr) for attr in _PYDANTIC_ATTRS}
    facts['is_required'] = field.is_required()
    facts['schema_property'] = _field_level_schema(schema['properties'][key])
    facts['schema_required'] = key in schema.get('required', [])
    return facts


def _assert_schema_agrees_with_fields(model: type[SQLModel]) -> None:
    """The JSON schema's ``required`` / ``default`` match ``model_fields``."""
    schema = model.model_json_schema()
    required = set(schema.get('required', []))
    for name, field in model.model_fields.items():
        if field.exclude:
            continue
        key = field.alias if field.alias is not None else name
        assert (key in required) == field.is_required(), f"{model.__name__}.{name}: required"
        if field.default is not PydanticUndefined:
            assert 'default' in schema['properties'][key], f"{model.__name__}.{name}: default"


def _column(model: type[SQLModel], name: str) -> Column[Any]:
    return model.__table__.c[name]  # pyright: ignore[reportAttributeAccessIssue]


# ================================================================ A: several Field in one Annotated

Legacy = Annotated[str, Field(alias='legacy', title='T', description='D', max_length=32), Field(unique=True)]
FiveTitled = Annotated[int, Field(default=5, title='five'), Field(index=True)]
ListWithFactory = Annotated[list[str], Field(default_factory=list, sa_type=JSON), Field(nullable=True)]
LegacyInUnion = Annotated[str, Field(alias='u', title='U', max_length=8), Field(unique=True)]
NumericOverride = Annotated[NonNegativeDecimal38_18, Field(sa_type=Numeric(20, 2))]
UniqueThenNotUnique = Annotated[str, Field(unique=True, max_length=8), Field(unique=False)]
PathWithFields = Annotated[FilePathType, Field(alias='p', title='P'), Field(index=True)]


class R052APlain(SQLModel):
    legacy: Legacy
    five: FiveTitled
    listed: ListWithFactory
    unioned: LegacyInUnion | None
    override: NumericOverride
    uniq: UniqueThenNotUnique
    path: PathWithFields
    opt_union: OptionalNonNegativeDecimal38_18 | None


class R052ABase(SQLModelBase):
    legacy: Legacy
    five: FiveTitled
    listed: ListWithFactory
    unioned: LegacyInUnion | None
    override: NumericOverride
    uniq: UniqueThenNotUnique
    path: PathWithFields
    opt_union: OptionalNonNegativeDecimal38_18 | None


class R052AInherited(R052ABase, UUIDTableBaseMixin, table=True):
    pass


class R052ADirect(SQLModelBase, UUIDTableBaseMixin, table=True):
    legacy: Legacy
    five: FiveTitled
    listed: ListWithFactory
    unioned: LegacyInUnion | None
    override: NumericOverride
    uniq: UniqueThenNotUnique
    path: PathWithFields
    opt_union: OptionalNonNegativeDecimal38_18 | None


class R052AMid(R052ABase):
    extra: int = 0


class R052AMultiLevel(R052AMid, UUIDTableBaseMixin, table=True):
    pass


class R052AStiRoot(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    name: str = 'root'


class R052AStiChild(R052AStiRoot, AutoPolymorphicIdentityMixin, table=True):
    legacy: Legacy | None = None
    five: FiveTitled


class R052AStiChildPlainTwin(SQLModel):
    legacy: Legacy | None = None
    five: FiveTitled


class R052APatch(R052ABase, partial=True):
    pass


_A_FIELDS = list(R052ABase.model_fields)
_A_TABLES = [R052ADirect, R052AInherited, R052AMultiLevel]


class TestSeveralFieldsInOneAnnotated:
    @pytest.mark.parametrize('name', _A_FIELDS)
    @pytest.mark.parametrize('table', _A_TABLES, ids=lambda c: c.__name__)
    def test_table_field_matches_plain_sqlmodel(self, table: type[SQLModel], name: str) -> None:
        assert _facts(table, name) == _facts(R052APlain, name)

    @pytest.mark.parametrize('name', _A_FIELDS)
    def test_non_table_field_matches_plain_sqlmodel(self, name: str) -> None:
        """Includes ``path``: its ``sa_type`` is injected through a right-hand ``Field`` that must change nothing."""
        assert _facts(R052ABase, name) == _facts(R052APlain, name)

    def test_later_field_clears_pydantic_attributes(self) -> None:
        field = R052ADirect.model_fields['legacy']
        assert (field.alias, field.title, field.description) == (None, None, None)
        assert R052ADirect.model_validate({'legacy': 'x', 'five': 1, 'listed': [], 'unioned': None,
                                           'override': '1', 'uniq': 'u', 'path': 'a.txt',
                                           'opt_union': None}).legacy == 'x'

    def test_field_inside_union_member_is_not_field_level(self) -> None:
        """Pydantic does not merge a union member's ``Field``: its ``default=None`` does not apply."""
        assert R052ADirect.model_fields['opt_union'].is_required()
        assert R052ADirect.model_fields['unioned'].alias is None

    def test_later_field_keeps_unset_default_and_clears_default_factory(self) -> None:
        """sqlmodel's ``Field()`` passes ``default_factory=None`` explicitly, ``default`` not at all."""
        assert R052ADirect.model_fields['five'].default == 5
        assert R052ADirect.model_fields['five'].title is None
        assert R052ADirect.model_fields['listed'].default_factory is None
        assert R052ADirect.model_fields['listed'].is_required()

    @pytest.mark.parametrize('table', _A_TABLES, ids=lambda c: c.__name__)
    def test_column_attributes_are_folded(self, table: type[SQLModel]) -> None:
        legacy = _column(table, 'legacy')
        assert legacy.unique is True
        assert repr(legacy.type) == 'AutoString(length=32)'
        assert _column(table, 'five').index is True
        assert repr(_column(table, 'unioned').type) == 'AutoString(length=8)'
        assert _column(table, 'unioned').unique is True
        assert _column(table, 'uniq').unique is True
        assert _column(table, 'path').index is True

    @pytest.mark.parametrize('table', _A_TABLES, ids=lambda c: c.__name__)
    def test_later_annotation_field_wins_column_conflicts(self, table: type[SQLModel]) -> None:
        """``Annotated[Alias, Field(sa_type=...)]`` overrides the alias's ``sa_type``."""
        column_type = _column(table, 'override').type
        assert isinstance(column_type, Numeric)
        assert (column_type.precision, column_type.scale) == (20, 2)

    def test_alias_carriers_are_not_mutated(self) -> None:
        """Folding works on a copy: the carriers shared by every user of the aliases keep their values."""
        carriers = [
            item
            for arg in get_args(NumericOverride)[1:] if isinstance(arg, FieldInfo)
            for item in arg.metadata if isinstance(item, FieldInfoMetadata)
        ]
        assert [repr(c.sa_type) for c in carriers] == [
            'Numeric(precision=38, scale=18)', 'Numeric(precision=20, scale=2)',
        ]
        legacy_carriers = [
            item
            for arg in get_args(Legacy)[1:] if isinstance(arg, FieldInfo)
            for item in arg.metadata if isinstance(item, FieldInfoMetadata)
        ]
        assert [c.unique for c in legacy_carriers] == [PydanticUndefined, True]

    @pytest.mark.parametrize('name', ['legacy', 'five'])
    def test_sti_child_matches_plain_sqlmodel(self, name: str) -> None:
        assert _facts(R052AStiChild, name) == _facts(R052AStiChildPlainTwin, name)

    def test_partial_follows_base_resolution(self) -> None:
        for name in _A_FIELDS:
            field = R052APatch.model_fields[name]
            assert field.default is Unset
            assert field.alias == R052ABase.model_fields[name].alias

    @pytest.mark.parametrize('table', [*_A_TABLES, R052AStiChild], ids=lambda c: c.__name__)
    def test_schema_agrees_with_fields(self, table: type[SQLModel]) -> None:
        _assert_schema_agrees_with_fields(table)


# ================================================================ B: plain right-hand value

class R052BNonTable(SQLModelBase):
    dec: NonNegativeDecimal38_18 = Decimal(0)
    s: Str64 = 'a'
    n: Str64 | None = None
    m: Str64 | None = 'z'
    opt: OptionalNonNegativeDecimal38_18 = Decimal(1)
    i: int = 3
    tags: Annotated[list[str], Field(sa_type=JSON)] = Field(default_factory=list)


class R052BDirect(SQLModelBase, UUIDTableBaseMixin, table=True):
    dec: NonNegativeDecimal38_18 = Decimal(0)
    s: Str64 = 'a'
    n: Str64 | None = None
    m: Str64 | None = 'z'
    opt: OptionalNonNegativeDecimal38_18 = Decimal(1)
    i: int = 3
    tags: Annotated[list[str], Field(sa_type=JSON)] = Field(default_factory=list)


class R052BInherited(R052BNonTable, UUIDTableBaseMixin, table=True):
    pass


class R052BMid(R052BNonTable):
    extra: int = 0


class R052BMultiLevel(R052BMid, UUIDTableBaseMixin, table=True):
    pass


class R052BStiRoot(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    s: Str64 = 'a'


class R052BStiChild(R052BStiRoot, AutoPolymorphicIdentityMixin, table=True):
    dec: NonNegativeDecimal38_18 = Decimal(0)


class R052BStiChildTwin(SQLModelBase):
    s: Str64 = 'a'
    dec: NonNegativeDecimal38_18 = Decimal(0)


class R052BJtiRoot(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    s: Str64 = 'a'


R052BJtiIdMixin = create_subclass_id_mixin('r052bjtiroot')


class R052BJtiChild(R052BJtiIdMixin, R052BJtiRoot, AutoPolymorphicIdentityMixin, table=True):
    dec: NonNegativeDecimal38_18 = Decimal(0)


class R052BRedeclaredBase(SQLModelBase):
    s: Str64 = 'a'


class R052BRedeclared(R052BRedeclaredBase, UUIDTableBaseMixin, table=True):
    s: Str64


class R052BPatch(R052BNonTable, partial=True):
    pass


_B_FIELDS = list(R052BNonTable.model_fields)
_B_TABLES = [R052BDirect, R052BInherited, R052BMultiLevel]


class TestPlainRightHandValue:
    @pytest.mark.parametrize('name', _B_FIELDS)
    @pytest.mark.parametrize('table', _B_TABLES, ids=lambda c: c.__name__)
    def test_table_field_matches_non_table(self, table: type[SQLModel], name: str) -> None:
        assert _facts(table, name) == _facts(R052BNonTable, name)

    @pytest.mark.parametrize('table', _B_TABLES, ids=lambda c: c.__name__)
    def test_defaults_in_schema_and_validation(self, table: type[SQLModel]) -> None:
        schema = table.model_json_schema()
        assert set(schema.get('required', [])) == set()
        assert schema['properties']['s']['default'] == 'a'
        assert schema['properties']['dec']['default'] == '0'
        validated = table.model_validate({})
        assert (validated.s, validated.dec, validated.m, validated.i) == ('a', Decimal(0), 'z', 3)  # pyright: ignore[reportAttributeAccessIssue]

    @pytest.mark.parametrize('table', [R052BStiChild, R052BJtiChild], ids=lambda c: c.__name__)
    def test_polymorphic_children_match_non_table(self, table: type[SQLModel]) -> None:
        """The child's own field; ``s`` inherited from the table root is outside this fix (see module docstring)."""
        assert _facts(table, 'dec') == _facts(R052BStiChildTwin, 'dec')

    @pytest.mark.parametrize('root', [R052BStiRoot, R052BJtiRoot], ids=lambda c: c.__name__)
    def test_polymorphic_roots_match_non_table(self, root: type[SQLModel]) -> None:
        assert _facts(root, 's') == _facts(R052BStiChildTwin, 's')

    def test_redeclared_without_right_hand_side_keeps_inherited_default_in_schema(self) -> None:
        """The table class inherits the base's default (a library rule); the schema must say so too."""
        field = R052BRedeclared.model_fields['s']
        assert field.default == 'a'
        assert 's' not in R052BRedeclared.model_json_schema().get('required', [])
        assert R052BRedeclared.model_validate({}).s == 'a'

    def test_column_attributes_kept(self) -> None:
        dec_type = _column(R052BDirect, 'dec').type
        assert isinstance(dec_type, Numeric)
        assert (dec_type.precision, dec_type.scale) == (38, 18)
        assert repr(_column(R052BDirect, 's').type) == 'AutoString(length=64)'
        assert _column(R052BDirect, 'n').nullable is True

    def test_constraints_still_validated(self) -> None:
        with pytest.raises(ValidationError):
            _ = R052BDirect.model_validate({'s': 'x' * 65})
        with pytest.raises(ValidationError):
            _ = R052BDirect.model_validate({'dec': '-1'})

    def test_partial_unaffected(self) -> None:
        for name in _B_FIELDS:
            assert R052BPatch.model_fields[name].default is Unset
        assert R052BPatch.model_json_schema().get('required', []) == []

    @pytest.mark.parametrize(
        'table',
        [*_B_TABLES, R052BStiRoot, R052BStiChild, R052BJtiRoot, R052BJtiChild, R052BRedeclared],
        ids=lambda c: c.__name__,
    )
    def test_schema_agrees_with_fields(self, table: type[SQLModel]) -> None:
        _assert_schema_agrees_with_fields(table)

    @pytest.mark.asyncio
    async def test_defaults_persisted(self, session: AsyncSession) -> None:
        row = await R052BDirect.model_validate({}).save(session)
        stored = await R052BDirect.get_one(session, row.id)
        assert (stored.s, stored.dec, stored.m, stored.tags) == ('a', Decimal(0), 'z', [])
