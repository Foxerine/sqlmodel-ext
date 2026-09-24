"""
Inherited fields keep the column attributes of their base-class ``Field(...)``.

A table class inheriting ``language: Str64 = Field(index=True)`` from a
non-table base used to get a column without the index: the metaclass rebuilt
the inherited ``Annotated`` field from the type alias's ``Field`` only, and the
base's right-hand ``Field(...)`` (index / unique / primary_key / foreign_key /
nullable / sa_type / sa_column_kwargs / ondelete ...) was dropped.

Criteria:

1. an inherited field produces the same column and Pydantic field as the same
   declaration made directly in the table class (for a right-hand ``Field``,
   both resolve the Pydantic attributes as Pydantic does, ``None`` included);
2. both match the identical declaration on plain ``SQLModel`` -- except where
   plain SQLModel itself drops an attribute the library recovers
   (``_PLAIN_SQLMODEL_DROPS``); there the library's value is asserted
   explicitly.
"""
from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import AfterValidator
from sqlalchemy import BigInteger, Column
from sqlmodel import Field, SQLModel

from sqlmodel_ext import (
    AutoPolymorphicIdentityMixin,
    OptionalNonNegativeDecimal38_18,
    PolymorphicBaseMixin,
    SQLModelBase,
    Str64,
    TableBaseMixin,
    UUIDTableBaseMixin,
    Unset,
    create_subclass_id_mixin,
)

Alias = Annotated[str, Field(max_length=64)]
IndexedAlias = Annotated[str, Field(max_length=32, index=True)]
NonNegative = Annotated[int, Field(ge=0)]
BigAlias = Annotated[int, Field(sa_type=BigInteger)]


class InhColParent(SQLModel, table=True):
    """Foreign-key target shared by both flavours."""

    __tablename__ = "inh_col_parent"  # pyright: ignore[reportAssignmentType]
    id: int | None = Field(default=None, primary_key=True)


# ---------------------------------------------------------------- single level

class InhColPureBase(SQLModel):
    al_index: Alias = Field(index=True)
    al_unique: Alias = Field(unique=True)
    al_optional: Alias | None = Field(default=None, index=True)
    al_nullable: Alias = Field(default="x", nullable=True)
    al_kwargs: Alias = Field(default="k", sa_column_kwargs={"comment": "commented"})
    al_described: Alias = Field(default="d", description="described", title="Described")
    al_indexed_alias: IndexedAlias = Field(unique=True)
    al_fk: NonNegative | None = Field(default=None, foreign_key="inh_col_parent.id", ondelete="SET NULL")
    al_big: BigAlias = Field(default=0, index=True)
    plain_index: int = Field(default=0, index=True)
    plain_unique: str = Field(unique=True)
    plain_fk: int | None = Field(default=None, foreign_key="inh_col_parent.id", ondelete="CASCADE")


class InhColExtBase(SQLModelBase):
    al_index: Alias = Field(index=True)
    al_unique: Alias = Field(unique=True)
    al_optional: Alias | None = Field(default=None, index=True)
    al_nullable: Alias = Field(default="x", nullable=True)
    al_kwargs: Alias = Field(default="k", sa_column_kwargs={"comment": "commented"})
    al_described: Alias = Field(default="d", description="described", title="Described")
    al_indexed_alias: IndexedAlias = Field(unique=True)
    al_fk: NonNegative | None = Field(default=None, foreign_key="inh_col_parent.id", ondelete="SET NULL")
    al_big: BigAlias = Field(default=0, index=True)
    plain_index: int = Field(default=0, index=True)
    plain_unique: str = Field(unique=True)
    plain_fk: int | None = Field(default=None, foreign_key="inh_col_parent.id", ondelete="CASCADE")


class InhColPure(InhColPureBase, table=True):
    id: int | None = Field(default=None, primary_key=True)


class InhColExt(InhColExtBase, TableBaseMixin, table=True):
    pass


class InhColExtDirect(SQLModelBase, TableBaseMixin, table=True):
    """The inherited declarations of ``InhColExtBase`` / ``InhColExtL2``, made in the table class itself."""

    al_index: Alias = Field(index=True)
    al_unique: Alias = Field(unique=True)
    al_optional: Alias | None = Field(default=None, index=True)
    al_nullable: Alias = Field(default="x", nullable=True)
    al_kwargs: Alias = Field(default="k", sa_column_kwargs={"comment": "commented"})
    al_described: Alias = Field(default="d", description="described", title="Described")
    al_indexed_alias: IndexedAlias = Field(unique=True)
    al_fk: NonNegative | None = Field(default=None, foreign_key="inh_col_parent.id", ondelete="SET NULL")
    al_big: BigAlias = Field(default=0, index=True)
    plain_index: int = Field(default=0, index=True)
    plain_unique: str = Field(unique=True)
    plain_fk: int | None = Field(default=None, foreign_key="inh_col_parent.id", ondelete="CASCADE")
    first: Alias = Field(index=True)
    second: Alias | None = Field(default=None, unique=True)


# ---------------------------------------------------------------- multi level

class InhColPureL1(SQLModel):
    first: Alias = Field(index=True)


class InhColPureL2(InhColPureL1):
    second: Alias | None = Field(default=None, unique=True)


class InhColPureMulti(InhColPureL2, table=True):
    id: int | None = Field(default=None, primary_key=True)


class InhColExtL1(SQLModelBase):
    first: Alias = Field(index=True)


class InhColExtL2(InhColExtL1):
    second: Alias | None = Field(default=None, unique=True)


class InhColExtMulti(InhColExtL2, TableBaseMixin, table=True):
    pass


# ---------------------------------------------------------------- primary key

class InhColPureKeyBase(SQLModel):
    key: Alias = Field(primary_key=True)
    label: Alias = Field(default="l", index=True)


class InhColPureKey(InhColPureKeyBase, table=True):
    pass


class InhColExtKeyBase(SQLModelBase):
    key: Alias = Field(primary_key=True)
    label: Alias = Field(default="l", index=True)


class InhColExtKey(InhColExtKeyBase, table=True):
    pass


# ---------------------------------------------------------------- overridden in the table class

class InhColPureOverrideBase(SQLModel):
    swapped: Alias = Field(index=True)
    redeclared: Alias = Field(index=True)
    kept: Alias = Field(index=True)


class InhColPureOverride(InhColPureOverrideBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    swapped: Alias = Field(unique=True)
    redeclared: Alias


class InhColExtOverrideBase(SQLModelBase):
    swapped: Alias = Field(index=True)
    redeclared: Alias = Field(index=True)
    kept: Alias = Field(index=True)


class InhColExtOverride(InhColExtOverrideBase, TableBaseMixin, table=True):
    swapped: Alias = Field(unique=True)
    redeclared: Alias


_PAIRS: list[tuple[type[SQLModel], type[SQLModel], list[str]]] = [
    (InhColPure, InhColExt, list(InhColPureBase.model_fields)),
    (InhColPureMulti, InhColExtMulti, ["first", "second"]),
    (InhColPureKey, InhColExtKey, ["key", "label"]),
    (InhColPureOverride, InhColExtOverride, ["swapped", "redeclared", "kept"]),
]
_CASES = [
    pytest.param(pure, ext, column, id=f"{ext.__name__}.{column}")
    for pure, ext, columns in _PAIRS
    for column in columns
]


def _column_facts(column: Column[Any]) -> dict[str, Any]:
    return {
        "type": repr(column.type),
        "primary_key": column.primary_key,
        "nullable": column.nullable,
        "index": column.index,
        "unique": column.unique,
        "foreign_keys": sorted((fk.target_fullname, fk.ondelete) for fk in column.foreign_keys),
        "comment": column.comment,
        "default": None if column.default is None else repr(getattr(column.default, "arg", column.default)),
    }


def _table_column(model: type[SQLModel], name: str) -> Column[Any]:
    return model.__table__.c[name]  # pyright: ignore[reportAttributeAccessIssue]


_PLAIN_SQLMODEL_DROPS: dict[tuple[str, str], dict[str, Any]] = {
    # ``Alias | None``: plain SQLModel loses the alias's ``max_length``.
    ("InhColExt", "al_optional"): {"type": "AutoString(length=64)"},
    ("InhColExtMulti", "second"): {"type": "AutoString(length=64)"},
    # An alias carrying ``index=True`` plus a right-hand ``Field(unique=True)``:
    # plain SQLModel keeps only the first ``FieldInfoMetadata`` carrier.
    ("InhColExt", "al_indexed_alias"): {"index": True},
    # ``sa_type`` inside the alias: plain SQLModel falls back to ``Integer``.
    ("InhColExt", "al_big"): {"type": "BigInteger()"},
}
"""Column facts plain SQLModel gets wrong for the same declaration (the library recovers them)."""


@pytest.mark.parametrize(("pure", "ext", "column"), _CASES)
def test_column_matches_plain_sqlmodel(pure: type[SQLModel], ext: type[SQLModel], column: str) -> None:
    ext_facts = _column_facts(_table_column(ext, column))
    pure_facts = _column_facts(_table_column(pure, column))
    recovered = _PLAIN_SQLMODEL_DROPS.get((ext.__name__, column), {})
    for key, value in recovered.items():
        assert ext_facts.pop(key) == value
        assert pure_facts.pop(key) != value
    assert ext_facts == pure_facts


_DIRECT_CASES = [
    pytest.param(ext, column, id=f"{ext.__name__}.{column}")
    for ext, columns in ((InhColExt, list(InhColExtBase.model_fields)), (InhColExtMulti, ["first", "second"]))
    for column in columns
]


@pytest.mark.parametrize(("ext", "column"), _DIRECT_CASES)
def test_inherited_column_matches_direct_declaration(ext: type[SQLModel], column: str) -> None:
    assert _column_facts(_table_column(ext, column)) == _column_facts(_table_column(InhColExtDirect, column))
    inherited = ext.model_fields[column]
    direct = InhColExtDirect.model_fields[column]
    assert inherited.is_required() == direct.is_required()
    assert inherited.default == direct.default
    assert ext.model_json_schema()["properties"][column] == InhColExtDirect.model_json_schema()["properties"][column]


@pytest.mark.parametrize(("pure", "ext", "column"), _CASES)
def test_pydantic_field_matches_plain_sqlmodel(pure: type[SQLModel], ext: type[SQLModel], column: str) -> None:
    pure_field = pure.model_fields[column]
    ext_field = ext.model_fields[column]
    assert ext_field.is_required() == pure_field.is_required()
    assert ext_field.default == pure_field.default
    assert ext_field.description == pure_field.description
    assert ext_field.title == pure_field.title
    pure_schema = pure.model_json_schema()["properties"][column]
    ext_schema = ext.model_json_schema()["properties"][column]
    assert ext_schema == pure_schema


def test_reported_reproduction() -> None:
    class InhColNewsBase(SQLModelBase):
        language: Str64 = Field(index=True)
        code: Str64 = Field(unique=True)

    class InhColNews(InhColNewsBase, TableBaseMixin, table=True):
        pass

    assert _table_column(InhColNews, "language").index is True
    assert _table_column(InhColNews, "code").unique is True


# ---------------------------------------------------------------- polymorphic

class InhColStiRoot(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    name: str


class InhColStiFields(SQLModelBase):
    breed: Alias | None = Field(default=None, index=True)
    tag: Alias | None = Field(default=None, unique=True, sa_column_kwargs={"comment": "tag"})


class InhColStiChild(InhColStiFields, InhColStiRoot, AutoPolymorphicIdentityMixin, table=True):
    pass


class InhColJtiRoot(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    name: str


InhColJtiIdMixin = create_subclass_id_mixin("inhcoljtiroot")


class InhColJtiFields(SQLModelBase):
    plate: Alias = Field(unique=True)
    make: Alias = Field(default="m", index=True)


class InhColJtiChild(InhColJtiIdMixin, InhColJtiFields, InhColJtiRoot, AutoPolymorphicIdentityMixin, table=True):
    pass


def test_sti_child_inherited_columns_keep_attributes() -> None:
    table = InhColStiRoot.__table__  # pyright: ignore[reportAttributeAccessIssue]
    assert table.c.breed.index is True
    assert table.c.tag.unique is True
    assert table.c.tag.comment == "tag"
    assert table.c.breed.nullable is True


def test_jti_child_inherited_columns_keep_attributes() -> None:
    table = InhColJtiChild.__table__  # pyright: ignore[reportAttributeAccessIssue]
    assert table.c.plate.unique is True
    assert table.c.plate.nullable is False
    assert table.c.make.index is True


# ---------------------------------------------------------------- partial

class InhColPartial(InhColExtBase, partial=True):
    pass


def test_partial_derivation_is_unaffected() -> None:
    patch = InhColPartial()
    assert patch.al_index is Unset
    assert not InhColPartial.model_fields["al_index"].is_required()
    with pytest.raises(ValueError):
        _ = InhColPartial(al_index="x" * 65)
    assert InhColPartial.model_fields["al_described"].description == "described"
    # The table class derived from the same base is unaffected by the partial.
    assert _table_column(InhColExt, "al_index").index is True


# ---------------------------------------------------------------- Pydantic attributes of the base's resolved field

FullAlias = Annotated[
    str,
    Field(
        alias="al_legacy",
        validation_alias="al_legacy_in",
        serialization_alias="al_legacy_out",
        title="Alias title",
        description="alias description",
        max_length=16,
        index=True,
    ),
]
"""An alias carrying every Pydantic naming / documentation attribute plus a column attribute."""

_PYDANTIC_ATTRS = ("alias", "validation_alias", "serialization_alias", "title", "description")

_RHS_CASES: dict[str, dict[str, Any] | None] = {
    # ``None``: the base declares the alias alone, no right-hand ``Field``.
    "alias_only": None,
    "column_only": {"unique": True},
    **{f"clear_{attr}": {attr: None} for attr in _PYDANTIC_ATTRS},
    **{f"override_{attr}": {attr: f"over_{attr}"} for attr in _PYDANTIC_ATTRS},
}
"""Right-hand ``Field(...)`` of the base declaration, per case."""


def _build_attr_case(case: str, rhs: dict[str, Any] | None) -> tuple[type[SQLModel], type[SQLModel], type[SQLModel]]:
    namespace: dict[str, Any] = {"__annotations__": {"name": FullAlias}}
    if rhs is not None:
        namespace["name"] = Field(**rhs)
    pure_base = type(f"InhAttrPureBase_{case}", (SQLModel,), dict(namespace))
    pure = type(
        f"InhAttrPure_{case}",
        (pure_base,),
        {"__annotations__": {"id": int | None}, "id": Field(default=None, primary_key=True)},
        table=True,
    )
    ext_base = type(f"InhAttrExtBase_{case}", (SQLModelBase,), dict(namespace))
    ext = type(f"InhAttrExt_{case}", (ext_base, TableBaseMixin), {}, table=True)
    return pure, ext_base, ext


_ATTR_MODELS = {case: _build_attr_case(case, rhs) for case, rhs in _RHS_CASES.items()}


_TABLE_ONLY_PROPERTIES = frozenset({"id", "created_at", "updated_at"})


def _field_schema(model: type[SQLModel], mode: Any) -> dict[str, Any]:
    properties: dict[str, Any] = model.model_json_schema(mode=mode)["properties"]
    return {key: value for key, value in properties.items() if key not in _TABLE_ONLY_PROPERTIES}


@pytest.mark.parametrize("case", list(_RHS_CASES))
def test_inherited_pydantic_attributes_follow_the_base_field(case: str) -> None:
    pure, ext_base, ext = _ATTR_MODELS[case]
    inherited = ext.model_fields["name"]
    for attr in _PYDANTIC_ATTRS:
        expected = getattr(ext_base.model_fields["name"], attr)
        assert getattr(inherited, attr) == expected, attr
        assert getattr(pure.model_fields["name"], attr) == expected, attr
    for mode in ("validation", "serialization"):
        assert _field_schema(ext, mode) == _field_schema(pure, mode) == _field_schema(ext_base, mode)
    # The column attributes of both declarations survive.
    column = _table_column(ext, "name")
    assert column.index is True
    assert column.unique is (case == "column_only")
    assert repr(column.type) == "AutoString(length=16)"
    # ...and are readable as attributes of the field (JTI detection reads ``primary_key`` / ``foreign_key`` so).
    assert getattr(inherited, "index") is True
    assert getattr(inherited, "unique") is (case == "column_only")


@pytest.mark.parametrize("case", list(_RHS_CASES))
def test_inherited_field_validates_and_dumps_like_the_base(case: str) -> None:
    _, ext_base, ext = _ATTR_MODELS[case]
    base_field = ext_base.model_fields["name"]
    input_key = base_field.validation_alias or base_field.alias or "name"
    assert isinstance(input_key, str)
    row = ext.model_validate({input_key: "v"})
    assert row.name == "v"  # pyright: ignore[reportAttributeAccessIssue]
    dumped = row.model_dump(by_alias=True, exclude=set(_TABLE_ONLY_PROPERTIES))
    assert dumped == ext_base.model_validate({input_key: "v"}).model_dump(by_alias=True)


def test_explicit_alias_clear_in_base_is_kept() -> None:
    """The reported case: ``Field(alias=None)`` in the base must not be undone by the alias's ``alias``."""
    _, ext_base, ext = _ATTR_MODELS["clear_alias"]
    assert ext_base.model_fields["name"].alias is None
    assert ext.model_fields["name"].alias is None
    assert "al_legacy" not in _field_schema(ext, "validation")


Conflicting = Annotated[str, Field(nullable=False, sa_column_kwargs={"comment": "alias"})]


def test_right_hand_column_attributes_win_over_the_alias() -> None:
    """The base's right-hand ``Field`` takes precedence over the alias's column attributes, as in a direct declaration."""
    class InhColConflictBase(SQLModelBase):
        name: Conflicting = Field(nullable=True, sa_column_kwargs={"comment": "rhs"})

    class InhColConflict(InhColConflictBase, TableBaseMixin, table=True):
        pass

    class InhColConflictDirect(SQLModelBase, TableBaseMixin, table=True):
        name: Conflicting = Field(nullable=True, sa_column_kwargs={"comment": "rhs"})

    inherited = _column_facts(_table_column(InhColConflict, "name"))
    assert inherited == _column_facts(_table_column(InhColConflictDirect, "name"))
    assert inherited["nullable"] is True
    assert inherited["comment"] == "rhs"


# ---------------------------------------------------------------- final-review regression (0.5.1)

AliasWithLegacyName = Annotated[str, Field(alias="legacy_name", max_length=16)]


class Review2PureBase(SQLModel):
    """Plain SQLModel control: the right-hand Field explicitly clears the alias."""

    name: AliasWithLegacyName = Field(alias=None, index=True)


class Review2PureRow(Review2PureBase, table=True):
    id: int | None = Field(default=None, primary_key=True)


class Review2ExtBase(SQLModelBase):
    """sqlmodel-ext base with the same declaration as the plain control."""

    name: AliasWithLegacyName = Field(alias=None, index=True)


class Review2ExtRow(Review2ExtBase, TableBaseMixin, table=True):
    pass


def test_inherited_explicit_alias_clear_matches_plain_sqlmodel() -> None:
    """The inherited resolved FieldInfo must preserve an explicit ``alias=None``."""
    assert Review2PureBase.model_fields["name"].alias is None
    assert Review2PureRow.model_fields["name"].alias is None
    assert Review2ExtBase.model_fields["name"].alias is None
    assert Review2ExtRow.model_fields["name"].alias is None
    assert "name" in Review2ExtRow.model_json_schema()["properties"]
    assert "legacy_name" not in Review2ExtRow.model_json_schema()["properties"]
    assert _table_column(Review2ExtRow, "name").index is True


# ---------------------------------------------------------------- direct declaration with a right-hand Field (0.5.1)
#
# A table class declaring ``name: Alias = Field(...)`` itself must resolve the
# Pydantic attributes exactly like plain SQLModel does for the same line, and
# exactly like a table class inheriting that line from a base: the right-hand
# ``Field`` overrides the alias's attributes, ``None`` included (sqlmodel's
# ``Field()`` passes every Pydantic attribute explicitly).

_RESOLVED_ATTRS = (*_PYDANTIC_ATTRS, "default")


def _build_direct_case(
        case: str,
        annotation: Any,
        rhs: dict[str, Any] | None,
) -> tuple[type[SQLModel], type[SQLModel], type[SQLModel]]:
    """Return (plain SQLModel table, sqlmodel-ext table, sqlmodel-ext table inheriting the same line)."""
    namespace: dict[str, Any] = {"__annotations__": {"name": annotation}}
    if rhs is not None:
        namespace["name"] = Field(**rhs)
    pure = type(
        f"DirAttrPure_{case}",
        (SQLModel,),
        {
            **namespace,
            "__annotations__": {"name": annotation, "id": int | None},
            "id": Field(default=None, primary_key=True),
        },
        table=True,
    )
    direct = type(f"DirAttrExt_{case}", (SQLModelBase, TableBaseMixin), dict(namespace), table=True)
    base = type(f"DirAttrExtBase_{case}", (SQLModelBase,), dict(namespace))
    inherited = type(f"DirAttrExtInh_{case}", (base, TableBaseMixin), {}, table=True)
    return pure, direct, inherited


_DIRECT_ATTR_MODELS = {
    **{f"plain_{case}": _build_direct_case(f"plain_{case}", FullAlias, rhs) for case, rhs in _RHS_CASES.items()},
    **{
        f"union_{case}": _build_direct_case(f"union_{case}", FullAlias | None, rhs)
        for case, rhs in _RHS_CASES.items()
        if rhs is not None
    },
}


@pytest.mark.parametrize("case", list(_DIRECT_ATTR_MODELS))
def test_direct_pydantic_attributes_match_plain_sqlmodel_and_inheritance(case: str) -> None:
    pure, direct, inherited = _DIRECT_ATTR_MODELS[case]
    for attr in _RESOLVED_ATTRS:
        expected = getattr(pure.model_fields["name"], attr)
        assert getattr(direct.model_fields["name"], attr) == expected, attr
        assert getattr(inherited.model_fields["name"], attr) == expected, attr
    assert direct.model_fields["name"].is_required() == pure.model_fields["name"].is_required()
    assert inherited.model_fields["name"].is_required() == pure.model_fields["name"].is_required()
    for mode in ("validation", "serialization"):
        assert _field_schema(direct, mode) == _field_schema(inherited, mode)
        if case.startswith("plain_"):
            # ``Alias | None``: plain SQLModel also documents the alias's
            # ``title`` / ``description`` inside the union member's schema; the
            # library moves the alias's ``Field`` out of the annotation, so only
            # the field-level schema (identical Pydantic attributes) is compared.
            assert _field_schema(direct, mode) == _field_schema(pure, mode)


@pytest.mark.parametrize("case", list(_DIRECT_ATTR_MODELS))
def test_direct_column_matches_inheritance(case: str) -> None:
    pure, direct, inherited = _DIRECT_ATTR_MODELS[case]
    direct_facts = _column_facts(_table_column(direct, "name"))
    assert direct_facts == _column_facts(_table_column(inherited, "name"))
    # The library keeps the alias's column attributes plain SQLModel drops
    # (it reads only the right-hand ``FieldInfoMetadata`` carrier).
    assert direct_facts["index"] is True
    assert _table_column(pure, "name").index is (case == "plain_alias_only")
    assert direct_facts["unique"] is case.endswith("column_only")
    assert direct_facts["type"] == "AutoString(length=16)"


def test_direct_explicit_alias_clear_matches_plain_sqlmodel() -> None:
    """The reported case, declared in the table class itself."""
    class Review3ExtRow(SQLModelBase, TableBaseMixin, table=True):
        name: AliasWithLegacyName = Field(alias=None, index=True)

    assert Review3ExtRow.model_fields["name"].alias is None
    assert Review3ExtRow.model_fields["name"].alias == Review2PureRow.model_fields["name"].alias
    assert "legacy_name" not in Review3ExtRow.model_json_schema()["properties"]
    assert _table_column(Review3ExtRow, "name").index is True
    assert Review3ExtRow.model_validate({"name": "v"}).name == "v"


# ---------------------------------------------------------------- annotation validators run once

_VALIDATOR_CALLS: list[str] = []


def _record_call(value: str) -> str:
    _VALIDATOR_CALLS.append(value)
    return value + "!"


Recorded = Annotated[str, AfterValidator(_record_call), Field(max_length=16)]


class OnceBase(SQLModelBase):
    rhs: Recorded = Field(index=True)


class OnceInherited(OnceBase, TableBaseMixin, table=True):
    pass


class OnceDirect(SQLModelBase, TableBaseMixin, table=True):
    rhs: Recorded = Field(index=True)


class OncePure(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    rhs: Recorded = Field(index=True)


@pytest.mark.parametrize("model", [OncePure, OnceBase, OnceDirect, OnceInherited], ids=lambda m: m.__name__)
def test_annotation_validator_runs_once(model: type[SQLModel]) -> None:
    """An ``AfterValidator`` next to the alias's ``Field`` must not be applied twice by the recovered field."""
    _VALIDATOR_CALLS.clear()
    row = model.model_validate({"rhs": "v"})
    assert row.rhs == "v!"  # pyright: ignore[reportAttributeAccessIssue]
    assert _VALIDATOR_CALLS == ["v"]


# ---------------------------------------------------------------- alias default under a union wrapper

class UnionDefaultBase(SQLModelBase):
    wrapped: OptionalNonNegativeDecimal38_18 | None = Field(index=True)
    bare: OptionalNonNegativeDecimal38_18 = Field(index=True)


class UnionDefaultInherited(UnionDefaultBase, TableBaseMixin, table=True):
    pass


class UnionDefaultDirect(SQLModelBase, TableBaseMixin, table=True):
    wrapped: OptionalNonNegativeDecimal38_18 | None = Field(index=True)
    bare: OptionalNonNegativeDecimal38_18 = Field(index=True)


@pytest.mark.parametrize("model", [UnionDefaultDirect, UnionDefaultInherited], ids=lambda m: m.__name__)
def test_alias_default_follows_pydantic_under_union(model: type[SQLModel]) -> None:
    """
    Pydantic does not merge a ``Field`` nested in a union member, so the alias's
    ``default=None`` does not apply to ``Alias | None = Field(...)`` -- the
    field is required, as in the non-table base. Without the union the
    alias's default applies.
    """
    for name in ("wrapped", "bare"):
        expected = UnionDefaultBase.model_fields[name]
        field = model.model_fields[name]
        assert field.is_required() == expected.is_required(), name
        assert field.default == expected.default, name
        assert _table_column(model, name).index is True
    assert model.model_fields["wrapped"].is_required() is True
    assert model.model_fields["bare"].default is None
