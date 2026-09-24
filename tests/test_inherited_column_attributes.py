"""
Inherited fields keep the column attributes of their base-class ``Field(...)``.

A table class inheriting ``language: Str64 = Field(index=True)`` from a
non-table base used to get a column without the index: the metaclass rebuilt
the inherited ``Annotated`` field from the type alias's ``Field`` only, and the
base's right-hand ``Field(...)`` (index / unique / primary_key / foreign_key /
nullable / sa_type / sa_column_kwargs / ondelete ...) was dropped.

Criteria:

1. an inherited field produces the same column and Pydantic field as the same
   declaration made directly in the table class;
2. both match the identical declaration on plain ``SQLModel`` -- except where
   plain SQLModel itself drops an attribute the library recovers
   (``_PLAIN_SQLMODEL_DROPS``); there the library's value is asserted
   explicitly.
"""
from __future__ import annotations

from typing import Annotated, Any

import pytest
from sqlalchemy import BigInteger, Column
from sqlmodel import Field, SQLModel

from sqlmodel_ext import (
    AutoPolymorphicIdentityMixin,
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
