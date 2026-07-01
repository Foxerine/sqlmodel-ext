"""
Regression: explicit ``Array[T] = Field(default_factory=list)`` must build a
PostgreSQL ``ARRAY`` column (not raise ``<class 'list'> has no matching
SQLAlchemy type``).

Root cause (sqlmodel_ext.base metaclass sa_type injection loop, the
``elif isinstance(field_value, FieldInfo)`` branch — i.e. the user wrote an
explicit ``= Field(...)``):

c00696c fixed only the ``field_value is Undefined`` (no ``= Field(...)``)
branch. The explicit-Field branch still did a plain
``setattr(field_value, 'sa_type', sa_type)``. That attribute is LOST:
``SQLModelMetaclass.__new__`` (reached via the metaclass ``super().__new__``)
runs ``get_column_from_field`` *before* step-7's SQLModelFieldInfo restore,
and Pydantic has by then rebuilt ``model_fields`` into fresh FieldInfo
objects that never saw the post-hoc attribute. ``get_sqlalchemy_type``
reads ``sa_type`` via ``_get_sqlmodel_field_value``, which prefers a
``FieldInfoMetadata`` entry in the FieldInfo's pydantic ``metadata`` list
(the rebuild-safe channel SQLModel's own ``Field(sa_type=...)`` uses).

Fix: ``_durably_set_sa_type()`` writes ``sa_type`` into a
``FieldInfoMetadata`` on the FieldInfo's ``metadata`` list (plus the
instance attribute as a fallback), so it survives Pydantic's rebuild into
the column build. Applied to both the explicit-Field branch and the
recovered-Annotated-FieldInfo path.

This locks in the real-world trigger: an ``Array[StrEnum] =
Field(default_factory=list)`` column on a ``table=True`` model that also
inherits a non-table base declaring the same field as a plain ``list``
(the contract-base / persistence-table split). Such fields were silently
mistyped (or raised ``<class 'list'> has no matching SQLAlchemy type``),
500-ing every INSERT into that table.
"""
from __future__ import annotations

from enum import StrEnum

import pytest
import sqlalchemy as sa
from sqlmodel import Field, SQLModel

from sqlmodel_ext import SQLModelBase, TableBaseMixin, UUIDTableBaseMixin
from sqlmodel_ext.field_types.dialects.postgresql import Array
from sqlmodel_ext.field_types.dialects.postgresql.array import (
    _TolerantEnum,
    _TolerantEnumArray,
)


# Module-level enums: with ``from __future__ import annotations`` all
# annotations are lazy strings; the metaclass resolves them via
# ``get_type_hints`` against module globals, so referenced names MUST be
# module-level (function-local names are unresolvable).
class ScopeLikeEnum(StrEnum):
    a = "a:b"
    c = "c:d"


DynScopeEnum = StrEnum(
    "DynScopeEnum",
    {f"r{i}_{a}": f"r{i}:{a}" for i in range(3) for a in ("read", "write")},
)


@pytest.fixture(autouse=True)
def _drop_array_tables_from_shared_metadata():
    # Each test in this file declares ``table=True`` models with PostgreSQL
    # ``ARRAY`` columns. SQLModel auto-registers them on the shared
    # ``SQLModel.metadata``; left there, conftest's SQLite ``create_all``
    # in later tests trips on ``visit_ARRAY``. Snapshot the table names
    # before each test and drop anything new on teardown. The leftover
    # mapper class is harmless — only the Table entry blocks create_all.
    tables_before = set(SQLModel.metadata.tables)
    yield
    for name in set(SQLModel.metadata.tables) - tables_before:
        SQLModel.metadata.remove(SQLModel.metadata.tables[name])


class TestExplicitFieldArraySaType:
    def test_lib_documented_array_str_pattern(self) -> None:
        """The README/array.py documented form on a table model."""

        class DocTags(SQLModelBase, TableBaseMixin, table=True):
            tags: Array[str] = Field(default_factory=list)

        col_type = DocTags.__table__.c.tags.type
        assert isinstance(col_type, sa.ARRAY)
        assert isinstance(col_type.item_type, sa.String)

    def test_array_enum_overrides_inherited_plain_list(self) -> None:
        """
        Production shape: non-table contract base declares ``scopes`` as a
        plain ``list[Enum]``; the table subclass overrides it with
        ``Array[Enum]`` to get a real PG ``ARRAY(Enum)`` column.
        """
        class ContractBase(SQLModelBase):
            is_admin: bool = False
            scopes: list[ScopeLikeEnum] = Field(default_factory=list)

        class ScopeTable(ContractBase, TableBaseMixin, table=True):
            scopes: Array[ScopeLikeEnum] = Field(default_factory=list)
            tags: Array[str] = Field(default_factory=list)

        # Enum arrays are wrapped in read-tolerant TypeDecorators
        # (_TolerantEnumArray -> ARRAY -> _TolerantEnum -> sa.Enum) so an unknown
        # DB enum value is dropped rather than raising LookupError during a
        # rolling-deploy version-skew window.
        scopes_type = ScopeTable.__table__.c.scopes.type
        assert isinstance(scopes_type, _TolerantEnumArray)
        assert isinstance(scopes_type.impl_instance, sa.ARRAY)
        assert isinstance(scopes_type.impl_instance.item_type, _TolerantEnum)
        assert isinstance(scopes_type.impl_instance.item_type.impl_instance, sa.Enum)
        # Plain str arrays stay a native ARRAY (only enum items get wrapped)
        assert isinstance(ScopeTable.__table__.c.tags.type, sa.ARRAY)

        # default_factory survives (not silently is_required=True)
        inst = ScopeTable(scopes=[ScopeLikeEnum.a], tags=["x"])
        assert inst.scopes == [ScopeLikeEnum.a]
        assert ScopeTable().scopes == []

    def test_array_enum_on_uuid_mixin(self) -> None:
        """Dynamically-built StrEnum (mirrors ScopeValueEnum) + UUID PK mixin."""
        class ScopeRow(SQLModelBase, UUIDTableBaseMixin, table=True):
            scopes: Array[DynScopeEnum] = Field(default_factory=list)

        # Read-tolerant TypeDecorator chain (see test above)
        col_type = ScopeRow.__table__.c.scopes.type
        assert isinstance(col_type, _TolerantEnumArray)
        assert isinstance(col_type.impl_instance, sa.ARRAY)
        assert isinstance(col_type.impl_instance.item_type, _TolerantEnum)
        assert isinstance(col_type.impl_instance.item_type.impl_instance, sa.Enum)

    def test_non_array_field_unaffected(self) -> None:
        """Guard: ordinary scalar Field columns keep working unchanged."""

        class Plain(SQLModelBase, TableBaseMixin, table=True):
            name: str = Field(max_length=32, index=True)
            qty: int = 0

        assert Plain.__table__.c.name.type.__class__.__name__ in (
            "AutoString",
            "String",
            "VARCHAR",
        )
        assert isinstance(Plain.__table__.c.qty.type, sa.Integer)
