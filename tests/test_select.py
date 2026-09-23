"""
``sqlmodel_ext.select``: type-only overloads for 5-9 column projections.

The overloads themselves are verified by a type checker; at runtime the only
invariant is that ``select`` is SQLModel's own function object, so every
projection width behaves exactly as upstream.
"""
import pytest
import sqlmodel
from sqlmodel import col
from sqlmodel.sql.expression import Select, SelectOfScalar

from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, select
from sqlmodel_ext import select as exported_select
from sqlmodel_ext.select import select as module_select


class SelectWideRow(SQLModelBase, UUIDTableBaseMixin, table=True):
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0
    e: int = 0
    f: int = 0
    g: int = 0
    h: int = 0


_COLUMNS = (
    SelectWideRow.id, SelectWideRow.a, SelectWideRow.b, SelectWideRow.c, SelectWideRow.d,
    SelectWideRow.e, SelectWideRow.f, SelectWideRow.g, SelectWideRow.h,
)


def test_runtime_identity() -> None:
    assert select is sqlmodel.select
    assert exported_select is module_select is sqlmodel.select


def test_single_entity_is_scalar_select() -> None:
    assert isinstance(select(SelectWideRow), SelectOfScalar)


@pytest.mark.parametrize('width', range(2, 10))
def test_multi_column_projection(width: int) -> None:
    stmt = select(*_COLUMNS[:width])
    assert isinstance(stmt, Select)
    assert len(stmt.selected_columns) == width


def test_col_expressions_accepted() -> None:
    stmt = select(
        col(SelectWideRow.a), col(SelectWideRow.b), col(SelectWideRow.c),
        col(SelectWideRow.d), col(SelectWideRow.e),
    )
    assert len(stmt.selected_columns) == 5
