"""
``sqlmodel_ext.select``: type-only overloads for 5-9 column projections.

At runtime the only invariant is that ``select`` is SQLModel's own function
object, so every projection width behaves exactly as upstream. The overloads
themselves are verified at the end of this file by running basedpyright on a
probe module and comparing the inferred types against ``sqlmodel.select``.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Static parity (the module's invariants 2 and 3 are claims about the type
# checker, so only a type checker can verify them).
# ---------------------------------------------------------------------------

_PROBE_HEADER = '''\
import sqlmodel
from sqlmodel import col, func
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, select


class ParityRow(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64
    n: int
'''

# Argument lists for 1-4 entities covering every shape upstream accepts:
# model class, bare attribute (scalar-typed to the checker), col(...), and
# SQL function expressions, alone and mixed.
_PARITY_ARGS = (
    'ParityRow',
    'ParityRow.name',
    'col(ParityRow.name)',
    'func.count()',
    'ParityRow.id, ParityRow.name',
    'ParityRow, func.count()',
    'ParityRow.name, func.count(col(ParityRow.id))',
    'col(ParityRow.id), ParityRow.n',
    'ParityRow.id, ParityRow.name, ParityRow.n',
    'ParityRow, ParityRow.name, func.max(col(ParityRow.n))',
    'ParityRow.id, col(ParityRow.name), ParityRow.n, ParityRow.created_at',
    'ParityRow.name, func.count(), ParityRow.n, func.max(col(ParityRow.n))',
)

# 5-9 entities: bare attributes must be accepted (the fallback overload), and
# all-col() projections must infer every element exactly.
_WIDE_BARE = 'ParityRow.id, ParityRow.name, ParityRow.n, ParityRow.created_at, ParityRow.updated_at'
_WIDE_COL = 'col(ParityRow.id), col(ParityRow.name), col(ParityRow.n), col(ParityRow.created_at), func.count()'


def _run_basedpyright(tmp_path: Path, body: str) -> dict[str, Any]:
    executable = shutil.which('basedpyright', path=str(Path(sys.executable).parent)) or shutil.which('basedpyright')
    if executable is None:
        pytest.skip("basedpyright is not installed (it ships with the [dev] extra)")
    probe = tmp_path / 'probe_select.py'
    probe.write_text(_PROBE_HEADER + body, encoding='utf-8')
    completed = subprocess.run(
        [executable, '--outputjson', '--pythonpath', sys.executable, str(probe)],
        capture_output=True, text=True, cwd=tmp_path, check=False,
    )
    report: dict[str, Any] = json.loads(completed.stdout)
    # Guard against an empty scan, which reports zero errors too.
    assert report['summary']['filesAnalyzed'] == 1, completed.stdout
    return report


def _revealed_types(report: dict[str, Any]) -> list[str]:
    pattern = re.compile(r'^Type of ".*" is "(?P<type>.*)"$')
    types: list[str] = []
    for diagnostic in report['generalDiagnostics']:
        match = pattern.match(diagnostic['message'])
        if diagnostic['severity'] == 'information' and match is not None:
            types.append(match.group('type'))
    return types


def test_static_parity_with_upstream_for_up_to_four_entities(tmp_path: Path) -> None:
    body = ''.join(
        f'reveal_type(select({args}))\nreveal_type(sqlmodel.select({args}))\n'
        for args in _PARITY_ARGS
    )
    report = _run_basedpyright(tmp_path, body)
    errors = [d['message'] for d in report['generalDiagnostics'] if d['severity'] == 'error']
    assert errors == []
    types = _revealed_types(report)
    assert len(types) == 2 * len(_PARITY_ARGS)
    ours, upstream = types[0::2], types[1::2]
    assert ours == upstream


def test_static_wide_projections(tmp_path: Path) -> None:
    body = f'reveal_type(select({_WIDE_BARE}))\nreveal_type(select({_WIDE_COL}))\n'
    report = _run_basedpyright(tmp_path, body)
    errors = [d['message'] for d in report['generalDiagnostics'] if d['severity'] == 'error']
    assert errors == []
    assert _revealed_types(report) == [
        'Select[tuple[UUID, str, int, datetime, datetime]]',
        'Select[tuple[UUID, str, int, datetime, int]]',
    ]
