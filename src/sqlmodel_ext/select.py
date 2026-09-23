"""
Type-level extension of ``sqlmodel.select``: overloads for 5 to 9 column projections.

At runtime ``select`` **is** ``sqlmodel.select`` (the same function object):
no wrapper, no overhead, no behavioral difference. Only the type annotations
are extended.

**Upstream limitation**: SQLModel generates its ``select`` overloads from a
template that stops at **4 entities**, while the implementation accepts any
number::

    def select(*entities: Any) -> Union[Select, SelectOfScalar]:
        if len(entities) == 1:
            return SelectOfScalar(*entities)
        return Select(*entities)

So a perfectly valid 5+ column projection is reported by type checkers as a
call with no matching overload. This module fixes the annotations only.

**Invariants**:

1. ``sqlmodel_ext.select.select is sqlmodel.select`` at runtime.
2. For 1 to 4 entities the overload set is **exactly** upstream's: every
   position accepts either a column-clause argument (``_TCCA``: model classes,
   ``col(...)``, SQL expressions) or a scalar-typed attribute (``_TScalar_i``:
   what a type checker sees for a bare ``Model.field``), in all 2**n
   combinations. Anything ``sqlmodel.select`` accepts here, this accepts with
   the same inferred type. ``tests/test_select.py`` pins this parity.
3. For 5 to 9 entities there are two overloads per arity: one where every
   argument is a column clause (precise), then a per-position fallback that
   also accepts bare attributes. Mixing a bare attribute with a SQL function
   expression (e.g. ``func.count()``) in a 5+ projection resolves through the
   fallback and widens that expression's element to a union; wrap bare
   attributes in ``col(...)`` to stay on the precise overload. (Generating all
   2**n shape combinations for n up to 9 would mean ~1000 overloads.)

Overloads stop at 9 columns on purpose: beyond that the type checker reports
an error instead of silently degrading to ``Any``.

Exit condition: once SQLModel's generated overloads reach 9 entities, this
module can be removed in favor of ``from sqlmodel import select``.

Usage::

    from sqlmodel import col
    from sqlmodel_ext import select

    stmt = select(User.id, User.name)                    # Select[tuple[UUID, str]]
    stmt = select(
        col(User.id), col(User.name), col(User.email),
        col(User.created_at), col(User.role),
    )                                                    # 5 columns, precise
"""
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar, overload
from uuid import UUID

from sqlalchemy import Column
from sqlalchemy.sql.elements import SQLCoreOperations
from sqlalchemy.sql.roles import TypedColumnsClauseRole
from sqlmodel import select as _upstream_select
from sqlmodel.sql.expression import Select, SelectOfScalar

_T0 = TypeVar('_T0')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')
_T4 = TypeVar('_T4')
_T5 = TypeVar('_T5')
_T6 = TypeVar('_T6')
_T7 = TypeVar('_T7')
_T8 = TypeVar('_T8')

_T = TypeVar('_T')
_TCCA = TypedColumnsClauseRole[_T] | SQLCoreOperations[_T] | type[_T]
"""
Column-clause argument shapes accepted by ``select``.

Same members as upstream's private ``sqlmodel.sql._expression_select_gen._TCCA``.
Copied rather than imported because the upstream symbol is private; compare
the member set when upgrading SQLModel.
"""

# Scalar-typed argument shapes, one TypeVar per position as upstream does
# (a constrained TypeVar cannot be reused across positions without forcing
# them to the same type). Same constraint set as upstream's ``_TScalar_i``;
# the explicit ``[Any]`` arguments spell out what upstream's bare generics mean.
_TScalar_0 = TypeVar('_TScalar_0', Column[Any], Sequence[Any], Mapping[Any, Any], UUID, datetime, float, int, bool, bytes, str, None)
_TScalar_1 = TypeVar('_TScalar_1', Column[Any], Sequence[Any], Mapping[Any, Any], UUID, datetime, float, int, bool, bytes, str, None)
_TScalar_2 = TypeVar('_TScalar_2', Column[Any], Sequence[Any], Mapping[Any, Any], UUID, datetime, float, int, bool, bytes, str, None)
_TScalar_3 = TypeVar('_TScalar_3', Column[Any], Sequence[Any], Mapping[Any, Any], UUID, datetime, float, int, bool, bytes, str, None)

if TYPE_CHECKING:
    @overload
    def select(
        ent0: _TCCA[_T0],
        /,
    ) -> SelectOfScalar[_T0]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        /,
    ) -> SelectOfScalar[_TScalar_0]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        /,
    ) -> Select[tuple[_T0, _T1]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        entity_1: _TScalar_1,
        /,
    ) -> Select[tuple[_T0, _TScalar_1]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        ent1: _TCCA[_T1],
        /,
    ) -> Select[tuple[_TScalar_0, _T1]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        entity_1: _TScalar_1,
        /,
    ) -> Select[tuple[_TScalar_0, _TScalar_1]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        /,
    ) -> Select[tuple[_T0, _T1, _T2]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        entity_2: _TScalar_2,
        /,
    ) -> Select[tuple[_T0, _T1, _TScalar_2]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        entity_1: _TScalar_1,
        ent2: _TCCA[_T2],
        /,
    ) -> Select[tuple[_T0, _TScalar_1, _T2]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        entity_1: _TScalar_1,
        entity_2: _TScalar_2,
        /,
    ) -> Select[tuple[_T0, _TScalar_1, _TScalar_2]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        /,
    ) -> Select[tuple[_TScalar_0, _T1, _T2]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        ent1: _TCCA[_T1],
        entity_2: _TScalar_2,
        /,
    ) -> Select[tuple[_TScalar_0, _T1, _TScalar_2]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        entity_1: _TScalar_1,
        ent2: _TCCA[_T2],
        /,
    ) -> Select[tuple[_TScalar_0, _TScalar_1, _T2]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        entity_1: _TScalar_1,
        entity_2: _TScalar_2,
        /,
    ) -> Select[tuple[_TScalar_0, _TScalar_1, _TScalar_2]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _TScalar_3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        entity_2: _TScalar_2,
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_T0, _T1, _TScalar_2, _T3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        entity_2: _TScalar_2,
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_T0, _T1, _TScalar_2, _TScalar_3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        entity_1: _TScalar_1,
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_T0, _TScalar_1, _T2, _T3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        entity_1: _TScalar_1,
        ent2: _TCCA[_T2],
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_T0, _TScalar_1, _T2, _TScalar_3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        entity_1: _TScalar_1,
        entity_2: _TScalar_2,
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_T0, _TScalar_1, _TScalar_2, _T3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        entity_1: _TScalar_1,
        entity_2: _TScalar_2,
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_T0, _TScalar_1, _TScalar_2, _TScalar_3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_TScalar_0, _T1, _T2, _T3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_TScalar_0, _T1, _T2, _TScalar_3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        ent1: _TCCA[_T1],
        entity_2: _TScalar_2,
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_TScalar_0, _T1, _TScalar_2, _T3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        ent1: _TCCA[_T1],
        entity_2: _TScalar_2,
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_TScalar_0, _T1, _TScalar_2, _TScalar_3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        entity_1: _TScalar_1,
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_TScalar_0, _TScalar_1, _T2, _T3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        entity_1: _TScalar_1,
        ent2: _TCCA[_T2],
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_TScalar_0, _TScalar_1, _T2, _TScalar_3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        entity_1: _TScalar_1,
        entity_2: _TScalar_2,
        ent3: _TCCA[_T3],
        /,
    ) -> Select[tuple[_TScalar_0, _TScalar_1, _TScalar_2, _T3]]: ...

    @overload
    def select(
        entity_0: _TScalar_0,
        entity_1: _TScalar_1,
        entity_2: _TScalar_2,
        entity_3: _TScalar_3,
        /,
    ) -> Select[tuple[_TScalar_0, _TScalar_1, _TScalar_2, _TScalar_3]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        ent4: _TCCA[_T4],
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0] | _T0,
        ent1: _TCCA[_T1] | _T1,
        ent2: _TCCA[_T2] | _T2,
        ent3: _TCCA[_T3] | _T3,
        ent4: _TCCA[_T4] | _T4,
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        ent4: _TCCA[_T4],
        ent5: _TCCA[_T5],
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0] | _T0,
        ent1: _TCCA[_T1] | _T1,
        ent2: _TCCA[_T2] | _T2,
        ent3: _TCCA[_T3] | _T3,
        ent4: _TCCA[_T4] | _T4,
        ent5: _TCCA[_T5] | _T5,
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        ent4: _TCCA[_T4],
        ent5: _TCCA[_T5],
        ent6: _TCCA[_T6],
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0] | _T0,
        ent1: _TCCA[_T1] | _T1,
        ent2: _TCCA[_T2] | _T2,
        ent3: _TCCA[_T3] | _T3,
        ent4: _TCCA[_T4] | _T4,
        ent5: _TCCA[_T5] | _T5,
        ent6: _TCCA[_T6] | _T6,
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        ent4: _TCCA[_T4],
        ent5: _TCCA[_T5],
        ent6: _TCCA[_T6],
        ent7: _TCCA[_T7],
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0] | _T0,
        ent1: _TCCA[_T1] | _T1,
        ent2: _TCCA[_T2] | _T2,
        ent3: _TCCA[_T3] | _T3,
        ent4: _TCCA[_T4] | _T4,
        ent5: _TCCA[_T5] | _T5,
        ent6: _TCCA[_T6] | _T6,
        ent7: _TCCA[_T7] | _T7,
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0],
        ent1: _TCCA[_T1],
        ent2: _TCCA[_T2],
        ent3: _TCCA[_T3],
        ent4: _TCCA[_T4],
        ent5: _TCCA[_T5],
        ent6: _TCCA[_T6],
        ent7: _TCCA[_T7],
        ent8: _TCCA[_T8],
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8]]: ...

    @overload
    def select(
        ent0: _TCCA[_T0] | _T0,
        ent1: _TCCA[_T1] | _T1,
        ent2: _TCCA[_T2] | _T2,
        ent3: _TCCA[_T3] | _T3,
        ent4: _TCCA[_T4] | _T4,
        ent5: _TCCA[_T5] | _T5,
        ent6: _TCCA[_T6] | _T6,
        ent7: _TCCA[_T7] | _T7,
        ent8: _TCCA[_T8] | _T8,
        /,
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8]]: ...

    def select(*_entities: object) -> object: ...
else:
    # At runtime this *is* the upstream function object -- no wrapping.
    select = _upstream_select
