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
2. The overloads here are a strict superset of upstream's: one entity
   returns ``SelectOfScalar``, two or more return ``Select``; 5 to 9 entities
   are added.
3. The covered argument shape is ``_TCCA`` (model classes, ``col(...)``,
   column expressions). Upstream's ``_TScalar_*`` variants (raw ``Column``
   instances, bare scalar types) are not covered; use ``sqlmodel.select`` for
   those.

Overloads stop at 9 columns on purpose: beyond that the type checker reports
an error instead of silently degrading to ``Any``.

Exit condition: once SQLModel's generated overloads reach 9 entities, this
module can be removed in favor of ``from sqlmodel import select``.

Usage::

    from sqlmodel_ext import select

    stmt = select(User.id, User.name, User.email, User.created_at, User.role)
"""
from typing import TYPE_CHECKING, TypeVar, overload

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

if TYPE_CHECKING:
    @overload
    def select(__ent0: _TCCA[_T0]) -> SelectOfScalar[_T0]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1],
    ) -> Select[tuple[_T0, _T1]]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2],
    ) -> Select[tuple[_T0, _T1, _T2]]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
    ) -> Select[tuple[_T0, _T1, _T2, _T3]]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3], __ent4: _TCCA[_T4],
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4]]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5],
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6], __ent7: _TCCA[_T7],
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...

    @overload
    def select(
        __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3], __ent4: _TCCA[_T4], __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6], __ent7: _TCCA[_T7], __ent8: _TCCA[_T8],
    ) -> Select[tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8]]: ...

    def select(*_entities: object) -> object: ...
else:
    # At runtime this *is* the upstream function object -- no wrapping.
    select = _upstream_select
