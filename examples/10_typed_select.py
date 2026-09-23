"""
10 -- Typed queries: ``select()`` up to 9 columns, ``col()``, ``cond()``, parameterized ``get()``.

Run::

    python examples/10_typed_select.py

* ``sqlmodel_ext.select`` **is** ``sqlmodel.select`` at runtime; only the type
  overloads are extended from 4 to 9 columns, so a 5-column projection is typed
  ``Select[tuple[...]]`` instead of "no matching overload". For 1 to 4 columns
  the overloads are exactly upstream's (``select(User.id, User.name)`` is
  ``Select[tuple[UUID, str]]``); bare attributes also work for 5 to 9 columns.
  Only when a 5+ column projection mixes in a SQL function expression (such as
  ``func.count()``) should every column go through ``col()`` to keep the precise
  type. Beyond 9 columns the checker reports an error rather than degrading to
  ``Any``.
* ``Model.field == value`` is inferred as ``bool`` by type checkers. Wrap the
  column with ``col()`` to get a real column expression (``.in_()``,
  ``.is_(None)``, ``.asc()``), or wrap a whole comparison with ``cond()`` to
  combine conditions with ``&`` / ``|``.
* Queries go through the parameterized ``get(session, condition, ...)``. Do
  not write ``find_by_xxx()`` wrappers around it -- the condition *is* the
  parameter.
"""
import asyncio
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, col
from sqlmodel.sql.expression import Select

from sqlmodel_ext import (
    AsyncSession,
    NonNegativeInt,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
    cond,
    select,
)


class Employee(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64
    """Full name."""

    team: Str64
    """Team name."""

    level: NonNegativeInt
    """Seniority level."""

    manager_id: UUID | None = None
    """Manager (null for the top of the hierarchy)."""


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        boss = await Employee(name="Ada", team="core", level=5).save(session)
        boss_id = boss.id
        for name, team, level in (("Bo", "core", 2), ("Cy", "web", 3), ("Di", "web", 1)):
            _ = await Employee(name=name, team=team, level=level, manager_id=boss_id).save(
                session, refresh=False,
            )

        # --- 6-column projection, fully typed ----------------------------------
        stmt: Select[tuple[UUID, str, str, int, UUID | None, datetime]] = select(
            col(Employee.id), col(Employee.name), col(Employee.team), col(Employee.level),
            col(Employee.manager_id), col(Employee.created_at),
        ).order_by(col(Employee.name).asc())
        rows = (await session.exec(stmt)).all()
        assert [row[1] for row in rows] == ["Ada", "Bo", "Cy", "Di"]

        # --- col(): column methods the checker accepts --------------------------
        web_or_core = await Employee.get(
            session, col(Employee.team).in_(["web", "core"]), fetch_mode='all',
        )
        assert len(web_or_core) == 4
        top = await Employee.get(session, col(Employee.manager_id).is_(None), fetch_mode='one')
        assert top.name == "Ada"

        # --- cond(): combine comparisons with & / | -----------------------------
        senior_web = cond(Employee.team == "web") & cond(Employee.level >= 2)
        found = await Employee.get(session, senior_web, fetch_mode='all')
        assert [e.name for e in found] == ["Cy"]

        # --- fetch_mode decides the return type ---------------------------------
        maybe = await Employee.get(session, col(Employee.name) == "Nobody")  # -> Employee | None
        assert maybe is None
        count = await Employee.count(session, cond(Employee.manager_id == boss_id))
        assert count == 3

    await engine.dispose()
    print("[OK] 10_typed_select: typed projections and conditions")


if __name__ == "__main__":
    asyncio.run(main())
