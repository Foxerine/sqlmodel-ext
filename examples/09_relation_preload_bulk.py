"""
09 -- Relationship preloading: ``load=``, ``@requires_relations``, ``ensure_relations_loaded_bulk``.

Run::

    python examples/09_relation_preload_bulk.py

In async SQLAlchemy a lazy relationship access would issue synchronous I/O
(``MissingGreenlet``). sqlmodel-ext therefore defaults every ``Relationship``
to ``lazy='raise_on_sql'``: reading an unloaded relationship raises
``InvalidRequestError`` immediately and deterministically. The rule is:
**every relationship you read must be loaded explicitly**. Three tools, from simplest to most scalable:

1. ``get(..., load=rel(Model.relation))`` (``selectinload``; nested chains via
   a list: ``load=[rel(A.b), rel(B.c)]``). ``rel()`` narrows the attribute
   type for the checker.
2. ``@requires_relations('relation')`` on a model method: the relationships the
   method reads are declared on the method and loaded on entry (incrementally;
   already-loaded ones cost nothing). Requires ``RelationPreloadMixin``.
3. ``ensure_relations_loaded_bulk(session, instances, {Class: ('relation',)})``:
   one ``IN (...)`` query per target table for a whole list, instead of one
   query per instance (N+1). It is a query-count optimization; the per-instance
   path stays the source of correctness.
"""
import asyncio
from uuid import UUID

from sqlalchemy import event
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, Relationship, SQLModel, col

from sqlmodel_ext import (
    AsyncSession,
    RelationPreloadMixin,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
    rel,
    requires_relations,
)


class Team(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64
    """Team name."""

    members: list['Member'] = Relationship(back_populates='team')


class Member(RelationPreloadMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64
    """Member name."""

    team_id: UUID = Field(foreign_key='team.id', index=True)
    """Owning team."""

    team: Team = Relationship(back_populates='members')

    @requires_relations('team')
    async def badge(self, session: AsyncSession) -> str:
        """Reads ``self.team`` -- declared above, loaded on entry from ``session``."""
        size = await Member.count(session, col(Member.team_id) == self.team_id)
        return f"{self.name} @ {self.team.name} ({size} members)"


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    selects: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _count_selects(_conn: object, _cursor: object, statement: str, *_args: object) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    async with AsyncSession(engine) as session:
        # Read what you need from each returned instance right away: every
        # later commit expires it, and touching an expired attribute is I/O.
        team_ids = [(await Team(name=f"team-{i}").save(session)).id for i in range(3)]
        for i in range(9):
            _ = await Member(name=f"m{i}", team_id=team_ids[i % 3]).save(session, refresh=False)

        # --- the failure mode ------------------------------------------------
        member = await Member.get(session, col(Member.name) == "m0", fetch_mode='one')
        try:
            _ = member.team.name
        except InvalidRequestError as exc:
            assert "raise_on_sql" in str(exc)
        else:
            raise AssertionError("an unloaded relationship must not be readable in async code")

        # --- 1. load= at query time ------------------------------------------
        member = await Member.get(session, col(Member.name) == "m0", fetch_mode='one', load=rel(Member.team))
        assert member.team.name == "team-0"

        # --- 2. method-declared relationships --------------------------------
        member = await Member.get(session, col(Member.name) == "m1", fetch_mode='one')
        assert await member.badge(session) == "m1 @ team-1 (3 members)"

        # --- 3. bulk preload for a list: 1 query instead of 9 ----------------
        members = await Member.get(session, fetch_mode='all', order_by=[col(Member.name).asc()])
        selects.clear()
        await Member.ensure_relations_loaded_bulk(session, members, {Member: ('team',)})
        assert len(selects) == 1, f"one IN(...) query for all teams, got {len(selects)}"
        team_names = [m.team.name for m in members]  # already loaded: zero extra SQL
        assert len(selects) == 1
        assert team_names[0] == "team-0" and team_names[8] == "team-2"

    await engine.dispose()
    print("[OK] 09_relation_preload_bulk: unloaded access fails loudly; no N+1")


if __name__ == "__main__":
    asyncio.run(main())
