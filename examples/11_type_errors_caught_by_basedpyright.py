"""
11 -- Misuse that basedpyright reports before the code ever runs.

**This file is intentionally wrong. Do not run it; type-check it**::

    basedpyright examples/11_type_errors_caught_by_basedpyright.py

Each numbered line below is a real mistake. Expected diagnostics (``error`` unless
noted; the exact output is reproduced in ``examples/README.md``):

1. ``shout(patch.name)`` -- ``Str64 | MISSING`` is not assignable to ``str``:
   an omissible field was used without checking ``is not Unset``.
2. ``if patch.nickname is not None: shout(patch.nickname)`` -- still
   ``str | MISSING``: ``is not None`` does not rule out "not sent".
3. ``shout(maybe.name)`` -- ``get()`` defaults to ``fetch_mode='first'`` and
   returns ``Member | None``.
4. ``member: Member = await Member.get(...)`` -- same, on assignment.
5. ``await Member.delete(session)`` -- no overload matches: ``delete()``
   needs either instances or ``condition=``.
6. ``load=Member.team`` -- a relationship attribute is typed as its target
   model; pass ``rel(Member.team)``.
7. ``Member.name.in_(...)`` without ``col()`` -- the checker types
   ``Member.name`` as ``Str64``, which has no ``in_``; write ``col(Member.name).in_(...)``.
8. ``select(...)`` with 10 columns -- the overloads stop at 9 on purpose.
9. (warning, ``reportUnusedCallResult``) ``await member.save(session)`` without
   using the return value -- the instance you still hold is expired.

The fix for each is shown in examples 01, 09 and 10. Note that ``partial=True``
DTOs are *not* covered by (1)/(2): their tri-state annotations are created at
runtime, so the checker sees the base annotations. Declare ``Unset | T = Unset``
(as ``MemberPatch`` does) or run the experimental ``sqlmodel_ext.check_derived``.
"""
from uuid import UUID

from sqlmodel import Field, Relationship, col

from sqlmodel_ext import AsyncSession, SQLModelBase, Str64, UUIDTableBaseMixin, Unset, select


class Team(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64
    members: list['Member'] = Relationship(back_populates='team')


class Member(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64
    nickname: str | None = None
    team_id: UUID = Field(foreign_key='team.id', index=True)
    team: Team = Relationship(back_populates='members')


class MemberPatch(SQLModelBase):
    name: Unset | Str64 = Unset
    nickname: Unset | str | None = Unset


def shout(text: str) -> str:
    return text.upper()


async def misuse(session: AsyncSession, patch: MemberPatch, member_id: UUID) -> None:
    _ = shout(patch.name)                                                   # 1
    if patch.nickname is not None:
        _ = shout(patch.nickname)                                           # 2
    maybe = await Member.get(session, col(Member.id) == member_id)
    _ = shout(maybe.name)                                                   # 3
    member: Member = await Member.get(session, col(Member.id) == member_id)  # 4
    _ = await Member.delete(session)                                        # 5
    _ = await Member.get(session, fetch_mode='all', load=Member.team)       # 6
    _ = await Member.get(session, Member.name.in_(["a", "b"]))              # 7
    _ = select(                                                             # 8
        col(Member.id), col(Member.name), col(Member.nickname), col(Member.team_id),
        col(Member.created_at), col(Member.updated_at), col(Member.id), col(Member.name),
        col(Member.nickname), col(Member.team_id),
    )
    await member.save(session)                                              # 9


if __name__ == "__main__":
    raise SystemExit("This file demonstrates type errors; run basedpyright on it instead.")
