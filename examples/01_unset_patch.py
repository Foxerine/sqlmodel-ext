"""
01 -- Tri-state PATCH with ``partial=True`` and ``Unset``.

Run::

    python examples/01_unset_patch.py

**The problem.** The classic PATCH DTO is written as ``field: T | None = None``.
That one spelling has to carry two opposite intents:

* "the client did not send this field -- leave the column alone", and
* "the client sent ``null`` -- clear the column".

The code cannot tell them apart, so every handler ends up with ad-hoc rules
(``exclude_unset=True`` here, ``if x is not None`` there) and a second copy of
every field declaration, whose constraints slowly drift from the table model.

**The sqlmodel-ext answer.** Every fact is declared once and everything else is
derived from it:

1. Constraints live in the type (``Str64``, ``Text1K``, ``Annotated[..., Field(ge=0)]``)
   on the shared ``ArticleBase``. The table model and every DTO inherit them.
2. The PATCH DTO is *derived* with ``partial=True`` -- zero re-declared fields.
   Each inherited field becomes ``Unset | T = Unset``: omissible, but its
   nullability is carried over unchanged (``T`` stays non-nullable,
   ``T | None`` keeps ``null`` as a real value).
3. "Not sent", "sent null" and "sent a value" are three different runtime
   states: ``Unset``, ``None`` and the value. Check with ``is Unset`` /
   ``is not Unset`` -- never with ``is None``.
4. ``Unset`` fields never appear in ``model_dump()``; there is no
   ``exclude_unset`` to remember, so ``update()`` only touches submitted columns.

Type-checker note: ``partial=True`` rewrites the annotations at class-creation
time, so basedpyright still sees the *base* annotations on ``ArticleUpdate``
(``title: str``). The ``is Unset`` checks below are therefore a runtime
discipline, not something basedpyright alone forces. Two ways to get static
enforcement: spell ``Unset | T = Unset`` explicitly in the DTO (checked
statically -- see ``11_type_errors_caught_by_basedpyright.py``), or run the
experimental ``python -m sqlmodel_ext.check_derived`` (see
``docs/how-to/check-partial-dtos.md``).

This script asserts every claim above and prints ``[OK]`` at the end.
"""
import asyncio
from typing import Annotated, ClassVar

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, SQLModel

from sqlmodel_ext import (
    AsyncSession,
    OMITTED_SENTINEL,
    SQLModelBase,
    SQLModelExtConfig,
    Str64,
    Text1K,
    UUIDTableBaseMixin,
    Unset,
)


# ---------------------------------------------------------------------------
# 1. The single declaration: every constraint is written exactly once.
# ---------------------------------------------------------------------------

class ArticleBase(SQLModelBase):
    title: Str64
    """Title. Required, at most 64 characters, never null."""

    summary: Text1K | None
    """Optional summary. ``null`` is a real value: "this article has no summary"."""

    view_limit: Annotated[int, Field(ge=0)] = 100
    """Maximum number of views. Has a natural default; never null."""


class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    """Table model: columns (VARCHAR(64), VARCHAR(1000), ...) come from the same types."""


class ArticleCreate(ArticleBase):
    """POST body: identical to the base, so nothing is re-declared."""


class ArticleUpdate(ArticleBase, partial=True):
    """PATCH body, derived. Effective annotations at runtime:

    * ``title: Unset | Str64 = Unset``            -- omissible, null rejected
    * ``summary: Unset | Text1K | None = Unset``  -- omissible, null clears the column
    * ``view_limit: Unset | Annotated[int, Field(ge=0)] = Unset``
    """


# ---------------------------------------------------------------------------
# 2. Domain logic that inspects individual fields: branch on ``Unset``.
# ---------------------------------------------------------------------------

def describe_patch(patch: ArticleUpdate) -> list[str]:
    """Human-readable change list -- the three states are handled separately."""
    changes: list[str] = []
    if patch.title is not Unset:
        changes.append(f"rename to {patch.title!r}")
    # WRONG: ``if patch.summary is not None`` would treat "not sent" as a change
    # and could never express "clear it".
    if patch.summary is Unset:
        pass  # not sent: leave it alone
    elif patch.summary is None:
        changes.append("clear summary")
    else:
        changes.append("replace summary")
    return changes


# ---------------------------------------------------------------------------
# 3. An opt-in wire value for callers that cannot omit keys (e.g. LLM strict mode).
# ---------------------------------------------------------------------------

class ArticleUpdateToolArgs(ArticleUpdate):
    """Same fields; the model additionally accepts ``"__omitted__"`` for any omissible field."""
    model_config: ClassVar[SQLModelExtConfig] = SQLModelExtConfig(omitted_sentinel=True)


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # --- the three states, without touching the database -----------------
    empty = ArticleUpdate.model_validate({})
    assert empty.title is Unset and empty.summary is Unset and empty.view_limit is Unset
    assert empty.model_dump() == {}, "Unset fields never reach model_dump()"

    clear = ArticleUpdate.model_validate({'summary': None})
    assert clear.summary is None
    assert clear.model_dump() == {'summary': None}, "explicit null survives -> clears the column"

    rename = ArticleUpdate.model_validate({'title': "Renamed"})
    assert rename.model_dump() == {'title': "Renamed"}
    assert describe_patch(rename) == ["rename to 'Renamed'"]
    assert describe_patch(clear) == ["clear summary"]
    assert describe_patch(empty) == []

    # --- constraints are inherited, never re-declared ---------------------
    for bad_payload in ({'title': None}, {'title': 'x' * 65}, {'view_limit': -1}):
        try:
            _ = ArticleUpdate.model_validate(bad_payload)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"{bad_payload} must be rejected")

    # --- JSON Schema (OpenAPI) is derived from the same declaration -------
    schema = ArticleUpdate.model_json_schema()
    assert 'required' not in schema, "every PATCH field is omissible"
    assert schema['properties']['title']['maxLength'] == 64
    title_types = str(schema['properties']['title'])
    assert "'null'" not in title_types, "title never accepts null"
    assert "'null'" in str(schema['properties']['summary']), "summary accepts null"

    # --- persisting: update() only writes submitted columns ---------------
    async with AsyncSession(engine) as session:
        article = Article.model_validate(
            ArticleCreate(title="Hello", summary="First draft", view_limit=10),
        )
        article = await article.save(session)  # always keep the return value

        article = await article.update(session, ArticleUpdate.model_validate({'title': "Hello, world"}))
        assert article.title == "Hello, world"
        assert article.summary == "First draft", "an omitted field is left alone"
        assert article.view_limit == 10

        article = await article.update(session, ArticleUpdate.model_validate({'summary': None}))
        assert article.summary is None, "explicit null clears the column"
        assert article.title == "Hello, world"

    # --- the wire value, opt-in per model ----------------------------------
    tool_args = ArticleUpdateToolArgs.model_validate({
        'title': OMITTED_SENTINEL,
        'summary': None,
        'view_limit': 5,
    })
    assert tool_args.title is Unset, "'__omitted__' is normalized to Unset on entry"
    assert tool_args.model_dump() == {'summary': None, 'view_limit': 5}
    tool_schema = ArticleUpdateToolArgs.model_json_schema()
    assert tool_schema['properties']['title']['default'] == OMITTED_SENTINEL
    # The REST model is unaffected: its schema has no sentinel branch.
    assert OMITTED_SENTINEL not in str(ArticleUpdate.model_json_schema())

    await engine.dispose()
    print("[OK] 01_unset_patch: omitted / null / value stay distinct end to end")


if __name__ == "__main__":
    asyncio.run(main())
