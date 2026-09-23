"""
06 -- Deleting a row that is still referenced: ``ResourceReferencedError`` + message registry.

Run::

    python examples/06_referenced_delete.py

On **PostgreSQL**, ``delete()`` recognizes a foreign-key violation caused by
its own ``DELETE`` statement (SQLSTATE 23503) and raises
``ResourceReferencedError`` (``status_code = 409``) carrying a user-facing
message. The message is registered once, next to the parent model, with
``TableBaseMixin.register_fk_delete_restrict_message(constraint_name, message)``;
unregistered constraints fall back to ``FK_DELETE_RESTRICT_FALLBACK_MESSAGE``,
which never leaks table or column names.

**SQLite cannot demonstrate the translation**: its driver reports no SQLSTATE,
so ``delete()`` re-raises the raw ``IntegrityError`` untouched (the library
only translates what it can attribute with certainty). This script therefore
asserts the SQLite behavior, exercises the registry API, and shows the endpoint
mapping you would write for PostgreSQL in ``delete_author_or_409``.
"""
import asyncio
from typing import Any
from uuid import UUID

from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, SQLModel, col

from sqlmodel_ext import (
    FK_DELETE_RESTRICT_FALLBACK_MESSAGE,
    AsyncSession,
    ResourceReferencedError,
    SQLModelBase,
    Str64,
    TableBaseMixin,
    UUIDTableBaseMixin,
)

BOOK_AUTHOR_FK = 'book_author_id_fkey'
"""PostgreSQL's default name for the FK below (``<table>_<column>_fkey``).
Verify the real name in the running database -- a typo silently falls back."""


class Author(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64
    """Author name."""


TableBaseMixin.register_fk_delete_restrict_message(
    BOOK_AUTHOR_FK, "This author still has books; delete or reassign the books first.",
)


class Book(SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64
    """Book title."""

    author_id: UUID = Field(foreign_key='author.id', ondelete='RESTRICT', index=True)
    """Owning author; deleting an author with books is refused."""


async def delete_author_or_409(session: AsyncSession, author: Author) -> tuple[int, str]:
    """What an endpoint does on PostgreSQL: map the domain error to an HTTP answer."""
    try:
        _ = await Author.delete(session, author)
    except ResourceReferencedError as exc:
        return exc.status_code, exc.friendly_message
    return 204, ""


async def main() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fks(dbapi_connection: Any, _record: object) -> None:
        # SQLite ships with FK enforcement off; turn it on for every connection.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # --- registry: one declaration next to the parent model ------------------
    assert TableBaseMixin.lookup_fk_delete_restrict_message(BOOK_AUTHOR_FK) is not None
    assert TableBaseMixin.lookup_fk_delete_restrict_message('never_registered') is None
    assert "author" not in FK_DELETE_RESTRICT_FALLBACK_MESSAGE.lower(), "fallback leaks no schema names"
    assert ResourceReferencedError.status_code == 409

    async with AsyncSession(engine) as session:
        author = await Author(name="Ursula").save(session)
        author_id = author.id
        _ = await Book(title="The Dispossessed", author_id=author_id).save(session)

        # --- SQLite: the FK is enforced, but not translated ------------------
        author = await Author.get(session, col(Author.id) == author_id, fetch_mode='one')
        try:
            _ = await delete_author_or_409(session, author)
        except IntegrityError:
            await session.rollback()  # SQLite: no SQLSTATE -> raw error, as documented
        else:
            raise AssertionError("the FK must block the delete")
        assert await Author.count(session) == 1, "the author still exists"

        # --- remove the reference first, then the delete succeeds ------------
        assert await Book.delete(session, condition=col(Book.author_id) == author_id) == 1
        author = await Author.get(session, col(Author.id) == author_id, fetch_mode='one')
        assert await delete_author_or_409(session, author) == (204, "")
        assert await Author.count(session) == 0

    await engine.dispose()
    print("[OK] 06_referenced_delete: FK enforced; registry and 409 mapping in place")


if __name__ == "__main__":
    asyncio.run(main())
