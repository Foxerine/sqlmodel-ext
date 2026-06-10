"""
Extended behavioral tests for the CRUD API in
``sqlmodel_ext.mixins.table`` (TableBaseMixin) + ``sqlmodel_ext.base``.

Covers: save (refresh / no-refresh / commit=False), add (single + batch),
get (fetch_mode one/first/all, condition, filter, order_by, offset/limit,
join, load + nested load chains, populate_existing, with_for_update
tracking), get_one, get_exist_one (404 + detail), update (partial DTO
update, extra_data, exclude), delete (instance / list / condition /
argument validation), count (condition + time_filter priority),
get_with_count (pagination + total), and the ``cond`` / ``rel`` helpers.
NOTE: no ``from __future__ import annotations`` here -- PEP 563 string
annotations break SQLAlchemy's resolution of ``list['CrudBook']``
relationship targets.
"""
from datetime import datetime

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import asc, desc
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import Field, Relationship, col
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    SESSION_FOR_UPDATE_KEY,
    SQLModelBase,
    TableBaseMixin,
    TableViewRequest,
    TimeFilterRequest,
    cond,
    rel,
)
from sqlmodel_ext import AsyncSession as ExtAsyncSession


# --------------------------------------------------------------------------
# Module-level models (unique names: Crud* prefix)
# --------------------------------------------------------------------------

class CrudAuthor(SQLModelBase, TableBaseMixin, table=True):
    """Author with int PK and a one-to-many relationship to books."""
    name: str
    age: int = 0
    books: list["CrudBook"] = Relationship(back_populates="author")


class CrudBook(SQLModelBase, TableBaseMixin, table=True):
    """Book referencing CrudAuthor."""
    title: str
    author_id: int | None = Field(default=None, foreign_key="crudauthor.id")
    author: CrudAuthor | None = Relationship(back_populates="books")
    chapters: list["CrudChapter"] = Relationship(back_populates="book")


class CrudChapter(SQLModelBase, TableBaseMixin, table=True):
    """Chapter referencing CrudBook (third level for nested load chains)."""
    title: str
    book_id: int | None = Field(default=None, foreign_key="crudbook.id")
    book: CrudBook | None = Relationship(back_populates="chapters")


class CrudAuthorPatch(SQLModelBase):
    """Partial-update DTO for CrudAuthor (non-table)."""
    name: str | None = None
    age: int | None = None


# --------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------

@pytest_asyncio.fixture
async def enhanced_session(engine: AsyncEngine):
    """sqlmodel-ext enhanced AsyncSession bound to the fresh per-test engine."""
    async with ExtAsyncSession(engine) as s:
        yield s


async def _seed_authors(session: AsyncSession) -> list[CrudAuthor]:
    """Insert three authors with deterministic created_at/updated_at/age."""
    authors = []
    for i, name in enumerate(("alice", "bob", "carol"), start=1):
        a = CrudAuthor(
            name=name,
            age=i * 10,
            created_at=datetime(2024, 1, i),
            updated_at=datetime(2024, 6, 4 - i),  # inverse order vs created_at
        )
        authors.append(await a.save(session))
    return authors


# --------------------------------------------------------------------------
# save / add
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSaveAdd:
    async def test_save_assigns_pk_and_timestamps(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="x").save(session)
        assert isinstance(a.id, int)
        assert a.created_at is not None
        assert a.updated_at is not None

    async def test_save_refresh_false_returns_same_object(self, session: AsyncSession) -> None:
        a = CrudAuthor(name="same")
        returned = await a.save(session, refresh=False)
        assert returned is a

    async def test_save_commit_false_flushes_but_is_uncommitted(
        self, session: AsyncSession
    ) -> None:
        a = await CrudAuthor(name="pending").save(session, commit=False, refresh=False)
        assert a.id is not None  # flush assigned a PK

        # Only flushed, not committed: a rollback must discard the row.
        await session.rollback()
        assert await CrudAuthor.get(session, col(CrudAuthor.name) == "pending") is None
        assert await CrudAuthor.count(session) == 0

    async def test_save_existing_persists_changes(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="before", age=1).save(session)
        a.name = "after"
        a = await a.save(session)
        fetched = await CrudAuthor.get_one(session, a.id)
        assert fetched.name == "after"

    async def test_add_single_and_batch(self, session: AsyncSession) -> None:
        single = await CrudAuthor.add(session, CrudAuthor(name="solo"))
        assert isinstance(single, CrudAuthor)
        assert single.id is not None

        batch = await CrudAuthor.add(
            session, [CrudAuthor(name="b1"), CrudAuthor(name="b2")]
        )
        assert isinstance(batch, list)
        assert len(batch) == 2
        assert all(isinstance(x.id, int) for x in batch)
        assert await CrudAuthor.count(session) == 3


# --------------------------------------------------------------------------
# get: fetch modes, condition, filter, ordering, pagination
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGet:
    async def test_first_returns_none_when_empty(self, session: AsyncSession) -> None:
        assert await CrudAuthor.get(session) is None

    async def test_first_with_condition(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        a = await CrudAuthor.get(session, col(CrudAuthor.name) == "bob")
        assert a is not None
        assert a.age == 20

    async def test_one_raises_no_result_found(self, session: AsyncSession) -> None:
        with pytest.raises(NoResultFound):
            await CrudAuthor.get(session, col(CrudAuthor.name) == "ghost", fetch_mode="one")

    async def test_one_raises_multiple_results_found(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        with pytest.raises(MultipleResultsFound):
            await CrudAuthor.get(session, col(CrudAuthor.age) >= 10, fetch_mode="one")

    async def test_all_with_combined_cond(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        condition = cond(CrudAuthor.age >= 10) & cond(CrudAuthor.age < 30)
        results = await CrudAuthor.get(session, condition, fetch_mode="all")
        assert {a.name for a in results} == {"alice", "bob"}

    async def test_filter_param_is_anded(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        results = await CrudAuthor.get(
            session,
            cond(CrudAuthor.age >= 10),
            filter=cond(CrudAuthor.name != "bob"),
            fetch_mode="all",
        )
        assert {a.name for a in results} == {"alice", "carol"}

    async def test_order_by_asc_desc(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        asc_names = [
            a.name
            for a in await CrudAuthor.get(
                session, fetch_mode="all", order_by=[asc(col(CrudAuthor.age))]
            )
        ]
        desc_names = [
            a.name
            for a in await CrudAuthor.get(
                session, fetch_mode="all", order_by=[desc(col(CrudAuthor.age))]
            )
        ]
        assert asc_names == ["alice", "bob", "carol"]
        assert desc_names == ["carol", "bob", "alice"]

    async def test_offset_and_limit(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        page = await CrudAuthor.get(
            session,
            fetch_mode="all",
            order_by=[asc(col(CrudAuthor.age))],
            offset=1,
            limit=1,
        )
        assert [a.name for a in page] == ["bob"]

    async def test_explicit_limit_beats_table_view_limit(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        tv = TableViewRequest(limit=50, desc=False)
        page = await CrudAuthor.get(session, fetch_mode="all", table_view=tv, limit=2)
        assert len(page) == 2

    async def test_time_filters_via_kwargs(self, session: AsyncSession) -> None:
        await _seed_authors(session)  # created_at = Jan 1 / 2 / 3
        results = await CrudAuthor.get(
            session,
            fetch_mode="all",
            created_after_datetime=datetime(2024, 1, 2),
            created_before_datetime=datetime(2024, 1, 3),
        )
        # created_at >= Jan 2 and < Jan 3 -> only bob
        assert [a.name for a in results] == ["bob"]

    async def test_table_view_sort_by_updated_at(self, session: AsyncSession) -> None:
        await _seed_authors(session)  # updated_at = Jun 3 / 2 / 1 (alice newest)
        tv = TableViewRequest(order="updated_at", desc=True)
        results = await CrudAuthor.get(session, fetch_mode="all", table_view=tv)
        assert [a.name for a in results] == ["alice", "bob", "carol"]

    async def test_table_view_default_sort_created_at_desc(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        results = await CrudAuthor.get(session, fetch_mode="all", table_view=TableViewRequest())
        assert [a.name for a in results] == ["carol", "bob", "alice"]

    async def test_table_view_limit_none_returns_everything(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        tv = TableViewRequest(limit=None)
        results = await CrudAuthor.get(session, fetch_mode="all", table_view=tv)
        assert len(results) == 3

    async def test_join_with_related_table(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        # Re-fetch and snapshot ids before further commits expire the objects.
        alice = await CrudAuthor.get(session, cond(CrudAuthor.name == "alice"), fetch_mode="one")
        bob = await CrudAuthor.get(session, cond(CrudAuthor.name == "bob"), fetch_mode="one")
        alice_id, bob_id = alice.id, bob.id
        await CrudBook(title="t1", author_id=alice_id).save(session)
        await CrudBook(title="t2", author_id=bob_id).save(session)

        books = await CrudBook.get(
            session, cond(CrudAuthor.name == "alice"), join=CrudAuthor, fetch_mode="all"
        )
        assert [b.title for b in books] == ["t1"]

    async def test_populate_existing_overwrites_dirty_identity(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="p", age=1).save(session)
        a_id = a.id
        a.age = 99  # dirty, in identity map, not committed

        # no_autoflush: keep the dirty value out of the DB while querying
        with session.no_autoflush:
            stale = await CrudAuthor.get(session, col(CrudAuthor.id) == a_id)
            assert stale is a
            assert stale.age == 99  # identity map wins by default

            fresh = await CrudAuthor.get(
                session, col(CrudAuthor.id) == a_id, populate_existing=True
            )
            assert fresh is a
            assert fresh.age == 1  # DB value forced back in


# --------------------------------------------------------------------------
# get: relationship preloading
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetLoad:
    async def test_load_single_relationship(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="rich").save(session)
        a_id = a.id  # snapshot before later commits expire the instance
        await CrudBook(title="b1", author_id=a_id).save(session)
        await CrudBook(title="b2", author_id=a_id).save(session)

        fetched = await CrudAuthor.get(
            session, col(CrudAuthor.id) == a_id, load=rel(CrudAuthor.books)
        )
        assert fetched is not None
        # Eagerly loaded: attribute is populated, no lazy IO needed.
        assert "books" in fetched.__dict__
        assert {b.title for b in fetched.books} == {"b1", "b2"}

    async def test_load_nested_chain(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="chain").save(session)
        a_id = a.id
        book = await CrudBook(title="cb", author_id=a_id).save(session)
        book_id = book.id
        await CrudChapter(title="ch1", book_id=book_id).save(session)

        fetched = await CrudAuthor.get(
            session,
            col(CrudAuthor.id) == a_id,
            load=[rel(CrudAuthor.books), rel(CrudBook.chapters)],
        )
        assert fetched is not None
        loaded_book = fetched.books[0]
        assert "chapters" in loaded_book.__dict__  # second hop eagerly loaded
        assert [c.title for c in loaded_book.chapters] == ["ch1"]

    async def test_load_bidirectional_pair_builds_chain(
        self, session: AsyncSession
    ) -> None:
        """A bidirectional pair (Author.books + Book.author) must not be
        treated as an unloadable cycle: _build_load_chains breaks the cycle
        at the first-listed member, yielding the chain books -> author."""
        a = await CrudAuthor(name="cycle").save(session)
        a_id = a.id
        await CrudBook(title="cy", author_id=a_id).save(session)

        fetched = await CrudAuthor.get(
            session,
            col(CrudAuthor.id) == a_id,
            load=[rel(CrudAuthor.books), rel(CrudBook.author)],
        )
        assert fetched is not None
        assert [b.title for b in fetched.books] == ["cy"]
        # Second hop (the books' author back-reference) is eagerly loaded too.
        assert "author" in fetched.books[0].__dict__
        assert fetched.books[0].author.id == a_id

    async def test_save_with_load_returns_preloaded(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="sl").save(session)
        await CrudBook(title="x", author_id=a.id).save(session)
        a = await a.save(session, load=rel(CrudAuthor.books))
        assert "books" in a.__dict__
        assert [b.title for b in a.books] == ["x"]

    async def test_jti_subclasses_without_load_raises(self, session: AsyncSession) -> None:
        with pytest.raises(ValueError, match="requires the load parameter"):
            await CrudAuthor.get(session, fetch_mode="all", jti_subclasses="all")


# --------------------------------------------------------------------------
# get_one / get_exist_one
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGetOne:
    async def test_get_one_returns_instance(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="go").save(session)
        fetched = await CrudAuthor.get_one(session, a.id)
        assert fetched.id == a.id

    async def test_get_one_missing_raises_no_result_found(self, session: AsyncSession) -> None:
        with pytest.raises(NoResultFound):
            await CrudAuthor.get_one(session, 424242)

    async def test_get_exist_one_found(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="ge").save(session)
        fetched = await CrudAuthor.get_exist_one(session, a.id)
        assert fetched.name == "ge"

    async def test_get_exist_one_404_default_detail(self, session: AsyncSession) -> None:
        with pytest.raises(HTTPException) as exc_info:
            await CrudAuthor.get_exist_one(session, 999999)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"

    async def test_get_exist_one_custom_detail(self, session: AsyncSession) -> None:
        with pytest.raises(HTTPException) as exc_info:
            await CrudAuthor.get_exist_one(session, 999999, detail="Author missing")
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Author missing"


# --------------------------------------------------------------------------
# update with DTO
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestUpdate:
    async def test_partial_update_only_touches_set_fields(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="orig", age=30).save(session)
        a = await a.update(session, CrudAuthorPatch(age=31))
        assert a.age == 31
        assert a.name == "orig"  # unset DTO field not applied

        fetched = await CrudAuthor.get_one(session, a.id)
        assert fetched.age == 31
        assert fetched.name == "orig"

    async def test_update_with_extra_data(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="e", age=1).save(session)
        a = await a.update(session, CrudAuthorPatch(age=2), extra_data={"name": "renamed"})
        assert a.age == 2
        assert a.name == "renamed"

    async def test_update_with_exclude(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="keep", age=5).save(session)
        a = await a.update(session, CrudAuthorPatch(name="new", age=6), exclude={"age"})
        assert a.name == "new"
        assert a.age == 5  # excluded from the merge

    async def test_update_refresh_false_returns_same_instance(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="nr", age=1).save(session)
        a_id = a.id  # snapshot: update() commits and expires the instance
        returned = await a.update(session, CrudAuthorPatch(age=2), refresh=False)
        assert returned is a
        fetched = await CrudAuthor.get_one(session, a_id)
        assert fetched.age == 2


# --------------------------------------------------------------------------
# delete
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDelete:
    async def test_delete_single_instance(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="d1").save(session)
        assert await CrudAuthor.delete(session, a) == 1
        assert await CrudAuthor.count(session) == 0

    async def test_delete_instance_list(self, session: AsyncSession) -> None:
        authors = await _seed_authors(session)
        deleted = await CrudAuthor.delete(session, authors[:2])
        assert deleted == 2
        remaining = await CrudAuthor.get(session, fetch_mode="all")
        assert [a.name for a in remaining] == ["carol"]

    async def test_delete_by_condition_returns_rowcount(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        deleted = await CrudAuthor.delete(session, condition=cond(CrudAuthor.age >= 20))
        assert deleted == 2
        assert await CrudAuthor.count(session) == 1

    async def test_delete_by_condition_no_match_returns_zero(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        deleted = await CrudAuthor.delete(session, condition=cond(CrudAuthor.age > 1000))
        assert deleted == 0
        assert await CrudAuthor.count(session) == 3

    async def test_delete_both_args_raises(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="dup").save(session)
        with pytest.raises(ValueError, match="both"):
            await CrudAuthor.delete(session, a, condition=cond(CrudAuthor.id == a.id))

    async def test_delete_neither_arg_raises(self, session: AsyncSession) -> None:
        with pytest.raises(ValueError, match="either"):
            await CrudAuthor.delete(session)  # type: ignore[call-overload]


# --------------------------------------------------------------------------
# count / get_with_count
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCount:
    async def test_count_empty_and_total(self, session: AsyncSession) -> None:
        assert await CrudAuthor.count(session) == 0
        await _seed_authors(session)
        assert await CrudAuthor.count(session) == 3

    async def test_count_with_condition(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        assert await CrudAuthor.count(session, cond(CrudAuthor.age >= 20)) == 2

    async def test_count_time_filter_object_takes_priority(self, session: AsyncSession) -> None:
        await _seed_authors(session)  # created Jan 1/2/3
        tf = TimeFilterRequest(created_after_datetime=datetime(2024, 1, 3))
        # the explicit kwarg (Jan 1, matches all) must lose to the time_filter
        n = await CrudAuthor.count(
            session, time_filter=tf, created_after_datetime=datetime(2024, 1, 1)
        )
        assert n == 1

    async def test_get_with_count_pagination(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        tv = TableViewRequest(limit=2, desc=False)  # oldest first
        resp = await CrudAuthor.get_with_count(session, table_view=tv)
        assert resp.count == 3  # total, not page size
        assert [a.name for a in resp.items] == ["alice", "bob"]

    async def test_get_with_count_offset_page(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        tv = TableViewRequest(limit=2, offset=2, desc=False)
        resp = await CrudAuthor.get_with_count(session, table_view=tv)
        assert resp.count == 3
        assert [a.name for a in resp.items] == ["carol"]

    async def test_get_with_count_time_filter_applies_to_both(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        tv = TableViewRequest(created_after_datetime=datetime(2024, 1, 2), desc=False)
        resp = await CrudAuthor.get_with_count(session, table_view=tv)
        assert resp.count == 2
        assert [a.name for a in resp.items] == ["bob", "carol"]

    async def test_get_with_count_condition(self, session: AsyncSession) -> None:
        await _seed_authors(session)
        resp = await CrudAuthor.get_with_count(session, cond(CrudAuthor.age < 30))
        assert resp.count == 2
        assert {a.name for a in resp.items} == {"alice", "bob"}


# --------------------------------------------------------------------------
# cond / rel helpers
# --------------------------------------------------------------------------

class TestHelpers:
    def test_cond_is_identity_at_runtime(self) -> None:
        expr = CrudAuthor.age >= 1
        narrowed = cond(expr)
        assert narrowed is expr
        assert isinstance(narrowed, ColumnElement)

    def test_cond_result_supports_boolean_operators(self) -> None:
        combined = cond(CrudAuthor.age >= 1) & cond(CrudAuthor.age <= 2)
        assert isinstance(combined, ColumnElement)

    def test_rel_returns_queryable_attribute(self) -> None:
        attr = rel(CrudAuthor.books)
        assert attr is CrudAuthor.books

    def test_rel_rejects_non_relationship(self) -> None:
        with pytest.raises(AttributeError, match="Expected a Relationship field"):
            rel(42)
        with pytest.raises(AttributeError):
            rel("CrudAuthor.books")


# --------------------------------------------------------------------------
# FOR UPDATE tracking + enhanced session reset integration
# --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestForUpdateTracking:
    async def test_get_with_for_update_records_instance(self, session: AsyncSession) -> None:
        a = await CrudAuthor(name="lock").save(session)
        locked = await CrudAuthor.get(
            session, col(CrudAuthor.id) == a.id, fetch_mode="one", with_for_update=True
        )
        assert id(locked) in session.info[SESSION_FOR_UPDATE_KEY]

    async def test_get_all_with_for_update_records_every_instance(
        self, session: AsyncSession
    ) -> None:
        await _seed_authors(session)
        instances = await CrudAuthor.get(session, fetch_mode="all", with_for_update=True)
        tracked = session.info[SESSION_FOR_UPDATE_KEY]
        assert {id(i) for i in instances} <= tracked
        assert len(tracked) == 3

    async def test_no_tracking_without_for_update(self, session: AsyncSession) -> None:
        await CrudAuthor(name="free").save(session)
        await CrudAuthor.get(session, fetch_mode="all")
        assert SESSION_FOR_UPDATE_KEY not in session.info

    async def test_enhanced_reset_clears_for_update_tracking(
        self, enhanced_session: ExtAsyncSession
    ) -> None:
        a = await CrudAuthor(name="rl").save(enhanced_session)
        await CrudAuthor.get(
            enhanced_session,
            col(CrudAuthor.id) == a.id,
            fetch_mode="one",
            with_for_update=True,
        )
        assert SESSION_FOR_UPDATE_KEY in enhanced_session.info

        await enhanced_session.reset()
        assert SESSION_FOR_UPDATE_KEY not in enhanced_session.info

        # Session stays usable after reset.
        again = await CrudAuthor.get(enhanced_session, col(CrudAuthor.id) == a.id)
        assert again is not None
