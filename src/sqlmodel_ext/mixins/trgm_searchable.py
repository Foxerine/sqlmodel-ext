"""Fuzzy search on name / text columns backed by PostgreSQL ``pg_trgm`` (``TrgmSearchableMixin``). **PostgreSQL only.**

An opt-in capability mixin: the host declares its searchable columns through
two ClassVars and endpoints call ``trgm_search_condition`` without repeating
column names.

**Filtering only, never ordering**: the mixin produces a ``WHERE`` condition,
not an ``ORDER BY`` -- ordering stays under the caller's explicit control
(e.g. ``table_view``); a search never silently changes the sort order.

Matching strategy (combined into one SQL condition):

- **name column** (``__trgm_name_column__``): ``ILIKE %query%`` substring match
  **OR** the pg_trgm similarity operator ``name % query`` (index-backed).
  Names are short, so trigram similarity usefully tolerates typos.
- **text columns** (``__trgm_text_columns__``): ``ILIKE %query%`` only -- the
  similarity of a long text to a short query is almost always tiny and
  carries no signal.

Columns are resolved through ``sa_inspect(cls).columns[name]`` rather than
``cls.<field>``, so STI base classes (whose subclass columns are registered
on the parent mapper without class attributes) and plain tables work alike.

**Host contract** (not enforced by the mixin):

1. The database must have ``CREATE EXTENSION pg_trgm`` and the
   ``public.bigrams(text)`` function (:data:`BIGRAM_FUNCTION_SQL`) installed
   before these conditions are used.
2. Create GIN indexes for the searched columns, e.g.
   ``CREATE INDEX ... USING gin (name gin_trgm_ops)`` and
   ``CREATE INDEX ... USING gin (public.bigrams(description))``. Without
   them the search still works but degrades to sequential scans.
3. Searched columns must use the database default collation (see
   :class:`TrgmSearchableMixin`).

The ``%`` threshold is the ``pg_trgm.similarity_threshold`` setting
(PostgreSQL default 0.3).

**Safety**: ``icontains(autoescape=True)`` escapes ``%`` / ``_`` / backslash,
so user input cannot inject LIKE wildcards. :class:`TrgmSearchRequest` uses
``Str64`` (length-bounded, rejects NUL bytes, which PostgreSQL text cannot
store).
"""
from typing import Any, ClassVar, cast

from sqlalchemy import ColumnElement, Text, and_, func, inspect as sa_inspect
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapper

from sqlmodel_ext.base import SQLModelBase
from sqlmodel_ext.field_types import Str64

_MIN_BIGRAM_QUERY_CHARS = 2
"""Shortest query that yields at least one bigram.

For shorter queries ``bigrams(query)`` is empty: the bigram guard filters
nothing yet PostgreSQL would still compute ``bigrams(column)`` for every
candidate row -- so such queries are short-circuited when the condition is
built (see ``_bigram_bounded_icontains``). This is a property of the bigram
algorithm, unrelated to any minimum search length a host may enforce."""

BIGRAM_FUNCTION_SQL = '''
CREATE OR REPLACE FUNCTION public.bigrams(t text) RETURNS text[] AS $$
    SELECT COALESCE(
        (SELECT array_agg(DISTINCT c || nxt) FROM (
            SELECT c, lead(c) OVER (ORDER BY ord) AS nxt
            FROM unnest(regexp_split_to_array(lower(t), '')) WITH ORDINALITY AS s(c, ord)
         ) AS paired WHERE nxt IS NOT NULL),
        '{}'::text[])
$$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
'''
"""DDL for ``bigrams(text) -> text[]``: the de-duplicated, lower-cased set of 2-grams of a string.

Execute it once (e.g. in a migration, before creating function indexes on
it). ``IMMUTABLE`` (required for function indexes), ``STRICT`` (NULL in ->
NULL out), ``PARALLEL SAFE``. Inputs shorter than 2 characters return ``'{}'``.

**Never change the body once indexes use it**: a ``gin (bigrams(col))`` index
stores keys computed by the old definition; a changed definition silently
misses rows until ``REINDEX``.

Why "split once + pair with ``lead()``" instead of ``substr(s, i, 2)`` in a
loop: for multi-byte UTF-8, ``substr`` must scan from the start to reach
position ``i`` (O(n) each, O(n²) overall), which makes writes of long texts
very expensive when a function index recomputes it. Splitting the already
lower-cased text also avoids collations where ``lower()`` changes the string
length."""


class TrgmSearchableMixin:
    """Opt-in capability mixin: pg_trgm fuzzy search on name / text columns (PostgreSQL only).

    The host must be a SQLModel table class containing the columns named by
    ``__trgm_name_column__`` / ``__trgm_text_columns__``.

    Host contract: searched columns must **not** use a non-default collation
    (e.g. ``String(collation='tr-x-icu')``). The two branches of the
    condition treat collations asymmetrically -- ``ILIKE`` case folding
    follows the **column's** collation while ``bigrams(query)`` uses the
    database default -- so with a special collation ``ILIKE`` may match while
    the bigram guard (``@>``) rejects the row first, silently losing results.
    """

    __trgm_name_column__: ClassVar[str] = 'name'
    """Main name column -- ILIKE substring + trigram similarity."""

    __trgm_text_columns__: ClassVar[tuple[str, ...]] = ()
    """Additional text columns (e.g. ``('description',)``) -- ILIKE substring only. Default: none."""

    @classmethod
    def _trgm_column(cls, column_name: str) -> ColumnElement[Any]:
        """Resolve a physical column of this table by name (works for STI base classes and plain tables).

        :raises RuntimeError: the column is not on the mapper
        """
        columns = cast('Mapper[Any]', sa_inspect(cls)).columns
        try:
            return columns[column_name]
        except KeyError as e:
            raise RuntimeError(
                f"{cls.__name__}: trgm search column {column_name!r} is not on the mapper; "
                + "check __trgm_name_column__ / __trgm_text_columns__"
            ) from e

    @classmethod
    def trgm_search_condition(cls, query: str) -> ColumnElement[bool]:
        """Build the pg_trgm fuzzy ``WHERE`` condition for the name column (+ ``__trgm_text_columns__``).

        The name column matches by ``ILIKE`` substring **OR** the ``%``
        similarity operator (``similarity(name, query) >
        pg_trgm.similarity_threshold``). The operator form is used instead of
        ``similarity() >= threshold`` because only the operator can use a GIN
        ``gin_trgm_ops`` index. Text columns match by ``ILIKE`` substring only.

        :param query: a stripped, non-empty search term (callers filter empty / blank input)
        :returns: a condition usable as ``get()`` / ``get_with_count()`` ``condition``
        """
        name_col = cls._trgm_column(cls.__trgm_name_column__)
        clause: ColumnElement[bool] = (
            cls._bigram_bounded_icontains(name_col, query)
            | name_col.op('%', is_comparison=True)(query)
        )
        for column_name in cls.__trgm_text_columns__:
            clause = clause | cls._bigram_bounded_icontains(cls._trgm_column(column_name), query)
        return clause

    @staticmethod
    def _bigram_bounded_icontains(column: ColumnElement[Any], query: str) -> ColumnElement[bool]:
        """An index-bounded equivalent of ``column ILIKE '%query%'``: narrow by bigram containment, then recheck exactly.

        Why: pg_trgm accelerates ``LIKE`` only when it can extract complete
        trigrams from the pattern, and substring patterns are not padded -- a
        2-character query yields no trigram and falls back to a sequential
        scan (a cheap way to make every search a full scan, common for
        2-character words in CJK text).

        Equivalence: any string containing substring ``S`` (``|S| >= 2``)
        contains every 2-gram of ``S``, so ``bigrams(col) @> bigrams(q)`` is a
        strict superset of ``col ILIKE '%q%'``; the ``ILIKE`` recheck restores
        exactness.

        Both sides call the **same SQL function** (instead of computing the
        query's bigrams in Python): Python's ``str.lower()`` and PostgreSQL's
        ``lower()`` differ on some characters, which would silently misalign
        the sets.

        Queries shorter than 2 characters short-circuit to a plain
        ``icontains`` **without emitting ``bigrams(column)``**: the
        "empty array is always contained" form is semantically equivalent
        but PostgreSQL would still evaluate ``bigrams(column)`` per row (orders
        of magnitude slower on long texts).

        The ``ILIKE`` recheck is not redundant: bigram sets are unordered, so
        a text containing both ``ab`` and ``bc`` in the wrong order matches
        ``@> bigrams('abc')`` without containing ``abc``.

        The function is schema-qualified (``public.bigrams``): with the
        default ``search_path`` (``"$user", public``) an unqualified name
        could resolve to a same-named function in the role's schema, binding
        the query and the index to different functions.
        """
        if len(query) < _MIN_BIGRAM_QUERY_CHARS:
            return column.icontains(query, autoescape=True)

        query_bigrams = func.public.bigrams(query, type_=ARRAY(Text))
        return (
            func.public.bigrams(column, type_=ARRAY(Text)).contains(query_bigrams)
            & column.icontains(query, autoescape=True)
        )


class TrgmSearchRequest(SQLModelBase):
    """Fuzzy-search request DTO -- carries the ``?query=`` query parameter and applies it to a query.

    Same "query-parameter DTO with behavior" pattern as ``TableViewRequest``:
    inject it into an endpoint, then call ``apply_condition`` to AND the fuzzy
    filter into the condition for any ``TrgmSearchableMixin`` host.

    Filtering only: ordering stays with the endpoint's ``table_view``.
    """

    query: Str64 | None = None
    """Fuzzy search term; ``None`` or blank = no filtering (``Str64`` bounds the length and rejects NUL bytes)."""

    @property
    def normalized_query(self) -> str | None:
        """The stripped, non-empty search term; ``None`` means "no effective search term"."""
        if self.query is None:
            return None
        stripped = self.query.strip()
        return stripped or None

    def apply_condition(
            self,
            model: type[TrgmSearchableMixin],
            condition: ColumnElement[bool] | None = None,
    ) -> ColumnElement[bool] | None:
        """AND ``model``'s fuzzy search condition into an existing ``condition``.

        Without an effective search term ``condition`` is returned unchanged.

        :param model: a table class mixing in ``TrgmSearchableMixin``
        :param condition: the caller's scope / other filters (may be ``None``)
        :returns: the combined condition
        """
        query = self.normalized_query
        if query is None:
            return condition
        fuzzy = model.trgm_search_condition(query)
        return fuzzy if condition is None else and_(condition, fuzzy)
