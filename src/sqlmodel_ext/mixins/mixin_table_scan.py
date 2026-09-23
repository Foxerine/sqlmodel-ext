"""
Cross-table scan for a field-level mixin mounted on several tables.

**Scope**: reflective discovery + parametrized condition scans for "the same
field-level mixin mounted on several unrelated physical tables" -- e.g. a
mixin that adds a ``remote_file_key`` column to N vendor tables, where you
need to "query / update rows by some condition across every concrete table
that mounts it". The discovery logic is independent of any specific field, so
it lives here once and can be mixed into any such mixin. It is **not**
responsible for the query conditions or their business meaning -- the
consuming mixin supplies those.

**Invariants**:

1. ``_concrete_mixin_subclasses()`` de-duplicates by ``__table__`` object (not
   by Python class): several concrete classes of one STI family share one
   physical table and all have a ``__table__``; only the first one visited in
   the depth-first pre-order walk is kept. A parent is always visited before
   its children, so for STI the polymorphic root is selected -- querying it
   adds no ``polymorphic_identity`` filter and covers the rows of every
   subclass. JTI subclasses own separate physical tables and are unaffected.
2. ``_scan_rows_where`` takes its condition as a
   ``Callable[[type[Self]], ColumnElement[bool]]`` and **never accepts string
   field names** -- callers build the condition with type-safe expressions
   such as ``col(table_cls.some_field)``; this module knows no field names.
"""
from collections.abc import AsyncIterator, Callable
from typing import Self, cast

from sqlalchemy import ColumnElement
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext.mixins.table import TableBaseMixin


class MixinTableScanMixin:
    """Utility mixin: a field-level mixin that mixes this in gains cross-table discovery + parametrized scans.

    It has no data fields and takes no part in CRUD -- purely a
    reflection/scan tool.

    It only makes sense when the consuming mixin is eventually mixed into
    concrete table classes (with ``TableBaseMixin``). Discovered table classes
    are ``cast`` to ``TableBaseMixin`` to call ``.get()`` -- a typing device
    (every discovered class has a ``__table__``, see
    ``_concrete_mixin_subclasses``), not a requirement that this mixin itself
    inherits ``TableBaseMixin``: field-level mixins are also mixed into plain
    DTO classes and must not be bound to it at the type level.
    """

    @classmethod
    def _concrete_mixin_subclasses(cls) -> list[type[Self]]:
        """Recursively discover every **concrete table class** (``__table__`` is not ``None``) that mounts this mixin.

        Non-table intermediate bases are skipped. Discovery is dynamic, so a
        new table that mounts the mixin is covered automatically.
        """
        collected: list[type[Self]] = []
        seen_classes: set[type] = set()
        seen_tables: set[object] = set()

        def visit(c: type[Self]) -> None:
            if c in seen_classes:
                return
            seen_classes.add(c)
            table = getattr(c, '__table__', None)
            if table is not None and table not in seen_tables:
                seen_tables.add(table)
                collected.append(c)
            for sub in c.__subclasses__():
                visit(sub)

        visit(cls)
        return collected

    @classmethod
    async def _scan_rows_where(
            cls,
            session: AsyncSession,
            condition_factory: Callable[[type[Self]], ColumnElement[bool]],
    ) -> AsyncIterator[tuple[type[Self], Self]]:
        """Scan every concrete table mounting this mixin and ``yield (table_class, row)`` for each matching row.

        ``condition_factory(table_cls)`` builds each table's WHERE condition
        (type-safe, no string field names), so several consumers of the same
        field ("query only" vs. "query then update") can share one scan
        implementation.
        """
        for table_cls in cls._concrete_mixin_subclasses():
            typed_cls = cast('type[TableBaseMixin]', table_cls)
            rows = await typed_cls.get(session, condition_factory(table_cls), fetch_mode='all')
            for row in rows:
                # At runtime the row is both a TableBaseMixin (produced by
                # .get()) and a Self (table_cls mounts this mixin); the
                # intersection is not expressible, so widen through object.
                yield table_cls, cast(Self, cast(object, row))
