"""
PostgreSQL ARRAY type support.

Provides a generic ``Array[T]`` type for using PostgreSQL's native ARRAY type
in SQLModel models.

Usage::

    # No length limit
    tags: Array[str] = Field(default_factory=list)

    # Limit to 20 elements
    version_vector: Array[dict, 20] = Field(default_factory=list)
"""
import logging
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable, Generic, TypeVar, final, get_origin
from uuid import UUID

import sqlalchemy as sa
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.engine.interfaces import Dialect

logger = logging.getLogger(__name__)

# --- Generic Type Definitions ---
T = TypeVar('T')


_UNKNOWN_ENUM_ELEM = object()
"""Internal sentinel yielded by ``_TolerantEnum`` when it reads an unknown DB
value; filtered out at the array level by ``_TolerantEnumArray``. A dedicated
sentinel (rather than None) is used so it stays distinct from a genuine NULL
element inside the array."""


class _TolerantEnum(sa.TypeDecorator[Any]):
    """Read-tolerant enum (a ``TypeDecorator`` wrapping ``sa.Enum``): when the
    value returned by the database is not a member of the Python enum, it is
    downgraded to a sentinel instead of raising ``LookupError``.

    Motivation: during a rolling deployment's version-skew window, a newer pod
    may run a migration that backfills a new enum value into an ``ARRAY(enum)``
    column while an older pod -- still serving traffic with a Python enum that
    lacks that value -- reads the row. ``sa.Enum`` raises ``LookupError`` on an
    unknown value by default, so **every** request on the old pod that loads
    such a row would 500 until the rollout completes.

    Why a ``TypeDecorator`` rather than subclassing ``sa.Enum`` directly:
    dialect adaptation (``dialect.colspecs`` -> ``type.adapt(impltype)``) copies
    a ``sa.Enum`` subclass instance into a ``postgresql.ENUM`` / asyncpg
    ``AsyncPgEnum`` instance, discarding the subclass overrides. ``TypeDecorator``
    is SQLAlchemy's official mechanism for "behavior wrapping that must survive
    dialect adaptation": the decorator itself stays as the column type, its impl
    is adapted internally, and the ``result_processor`` hook is guaranteed to be
    called.

    Semantics: unknown values are dropped on the **read path only** (with a
    warning); the write path still goes through Pydantic / PG enum strict
    validation, so no dirty writes are introduced. Dropped values remain in the
    database and become visible again once code that knows them is deployed (the
    DB is the superset source of truth; each code version consumes the subset it
    recognizes -- consistent with Pydantic's ``extra='ignore'`` handling of
    unknown fields).
    """

    impl = sa.Enum
    cache_ok = True

    def __init__(self, enum_class: type[Enum], name: str):
        super().__init__(
            enum_class,
            name=name,
            values_callable=lambda e: [m.value for m in e],
        )
        self._enum_name = name

    def result_processor(
            self, dialect: Dialect, coltype: object,
    ) -> 'Callable[[Any], Any]':
        impl_processor = self.impl_instance.result_processor(dialect, coltype)

        def process(value: Any) -> Any:
            if value is None:
                return None
            try:
                return impl_processor(value) if impl_processor is not None else value
            except LookupError:
                logger.warning(
                    "Enum column read an unknown value %r (enum %s); ignored -- "
                    "likely a rolling-deployment version-skew window (the DB holds "
                    "a newer enum value that this process's code does not yet know).",
                    value, self._enum_name,
                )
                return _UNKNOWN_ENUM_ELEM

        return process


class _TolerantEnumArray(sa.TypeDecorator[Any]):
    """ARRAY companion to ``_TolerantEnum`` (a ``TypeDecorator`` wrapping
    ``ARRAY``): filters out the unknown-enum sentinel elements while processing
    results.

    The sentinel is matched by identity (``is``) so genuine NULL elements
    (``None``) inside the array are unaffected.
    """

    impl = ARRAY
    cache_ok = True

    def __init__(self, item_type: '_TolerantEnum'):
        super().__init__(item_type)

    def process_result_value(
            self, value: 'list[Any] | None', dialect: Dialect,
    ) -> 'list[Any] | None':
        if value is None:
            return None
        return [elem for elem in value if elem is not _UNKNOWN_ENUM_ELEM]


# --- Pydantic/SQLModel Integration Layer ---
@final
class _ArrayTypeHandler:
    """(Internal) Provides Pydantic/SQLAlchemy config for Array[T]."""

    def __init__(self, item_type: type[T], max_length: int | None = None):
        self.max_length = max_length
        # Allow item_type to be a generic alias, e.g. dict[str, Any]
        origin_type = get_origin(item_type) or item_type

        if issubclass(origin_type, Enum):
            # Enum.value is always Any in Python's type system (untyped .value attribute)
            self.item_schema = core_schema.literal_schema(
                [member.value for member in origin_type]  # pyright: ignore[reportAny]
            )
            # _TolerantEnum carries values_callable so the PG ENUM is populated
            # with the member string values (StrEnum.value) rather than the
            # Python identifiers (Enum.name). Required so StrEnum subclasses
            # serialize as their user-facing strings (e.g. 'read:own' instead of
            # 'READ_OWN'). The _TolerantEnumArray/_TolerantEnum pair also makes
            # the read path tolerant of unknown DB enum values (see their
            # docstrings) instead of raising LookupError during version skew.
            self.sa_array_type = _TolerantEnumArray(_TolerantEnum(
                origin_type,
                name=origin_type.__name__.lower(),
            ))
            return

        # Per-type dispatch to avoid union types that cannot satisfy ARRAY's invariant _T
        match origin_type:
            case t if t is str:
                self.item_schema = core_schema.str_schema()
                self.sa_array_type = ARRAY(String)
            case t if t is int:
                self.item_schema = core_schema.int_schema()
                self.sa_array_type = ARRAY(Integer)
            case t if t is dict:
                self.item_schema = core_schema.dict_schema(
                    core_schema.str_schema(), core_schema.any_schema()
                )
                self.sa_array_type = ARRAY(JSONB)
            case t if t is UUID:
                self.item_schema = core_schema.uuid_schema()
                self.sa_array_type = ARRAY(PG_UUID(as_uuid=True))
            case _:
                raise TypeError(f"Unsupported inner type for Array: {item_type}")

    def __get_pydantic_core_schema__(
            self, source_type: type[T], handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        list_schema = core_schema.list_schema(
            self.item_schema,
            max_length=self.max_length,
        )
        return core_schema.json_or_python_schema(
            json_schema=list_schema,
            python_schema=list_schema,
            metadata={'sa_type': self.sa_array_type}
        )


# --- Public, User-Facing Type ---
#
# basedpyright does not evaluate custom __class_getitem__ and treats Array[T] as a
# nominal subtype of list[T], making list[T] unassignable to Array[T].
# By aliasing Array = list in the TYPE_CHECKING branch, type checkers correctly
# resolve Array[T] as list[T].
# Known limitation: Array[T, N] (two-param form) degrades to list[T, N] at type-check
# time, which basedpyright will flag. This is consistent with the original class
# definition (which also reported reportInvalidTypeArguments).
if TYPE_CHECKING:
    Array = list
else:
    class Array(list[T], Generic[T]):
        """
        A generic array type compatible with Pydantic and SQLModel, designed for PostgreSQL.

        Behaves as ``list[T]`` in Python code and is stored as a native ``ARRAY`` in PostgreSQL.
        Supports basic types like ``str``, ``int``, ``dict``, ``UUID``, and standard ``Enum`` classes.

        Two forms are supported::

            Array[str]       # no length limit
            Array[str, 20]   # max 20 elements
        """

        @classmethod
        def __class_getitem__(cls, params: type[T] | tuple[type[T], int]):
            if isinstance(params, tuple):
                item_type, max_length = params
            else:
                item_type, max_length = params, None
            return Annotated[list[item_type], _ArrayTypeHandler(item_type, max_length)]
