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
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar, final, get_origin
from uuid import UUID

import sqlalchemy as sa
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID

# --- Generic Type Definitions ---
T = TypeVar('T')


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
            self.sa_array_type = ARRAY(
                sa.Enum(origin_type, name=origin_type.__name__.lower())
            )
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
