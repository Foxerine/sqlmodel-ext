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
from typing import Annotated, Generic, TypeVar, get_origin
from uuid import UUID

import sqlalchemy as sa
from pydantic_core import core_schema
from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID

# --- Generic Type Definitions ---
T = TypeVar('T')

# Mapping from Python types to Pydantic core schemas
PY_TO_PYDANTIC_SCHEMA_MAP = {
    str: core_schema.str_schema,
    int: core_schema.int_schema,
    dict: lambda: core_schema.dict_schema(
        core_schema.str_schema(), core_schema.any_schema()
    ),
    UUID: core_schema.uuid_schema,
}

# Mapping from Python types to SQLAlchemy column types
PY_TO_SA_TYPE_MAP = {
    str: String,
    int: Integer,
    dict: JSONB,
    UUID: PG_UUID(as_uuid=True),
}


# --- Pydantic/SQLModel Integration Layer ---
class _ArrayTypeHandler(Generic[T]):
    """(Internal) Provides Pydantic/SQLAlchemy config for Array[T]."""

    def __init__(self, item_type: type[T], max_length: int | None = None):
        self.max_length = max_length
        # Allow item_type to be a generic alias, e.g. dict[str, Any]
        origin_type = get_origin(item_type) or item_type

        if issubclass(origin_type, Enum):
            members = [member.value for member in origin_type]
            self.item_schema = core_schema.literal_schema(members)
            self.sa_array_type = ARRAY(
                sa.Enum(origin_type, name=origin_type.__name__.lower())
            )
        elif origin_type in PY_TO_PYDANTIC_SCHEMA_MAP:
            self.item_schema = PY_TO_PYDANTIC_SCHEMA_MAP[origin_type]()
            sa_type = (
                JSONB if origin_type is dict
                else PY_TO_SA_TYPE_MAP[origin_type]
            )
            self.sa_array_type = ARRAY(sa_type)
        else:
            raise TypeError(f"Unsupported inner type for Array: {item_type}")

    def __get_pydantic_core_schema__(self, s, h):
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
