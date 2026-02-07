"""
PostgreSQL JSONB types with size limits.

Provides two JSONB types that enforce a maximum JSON string length of 100K characters:

- ``JSON100K`` -- stores a ``dict`` (JSON object)
- ``JSONList100K`` -- stores a ``list[dict]`` (JSON array)
"""
import typing

import orjson
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from sqlalchemy.dialects.postgresql import JSONB

MAX_JSON_LENGTH = 100_000
"""Maximum JSON string length (100K characters)."""


def _serialize_to_json(value: dict | list) -> str:
    """Serialize a dict or list to a JSON string."""
    return orjson.dumps(value).decode('utf-8')


def _parse_json_string(value: str, expected_type: type, type_name: str) -> dict | list:
    """
    Parse a JSON string.

    :param value: JSON string
    :param expected_type: Expected Python type (dict or list)
    :param type_name: Type name for error messages
    :returns: Parsed dict or list
    :raises ValueError: If length exceeds limit or format is invalid
    """
    if len(value) > MAX_JSON_LENGTH:
        raise ValueError(
            f"JSON string length exceeds limit: {len(value)} > {MAX_JSON_LENGTH}"
        )

    try:
        result = orjson.loads(value)
    except orjson.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e

    if not isinstance(result, expected_type):
        expected = "object" if expected_type is dict else "array"
        raise ValueError(
            f"JSON must be an {expected}, not {type(result).__name__}"
        )

    return result


class JSON100K(dict):
    """
    PostgreSQL JSONB type (object) with a 100K character input limit.

    - Behaves as ``dict[str, Any]`` in Python code
    - Accepts either a dict or a JSON string as input
    - Input string is limited to 100K characters
    - Stored as JSONB in PostgreSQL
    - Automatically serialized to a JSON string in API responses

    Usage::

        from sqlmodel_ext.field_types.dialects.postgresql import JSON100K

        class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
            canvas: JSON100K
            '''Canvas data (JSONB, max 100K chars)'''
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Pydantic v2 core schema for dict JSONB."""

        def validate(value: typing.Any) -> dict[str, typing.Any]:
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                return _parse_json_string(value, dict, "JSON100K")
            raise TypeError(
                f"JSON100K accepts str or dict, not {type(value).__name__}"
            )

        dict_schema = core_schema.dict_schema(
            core_schema.str_schema(),
            core_schema.any_schema()
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                validate,
                core_schema.str_schema(max_length=MAX_JSON_LENGTH),
            ),
            python_schema=core_schema.no_info_after_validator_function(
                validate,
                core_schema.union_schema([
                    dict_schema,
                    core_schema.str_schema(max_length=MAX_JSON_LENGTH),
                ]),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_to_json,
                info_arg=False,
                return_schema=core_schema.str_schema(),
                when_used='always',
            ),
            metadata={'sa_type': JSONB},
        )


class JSONList100K(list):
    """
    PostgreSQL JSONB type (array) with a 100K character input limit.

    - Behaves as ``list[dict[str, Any]]`` in Python code
    - Accepts either a list or a JSON string as input
    - Input string is limited to 100K characters
    - Stored as JSONB in PostgreSQL
    - Automatically serialized to a JSON string in API responses

    Usage::

        from sqlmodel_ext.field_types.dialects.postgresql import JSONList100K

        class Conversation(SQLModelBase, UUIDTableBaseMixin, table=True):
            messages: JSONList100K
            '''Message list (JSONB, max 100K chars)'''
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Pydantic v2 core schema for list JSONB."""

        def validate(value: typing.Any) -> list[dict[str, typing.Any]]:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                return _parse_json_string(value, list, "JSONList100K")
            raise TypeError(
                f"JSONList100K accepts str or list, not {type(value).__name__}"
            )

        dict_schema = core_schema.dict_schema(
            core_schema.str_schema(),
            core_schema.any_schema()
        )
        list_schema = core_schema.list_schema(dict_schema)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                validate,
                core_schema.str_schema(max_length=MAX_JSON_LENGTH),
            ),
            python_schema=core_schema.no_info_after_validator_function(
                validate,
                core_schema.union_schema([
                    list_schema,
                    core_schema.str_schema(max_length=MAX_JSON_LENGTH),
                ]),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_to_json,
                info_arg=False,
                return_schema=core_schema.str_schema(),
                when_used='always',
            ),
            metadata={'sa_type': JSONB},
        )
