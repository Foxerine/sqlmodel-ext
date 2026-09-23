"""
PostgreSQL JSONB types with size limits.

Provides two JSONB types whose canonical JSON encoding never exceeds 100K
characters, **whatever the input form** (JSON string, native ``dict`` or
native ``list``):

- ``JSON100K`` -- stores a ``dict`` (JSON object)
- ``JSONList100K`` -- stores a ``list[dict]`` (JSON array)

The limits apply to every input form through a single function,
:func:`ensure_json_within_limits`.

**Contract: object in, object out.** Inbound accepts a JSON object / array
(preferred) or a JSON *string* (compatibility form, see below); outbound is
**always** the object / array itself -- ``model_dump()`` and
``model_dump(mode='json')`` agree, and ``model_dump_json()`` embeds a nested
JSON value rather than an escaped string. The serialization JSON Schema is
declared as a pure object / array accordingly, so code generators do not have
to model an impossible ``string`` branch.

**Why inbound still accepts a JSON string**: callers that historically sent a
stringified JSON value (older clients, LLM tool calls taught to pass a JSON
string) keep working. The validation JSON Schema therefore honestly declares
``anyOf[object, string]`` -- that is the real inbound contract, not a leftover.
"""
import typing

import orjson
import pydantic_core
from pydantic import GetCoreSchemaHandler
from pydantic_core import PydanticSerializationError, core_schema
from sqlalchemy.dialects.postgresql import JSONB

MAX_JSON_LENGTH = 100_000
"""Maximum canonical JSON length (100K characters)."""


def ensure_json_within_limits(value: dict[str, typing.Any] | list[typing.Any]) -> None:
    """Verify that ``value`` can be JSON-encoded and that the encoding fits in
    ``MAX_JSON_LENGTH`` characters.

    Called by the ``JSON100K`` / ``JSONList100K`` Pydantic validators for every
    input form (dict, list, and the parsed result of a JSON string). It is also
    usable directly wherever Pydantic validation is bypassed -- notably
    ``table=True`` models, which skip field validators on construction.

    Two checks, one encoding pass:

    - **Encodability**: ``orjson.loads`` accepts much deeper nesting than
      ``orjson.dumps`` can produce (``dumps`` stops at 255 levels with
      ``JSONEncodeError: Recursion limit reached``). Without an input-side
      check, a deeply nested value would be stored silently and only fail later
      in ``model_dump(mode='json')`` / a response / a cache write -- far from
      whoever supplied it.
    - **Length**: measured as the number of **characters** of the canonical
      encoding (the same unit as the ``len(value)`` check on the string input
      path). Counting UTF-8 *bytes* would reject CJK text about 3x too early.

    The check runs the very operation that would fail downstream instead of a
    hand-written depth walk or length estimate, so it follows the serializers'
    limits automatically.

    **Both serializers are exercised.** Outbound values are serialized
    recursively by Pydantic, whose depth limit is platform-dependent and can be
    well below orjson's (``pydantic_core.to_json`` fails from about 98 nesting
    levels on Windows builds, and higher elsewhere); running the real
    serializer keeps the check equal to whatever limit applies where the code
    runs. Checking only orjson would accept a value that later fails at
    response time; checking only ``to_json`` would accept non-JSON types such
    as ``set`` (which it silently turns into an array while orjson rejects
    them). The union of both checks is enforced.

    :raises ValueError: If the value cannot be encoded (nested too deeply or
        contains an unsupported value), or its encoding exceeds
        ``MAX_JSON_LENGTH`` characters.
    """
    try:
        encoded = orjson.dumps(value)
    except orjson.JSONEncodeError as e:
        raise ValueError(f"JSON cannot be serialized (nested too deeply or contains an unsupported value): {e}") from e
    # UTF-8 byte count >= character count, so a value within the byte limit is
    # within the character limit. Only decode to count characters when the byte
    # count is over the limit.
    if len(encoded) > MAX_JSON_LENGTH:
        length = len(encoded.decode('utf-8'))
        if length > MAX_JSON_LENGTH:
            raise ValueError(f"JSON length exceeds limit: {length} > {MAX_JSON_LENGTH}")
    try:
        _ = pydantic_core.to_json(value)
    except PydanticSerializationError as e:
        raise ValueError(
            f"JSON is nested too deeply; outbound serialization would fail "
            + f"(beyond Pydantic's serializer depth limit on this platform): {e}"
        ) from e


def _parse_json_string(value: str, expected_type: type) -> dict[str, typing.Any] | list[typing.Any]:
    """
    Parse a JSON string.

    :param value: JSON string
    :param expected_type: Expected Python type (dict or list)
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


class JSON100K(dict[str, typing.Any]):
    """
    PostgreSQL JSONB type (object); the canonical encoding is at most 100K characters.

    - Behaves as ``dict[str, Any]`` in Python code
    - Inbound: accepts a JSON object (preferred) or a JSON string
    - Outbound: always an object (``model_dump`` python and json modes agree)
    - Stored as JSONB in PostgreSQL

    The 100K limit does not appear in the JSON Schema: it is measured on the
    canonical encoding, and JSON Schema cannot express that for an object
    (``maxLength`` applies to strings only). Document the limit in the field
    docstring; oversized values are rejected at validation time (see
    :func:`ensure_json_within_limits`).

    Usage::

        from sqlmodel_ext.field_types.dialects.postgresql import JSON100K

        class Project(SQLModelBase, UUIDTableBaseMixin, table=True):
            canvas: JSON100K
            '''Canvas data (JSONB object, canonical encoding <= 100K chars)'''
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
                # Native dict path -- what a deserialized request body takes.
                ensure_json_within_limits(value)
                return typing.cast(dict[str, typing.Any], value)
            if isinstance(value, str):
                result = typing.cast(dict[str, typing.Any], _parse_json_string(value, dict))
                # The raw string length was already checked before parsing (so a
                # huge input is never loaded); re-check the canonical encoding,
                # which is what gets stored, and the nesting depth.
                ensure_json_within_limits(result)
                return result
            raise TypeError(
                f"JSON100K accepts str or dict, not {type(value).__name__}"
            )

        dict_schema = core_schema.dict_schema(
            core_schema.str_schema(),
            core_schema.any_schema()
        )

        # JSON and Python inputs share one schema, so the inbound contract
        # ``anyOf[object, string]`` is exactly what the validator accepts.
        #
        # The serialization schema must be declared separately as a pure
        # object: without it Pydantic falls back to the validation schema and
        # would advertise an outbound ``string`` branch that can never occur.
        # The identity serializer is the only way to attach that return schema;
        # it returns the value unchanged.
        return core_schema.no_info_after_validator_function(
            validate,
            core_schema.union_schema([
                dict_schema,
                core_schema.str_schema(max_length=MAX_JSON_LENGTH),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value,
                info_arg=False,
                return_schema=dict_schema,
                when_used='always',
            ),
            metadata={'sa_type': JSONB},
        )


class JSONList100K(list[dict[str, typing.Any]]):
    """
    PostgreSQL JSONB type (array); the canonical encoding is at most 100K characters.

    - Behaves as ``list[dict[str, Any]]`` in Python code
    - Inbound: accepts a JSON array (preferred) or a JSON string
    - Outbound: always an array (``model_dump`` python and json modes agree)
    - Stored as JSONB in PostgreSQL

    The 100K limit does not appear in the JSON Schema, for the same reason as
    :class:`JSON100K`.

    Usage::

        from sqlmodel_ext.field_types.dialects.postgresql import JSONList100K

        class Conversation(SQLModelBase, UUIDTableBaseMixin, table=True):
            messages: JSONList100K
            '''Message list (JSONB array, canonical encoding <= 100K chars)'''
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
                # Native list path (see JSON100K's dict branch).
                ensure_json_within_limits(value)
                return typing.cast(list[dict[str, typing.Any]], value)
            if isinstance(value, str):
                result = typing.cast(list[dict[str, typing.Any]], _parse_json_string(value, list))
                # See JSON100K: re-check the canonical encoding and depth.
                ensure_json_within_limits(result)
                return result
            raise TypeError(
                f"JSONList100K accepts str or list, not {type(value).__name__}"
            )

        dict_schema = core_schema.dict_schema(
            core_schema.str_schema(),
            core_schema.any_schema()
        )
        list_schema = core_schema.list_schema(dict_schema)

        # Unified inbound schema + pure-array serialization schema (see JSON100K).
        return core_schema.no_info_after_validator_function(
            validate,
            core_schema.union_schema([
                list_schema,
                core_schema.str_schema(max_length=MAX_JSON_LENGTH),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value,
                info_arg=False,
                return_schema=list_schema,
                when_used='always',
            ),
            metadata={'sa_type': JSONB},
        )
