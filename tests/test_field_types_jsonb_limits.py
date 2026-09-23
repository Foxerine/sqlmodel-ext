"""
JSON100K / JSONList100K limits and schema contract -- no database required.

Covers ``sqlmodel_ext.field_types.dialects.postgresql.jsonb_types``:

1. ``ensure_json_within_limits``: the 100K limit applies to **every** input
   form (JSON string, dict, list, ``model_validate_json`` with a nested
   object), measured in characters of the canonical encoding, with exact
   boundaries.
2. Nesting depth: values that orjson or Pydantic cannot serialize are rejected
   at validation time instead of failing later at response time.
3. Schema contract: inbound ``anyOf[object|array, string]``, outbound pure
   object / array.
"""
from __future__ import annotations

from collections.abc import Callable

import orjson
import pydantic_core
import pytest
from pydantic import ValidationError

from sqlmodel_ext import SQLModelBase
from sqlmodel_ext.field_types.dialects.postgresql import (
    JSON100K,
    JSONList100K,
    ensure_json_within_limits,
)
from sqlmodel_ext.field_types.dialects.postgresql.jsonb_types import MAX_JSON_LENGTH


class FtJsonLimitDictModel(SQLModelBase):
    data: JSON100K


class FtJsonLimitListModel(SQLModelBase):
    items: JSONList100K


def _serializes(serializer: Callable[[object], bytes], value: object) -> bool:
    try:
        _ = serializer(value)
    except (orjson.JSONEncodeError, pydantic_core.PydanticSerializationError):
        return False
    return True


def _deep_dict(depth: int) -> dict[str, object]:
    value: dict[str, object] = {"leaf": 1}
    for _ in range(depth):
        value = {"d": value}
    return value


def _oversized() -> dict[str, str]:
    """Encodes to ~553K characters, 5.5x over the limit."""
    return {f"k{i}": "x" * 100 for i in range(5000)}


# ============================================================
# 1. Length limit covers every input form
# ============================================================

class TestLengthLimitCoversAllInputForms:
    def test_dict_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="length exceeds limit"):
            FtJsonLimitDictModel(data=_oversized())

    def test_list_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="length exceeds limit"):
            FtJsonLimitListModel(items=[_oversized()])

    def test_model_validate_json_with_nested_object_rejected(self) -> None:
        # A JSON *object* in the body takes the dict branch, not the string one.
        body = orjson.dumps({"data": _oversized()}).decode("utf-8")
        with pytest.raises(ValidationError, match="length exceeds limit"):
            FtJsonLimitDictModel.model_validate_json(body)

    def test_boundary_at_limit_accepted(self) -> None:
        payload = {"k": "x" * (MAX_JSON_LENGTH - len('{"k":""}'))}
        assert len(orjson.dumps(payload)) == MAX_JSON_LENGTH
        assert FtJsonLimitDictModel(data=payload).data == payload

    def test_boundary_one_over_limit_rejected(self) -> None:
        payload = {"k": "x" * (MAX_JSON_LENGTH - len('{"k":""}') + 1)}
        assert len(orjson.dumps(payload)) == MAX_JSON_LENGTH + 1
        with pytest.raises(ValidationError, match="length exceeds limit"):
            FtJsonLimitDictModel(data=payload)

    def test_multibyte_measured_in_characters_not_bytes(self) -> None:
        payload = {"t": chr(0x4E2D) * 40_000}  # CJK: 3 UTF-8 bytes per character
        encoded = orjson.dumps(payload)
        assert len(encoded) > MAX_JSON_LENGTH                  # bytes over the limit
        assert len(encoded.decode("utf-8")) < MAX_JSON_LENGTH  # characters within it
        assert FtJsonLimitDictModel(data=payload).data == payload

    def test_direct_call_on_valid_values(self) -> None:
        ensure_json_within_limits({"a": [1, 2, {"b": None}]})
        ensure_json_within_limits([{"a": 1}])

    def test_direct_call_rejects_oversized(self) -> None:
        with pytest.raises(ValueError, match="length exceeds limit"):
            ensure_json_within_limits(_oversized())


# ============================================================
# 2. Nesting depth / encodability
# ============================================================

class TestDeepNestingRejectedAtInput:
    def test_deep_json_string_rejected(self) -> None:
        deep_string = '{"d":' * 400 + "{}" + "}" * 400
        with pytest.raises(ValidationError, match="cannot be serialized"):
            FtJsonLimitDictModel(data=deep_string)

    def test_deep_dict_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be serialized"):
            FtJsonLimitDictModel(data=_deep_dict(400))

    def test_deep_list_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be serialized"):
            FtJsonLimitListModel(items=[_deep_dict(400)])

    def test_depth_beyond_pydantic_limit_rejected(self) -> None:
        # pydantic_core's recursion limit is platform-dependent (about 98
        # levels on Windows builds, higher elsewhere), so measure it here
        # instead of hard-coding a depth that only exceeds it on one platform.
        pydantic_limit = next(
            (depth for depth in range(1, 1000) if not _serializes(pydantic_core.to_json, _deep_dict(depth))),
            None,
        )
        if pydantic_limit is None or not _serializes(orjson.dumps, _deep_dict(pydantic_limit)):
            pytest.skip("on this platform pydantic_core is not stricter than orjson; the case cannot occur")
        # orjson CAN encode this depth, so the rejection must come from the
        # Pydantic serializer check -- otherwise this value would be accepted
        # and fail only when the response is serialized.
        value = _deep_dict(pydantic_limit)
        with pytest.raises(ValidationError, match="nested too deeply"):
            FtJsonLimitDictModel(data=value)

    def test_non_json_value_rejected(self) -> None:
        # pydantic_core.to_json silently turns a set into an array; orjson
        # rejects it, and that rejection must be kept.
        with pytest.raises(ValidationError, match="cannot be serialized"):
            FtJsonLimitDictModel(data={"s": {1, 2}})

    def test_reasonable_depth_accepted_and_dumpable(self) -> None:
        value = _deep_dict(50)
        m = FtJsonLimitDictModel(data=value)
        assert m.model_dump(mode="json")["data"] == value

    def test_wide_shallow_value_accepted(self) -> None:
        value = {f"k{i}": {"text": "x" * 80, "n": i} for i in range(700)}
        assert FtJsonLimitDictModel(data=value).data == value


# ============================================================
# 3. Schema contract: inbound two forms, outbound object only
# ============================================================

def _types_in(schema: dict[str, object]) -> set[str]:
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        return {t for m in any_of if isinstance(m, dict) and isinstance(t := m.get("type"), str)}
    t = schema.get("type")
    return {t} if isinstance(t, str) else set()


class TestSchemaContract:
    def test_json100k_inbound_both_outbound_object_only(self) -> None:
        props_in = FtJsonLimitDictModel.model_json_schema(mode="validation")["properties"]
        props_out = FtJsonLimitDictModel.model_json_schema(mode="serialization")["properties"]
        assert _types_in(props_in["data"]) == {"object", "string"}
        assert _types_in(props_out["data"]) == {"object"}

    def test_jsonlist100k_inbound_both_outbound_array_only(self) -> None:
        props_in = FtJsonLimitListModel.model_json_schema(mode="validation")["properties"]
        props_out = FtJsonLimitListModel.model_json_schema(mode="serialization")["properties"]
        assert _types_in(props_in["items"]) == {"array", "string"}
        assert _types_in(props_out["items"]) == {"array"}

    def test_json_string_input_via_model_validate_json(self) -> None:
        m = FtJsonLimitDictModel.model_validate_json('{"data": "{\\"k\\": \\"v\\"}"}')
        assert m.data == {"k": "v"}
        assert isinstance(m.model_dump(mode="json")["data"], dict)
