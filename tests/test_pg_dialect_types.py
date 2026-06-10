"""
PostgreSQL dialect field-type tests -- no real PostgreSQL required.

Covers ``sqlmodel_ext.field_types.dialects.postgresql``:

1. ``Array[T]`` / ``Array[T, N]``: SA ``ARRAY`` item-type dispatch
   (str/int/dict/UUID/Enum), unsupported inner types raise ``TypeError``,
   Pydantic max-length and item-type validation on a non-table model.
2. ``JSON100K`` / ``JSONList100K``: dict/list passthrough, JSON-string
   parsing, the 100K input-length limit (boundary: exactly 100_000 passes,
   100_001 fails), wrong-JSON-root rejection, invalid JSON rejection,
   serialization to a JSON string, and the JSONB sa_type metadata.
3. ``NumpyVector``: type-factory caching, invalid parameters, Pydantic
   validation of list/tuple/ndarray/string/base64-dict inputs, dimension and
   dtype enforcement (``VectorDimensionError`` / ``VectorDTypeError`` /
   ``VectorDecodeError`` from ``exceptions.py``), JSON base64 round-trip, and
   the ``_NumpyVectorSQLAlchemyType`` TypeDecorator
   ``process_bind_param`` / ``process_result_value`` against a real
   PostgreSQL dialect instance.

Only non-table models are used here, so nothing leaks into the shared
``SQLModel.metadata`` (PG-only column types would break conftest's SQLite
``create_all``).
"""
from __future__ import annotations

import json
import typing
from enum import StrEnum
from uuid import UUID

import numpy as np
import orjson
import pytest
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.dialects.postgresql.base import PGDialect

from sqlmodel_ext import SQLModelBase
from sqlmodel_ext.field_types.dialects.postgresql import (
    Array,
    JSON100K,
    JSONList100K,
    NumpyVector,
)
from sqlmodel_ext.field_types.dialects.postgresql.exceptions import (
    VectorDecodeError,
    VectorDimensionError,
    VectorDTypeError,
    VectorError,
)
from sqlmodel_ext.field_types.dialects.postgresql.jsonb_types import MAX_JSON_LENGTH

pg_dialect = PGDialect()


class FtPgScopeEnum(StrEnum):
    read = "scope:read"
    write = "scope:write"


class FtArrayModel(SQLModelBase):
    tags: Array[str, 2] = []
    nums: Array[int] = []
    scopes: Array[FtPgScopeEnum] = []


class FtJsonModel(SQLModelBase):
    canvas: JSON100K = {}
    messages: JSONList100K = []


Vec4 = NumpyVector[4, np.float32]


class FtVectorModel(SQLModelBase):
    embedding: Vec4


def _array_handler(alias: typing.Any):
    """Extract the _ArrayTypeHandler from an Array[...] Annotated alias."""
    return typing.get_args(alias)[1]


# ============================================================
# 1. Array[T]
# ============================================================

class TestArray:
    def test_sa_item_type_dispatch(self) -> None:
        cases = [
            (Array[str], sa.String),
            (Array[int], sa.Integer),
            (Array[dict], JSONB),
            (Array[UUID], PG_UUID),
            (Array[FtPgScopeEnum], sa.Enum),
        ]
        for alias, expected_item_type in cases:
            sa_type = _array_handler(alias).sa_array_type
            assert isinstance(sa_type, sa.ARRAY)
            assert isinstance(sa_type.item_type, expected_item_type)

    def test_enum_array_uses_member_values(self) -> None:
        """PG ENUM must be populated with .value strings, not Python names."""
        enum_type = _array_handler(Array[FtPgScopeEnum]).sa_array_type.item_type
        assert set(enum_type.enums) == {"scope:read", "scope:write"}

    def test_unsupported_inner_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported inner type"):
            Array[float]

    def test_max_length_boundary(self) -> None:
        assert FtArrayModel(tags=["a", "b"]).tags == ["a", "b"]
        with pytest.raises(ValidationError):
            FtArrayModel(tags=["a", "b", "c"])

    def test_unbounded_array_accepts_many(self) -> None:
        m = FtArrayModel(nums=list(range(500)))
        assert len(m.nums) == 500

    def test_item_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            FtArrayModel(tags=[123])
        with pytest.raises(ValidationError):
            FtArrayModel(nums=["not-an-int"])

    def test_enum_array_validates_values(self) -> None:
        m = FtArrayModel(scopes=[FtPgScopeEnum.read, "scope:write"])
        assert list(m.scopes) == ["scope:read", "scope:write"]
        with pytest.raises(ValidationError):
            FtArrayModel(scopes=["bogus:scope"])


# ============================================================
# 2. JSON100K / JSONList100K
# ============================================================

def _json_str_of_length(total: int) -> str:
    """Build a valid JSON object string of exactly ``total`` characters."""
    overhead = len('{"a": ""}')
    s = '{"a": "' + "x" * (total - overhead) + '"}'
    assert len(s) == total
    return s


class TestJson100K:
    def test_dict_passthrough(self) -> None:
        data = {"a": 1, "nested": {"b": [1, 2]}}
        assert FtJsonModel(canvas=data).canvas == data

    def test_json_string_parsed(self) -> None:
        m = FtJsonModel(canvas='{"k": [1, 2, 3]}')
        assert m.canvas == {"k": [1, 2, 3]}

    def test_string_at_exact_limit_passes(self) -> None:
        s = _json_str_of_length(MAX_JSON_LENGTH)
        m = FtJsonModel(canvas=s)
        assert len(m.canvas["a"]) == MAX_JSON_LENGTH - len('{"a": ""}')

    def test_string_over_limit_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FtJsonModel(canvas=_json_str_of_length(MAX_JSON_LENGTH + 1))

    def test_invalid_json_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Invalid JSON"):
            FtJsonModel(canvas="{not json")

    def test_json_array_string_rejected_for_object_type(self) -> None:
        with pytest.raises(ValidationError, match="object"):
            FtJsonModel(canvas="[1, 2, 3]")

    def test_serializes_to_json_string(self) -> None:
        m = FtJsonModel(canvas={"a": 1})
        dumped = m.model_dump()
        assert isinstance(dumped["canvas"], str)
        assert orjson.loads(dumped["canvas"]) == {"a": 1}
        # json mode: the field is a JSON string embedded in the payload
        parsed = json.loads(m.model_dump_json())
        assert isinstance(parsed["canvas"], str)
        assert orjson.loads(parsed["canvas"]) == {"a": 1}

    def test_sa_type_metadata_is_jsonb(self) -> None:
        schema = JSON100K.__get_pydantic_core_schema__(JSON100K, None)
        assert schema["metadata"]["sa_type"] is JSONB


class TestJsonList100K:
    def test_list_passthrough(self) -> None:
        data = [{"role": "user"}, {"role": "assistant"}]
        assert FtJsonModel(messages=data).messages == data

    def test_json_string_parsed(self) -> None:
        m = FtJsonModel(messages='[{"a": 1}]')
        assert m.messages == [{"a": 1}]

    def test_json_object_string_rejected_for_list_type(self) -> None:
        with pytest.raises(ValidationError, match="array"):
            FtJsonModel(messages='{"a": 1}')

    def test_string_over_limit_rejected(self) -> None:
        oversized = '[{"a": "' + "x" * MAX_JSON_LENGTH + '"}]'
        with pytest.raises(ValidationError):
            FtJsonModel(messages=oversized)

    def test_serializes_to_json_string(self) -> None:
        m = FtJsonModel(messages=[{"a": 1}])
        dumped = m.model_dump()
        assert isinstance(dumped["messages"], str)
        assert orjson.loads(dumped["messages"]) == [{"a": 1}]


# ============================================================
# 3. NumpyVector
# ============================================================

class TestNumpyVectorFactory:
    def test_default_dtype_is_float32_and_cached(self) -> None:
        assert NumpyVector[4] is NumpyVector[4, np.float32]
        assert NumpyVector[4] is Vec4

    def test_distinct_params_get_distinct_types(self) -> None:
        assert NumpyVector[4, np.float64] is not Vec4
        assert NumpyVector[8] is not NumpyVector[4]

    def test_debug_friendly_class_name(self) -> None:
        assert Vec4.__name__ == "NumpyVector_4_float32"

    def test_direct_instantiation_forbidden(self) -> None:
        with pytest.raises(TypeError, match="cannot be instantiated"):
            NumpyVector()

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(VectorDTypeError):
            NumpyVector[4, "definitely-not-a-dtype"]

    def test_invalid_params_raise_type_error(self) -> None:
        with pytest.raises(TypeError):
            NumpyVector[1, 2, 3]

    def test_exception_hierarchy(self) -> None:
        for exc in (VectorDimensionError, VectorDecodeError, VectorDTypeError):
            assert issubclass(exc, VectorError)


class TestNumpyVectorValidation:
    def test_list_input_converted_to_ndarray(self) -> None:
        m = FtVectorModel(embedding=[1, 2, 3, 4])
        assert isinstance(m.embedding, np.ndarray)
        assert m.embedding.dtype == np.float32
        np.testing.assert_array_equal(m.embedding, np.array([1, 2, 3, 4], dtype=np.float32))

    def test_tuple_input_accepted(self) -> None:
        m = FtVectorModel(embedding=(0.5, 1.5, 2.5, 3.5))
        assert m.embedding.shape == (4,)

    def test_ndarray_input_passthrough(self) -> None:
        arr = np.array([1, 2, 3, 4], dtype=np.float32)
        m = FtVectorModel(embedding=arr)
        np.testing.assert_array_equal(m.embedding, arr)

    def test_wrong_dtype_ndarray_warns_and_converts(self) -> None:
        arr = np.array([1, 2, 3, 4], dtype=np.float64)
        with pytest.warns(UserWarning, match="Converting vector dtype"):
            m = FtVectorModel(embedding=arr)
        assert m.embedding.dtype == np.float32

    def test_pgvector_string_repr_parsed(self) -> None:
        m = FtVectorModel(embedding="[1.0, 2.0, 3.0, 4.0]")
        np.testing.assert_array_equal(
            m.embedding, np.array([1, 2, 3, 4], dtype=np.float32)
        )

    def test_unparseable_string_raises_decode_error(self) -> None:
        with pytest.raises(VectorDecodeError):
            FtVectorModel(embedding="not a vector at all")

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(VectorDimensionError, match="dimension mismatch"):
            FtVectorModel(embedding=[1, 2, 3])

    def test_multidimensional_input_raises(self) -> None:
        with pytest.raises(VectorDimensionError, match="1-dimensional"):
            FtVectorModel(embedding=np.ones((2, 2), dtype=np.float32))

    def test_base64_dict_input(self) -> None:
        import base64

        arr = np.array([1, 2, 3, 4], dtype=np.float32)
        payload = {
            "dtype": "float32",
            "shape": 4,
            "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
        m = FtVectorModel(embedding=payload)
        np.testing.assert_array_equal(m.embedding, arr)

    def test_base64_dict_missing_key_raises(self) -> None:
        with pytest.raises(VectorDecodeError, match="missing required key"):
            FtVectorModel(embedding={"dtype": "float32", "shape": 4})

    def test_base64_dict_shape_mismatch_raises(self) -> None:
        import base64

        arr = np.array([1, 2], dtype=np.float32)
        payload = {
            "dtype": "float32",
            "shape": 4,
            "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
        with pytest.raises(VectorDecodeError, match="Shape mismatch"):
            FtVectorModel(embedding=payload)

    def test_json_serialization_roundtrip(self) -> None:
        arr = np.array([0.25, -1.5, 3.75, 42.0], dtype=np.float32)
        m = FtVectorModel(embedding=arr)
        payload = json.loads(m.model_dump_json())
        assert payload["embedding"]["dtype"] == "float32"
        assert payload["embedding"]["shape"] == 4
        restored = FtVectorModel.model_validate_json(m.model_dump_json())
        np.testing.assert_array_equal(restored.embedding, arr)
        assert restored.embedding.dtype == np.float32


class TestNumpyVectorSQLAlchemyType:
    sa_type = Vec4.__sqlmodel_sa_type__

    def test_dialect_impl_is_pgvector_with_dimensions(self) -> None:
        impl = self.sa_type.load_dialect_impl(pg_dialect)
        assert isinstance(impl, Vector)
        assert impl.dim == 4

    def test_process_bind_param_ndarray_to_list(self) -> None:
        arr = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        bound = self.sa_type.process_bind_param(arr, pg_dialect)
        assert bound == [1.5, 2.5, 3.5, 4.5]
        assert all(isinstance(x, float) for x in bound)

    def test_process_bind_param_plain_list(self) -> None:
        assert self.sa_type.process_bind_param([1, 2, 3, 4], pg_dialect) == [1.0, 2.0, 3.0, 4.0]

    def test_process_bind_param_none(self) -> None:
        assert self.sa_type.process_bind_param(None, pg_dialect) is None

    def test_process_result_value_list_to_ndarray(self) -> None:
        out = self.sa_type.process_result_value([1.0, 2.0, 3.0, 4.0], pg_dialect)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype=np.float32))

    def test_process_result_value_none(self) -> None:
        assert self.sa_type.process_result_value(None, pg_dialect) is None

    def test_bind_then_result_roundtrip(self) -> None:
        arr = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        bound = self.sa_type.process_bind_param(arr, pg_dialect)
        restored = self.sa_type.process_result_value(bound, pg_dialect)
        np.testing.assert_array_almost_equal(restored, arr)

    def test_process_result_value_bad_value_raises(self) -> None:
        with pytest.raises(VectorDecodeError):
            self.sa_type.process_result_value(object(), pg_dialect)
