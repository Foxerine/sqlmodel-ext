"""
PostgreSQL pgvector integration with NumPy.

Provides the ``NumpyVector`` type that stores data as pgvector's ``Vector`` type
in the database while exposing it as ``numpy.ndarray`` in Python/Pydantic.

Implementation overview:

- ``NumpyVector[dims, dtype]`` returns a dynamically-created type class
- The type class carries a ``__sqlmodel_sa_type__`` attribute for SQLModel recognition
- ``__get_pydantic_core_schema__`` provides Pydantic validation logic
- Does not inherit from ``np.ndarray`` to avoid SQLModel type-resolution conflicts
"""
import ast
import base64
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from pgvector.sqlalchemy import Vector
from pydantic_core import core_schema
from sqlalchemy import TypeDecorator

from .exceptions import VectorDTypeError, VectorDecodeError, VectorDimensionError

# --- Type cache (avoid recreating types for the same dimensions) ---
_numpy_vector_type_cache: dict[tuple[int, type], type] = {}


# --- Metaclass for Type Factory ---
class _NumpyVectorMeta(type):
    """
    Metaclass for NumpyVector that implements a type-factory pattern.

    Caches type instances for identical parameters to avoid redundant creation.
    """

    def __getitem__(cls, params: tuple[int, type] | int):
        """
        Support ``NumpyVector[1024, np.float32]`` or ``NumpyVector[1024]`` syntax.

        :param params: ``(dimensions, dtype)`` or a single ``dimensions`` integer
        :returns: Dynamically-created type class with SQLAlchemy type info
        """
        # Normalize parameters
        if isinstance(params, int):
            dimensions = params
            dtype = np.float32
        elif isinstance(params, tuple) and len(params) == 2:
            dimensions, dtype = params
        else:
            raise TypeError(
                f"NumpyVector requires (dimensions, dtype) or dimensions, "
                f"got {params}"
            )

        # Validate dtype is a numpy dtype
        try:
            dtype_normalized = np.dtype(dtype).type
        except Exception as e:
            raise VectorDTypeError(
                f"Invalid numpy dtype: {dtype}. Error: {e}"
            ) from e

        # Check cache
        cache_key = (dimensions, dtype_normalized)
        if cache_key in _numpy_vector_type_cache:
            return _numpy_vector_type_cache[cache_key]

        # Create SQLAlchemy type instance
        sa_type = _NumpyVectorSQLAlchemyType(
            dimensions=dimensions,
            dtype=dtype_normalized
        )

        # Create handler instance
        handler = _NumpyVectorTypeHandler(
            dimensions=dimensions,
            dtype=dtype_normalized,
            sa_type=sa_type
        )

        # Create a standalone type class (not inheriting np.ndarray)
        # so SQLModel can recognize it properly
        class _NumpyVectorType:
            """
            Dynamically-created NumpyVector type.

            Does not inherit from np.ndarray; instead provides Pydantic schema
            for validation and conversion. SQLModel identifies the SQLAlchemy
            type via the ``__sqlmodel_sa_type__`` class attribute.
            """
            __sqlmodel_sa_type__ = sa_type

            @classmethod
            def __get_pydantic_core_schema__(cls, source_type, schema_handler):
                """Delegate to the handler's schema definition."""
                return handler.__get_pydantic_core_schema__(source_type, schema_handler)

            def __class_getitem__(cls, item):
                """Support generic syntax in type hints if needed."""
                return cls

        # Set a meaningful class name for debugging
        _NumpyVectorType.__name__ = f'NumpyVector_{dimensions}_{dtype_normalized.__name__}'
        _NumpyVectorType.__qualname__ = _NumpyVectorType.__name__

        # Cache and return
        _numpy_vector_type_cache[cache_key] = _NumpyVectorType
        return _NumpyVectorType


# --- Pydantic/SQLModel Integration Layer ---
class _NumpyVectorTypeHandler:
    """
    (Internal) Provides Pydantic/SQLAlchemy config for NumpyVector.

    Handles conversion between ``numpy.ndarray`` and ``pgvector.Vector``,
    as well as Pydantic validation.
    """

    def __init__(
        self,
        dimensions: int,
        dtype: type = np.float32,
        sa_type: '_NumpyVectorSQLAlchemyType | None' = None
    ):
        """
        :param dimensions: Fixed vector dimensions
        :param dtype: numpy data type (default: float32)
        :param sa_type: SQLAlchemy type instance (optional, for reuse)
        """
        self.dimensions = dimensions
        self.dtype = np.dtype(dtype).type
        self.sa_type = sa_type or _NumpyVectorSQLAlchemyType(
            dimensions=dimensions,
            dtype=dtype
        )

    def _validate_and_convert(self, value: Any) -> npt.NDArray:
        """
        Validate and convert the input value to a numpy array.

        Supported input formats:

        1. ``numpy.ndarray`` -- validated directly
        2. ``list`` / ``tuple`` -- converted to numpy array
        3. ``dict`` with base64 -- decoded to numpy array
           ``{"dtype": "float32", "shape": 1024, "data_b64": "..."}``
        4. ``pgvector.Vector`` instance -- lazy conversion from database load

        :param value: Input value
        :returns: Validated numpy array
        :raises VectorDimensionError: If dimensions do not match
        :raises VectorDTypeError: If dtype conversion fails
        :raises VectorDecodeError: If base64 decoding fails
        """
        # Handle pgvector.Vector (loaded from database)
        if hasattr(value, 'as_numpy'):
            value = value.as_numpy()
        elif isinstance(value, str):
            # May be pgvector's string representation "[1.0, 2.0, ...]"
            try:
                value = ast.literal_eval(value)
            except Exception as e:
                raise VectorDecodeError(
                    f"Failed to parse vector string: {value}"
                ) from e

        # Handle base64-encoded dict format
        if isinstance(value, dict):
            try:
                encoded_dtype = value['dtype']
                shape = value['shape']
                data_b64 = value['data_b64']

                # Decode base64
                data_bytes = base64.b64decode(data_b64)

                # Reconstruct numpy array from bytes
                arr = np.frombuffer(data_bytes, dtype=encoded_dtype)

                # Validate shape
                if isinstance(shape, int):
                    expected_size = shape
                elif isinstance(shape, (list, tuple)):
                    expected_size = int(np.prod(shape))
                else:
                    raise VectorDecodeError(f'Invalid shape format: {shape}')

                if arr.size != expected_size:
                    raise VectorDecodeError(
                        f'Shape mismatch: expected {expected_size}, '
                        f'got {arr.size}'
                    )

                # Reshape to target shape
                if isinstance(shape, (list, tuple)):
                    arr = arr.reshape(shape)

                value = arr

            except KeyError as e:
                raise VectorDecodeError(
                    f'Base64 dict missing required key: {e}'
                ) from e
            except VectorDecodeError:
                raise
            except Exception as e:
                raise VectorDecodeError(
                    f'Failed to decode base64 vector: {e}'
                ) from e

        # Convert to numpy array
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value, dtype=self.dtype)
            except Exception as e:
                raise VectorDTypeError(
                    f'Failed to convert {type(value)} to numpy array: {e}'
                ) from e
        else:
            # Already a numpy array -- check dtype
            if value.dtype != self.dtype:
                warnings.warn(
                    f'Converting vector dtype from {value.dtype} to {self.dtype}',
                    UserWarning,
                    stacklevel=2
                )
                try:
                    value = value.astype(self.dtype)
                except Exception as e:
                    raise VectorDTypeError(
                        f'Failed to convert dtype from {value.dtype} '
                        f'to {self.dtype}: {e}'
                    ) from e

        # Validate dimensions (strict)
        if value.ndim != 1:
            raise VectorDimensionError(
                f'Vector must be 1-dimensional, got shape {value.shape}'
            )

        if value.size != self.dimensions:
            raise VectorDimensionError(
                f'Vector dimension mismatch: expected {self.dimensions}, '
                f'got {value.size}'
            )

        return value

    def __get_pydantic_core_schema__(self, source_type, handler):
        """Pydantic v2 core schema definition."""

        def validate_from_any(value: Any) -> npt.NDArray:
            """Pydantic validation function."""
            return self._validate_and_convert(value)

        def serialize_to_json(value: npt.NDArray) -> dict[str, Any]:
            """
            Serialize to a JSON-safe base64 format.

            Format: ``{"dtype": "float32", "shape": 1024, "data_b64": "..."}``
            """
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=self.dtype)

            return {
                'dtype': str(value.dtype),
                'shape': value.shape[0] if value.ndim == 1 else list(value.shape),
                'data_b64': base64.b64encode(value.tobytes()).decode('ascii')
            }

        python_schema = core_schema.with_info_plain_validator_function(
            lambda v, _: validate_from_any(v)
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema([
                core_schema.dict_schema(),
                core_schema.no_info_plain_validator_function(validate_from_any)
            ]),
            python_schema=python_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_to_json,
                when_used='json'
            ),
            metadata={
                'sa_type': self.sa_type
            }
        )


# --- SQLAlchemy TypeDecorator ---
class _NumpyVectorSQLAlchemyType(TypeDecorator):
    """
    SQLAlchemy type decorator mapping ``numpy.ndarray`` to ``pgvector.Vector``.

    Uses the TypeDecorator pattern to wrap pgvector's Vector type:

    - Database storage: pgvector's vector type
    - Python representation: numpy.ndarray
    - Automatic caching: managed by SQLAlchemy
    - Vector method proxying: cosine_distance, l2_distance, max_inner_product, etc.
    """

    impl = Vector
    cache_ok = True

    def __init__(self, dimensions: int, dtype: type = np.float32):
        """
        :param dimensions: Vector dimensions
        :param dtype: numpy data type
        """
        self.dimensions = dimensions
        self.dtype = np.dtype(dtype).type
        super().__init__()

    def load_dialect_impl(self, dialect):
        """
        Load the Vector implementation for the PostgreSQL dialect.

        Ensures the Vector instance is correctly created with the configured
        dimensions, preserving all pgvector vector methods.
        """
        return Vector(dim=self.dimensions)

    def process_bind_param(self, value: npt.NDArray | None, dialect) -> list[float] | None:
        """
        Python -> Database: convert ``numpy.ndarray`` to ``list[float]``.
        """
        if value is None:
            return None

        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=self.dtype)

        return value.tolist()

    def process_result_value(self, value: Any, dialect) -> npt.NDArray | None:
        """
        Database -> Python: convert ``pgvector.Vector`` to ``numpy.ndarray``.
        """
        if value is None:
            return None

        if isinstance(value, list):
            return np.array(value, dtype=self.dtype)

        try:
            return np.array(value, dtype=self.dtype)
        except Exception as e:
            raise VectorDecodeError(
                f'Failed to convert database value to numpy array: {e}'
            ) from e


# --- Public, User-Facing Type ---
class NumpyVector(metaclass=_NumpyVectorMeta):
    """
    PostgreSQL pgvector integration with NumPy.

    Stored as pgvector's ``Vector`` type in the database, exposed as
    ``numpy.ndarray`` in Pydantic/Python, supporting fixed-dimension
    vector data.

    Usage::

        from sqlmodel import Field
        from sqlmodel_ext.field_types.dialects.postgresql import NumpyVector
        import numpy as np

        # Direct usage
        class SpeakerInfo(SQLModelBase, UUIDTableBaseMixin, table=True):
            embedding: NumpyVector[1024, np.float32] = Field(...)

        # With Annotated
        from typing import Annotated
        EmbeddingVector = Annotated[
            np.ndarray,
            NumpyVector[1024, np.float32]
        ]

        class SpeakerInfo(SQLModelBase, UUIDTableBaseMixin, table=True):
            embedding: EmbeddingVector = Field(...)

        # Type alias
        EmbeddingVector = NumpyVector[1024, np.float32]

        class SpeakerInfo(SQLModelBase, UUIDTableBaseMixin, table=True):
            embedding: EmbeddingVector = Field(...)

    API Serialization Format:
        JSON serialization uses base64 encoding::

            {
                "dtype": "float32",
                "shape": 1024,
                "data_b64": "AAABAAA..."
            }

    Vector Search::

        from sqlalchemy import select

        # L2 distance
        stmt = select(SpeakerInfo).order_by(
            SpeakerInfo.embedding.l2_distance(query_vector)
        ).limit(10)

        # Cosine distance
        stmt = select(SpeakerInfo).order_by(
            SpeakerInfo.embedding.cosine_distance(query_vector)
        ).limit(10)

        # Max inner product
        stmt = select(SpeakerInfo).order_by(
            SpeakerInfo.embedding.max_inner_product(query_vector)
        ).limit(10)
    """

    def __init__(self):
        raise TypeError(
            "NumpyVector cannot be instantiated directly. "
            "Use NumpyVector[dimensions, dtype] syntax instead."
        )
