"""
sqlmodel_ext.field_types.dialects.postgresql -- PostgreSQL-specific type support.

Provides PostgreSQL-specific types for SQLModel:

- ``Array[T]`` -- PostgreSQL ARRAY type with optional length limit
- ``JSON100K`` -- JSONB dict type with 100K character limit (requires ``orjson``)
- ``JSONList100K`` -- JSONB list type with 100K character limit (requires ``orjson``)
- ``NumpyVector[dims, dtype]`` -- pgvector + NumPy integration (requires ``numpy``, ``pgvector``)
- Vector exception hierarchy

Install extras for full support::

    pip install sqlmodel-ext[postgresql]   # Array + JSON100K/JSONList100K
    pip install sqlmodel-ext[pgvector]     # NumpyVector (includes postgresql)
"""
# Array is always available (uses only sqlalchemy.dialects.postgresql)
from .array import Array

# Exceptions are always available (pure Python)
from .exceptions import (
    VectorDecodeError,
    VectorDimensionError,
    VectorDTypeError,
    VectorError,
)

# JSONB types require orjson; install sqlmodel-ext[postgresql]
try:
    from .jsonb_types import JSON100K, JSONList100K
except ImportError:
    pass

# NumpyVector requires numpy + pgvector; install sqlmodel-ext[pgvector]
try:
    from .numpy_vector import NumpyVector
except ImportError:
    pass
