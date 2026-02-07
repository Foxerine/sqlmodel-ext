"""
sqlmodel_ext.field_types -- Reusable type aliases and custom types for SQLModel.

Provides constrained string/numeric types, path types, URL types, and IP address types,
all compatible with Pydantic validation and SQLAlchemy column mapping.
"""
from pathlib import Path
from typing import Annotated, TypeAlias

from sqlmodel import Field

from ._internal.path import _DirectoryPathHandler, _FilePathHandler
from .ip_address import IPAddress
from .mixins import ModuleNameMixin
from .url import HttpUrl, SafeHttpUrl, Url, WebSocketUrl

# Re-export SSRF utilities
from ._ssrf import UnsafeURLError, validate_not_private_host

# ---------------------------------------------------------------------------
#  Public, Database-Agnostic Types
# ---------------------------------------------------------------------------

DirectoryPathType = Annotated[Path, _DirectoryPathHandler]
"""
A directory path type compatible with Pydantic and SQLModel.

Validates that the path should not contain a file extension,
while behaving as a ``pathlib.Path`` in Python code.
"""

FilePathType = Annotated[Path, _FilePathHandler]
"""
A file path type compatible with Pydantic and SQLModel.

Validates that the path must contain a filename component,
while behaving as a ``pathlib.Path`` in Python code.
"""


# ---------------------------------------------------------------------------
#  Field Constraint Type Aliases (Annotated Style)
# ---------------------------------------------------------------------------

# String length constraints
Str24: TypeAlias = Annotated[str, Field(max_length=24)]
"""24-character string field"""

Str32: TypeAlias = Annotated[str, Field(max_length=32)]
"""32-character string field"""

Str36: TypeAlias = Annotated[str, Field(max_length=36)]
"""36-character string field (UUID standard format length)"""

Str48: TypeAlias = Annotated[str, Field(max_length=48)]
"""48-character string field"""

Str64: TypeAlias = Annotated[str, Field(max_length=64)]
"""64-character string field"""

Str100: TypeAlias = Annotated[str, Field(max_length=100)]
"""100-character string field"""

Str128: TypeAlias = Annotated[str, Field(max_length=128)]
"""128-character string field"""

Str255: TypeAlias = Annotated[str, Field(max_length=255)]
"""255-character string field"""

Str256: TypeAlias = Annotated[str, Field(max_length=256)]
"""256-character string field"""

Text1K: TypeAlias = Annotated[str, Field(max_length=1000)]
"""1000-character text field"""

Text1024: TypeAlias = Annotated[str, Field(max_length=1024)]
"""1024-character text field"""

Text2K: TypeAlias = Annotated[str, Field(max_length=2000)]
"""2000-character text field"""

Text2500: TypeAlias = Annotated[str, Field(max_length=2500)]
"""2500-character text field"""

Text3K: TypeAlias = Annotated[str, Field(max_length=3000)]
"""3000-character text field"""

Text10K: TypeAlias = Annotated[str, Field(max_length=10000)]
"""10000-character text field"""

Text60K: TypeAlias = Annotated[str, Field(max_length=60000)]
"""60000-character text field"""

Text64K: TypeAlias = Annotated[str, Field(max_length=65536)]
"""65536-character text field"""

Text100K: TypeAlias = Annotated[str, Field(max_length=100000)]
"""100000-character text field"""

# Numeric range constraints
Port: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
"""Port number (1-65535)"""

Percentage: TypeAlias = Annotated[int, Field(ge=0, le=100)]
"""Percentage (0-100)"""

PositiveInt: TypeAlias = Annotated[int, Field(ge=1)]
"""Positive integer (>=1)"""

NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
"""Non-negative integer (>=0)"""

PositiveFloat: TypeAlias = Annotated[float, Field(gt=0.0)]
"""Positive float (>0)"""
