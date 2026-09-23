"""Path type handlers for Pydantic + SQLAlchemy integration.

Internal protocol, not part of the public API: the handler classes carry public
names only because :mod:`sqlmodel_ext.field_types` imports them across module
boundaries to build ``FilePathType`` / ``DirectoryPathType``; the module itself
stays private (``_internal``).
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, override

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from sqlalchemy.engine import Dialect
from sqlalchemy.types import String, TypeDecorator, TypeEngine


class _PathAsSQLString(TypeDecorator[Path]):
    """(Internal) Converts Path <-> str for the database."""
    impl: TypeEngine[Any] | type[TypeEngine[Any]] = String
    cache_ok: bool | None = True
    @override
    def process_bind_param(self, v: Path | None, d: Dialect) -> str | None: return str(v) if v else None
    @override
    def process_result_value(self, v: Any | None, d: Dialect) -> Path | None: return Path(v) if v else None


class _BasePathHandler(ABC):
    """(Internal) Base class for single-value type handlers like Path."""
    # Must be an *instance*, not the class: the metaclass extracts this value
    # from the core-schema metadata and hands it straight to the column
    # builder, which expects a TypeEngine instance (a bare class silently
    # falls back to AutoString, dropping the Path<->str result_processor).
    sa_type: TypeDecorator[Path] = _PathAsSQLString()

    @classmethod
    @abstractmethod
    def _validate(cls, value: Any) -> Any:
        """Subclasses must implement this to provide specific validation logic."""
        raise NotImplementedError

    @classmethod
    def __get_pydantic_core_schema__(cls, s: Any, h: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        validator = core_schema.no_info_plain_validator_function(cls._validate)
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=validator,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
            metadata={'sa_type': cls.sa_type}
        )


class FilePathHandler(_BasePathHandler):
    """(Internal) Validates that the path includes a filename."""
    @classmethod
    @override
    def _validate(cls, value: Any) -> Path:
        path = Path(str(value))
        if not path.name or path.name in ('.', '..'):
            raise ValueError(f"Path '{path}' must contain a valid filename component.")
        return path


class DirectoryPathHandler(_BasePathHandler):
    """(Internal) Validates that the path does not include a file extension."""
    @classmethod
    @override
    def _validate(cls, value: Any) -> Path:
        path = Path(str(value))
        if path.suffix:
            raise ValueError(f"Directory path '{path}' should not contain a file extension.")
        return path
