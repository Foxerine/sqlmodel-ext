"""Path type handlers for Pydantic + SQLAlchemy integration."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic_core import core_schema
from sqlalchemy.types import String, TypeDecorator


class _PathAsSQLString(TypeDecorator):
    """(Internal) Converts Path <-> str for the database."""
    impl = String
    cache_ok = True
    def process_bind_param(self, v, d): return str(v) if v else None
    def process_result_value(self, v, d): return Path(v) if v else None


class _BasePathHandler(ABC):
    """(Internal) Base class for single-value type handlers like Path."""
    sa_type: TypeDecorator = _PathAsSQLString

    @classmethod
    @abstractmethod
    def _validate(cls, value: Any) -> Any:
        """Subclasses must implement this to provide specific validation logic."""
        raise NotImplementedError

    @classmethod
    def __get_pydantic_core_schema__(cls, s, h):
        validator = core_schema.no_info_plain_validator_function(cls._validate)
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=validator,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
            metadata={'sa_type': cls.sa_type}
        )


class _FilePathHandler(_BasePathHandler):
    """(Internal) Validates that the path includes a filename."""
    @classmethod
    def _validate(cls, value: Any) -> Path:
        path = Path(str(value))
        if not path.name or path.name in ('.', '..'):
            raise ValueError(f"Path '{path}' must contain a valid filename component.")
        return path


class _DirectoryPathHandler(_BasePathHandler):
    """(Internal) Validates that the path does not include a file extension."""
    @classmethod
    def _validate(cls, value: Any) -> Path:
        path = Path(str(value))
        if path.suffix:
            raise ValueError(f"Directory path '{path}' should not contain a file extension.")
        return path
