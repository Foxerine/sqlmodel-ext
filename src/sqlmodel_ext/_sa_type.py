"""
Internal helpers for extracting SQLAlchemy types from annotations.

These are used by the metaclass to inject sa_type into Field definitions.
"""
import sys
import typing
from typing import Any, Mapping, get_args, get_origin, get_type_hints

from sqlalchemy.orm import Mapped


def _extract_sa_type_from_annotation(annotation: Any) -> Any | None:
    """
    Extract SQLAlchemy type from a type annotation.

    Supports:
    1. Types with __sqlmodel_sa_type__ attribute (e.g. NumpyVector[256, np.float32])
    2. Annotated[T, metadata] wrappers
    3. Types with __get_pydantic_core_schema__ returning metadata['sa_type']

    :param annotation: Field type annotation
    :returns: Extracted SQLAlchemy type, or None
    """
    # Method 1: Direct __sqlmodel_sa_type__ attribute
    if hasattr(annotation, '__sqlmodel_sa_type__'):
        return annotation.__sqlmodel_sa_type__

    # Method 2: Annotated type
    if get_origin(annotation) is typing.Annotated:
        args = get_args(annotation)
        if len(args) >= 2:
            metadata_items = args[1:]

            for item in metadata_items:
                if hasattr(item, '__sqlmodel_sa_type__'):
                    return item.__sqlmodel_sa_type__

                if hasattr(item, '__get_pydantic_core_schema__'):
                    try:
                        schema = item.__get_pydantic_core_schema__(
                            annotation,
                            lambda x: None,
                        )
                        if isinstance(schema, dict) and 'metadata' in schema:
                            sa_type = schema['metadata'].get('sa_type')
                            if sa_type is not None:
                                return sa_type
                    except (TypeError, AttributeError, KeyError, ValueError):
                        pass

    # Method 3: Type itself has __get_pydantic_core_schema__
    if hasattr(annotation, '__get_pydantic_core_schema__'):
        try:
            schema = annotation.__get_pydantic_core_schema__(
                annotation,
                lambda x: None,
            )
            if isinstance(schema, dict) and 'metadata' in schema:
                sa_type = schema['metadata'].get('sa_type')
                if sa_type is not None:
                    return sa_type
        except (TypeError, AttributeError, KeyError, ValueError):
            pass

    return None


def _resolve_annotations(attrs: dict[str, Any]) -> tuple[
    dict[str, Any],
    dict[str, str],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    """
    Resolve annotations from a class namespace with Python 3.14 (PEP 649) support.

    Prefers evaluated annotations (Format.VALUE) so that typing.Annotated
    metadata and custom types remain accessible. Forward references that cannot be
    evaluated are replaced with typing.ForwardRef placeholders.
    """
    raw_annotations = attrs.get('__annotations__') or {}
    try:
        base_annotations = dict(raw_annotations)
    except TypeError:
        base_annotations = {}

    module_name = attrs.get('__module__')
    module_globals: dict[str, Any]
    if module_name and module_name in sys.modules:
        module_globals = dict(sys.modules[module_name].__dict__)
    else:
        module_globals = {}

    module_globals.setdefault('__builtins__', __builtins__)
    localns: dict[str, Any] = dict(attrs)

    try:
        temp_cls = type('AnnotationProxy', (object,), dict(attrs))
        temp_cls.__module__ = module_name
        extras_kw = {'include_extras': True} if sys.version_info >= (3, 10) else {}
        evaluated = get_type_hints(
            temp_cls,
            globalns=module_globals,
            localns=localns,
            **extras_kw,
        )
    except (NameError, AttributeError, TypeError, RecursionError):
        evaluated = base_annotations

    return dict(evaluated), {}, module_globals, localns


def _evaluate_annotation_from_string(
    field_name: str,
    annotation_strings: dict[str, str],
    current_type: Any,
    globalns: Mapping[str, Any],
    localns: Mapping[str, Any],
) -> Any:
    """
    Attempt to re-evaluate the original annotation string for a field.

    Used as a fallback when the resolved annotation lost its metadata
    (e.g., Annotated wrappers) and we need to recover custom sa_type data.
    """
    if not annotation_strings:
        return current_type

    expr = annotation_strings.get(field_name)
    if not expr or not isinstance(expr, str):
        return current_type

    try:
        return eval(expr, globalns, localns)
    except (NameError, SyntaxError, AttributeError, TypeError):
        return current_type
