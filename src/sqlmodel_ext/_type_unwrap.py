"""
Type-annotation unwrapping helpers (shared mechanical operations).

Extracts the inner concrete type from wrappers such as ``Annotated[X, ...]``,
``X | None`` or ``ListResponse[X]``. Pure type-system operations with no domain
semantics and no state, hence plain module-level functions.

**Why a dedicated unwrapper is needed**: Pydantic v2 concrete generics (e.g.
``ListResponse[FooResponse]``) are fully concrete ``ModelMetaclass`` instances;
``typing.get_origin()`` / ``typing.get_args()`` **both return empty values** for
them. The real generic arguments live in
``__pydantic_generic_metadata__['args']``. Using only the standard ``typing``
API would make every generic response model opaque.

Consumer: :mod:`sqlmodel_ext.relation_load_checker` (RLC005 / RLC012 drill down
from a ``response_model`` to its DTO / ORM classes).
"""
import types
import typing
from typing import Annotated, Any, Final

# ``typing.Union`` as a runtime value: the ``get_origin()`` of ``Optional[X]`` /
# ``Union[X, Y]`` on Python < 3.14 (``X | Y`` has origin ``types.UnionType``).
# reportDeprecated targets spelling *annotations* with ``Union`` (use ``X | Y``);
# an identity-comparison target has no non-deprecated spelling, so the single
# reference lives here and every origin check in the package compares against
# this name (internal protocol, not part of the public API).
TYPING_UNION: Final = typing.Union  # pyright: ignore[reportDeprecated]


def unwrap_to_class(hint: Any) -> type | None:
    """
    Extract the actual class from a type annotation.

    Handles ``Annotated[X, ...]``, ``X | None``, ``Optional[X]`` and returns the
    innermost concrete type. Generic containers are **not** drilled into
    (``ListResponse[X]`` returns the container class itself); use
    :func:`unwrap_generic_to_dto_class` for that.

    :param hint: any type annotation object
    :returns: the actual type, or ``None`` if it cannot be extracted
    """
    origin = typing.get_origin(hint)

    # Annotated[X, ...] -> unwrap X
    if origin is Annotated:
        args = typing.get_args(hint)
        if args:
            return unwrap_to_class(args[0])

    # X | None (UnionType) or Optional[X] (Union[X, None])
    if origin is types.UnionType or origin is TYPING_UNION:
        args = typing.get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return unwrap_to_class(non_none[0])

    # Plain class
    if isinstance(hint, type):
        return hint

    return None


def get_pydantic_generic_args(hint: Any) -> tuple[Any, ...]:
    """
    Extract the type arguments of a generic (compatible with Pydantic v2 concrete generics).

    Pydantic v2's ``ListResponse[T]`` creates a fully concrete ``ModelMetaclass``
    instance for which ``typing.get_origin()`` / ``typing.get_args()`` return
    empty values. The real generic arguments are in
    ``__pydantic_generic_metadata__['args']``.

    :param hint: any type annotation object
    :returns: the generic arguments, or an empty tuple if there are none
    """
    # Standard typing API first
    args = typing.get_args(hint)
    if args:
        return args
    # Pydantic concrete-generic fallback
    pgm = getattr(hint, '__pydantic_generic_metadata__', None)
    if pgm is not None:
        return pgm.get('args', ())
    return ()


def unwrap_generic_to_dto_class(hint: Any) -> type | None:
    """
    Recursively extract a DTO class (a class with ``model_fields``) from a type annotation.

    Container types (Pydantic generic models such as ``ListResponse``) are skipped
    and their generic arguments are drilled into. For a union annotation the
    **first** member that yields a DTO wins; callers that need every member must
    iterate ``get_pydantic_generic_args`` / ``typing.get_args`` themselves.

    :param hint: any type annotation object
    :returns: the inner DTO class, or ``None`` if it cannot be extracted
    """
    # Pydantic generic container? (has __pydantic_generic_metadata__ with a non-empty origin)
    pgm = getattr(hint, '__pydantic_generic_metadata__', None)
    is_pydantic_generic = pgm is not None and pgm.get('origin') is not None

    if not is_pydantic_generic:
        cls = unwrap_to_class(hint)
        if cls is not None and hasattr(cls, 'model_fields'):
            return cls

    # Recurse into generic arguments
    for arg in get_pydantic_generic_args(hint):
        if arg is type(None):
            continue
        result = unwrap_generic_to_dto_class(arg)
        if result is not None:
            return result

    return None
