"""
FastAPI dependency factory for query-parameter DTOs.

``query_dependency(Model)`` turns a query-parameter DTO (``TableViewRequest``,
``PageWindowRequest``, ``TimeFilterRequest``, ``TrgmSearchRequest`` or a
subclass of your own) into a FastAPI dependency::

    from typing import Annotated
    from fastapi import Depends
    from sqlmodel_ext import TableViewRequest, query_dependency

    TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]

Why not ``Annotated[TableViewRequest, Depends()]``: FastAPI validates the
parameters of a class dependency one by one (a single-field error is a 422),
then calls the class to build the object. Cross-field rules (``after_id`` with
a non-zero ``offset``, an inverted time range) only run in that call, and the
``pydantic.ValidationError`` they raise is not a ``RequestValidationError`` --
FastAPI does not catch it and the request fails with a 500.

The dependency returned here declares one query parameter per model field
(same type, constraints, default, title, description, examples, deprecation
and ``json_schema_extra``, so the OpenAPI schema describes the model's fields), builds the model itself and re-raises a failed
construction as ``fastapi.exceptions.RequestValidationError`` with every
error location prefixed by ``'query'``. FastAPI's default handler then answers
422, in the same shape as any other query-parameter error.

FastAPI is an optional dependency: this module imports without it, and
``query_dependency()`` raises ``ImportError`` when FastAPI is missing.
"""
import inspect
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

# Conditional FastAPI import: the library must be usable without FastAPI installed.
try:
    from fastapi import Query as _FastAPIQuery
    from fastapi.exceptions import RequestValidationError as _FastAPIRequestValidationError
except ImportError:
    _FastAPIQuery = None
    _FastAPIRequestValidationError = None

ModelT = TypeVar("ModelT", bound=BaseModel)

_dependency_by_model: dict[type[BaseModel], Callable[..., Coroutine[Any, Any, BaseModel]]] = {}
"""``query_dependency`` results per model class (a plain dict rather than
``functools.cache``, which would erase the ``ModelT`` return type)."""


def query_dependency(model: type[ModelT]) -> Callable[..., Coroutine[Any, Any, ModelT]]:
    """
    Build a FastAPI dependency that parses ``model`` from the query string.

    Every field of ``model`` becomes a query parameter named after the field's
    alias (the field name when it has none). The dependency constructs
    ``model`` from the received values, so all of its validators -- including
    cross-field ones -- run, and a ``ValidationError`` is re-raised as
    ``RequestValidationError`` whose error locations start with ``'query'``
    (``('query', 'after_id')``; a model-level error without a field location
    becomes ``('query',)``).

    Query parameters the model does not declare are ignored, as with any
    FastAPI query parameter; the endpoint can declare further query parameters
    next to the dependency.

    The result is cached per model class: calling ``query_dependency(Model)``
    twice returns the same callable, so FastAPI's per-request dependency cache
    treats both uses as one dependency.

    :param model: a non-table Pydantic / SQLModel model whose fields are all
        expressible as query parameters (scalars, enums, ``Literal``, lists of
        scalars)
    :returns: an async dependency callable returning a validated ``model`` instance
    :raises ImportError: FastAPI is not installed
    :raises TypeError: ``model`` is a ``table=True`` model (table models skip
        validation), or a field has a ``default_factory`` (a query parameter
        default must be a value), a callable ``json_schema_extra`` or a
        ``Discriminator`` object (``Query`` takes a dict / a field name)

    Field attributes carried over to each query parameter: the type and its
    constraints (``field.metadata``), the default, ``title``, ``description``,
    ``examples``, ``deprecated``, ``json_schema_extra`` and ``discriminator``;
    the aliases decide the parameter name. The remaining ``FieldInfo``
    attributes describe the model rather than a request parameter
    (``serialization_alias``, ``exclude``, ``exclude_if``, ``frozen``,
    ``repr``, ``init``, ``init_var``, ``kw_only``) or are enforced when the
    dependency constructs the model (``validate_default``), so they have no
    ``Query`` counterpart.
    """
    cached = _dependency_by_model.get(model)
    if cached is not None:
        # Stored by this function for this very ``model`` below.
        return cast(Callable[..., Coroutine[Any, Any, ModelT]], cached)
    if _FastAPIQuery is None or _FastAPIRequestValidationError is None:
        raise ImportError(
            "query_dependency() requires FastAPI; install it with `pip install sqlmodel-ext[fastapi]`"
        )
    query_cls = _FastAPIQuery
    request_validation_error_cls = _FastAPIRequestValidationError

    if model.model_config.get('table', False):
        raise TypeError(f"query_dependency({model.__name__}): a table=True model skips validation; pass a request DTO")

    parameters: list[inspect.Parameter] = []
    query_name_by_field: dict[str, str] = {}
    for field_name, field in model.model_fields.items():
        if field.default_factory is not None:
            raise TypeError(f"query_dependency({model.__name__}): field {field_name!r} has a default_factory")
        # The query parameter name is the key the model validates this field
        # by, so the received values can be passed to ``model_validate`` as is.
        if field.validation_alias is None:
            query_name = field.alias if field.alias is not None else field_name
        elif isinstance(field.validation_alias, str):
            query_name = field.validation_alias
        else:
            raise TypeError(
                f"query_dependency({model.__name__}): field {field_name!r} needs a single-string validation alias"
            )
        query_name_by_field[field_name] = query_name
        # A callable ``json_schema_extra`` / a ``Discriminator`` object has no
        # ``Query`` counterpart (``Query`` takes a dict / a field name).
        if field.json_schema_extra is not None and not isinstance(field.json_schema_extra, dict):
            raise TypeError(
                f"query_dependency({model.__name__}): field {field_name!r} has a callable json_schema_extra; "
                + "a query parameter takes a dict"
            )
        if field.discriminator is not None and not isinstance(field.discriminator, str):
            raise TypeError(
                f"query_dependency({model.__name__}): field {field_name!r} has a Discriminator object; "
                + "a query parameter takes a field name"
            )
        # ``field.metadata`` carries the constraints (``ge`` / ``le`` / length /
        # ``strict`` / ...): FastAPI merges them into the parameter, so
        # single-field errors are rejected per parameter and published in the
        # OpenAPI schema. The documentation attributes go to ``Query`` itself.
        # ``title`` already holds a ``field_title_generator`` result (Pydantic
        # applies it when the model is built).
        query = query_cls(
            alias=query_name,
            title=field.title,
            description=field.description,
            examples=field.examples,
            deprecated=field.deprecated,
            json_schema_extra=field.json_schema_extra,
            discriminator=field.discriminator,
        )
        annotation = Annotated[(field.annotation, *field.metadata, query)]
        parameters.append(inspect.Parameter(
            field_name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if field.is_required() else field.default,
            annotation=annotation,
        ))

    async def dependency(**values: Any) -> ModelT:
        # FastAPI passes the values keyed by parameter (= field) name; the
        # model validates them under their query names (see above).
        data = {query_name_by_field[field_name]: value for field_name, value in values.items()}
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            # Field errors are located by the key they were validated under,
            # i.e. the query parameter name; model-level errors have an empty
            # location and become ``('query',)``.
            errors = [
                {**error, 'loc': ('query', *error['loc'])}
                for error in exc.errors(include_url=False)
            ]
            raise request_validation_error_cls(errors) from exc

    # FastAPI reads the parameters through ``inspect.signature()``, which honors ``__signature__``.
    dependency.__signature__ = inspect.Signature(parameters, return_annotation=model)
    dependency.__name__ = f"{model.__name__}_query_dependency"
    dependency.__qualname__ = dependency.__name__
    _dependency_by_model[model] = dependency
    return dependency
