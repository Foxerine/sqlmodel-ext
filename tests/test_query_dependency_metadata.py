"""
``query_dependency()`` carries every ``Query``-expressible ``FieldInfo`` attribute.

The OpenAPI schema of each generated query parameter must equal the model
field's own JSON schema (``model_json_schema(mode='validation')``), apart from
FastAPI's rendering of a ``None`` default (it omits it). Compared with a bare
``Depends()`` -- which FastAPI builds from Pydantic's ``__signature__`` and
which therefore loses the documentation attributes -- the parameters are the
same (name, location, requiredness, type, constraints, default); only the
documentation keys differ.
"""
import inspect
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI
from pydantic import Discriminator, Field as PydanticField, Tag
from pydantic.fields import FieldInfo
from sqlmodel import Field

from sqlmodel_ext import PaginationRequest, SQLModelBase, TableViewRequest, TimeFilterRequest, query_dependency

_DOC_KEYS = frozenset({"title", "description", "examples", "deprecated", "x-extra"})
"""Schema keys a bare ``Depends()`` does not render (it only sees Pydantic's ``__signature__``)."""


def _upper_title(name: str, _field: FieldInfo) -> str:
    return name.upper()


class QueryDepMetadataRequest(SQLModelBase):
    """Every documentation attribute a ``Query`` can express."""

    required_int: int = Field(ge=0, title="Required int", description="a required field")
    titled: int = Field(default=3, ge=1, le=9, title="Titled", schema_extra={"examples": [1, 2]})
    deprecated_flag: str = PydanticField(default="x", deprecated=True, max_length=5)
    deprecated_message: str | None = PydanticField(default=None, deprecated="use titled")
    extra: str = PydanticField(default="e", json_schema_extra={"x-extra": "yes"}, examples=["e", "f"])
    generated_title: int = PydanticField(default=0, field_title_generator=_upper_title)
    aliased: int = PydanticField(default=10, alias="aliasedName", title="Aliased", description="by alias")


class QueryDepPlainRequest(SQLModelBase):
    """No documentation attribute: a bare ``Depends()`` renders it fully."""

    count: int = Field(default=1, ge=1, le=5)
    name: str | None = None
    mode: str = Field(default="a", max_length=3)


class QueryDepCallableExtraRequest(SQLModelBase):
    value: int = PydanticField(default=0, json_schema_extra=lambda schema: schema.update({"x-dyn": 1}))


_MODELS: tuple[type[SQLModelBase], ...] = (
    QueryDepMetadataRequest,
    QueryDepPlainRequest,
    TableViewRequest,
    PaginationRequest,
    TimeFilterRequest,
)


def _endpoint(dependency: Any) -> Any:
    """An endpoint whose only parameter is ``r: Annotated[<dependency>]``."""

    async def endpoint(**_values: Any) -> None: ...

    # FastAPI reads the parameters through ``inspect.signature()``, which honors ``__signature__``.
    endpoint.__signature__ = inspect.Signature(
        [inspect.Parameter("r", inspect.Parameter.KEYWORD_ONLY, annotation=dependency)],
    )
    return endpoint


def _build_app() -> FastAPI:
    app = FastAPI()
    for model in _MODELS:
        app.get(f"/dep/{model.__name__}")(_endpoint(Annotated[(Any, Depends(query_dependency(model)))]))
        app.get(f"/bare/{model.__name__}")(_endpoint(Annotated[(model, Depends())]))
    return app


_APP = _build_app()
_OPENAPI: dict[str, Any] = _APP.openapi()


def _parameters(flavour: str, model: type[SQLModelBase]) -> list[dict[str, Any]]:
    return _OPENAPI["paths"][f"/{flavour}/{model.__name__}"]["get"]["parameters"]


@pytest.mark.parametrize("model", _MODELS, ids=lambda m: m.__name__)
def test_parameter_schema_equals_field_json_schema(model: type[SQLModelBase]) -> None:
    model_schema = model.model_json_schema(mode="validation")
    properties: dict[str, dict[str, Any]] = model_schema["properties"]
    required = set(model_schema.get("required", []))
    parameters = _parameters("dep", model)
    assert [p["name"] for p in parameters] == list(properties)
    for parameter in parameters:
        expected = dict(properties[parameter["name"]])
        if "default" in expected and expected["default"] is None:
            del expected["default"]  # FastAPI does not render a None default
        assert parameter["schema"] == expected, parameter["name"]
        assert parameter["required"] == (parameter["name"] in required)
        assert parameter.get("description") == expected.get("description")
        assert parameter.get("deprecated", False) == expected.get("deprecated", False)


@pytest.mark.parametrize("model", _MODELS, ids=lambda m: m.__name__)
def test_parameters_match_bare_depends_except_documentation(model: type[SQLModelBase]) -> None:
    def strip(parameter: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": parameter["name"],
            "in": parameter["in"],
            "required": parameter["required"],
            "schema": {k: v for k, v in parameter["schema"].items() if k not in _DOC_KEYS},
        }

    assert [strip(p) for p in _parameters("dep", model)] == [strip(p) for p in _parameters("bare", model)]


def test_plain_model_is_identical_to_bare_depends() -> None:
    assert _parameters("dep", QueryDepPlainRequest) == _parameters("bare", QueryDepPlainRequest)


def test_documentation_attributes_reach_openapi() -> None:
    by_name = {p["name"]: p for p in _parameters("dep", QueryDepMetadataRequest)}
    assert by_name["titled"]["schema"]["title"] == "Titled"
    assert by_name["titled"]["schema"]["examples"] == [1, 2]
    assert by_name["deprecated_flag"]["deprecated"] is True
    assert by_name["deprecated_message"]["deprecated"] is True
    assert by_name["extra"]["schema"]["x-extra"] == "yes"
    assert by_name["extra"]["schema"]["examples"] == ["e", "f"]
    assert by_name["generated_title"]["schema"]["title"] == "GENERATED_TITLE"
    assert by_name["aliasedName"]["schema"]["title"] == "Aliased"
    assert by_name["aliasedName"]["description"] == "by alias"
    assert by_name["required_int"]["required"] is True


def test_callable_json_schema_extra_is_rejected() -> None:
    with pytest.raises(TypeError, match="callable json_schema_extra"):
        _ = query_dependency(QueryDepCallableExtraRequest)


class QueryDepDiscriminatorObjectRequest(SQLModelBase):
    value: Annotated[int, Tag("int")] | Annotated[str, Tag("str")] = PydanticField(
        default=0,
        discriminator=Discriminator(lambda v: "int" if isinstance(v, int) else "str"),
    )


def test_discriminator_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="Discriminator object"):
        _ = query_dependency(QueryDepDiscriminatorObjectRequest)
