"""
``query_dependency()``: query-parameter DTOs as FastAPI dependencies.

Requests go through the real ASGI app (FastAPI routing, dependency solving,
the default ``RequestValidationError`` handler); a minimal ASGI driver is used
instead of ``TestClient`` so no extra HTTP client package is needed.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

import orjson
import pytest
from fastapi import Depends, FastAPI
from pydantic import Field as PydanticField
from sqlmodel import Field

import sqlmodel_ext.dependencies as dependencies_module
from sqlmodel_ext import (
    PageWindowRequest,
    PaginationRequest,
    SQLModelBase,
    TableViewRequest,
    TimeFilterRequest,
    TrgmSearchRequest,
    UUIDTableBaseMixin,
    Unset,
    query_dependency,
)


class QueryDepSortedRequest(TableViewRequest):
    """Subclass widening the ``order`` Literal with a domain column."""
    order: Literal["created_at", "updated_at", "id", "name"] | None = "created_at"


class QueryDepAliasedRequest(SQLModelBase):
    """DTO whose query parameter name differs from the field name."""
    page_size: int = Field(default=10, ge=1, le=20, alias="pageSize")


class QueryDepOmissibleRequest(SQLModelBase):
    """DTO with an omissible (``Unset``) field."""
    limit: Unset | int = Unset


class QueryDepFactoryRequest(SQLModelBase):
    tags: list[str] = PydanticField(default_factory=list)


class QueryDepRow(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str


TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]

app = FastAPI()


@app.get("/table-view")
async def table_view_endpoint(table_view: TableViewDep, status: str | None = None) -> dict[str, Any]:
    return {'table_view': table_view.model_dump(mode='json'), 'status': status}


@app.get("/page-window")
async def page_window_endpoint(
    window: Annotated[PageWindowRequest, Depends(query_dependency(PageWindowRequest))],
) -> dict[str, Any]:
    return window.model_dump(mode='json')


@app.get("/time-filter")
async def time_filter_endpoint(
    time_filter: Annotated[TimeFilterRequest, Depends(query_dependency(TimeFilterRequest))],
) -> dict[str, Any]:
    return time_filter.model_dump(mode='json')


@app.get("/sorted")
async def sorted_endpoint(
    table_view: Annotated[QueryDepSortedRequest, Depends(query_dependency(QueryDepSortedRequest))],
) -> dict[str, Any]:
    return table_view.model_dump(mode='json')


@app.get("/aliased")
async def aliased_endpoint(
    params: Annotated[QueryDepAliasedRequest, Depends(query_dependency(QueryDepAliasedRequest))],
) -> dict[str, Any]:
    return {'page_size': params.page_size}


@app.get("/omissible")
async def omissible_endpoint(
    params: Annotated[QueryDepOmissibleRequest, Depends(query_dependency(QueryDepOmissibleRequest))],
) -> dict[str, Any]:
    return {'omitted': params.limit is Unset, 'dump': params.model_dump()}


@app.get("/search")
async def search_endpoint(
    search: Annotated[TrgmSearchRequest, Depends(query_dependency(TrgmSearchRequest))],
) -> dict[str, Any]:
    return {'query': search.query}


async def _get(path: str, query: str = '') -> tuple[int, Any]:
    """Send one GET request through the ASGI app; return (status, decoded JSON body)."""
    messages: list[dict[str, Any]] = []
    scope: dict[str, Any] = {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': 'GET',
        'scheme': 'http',
        'path': path,
        'raw_path': path.encode(),
        'query_string': query.encode(),
        'root_path': '',
        'headers': [],
        'client': ('testclient', 50000),
        'server': ('testserver', 80),
    }

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(scope, receive, send)
    status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
    body = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
    return status, orjson.loads(body)


def _locs(body: Any) -> list[list[Any]]:
    return [error['loc'] for error in body['detail']]


# ---------------------------------------------------------------------------
# Cross-field errors -> 422 located in the query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_after_id_with_offset_is_422_at_after_id() -> None:
    status, body = await _get("/table-view", f"after_id={uuid.uuid4()}&offset=3")
    assert status == 422
    assert _locs(body) == [['query', 'after_id']]
    assert "after_id and offset cannot be combined" in body['detail'][0]['msg']


@pytest.mark.asyncio
async def test_after_id_with_mutable_order_is_422_at_after_id() -> None:
    status, body = await _get("/table-view", f"after_id={uuid.uuid4()}&order=updated_at")
    assert status == 422
    assert _locs(body) == [['query', 'after_id']]
    assert "does not support order=updated_at" in body['detail'][0]['msg']


@pytest.mark.asyncio
async def test_inverted_created_range_is_422_at_created_before() -> None:
    status, body = await _get(
        "/table-view",
        "created_after_datetime=2026-02-01T00:00:00Z&created_before_datetime=2026-01-01T00:00:00Z",
    )
    assert status == 422
    assert _locs(body) == [['query', 'created_before_datetime']]


@pytest.mark.asyncio
async def test_created_after_not_before_updated_before_is_422_at_updated_before() -> None:
    status, body = await _get(
        "/time-filter",
        "created_after_datetime=2026-03-01T00:00:00Z&updated_before_datetime=2026-01-01T00:00:00Z",
    )
    assert status == 422
    assert _locs(body) == [['query', 'updated_before_datetime']]


@pytest.mark.asyncio
async def test_two_violated_rules_are_both_reported() -> None:
    status, body = await _get(
        "/table-view",
        f"after_id={uuid.uuid4()}&offset=1"
        "&updated_after_datetime=2026-02-01T00:00:00Z&updated_before_datetime=2026-01-01T00:00:00Z",
    )
    assert status == 422
    assert sorted(_locs(body)) == [['query', 'after_id'], ['query', 'updated_before_datetime']]


# ---------------------------------------------------------------------------
# Valid requests and single-field errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_request_is_200_with_defaults() -> None:
    status, body = await _get("/table-view")
    assert status == 200
    assert body['table_view'] == TableViewRequest().model_dump(mode='json')


@pytest.mark.asyncio
async def test_valid_keyset_request_is_200() -> None:
    anchor = uuid.uuid4()
    status, body = await _get(
        "/table-view",
        f"after_id={anchor}&limit=5&desc=false&created_after_datetime=2026-01-01T00:00:00%2B08:00",
    )
    assert status == 200
    assert body['table_view']['after_id'] == str(anchor)
    assert body['table_view']['limit'] == 5
    assert body['table_view']['desc'] is False
    assert body['table_view']['offset'] == 0


@pytest.mark.asyncio
async def test_other_query_parameters_are_ignored_and_endpoint_parameters_still_work() -> None:
    # ``status`` is the endpoint's own parameter, ``_`` is undeclared (cache buster).
    status, body = await _get("/table-view", "status=draft&_=123&limit=7")
    assert status == 200
    assert body['status'] == 'draft'
    assert body['table_view']['limit'] == 7


@pytest.mark.asyncio
async def test_single_field_errors_stay_per_parameter_422() -> None:
    status, body = await _get("/table-view", "limit=0")
    assert status == 422
    assert _locs(body) == [['query', 'limit']]

    status, body = await _get("/table-view", "created_after_datetime=2026-01-01T00:00:00")
    assert status == 422
    assert _locs(body) == [['query', 'created_after_datetime']]


@pytest.mark.asyncio
async def test_page_window_request_dependency() -> None:
    status, body = await _get("/page-window", "offset=10&limit=20")
    assert status == 200
    assert body == {'offset': 10, 'limit': 20, 'desc': True}

    status, body = await _get("/page-window", "limit=101")
    assert status == 422
    assert _locs(body) == [['query', 'limit']]


@pytest.mark.asyncio
async def test_subclass_with_widened_order_keeps_the_keyset_rule() -> None:
    status, body = await _get("/sorted", "order=name")
    assert status == 200
    assert body['order'] == 'name'

    status, body = await _get("/sorted", f"order=name&after_id={uuid.uuid4()}")
    assert status == 422
    assert _locs(body) == [['query', 'after_id']]


@pytest.mark.asyncio
async def test_alias_is_the_query_parameter_name() -> None:
    status, body = await _get("/aliased", "pageSize=15")
    assert status == 200
    assert body == {'page_size': 15}

    status, body = await _get("/aliased", "pageSize=21")
    assert status == 422
    assert _locs(body) == [['query', 'pageSize']]


@pytest.mark.asyncio
async def test_trgm_search_request_dependency() -> None:
    status, body = await _get("/search", "query=abc")
    assert status == 200
    assert body == {'query': 'abc'}


@pytest.mark.asyncio
async def test_unset_default_means_omitted() -> None:
    status, body = await _get("/omissible")
    assert status == 200
    assert body == {'omitted': True, 'dump': {}}

    status, body = await _get("/omissible", "limit=3")
    assert status == 200
    assert body == {'omitted': False, 'dump': {'limit': 3}}


# ---------------------------------------------------------------------------
# OpenAPI and factory contract
# ---------------------------------------------------------------------------

def test_openapi_publishes_every_field_with_its_constraints() -> None:
    parameters = app.openapi()['paths']['/table-view']['get']['parameters']
    by_name = {p['name']: p for p in parameters}
    assert set(by_name) == set(TableViewRequest.model_fields) | {'status'}
    assert all(p['in'] == 'query' for p in parameters)
    limit_schema = by_name['limit']['schema']['anyOf'][0]
    assert limit_schema['minimum'] == 1
    assert limit_schema['maximum'] == 100
    assert by_name['limit']['schema']['default'] == 50
    assert by_name['limit']['description'] == TableViewRequest.model_fields['limit'].description


def test_same_model_returns_the_same_dependency() -> None:
    assert query_dependency(TableViewRequest) is query_dependency(TableViewRequest)
    assert query_dependency(TableViewRequest) is not query_dependency(PaginationRequest)


def test_table_model_is_rejected() -> None:
    with pytest.raises(TypeError, match="table=True"):
        query_dependency(QueryDepRow)


def test_default_factory_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="default_factory"):
        query_dependency(QueryDepFactoryRequest)


def test_missing_fastapi_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class QueryDepNoFastAPIRequest(SQLModelBase):
        value: int = 1

    monkeypatch.setattr(dependencies_module, '_FastAPIQuery', None)
    with pytest.raises(ImportError, match="requires FastAPI"):
        query_dependency(QueryDepNoFastAPIRequest)
