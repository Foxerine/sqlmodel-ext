"""URL types compatible with Pydantic and SQLModel."""
import typing

from pydantic import AnyUrl, GetCoreSchemaHandler, HttpUrl as PydanticHttpUrl, WebsocketUrl as PydanticWebsocketUrl
from pydantic_core import core_schema

from sqlmodel_ext.field_types._ssrf import validate_not_private_host, UnsafeURLError


class Url(str):
    """
    Generic URL type compatible with Pydantic and SQLModel.

    Validates any URL format (HTTP, HTTPS, WS, WSS, etc.).
    Stored as VARCHAR string in the database.

    Example::

        class ExternalService(UUIDTableBaseMixin, table=True):
            base_url: Url
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_url(value: typing.Any) -> str:
            if isinstance(value, str):
                AnyUrl(value)
                return value
            elif isinstance(value, AnyUrl):
                return str(value)
            else:
                url_str = str(value)
                AnyUrl(url_str)
                return url_str

        return core_schema.no_info_after_validator_function(
            validate_url,
            core_schema.str_schema(),
        )


class HttpUrl(str):
    """
    HTTP/HTTPS URL type compatible with Pydantic and SQLModel.

    Only accepts HTTP and HTTPS protocols.
    Stored as VARCHAR string in the database.

    Example::

        class APIConfig(UUIDTableBaseMixin, table=True):
            base_url: HttpUrl = "http://example.com/api"
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_http_url(value: typing.Any) -> str:
            if isinstance(value, str):
                PydanticHttpUrl(value)
                return value
            elif isinstance(value, PydanticHttpUrl):
                return str(value)
            else:
                url_str = str(value)
                PydanticHttpUrl(url_str)
                return url_str

        return core_schema.no_info_after_validator_function(
            validate_http_url,
            core_schema.str_schema(),
        )


class WebSocketUrl(str):
    """
    WebSocket URL type compatible with Pydantic and SQLModel.

    Only accepts WS and WSS protocols.
    Stored as VARCHAR string in the database.

    Example::

        class ASRService(UUIDTableBaseMixin, table=True):
            base_url: WebSocketUrl
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_websocket_url(value: typing.Any) -> str:
            if isinstance(value, str):
                PydanticWebsocketUrl(value)
                return value
            elif isinstance(value, PydanticWebsocketUrl):
                return str(value)
            else:
                url_str = str(value)
                PydanticWebsocketUrl(url_str)
                return url_str

        return core_schema.no_info_after_validator_function(
            validate_websocket_url,
            core_schema.str_schema(),
        )


class SafeHttpUrl(str):
    """
    Safe HTTP/HTTPS URL type with SSRF protection, compatible with Pydantic and SQLModel.

    Validates:
    1. HTTP/HTTPS protocol (via Pydantic HttpUrl)
    2. SSRF protection (blocks private IPs and localhost)

    Stored as VARCHAR string in the database.

    Example::

        class WebhookConfig(UUIDTableBaseMixin, table=True):
            callback_url: SafeHttpUrl

    Protection:
    - Blocks file://, gopher://, dict:// etc. (only allows http/https)
    - Blocks private IPs (10.x, 172.16-31.x, 192.168.x, 127.x, 169.254.x, etc.)
    - Blocks localhost and variants

    References:
    - https://docs.pydantic.dev/latest/api/networks/
    - https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_safe_http_url(value: typing.Any) -> str:
            url_str = str(value) if not isinstance(value, str) else value

            # 1. Validate HTTP URL format
            parsed_url = PydanticHttpUrl(url_str)

            # 2. SSRF protection: check hostname is not private
            hostname = parsed_url.host
            if not hostname:
                raise ValueError("URL must contain a valid hostname")

            try:
                validate_not_private_host(hostname)
            except UnsafeURLError as e:
                raise ValueError(f"SSRF protection: {e}") from e

            return url_str

        return core_schema.no_info_after_validator_function(
            validate_safe_http_url,
            core_schema.str_schema(),
        )
