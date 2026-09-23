"""IP address types compatible with Pydantic and SQLModel.

- :class:`IPAddress`: a **storage column** type (VARCHAR, behaves like ``str``)
- :data:`ClientIPAddress`: a **parse-time** type that validates untrusted text
  (e.g. a reverse-proxy request header) into ``IPv4Address | IPv6Address`` --
  Pydantic ``IPvAnyAddress`` plus rejection of IPv6 zone IDs
"""
import ipaddress
import typing
from typing import Annotated, TypeAlias

from pydantic import AfterValidator, IPvAnyAddress, GetCoreSchemaHandler
from pydantic_core import core_schema


class IPAddress(str):
    """
    IP address type compatible with Pydantic and SQLModel.

    - Behaves like a string in Python code
    - Pydantic validation supports IPv4 and IPv6 formats
    - Stored as VARCHAR string in the database
    - Displayed as str for type checkers (linters)

    Example::

        class User(SQLModel, table=True):
            register_ip: IPAddress
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Pydantic v2 schema: validate with IPvAnyAddress, store as string."""

        def validate_ip_address(value: typing.Any) -> str:
            """Validate IP address format and return as string."""
            if isinstance(value, str):
                IPvAnyAddress(value)  # pyright: ignore[reportCallIssue]  # pydantic declares IPvAnyAddress as a Union alias under TYPE_CHECKING; at runtime it is a callable class
                return value
            elif isinstance(value, (IPvAnyAddress, )):
                return str(value)
            else:
                ip_str = str(value)
                IPvAnyAddress(ip_str)  # pyright: ignore[reportCallIssue]  # pydantic declares IPvAnyAddress as a Union alias under TYPE_CHECKING; at runtime it is a callable class
                return ip_str

        return core_schema.no_info_after_validator_function(
            validate_ip_address,
            core_schema.str_schema(),
        )

    def is_private(self) -> bool:
        """Check if this IP address is a private address."""
        return IPvAnyAddress(self).is_private  # pyright: ignore[reportCallIssue]  # pydantic declares IPvAnyAddress as a Union alias under TYPE_CHECKING; at runtime it is a callable class


def _reject_ipv6_scope_id(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Reject IPv6 addresses carrying a zone ID (``fe80::1%eth0``).

    A zone ID names a local interface; it is not part of a network address.
    ``ipaddress`` also places no length limit on it (``'fe80::1%'`` followed by
    16,000 characters is accepted by ``IPvAnyAddress``), so on untrusted input it
    is an unbounded injection / resource-exhaustion channel.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.scope_id is not None:
        raise ValueError("an IPv6 zone ID is not part of a client address")
    return ip


ClientIPAddress: TypeAlias = Annotated[IPvAnyAddress, AfterValidator(_reject_ipv6_scope_id)]
"""A client IP literal parsed from **untrusted text** (reverse-proxy headers etc.).

Pydantic ``IPvAnyAddress`` performs the structural validation (IPv4 / IPv6 /
IPv4-mapped IPv6; brackets, port suffixes, leading zeros, whitespace and
oversized input are rejected), then IPv6 zone IDs are rejected on top. The
validated value is an ``IPv4Address | IPv6Address``; ``str()`` of it is the
normalized text (compressed form for IPv6).

Use :class:`IPAddress` for the database column that stores the result.
"""
