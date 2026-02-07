"""IP address type compatible with Pydantic and SQLModel."""
import typing

from pydantic import IPvAnyAddress, GetCoreSchemaHandler
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
                IPvAnyAddress(value)
                return value
            elif isinstance(value, (IPvAnyAddress, )):
                return str(value)
            else:
                ip_str = str(value)
                IPvAnyAddress(ip_str)
                return ip_str

        return core_schema.no_info_after_validator_function(
            validate_ip_address,
            core_schema.str_schema(),
        )

    def is_private(self) -> bool:
        """Check if this IP address is a private address."""
        return IPvAnyAddress(self).is_private
