"""
URL safety validation -- SSRF protection

Validates that URL hostnames are not private/internal addresses.
Designed for use with Pydantic HttpUrl validation.

References:
- https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html
"""
import ipaddress


class UnsafeURLError(ValueError):
    """URL contains an unsafe hostname (private IP or localhost)."""


def validate_not_private_host(hostname: str) -> None:
    """
    Validate that a hostname is not a private/internal address (SSRF protection).

    :param hostname: Hostname or IP address string
    :raises UnsafeURLError: When a private address is detected

    Protection scope:
    - Forbids localhost and variants
    - Forbids private IPs (10.x, 172.16-31.x, 192.168.x, fc00::/7)
    - Forbids loopback addresses (127.x, ::1)
    - Forbids link-local addresses (169.254.x, fe80::/10)
    - Forbids unspecified/reserved/multicast addresses
    """
    if not hostname:
        raise UnsafeURLError("Hostname must not be empty")

    # 1. Forbid special hostnames
    hostname_lower = hostname.lower()
    forbidden_hosts = {
        'localhost',
        'localhost.localdomain',
        'ip6-localhost',
        'ip6-loopback',
    }
    if hostname_lower in forbidden_hosts:
        raise UnsafeURLError(f"Forbidden hostname: {hostname}")

    # 2. Check IP addresses
    # IPv6 addresses in URLs appear as [::1], strip brackets
    hostname_for_ip = hostname
    if hostname.startswith('[') and hostname.endswith(']'):
        hostname_for_ip = hostname[1:-1]

    try:
        ip = ipaddress.ip_address(hostname_for_ip)
    except ValueError:
        # Not an IP address format, it's a domain name (allow)
        return

    # IP address format -- check if private
    if _is_private_ip(ip):
        raise UnsafeURLError(f"Forbidden private IP address: {hostname}")


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    Check if an IP is a private/internal address.

    Uses the ipaddress stdlib's built-in properties:
    - is_private: Private networks (10.x, 172.16-31.x, 192.168.x, fc00::/7)
    - is_loopback: Loopback (127.x, ::1)
    - is_link_local: Link-local (169.254.x, fe80::/10)
    - is_unspecified: Unspecified (0.0.0.0, ::)
    - is_reserved: Reserved ranges
    - is_multicast: Multicast
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )
