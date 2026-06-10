"""
URL field types + SSRF protection tests.

Covers ``sqlmodel_ext.field_types.url`` (``Url`` / ``HttpUrl`` /
``WebSocketUrl`` / ``SafeHttpUrl``) and ``sqlmodel_ext.field_types._ssrf``
(``validate_not_private_host`` / ``UnsafeURLError``):

1. ``Url``: any scheme accepted, malformed strings rejected, the validated
   value stays the original string (no trailing-slash normalization surprise).
2. ``HttpUrl``: only http/https; ftp / ws / file / javascript rejected.
3. ``WebSocketUrl``: only ws/wss; http rejected.
4. ``SafeHttpUrl``: public hosts pass; private/loopback/link-local/multicast
   IP literals, localhost variants and IPv6 private literals are rejected
   with an SSRF error; non-http schemes rejected.
5. ``validate_not_private_host`` direct unit tests, including the documented
   no-DNS-resolution property: arbitrary domain names pass without any
   network access (only IP literals and known-localhost names are blocked),
   so no socket monkeypatching is needed.

Pure Pydantic-layer tests -- no DB, no network.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sqlmodel_ext import SQLModelBase
from sqlmodel_ext.field_types import (
    HttpUrl,
    SafeHttpUrl,
    UnsafeURLError,
    Url,
    WebSocketUrl,
    validate_not_private_host,
)


class FtUrlModel(SQLModelBase):
    any_url: Url = "https://example.com"
    http_url: HttpUrl = "https://example.com"
    ws_url: WebSocketUrl = "wss://example.com"


class FtSafeUrlModel(SQLModelBase):
    callback: SafeHttpUrl


# ============================================================
# 1. Url (any scheme)
# ============================================================

@pytest.mark.parametrize("url", [
    "https://example.com",
    "http://example.com:8080/path?q=1",
    "ftp://files.example.com/file.txt",
    "ws://example.com/socket",
    "postgres://user:pass@db.example.com:5432/app",
])
def test_url_accepts_any_scheme(url: str) -> None:
    assert FtUrlModel(any_url=url).any_url == url


@pytest.mark.parametrize("bad", ["", "not a url", "example.com", "http://"])
def test_url_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValidationError):
        FtUrlModel(any_url=bad)


def test_url_preserves_original_string() -> None:
    """The validator returns the input string verbatim (no normalization)."""
    m = FtUrlModel(any_url="https://example.com")  # no trailing slash
    assert m.any_url == "https://example.com"
    assert isinstance(m.any_url, str)


# ============================================================
# 2. HttpUrl (http/https only)
# ============================================================

@pytest.mark.parametrize("url", [
    "http://example.com",
    "https://example.com/api/v1?x=1#frag",
    "https://sub.domain.example.co.uk:8443/",
])
def test_http_url_accepts_http_https(url: str) -> None:
    assert FtUrlModel(http_url=url).http_url == url


@pytest.mark.parametrize("bad", [
    "ftp://example.com",
    "ws://example.com",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "gopher://example.com",
    "not a url",
])
def test_http_url_rejects_other_schemes(bad: str) -> None:
    with pytest.raises(ValidationError):
        FtUrlModel(http_url=bad)


# ============================================================
# 3. WebSocketUrl (ws/wss only)
# ============================================================

@pytest.mark.parametrize("url", [
    "ws://example.com/socket",
    "wss://example.com:9443/stream",
])
def test_websocket_url_accepts_ws_wss(url: str) -> None:
    assert FtUrlModel(ws_url=url).ws_url == url


@pytest.mark.parametrize("bad", [
    "http://example.com",
    "https://example.com",
    "ftp://example.com",
    "not a url",
])
def test_websocket_url_rejects_other_schemes(bad: str) -> None:
    with pytest.raises(ValidationError):
        FtUrlModel(ws_url=bad)


# ============================================================
# 4. SafeHttpUrl (SSRF protection)
# ============================================================

@pytest.mark.parametrize("url", [
    "https://example.com/webhook",
    "http://api.example.org:8080/cb",
    "https://8.8.8.8/dns",          # public IPv4 literal
    "https://1.1.1.1",
    "https://[2606:4700:4700::1111]/",  # public IPv6 literal
])
def test_safe_http_url_accepts_public_hosts(url: str) -> None:
    assert FtSafeUrlModel(callback=url).callback == url


@pytest.mark.parametrize("url", [
    # loopback
    "http://127.0.0.1/",
    "http://127.0.0.1:8080/admin",
    "http://127.255.255.255/",
    # RFC1918 private ranges
    "http://10.0.0.1/",
    "http://10.255.255.255/",
    "http://172.16.0.1/",
    "http://172.31.255.254/",
    "http://192.168.1.1/router",
    # link-local (cloud metadata endpoint)
    "http://169.254.169.254/latest/meta-data/",
    # unspecified
    "http://0.0.0.0/",
    # localhost variants
    "http://localhost/",
    "http://LOCALHOST:8000/",
    "http://localhost.localdomain/",
    # IPv6 private/loopback literals
    "http://[::1]/",
    "http://[::]/",
    "http://[fc00::1]/",
    "http://[fd12:3456:789a::1]/",
    "http://[fe80::1]/",
])
def test_safe_http_url_blocks_private_hosts(url: str) -> None:
    with pytest.raises(ValidationError, match="SSRF protection"):
        FtSafeUrlModel(callback=url)


@pytest.mark.parametrize("bad", [
    "file:///etc/passwd",
    "gopher://example.com",
    "dict://example.com:11111/",
    "ftp://example.com",
])
def test_safe_http_url_blocks_non_http_schemes(bad: str) -> None:
    with pytest.raises(ValidationError):
        FtSafeUrlModel(callback=bad)


def test_safe_http_url_returns_original_string() -> None:
    m = FtSafeUrlModel(callback="https://hooks.example.com/x")
    assert m.callback == "https://hooks.example.com/x"
    assert isinstance(m.callback, str)


# ============================================================
# 5. validate_not_private_host direct unit tests
# ============================================================

class TestValidateNotPrivateHost:
    @pytest.mark.parametrize("host", [
        "example.com",
        "api.example.org",
        "8.8.8.8",
        "93.184.216.34",
        "2606:4700:4700::1111",
        "[2606:4700:4700::1111]",  # bracketed IPv6 form from URLs
    ])
    def test_public_hosts_pass(self, host: str) -> None:
        validate_not_private_host(host)  # must not raise

    @pytest.mark.parametrize("host", [
        "localhost",
        "LocalHost",                # case-insensitive
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    ])
    def test_localhost_variants_blocked(self, host: str) -> None:
        with pytest.raises(UnsafeURLError, match="Forbidden hostname"):
            validate_not_private_host(host)

    @pytest.mark.parametrize("host", [
        "127.0.0.1",        # loopback
        "10.1.2.3",         # RFC1918
        "172.16.0.1",
        "172.31.255.255",
        "192.168.0.1",
        "169.254.169.254",  # link-local
        "0.0.0.0",          # unspecified
        "224.0.0.1",        # multicast
        "240.0.0.1",        # reserved
        "::1",              # IPv6 loopback
        "[::1]",            # bracketed
        "::",               # IPv6 unspecified
        "fc00::1",          # IPv6 ULA (fc00::/7)
        "fe80::1",          # IPv6 link-local
        "[fe80::1]",
        "ff02::1",          # IPv6 multicast
    ])
    def test_private_ip_literals_blocked(self, host: str) -> None:
        with pytest.raises(UnsafeURLError, match="Forbidden private IP"):
            validate_not_private_host(host)

    def test_public_edge_of_172_range_passes(self) -> None:
        # 172.32.0.1 sits just outside 172.16.0.0/12
        validate_not_private_host("172.32.0.1")

    def test_empty_hostname_blocked(self) -> None:
        with pytest.raises(UnsafeURLError, match="must not be empty"):
            validate_not_private_host("")

    def test_domain_names_skip_dns_resolution(self) -> None:
        """Domain names are allowed without DNS resolution (no network I/O).

        The function only inspects IP literals; a non-resolvable internal
        name passes here -- resolution-time SSRF defenses must live at the
        HTTP client layer.
        """
        import socket

        def boom(*args, **kwargs):  # pragma: no cover
            raise AssertionError("validate_not_private_host must not touch DNS")

        original = socket.getaddrinfo
        socket.getaddrinfo = boom
        try:
            validate_not_private_host("intranet.internal")
            validate_not_private_host("some-private-name")
        finally:
            socket.getaddrinfo = original

    def test_unsafe_url_error_is_value_error(self) -> None:
        """UnsafeURLError must stay a ValueError so Pydantic converts it."""
        assert issubclass(UnsafeURLError, ValueError)
