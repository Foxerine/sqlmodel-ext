"""
Tests for the string / numeric aliases, ``max_length_of`` and
``ClientIPAddress`` added to ``sqlmodel_ext.field_types`` in 0.5.0.

1. ``max_length_of``: the reflected bound equals the line Pydantic enforces
   (last constraint wins, ``GroupedMetadata`` expanded recursively, unrelated
   metadata ignored, ``X | None`` unwrapped, ``Array[T, N]`` element bound,
   fail-loud without a bound).
2. New string aliases: ``Str1`` / ``Text3072`` / ``Text4K`` /
   ``NonEmptyStrippedStr32`` / ``SingleLineStr64`` / ``SearchQueryStr64`` /
   ``HttpHeaderName``.
3. New numeric pieces: ``INT32_MIN``, ``SignedBigInt``, and finite-only
   ``PositiveFloat`` / ``NonNegativeFloat``.
4. ``ClientIPAddress``: IPv4 / IPv6 parsing with IPv6 zone IDs rejected.

Pure Pydantic-layer tests -- no DB.
"""
from __future__ import annotations

import ipaddress
from collections.abc import Iterator
from typing import Annotated, Any

import pytest
from annotated_types import GroupedMetadata, Len
from pydantic import IPvAnyAddress, StringConstraints, TypeAdapter, ValidationError
from sqlalchemy import BigInteger
from sqlmodel import Field

import sqlmodel_ext.field_types as ft
from sqlmodel_ext.field_types import (
    INT32_MAX,
    INT32_MIN,
    JS_MAX_SAFE_INTEGER,
    ClientIPAddress,
    HttpHeaderName,
    NonEmptyStrippedStr32,
    NonNegativeFloat,
    PositiveFloat,
    SearchQueryStr64,
    SignedBigInt,
    SingleLineStr64,
    Str1,
    Text100K,
    Text3072,
    Text4K,
    max_length_of,
)
from sqlmodel_ext.field_types.dialects.postgresql import Array


# ============================================================
# 1. max_length_of
# ============================================================

def _assert_matches_pydantic(alias: Any) -> None:
    """The reflected bound is exactly the line Pydantic enforces."""
    bound = max_length_of(alias)
    adapter = TypeAdapter(alias)
    adapter.validate_python('x' * bound)
    with pytest.raises(ValidationError):
        adapter.validate_python('x' * (bound + 1))


class TestMaxLengthOf:
    @pytest.mark.parametrize('alias, expected', [
        (Str1, 1), (Text3072, 3072), (Text4K, 4000), (Text100K, 100_000),
        (NonEmptyStrippedStr32, 32), (SingleLineStr64, 64), (SearchQueryStr64, 64),
    ])
    def test_library_aliases(self, alias: Any, expected: int) -> None:
        assert max_length_of(alias) == expected
        _assert_matches_pydantic(alias)

    def test_unrelated_metadata_with_max_length_attribute_is_ignored(self) -> None:
        class UnrelatedMetadata:
            def __init__(self, max_length: int) -> None:
                self.max_length = max_length

        alias = Annotated[str, Field(max_length=10), UnrelatedMetadata(max_length=100)]
        assert max_length_of(alias) == 10
        _assert_matches_pydantic(alias)

    def test_grouped_metadata_len_is_expanded(self) -> None:
        _assert_matches_pydantic(Annotated[str, Len(max_length=7)])

    def test_grouped_metadata_len_participates_in_last_wins(self) -> None:
        alias = Annotated[str, Field(max_length=10), Len(max_length=13)]
        assert max_length_of(alias) == 13
        _assert_matches_pydantic(alias)

    def test_nested_grouped_metadata_is_expanded_recursively(self) -> None:
        # A one-level expansion would stop at ``Len`` and report the wider 10.
        class NestedGroup(GroupedMetadata):
            def __iter__(self) -> Iterator[Any]:
                yield Len(0, 7)

        alias = Annotated[str, Field(max_length=10), NestedGroup()]
        assert max_length_of(alias) == 7
        _assert_matches_pydantic(alias)

    @pytest.mark.parametrize('alias, expected', [
        (Annotated[str, Field(max_length=10), Field(max_length=11)], 11),
        (Annotated[str, Field(max_length=12), StringConstraints(max_length=13)], 13),
        (Annotated[str, StringConstraints(max_length=14), Field(max_length=15)], 15),
    ])
    def test_last_constraint_wins(self, alias: Any, expected: int) -> None:
        assert max_length_of(alias) == expected
        _assert_matches_pydantic(alias)

    def test_optional_alias_is_unwrapped(self) -> None:
        assert max_length_of(Text4K | None) == 4000

    def test_multi_member_union_fails_loud(self) -> None:
        with pytest.raises(TypeError, match='non-None members'):
            max_length_of(Str1 | Text4K)

    def test_alias_without_max_length_fails_loud(self) -> None:
        with pytest.raises(TypeError, match='max_length'):
            max_length_of(Annotated[str, StringConstraints(pattern=r'^x*$')])

    def test_array_alias_reflects_max_items(self) -> None:
        alias = Array[str, 50]
        bound = max_length_of(alias)
        assert bound == 50
        adapter = TypeAdapter(alias)
        adapter.validate_python(['x'] * bound)
        with pytest.raises(ValidationError):
            adapter.validate_python(['x'] * (bound + 1))

    def test_unbounded_array_fails_loud(self) -> None:
        with pytest.raises(TypeError, match='max_length'):
            max_length_of(Array[str])


# ============================================================
# 2. New string aliases
# ============================================================

class TestSingleLineStr64:
    def test_line_break_set_is_derived_from_splitlines(self) -> None:
        # Full code-point scan: the import-time scan bound must not miss any
        # line-break character.
        full = ''.join(
            chr(code) for code in range(0x110000)
            if len(('a' + chr(code) + 'b').splitlines()) > 1
        )
        assert full == ft._LINE_BREAK_CHARS
        assert len(full) == 10

    @pytest.mark.parametrize('char', list('\n\v\f\r\x1c\x1d\x1e\x85  '))
    def test_rejects_every_line_break(self, char: str) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(SingleLineStr64).validate_python(f'a{char}b')

    def test_rejects_nul(self) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(SingleLineStr64).validate_python('a\x00b')

    def test_allows_tab_and_strips(self) -> None:
        assert TypeAdapter(SingleLineStr64).validate_python('  a\tb  ') == 'a\tb'

    @pytest.mark.parametrize('bad', ['', '   '])
    def test_rejects_empty_and_whitespace(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(SingleLineStr64).validate_python(bad)

    def test_single_line_constraint_is_in_json_schema(self) -> None:
        schema = TypeAdapter(SingleLineStr64).json_schema()
        assert '\\u2028' in schema['pattern']
        assert '\\u000a' in schema['pattern']


class TestSearchQueryStr64:
    @pytest.mark.parametrize('bad', ['', ' ', 'a', ' a ', '     '])
    def test_rejects_under_two_chars_after_strip(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(SearchQueryStr64).validate_python(bad)

    def test_strips_and_accepts(self) -> None:
        assert TypeAdapter(SearchQueryStr64).validate_python(' ab ') == 'ab'

    def test_min_length_is_in_json_schema(self) -> None:
        schema = TypeAdapter(SearchQueryStr64).json_schema()
        assert schema['minLength'] == 2
        assert schema['maxLength'] == 64


class TestNonEmptyStrippedStr32:
    def test_bounds(self) -> None:
        adapter = TypeAdapter(NonEmptyStrippedStr32)
        assert adapter.validate_python(' ' + 'x' * 30 + ' ') == 'x' * 30
        with pytest.raises(ValidationError):
            adapter.validate_python('   ')
        with pytest.raises(ValidationError):
            adapter.validate_python('x' * 33)


class TestHttpHeaderName:
    @pytest.mark.parametrize('name', ['X-Forwarded-For', 'CF-Connecting-IP', 'x_real_ip', "a!#$%&'*+.^_`|~"])
    def test_accepts_tokens(self, name: str) -> None:
        assert TypeAdapter(HttpHeaderName).validate_python(name) == name

    @pytest.mark.parametrize('bad', ['', 'X Forwarded', 'X-Forwarded-For:', 'X-é', 'a\x00', 'x' * 65])
    def test_rejects_non_tokens(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(HttpHeaderName).validate_python(bad)


def test_str1_bounds() -> None:
    assert TypeAdapter(Str1).validate_python('') == ''
    assert TypeAdapter(Str1).validate_python('a') == 'a'
    with pytest.raises(ValidationError):
        TypeAdapter(Str1).validate_python('ab')


# ============================================================
# 3. Numeric additions
# ============================================================

def test_int32_min() -> None:
    assert INT32_MIN == -(2**31) == -INT32_MAX - 1


@pytest.mark.parametrize('value, ok', [
    (-JS_MAX_SAFE_INTEGER, True),
    (-JS_MAX_SAFE_INTEGER - 1, False),
    (0, True),
    (JS_MAX_SAFE_INTEGER, True),
    (JS_MAX_SAFE_INTEGER + 1, False),
])
def test_signed_big_int_bounds(value: int, ok: bool) -> None:
    adapter = TypeAdapter(SignedBigInt)
    if ok:
        assert adapter.validate_python(value) == value
    else:
        with pytest.raises(ValidationError):
            adapter.validate_python(value)


def test_signed_big_int_maps_to_biginteger() -> None:
    sa_types = [m.sa_type for m in SignedBigInt.__metadata__ if hasattr(m, 'sa_type')]
    assert sa_types == [BigInteger]


@pytest.mark.parametrize('alias', [PositiveFloat, NonNegativeFloat])
@pytest.mark.parametrize('bad', [float('inf'), float('nan')])
def test_floats_reject_inf_and_nan(alias: Any, bad: float) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(alias).validate_python(bad)


@pytest.mark.parametrize('alias', [PositiveFloat, NonNegativeFloat])
def test_floats_reject_json_overflow_to_inf(alias: Any) -> None:
    # The JSON number 1e309 parses to float('inf').
    with pytest.raises(ValidationError):
        TypeAdapter(alias).validate_json('1e309')


def test_floats_still_accept_finite_values() -> None:
    assert TypeAdapter(PositiveFloat).validate_python(1e308) == 1e308
    assert TypeAdapter(NonNegativeFloat).validate_python(0.0) == 0.0


# ============================================================
# 4. ClientIPAddress
# ============================================================

class TestClientIPAddress:
    @pytest.mark.parametrize('raw, normalized', [
        ('203.0.113.7', '203.0.113.7'),
        ('2001:DB8:0:0:0:0:0:1', '2001:db8::1'),
        ('::FFFF:203.0.113.7', '::ffff:203.0.113.7'),
    ])
    def test_accepts_and_normalizes(self, raw: str, normalized: str) -> None:
        value = TypeAdapter(ClientIPAddress).validate_python(raw)
        assert isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address))
        assert str(value) == normalized

    def test_rejects_zone_id(self) -> None:
        # Plain IPvAnyAddress accepts the zone ID -- the rejection is ours.
        assert TypeAdapter(IPvAnyAddress).validate_python('fe80::1%eth0')
        with pytest.raises(ValidationError, match='zone ID'):
            TypeAdapter(ClientIPAddress).validate_python('fe80::1%eth0')

    def test_rejects_unbounded_zone_id(self) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(ClientIPAddress).validate_python('fe80::1%' + 'a' * 16_000)

    @pytest.mark.parametrize('bad', ['[::1]', '203.0.113.7:80', ' 203.0.113.7', '01.2.3.4', 'example.com', ''])
    def test_rejects_non_literals(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(ClientIPAddress).validate_python(bad)
