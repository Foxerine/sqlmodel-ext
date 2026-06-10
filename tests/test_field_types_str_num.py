"""
String / numeric / list field-type alias tests for ``sqlmodel_ext.field_types``.

Covers:

1. ``Str<N>`` / ``Text<N>`` max-length boundaries (exactly N passes, N+1 fails)
   plus the ``_NO_NULL_BYTE`` constraint shared by every string alias.
2. ``NonEmptyStr*`` (rejects ``""``, lets pure whitespace through) vs
   ``NonEmptyStrippedStr*`` (rejects ``""`` and whitespace-only, strips
   leading/trailing whitespace).
3. ``Sha256Hex`` strict 64-char lowercase-hex digest validation.
4. ``BCP47LanguageCode`` syntax validation.
5. Numeric aliases: ``Port`` / ``Percentage`` / ``PositiveInt`` /
   ``NonNegativeInt`` / ``PositiveBigInt`` / ``NonNegativeBigInt`` /
   ``PositiveFloat`` / ``NonNegativeFloat`` boundaries, and the
   ``INT32_MAX`` / ``INT64_MAX`` / ``JS_MAX_SAFE_INTEGER`` constants.
6. ``List<N>[T]`` bounded-list aliases and the transparent ``List[T]`` alias.
7. SQLAlchemy column mapping on a ``table=True`` model
   (e.g. ``Str64`` -> ``VARCHAR(64)`` via ``AutoString(length=64)``).

Pure Pydantic-layer tests except the metadata inspection -- no DB required.
"""
from __future__ import annotations

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from sqlmodel_ext import SQLModelBase, TableBaseMixin
from sqlmodel_ext.field_types import (
    BCP47LanguageCode,
    INT32_MAX,
    INT64_MAX,
    JS_MAX_SAFE_INTEGER,
    List,
    List1,
    List3,
    List16,
    NonEmptyStr64,
    NonEmptyStrippedStr64,
    NonNegativeBigInt,
    NonNegativeFloat,
    NonNegativeInt,
    Percentage,
    Port,
    PositiveBigInt,
    PositiveFloat,
    PositiveInt,
    Sha256Hex,
    Str16,
    Str64,
    Text1K,
)


# ---------------------------------------------------------------------------
# Validation DTOs (non-table, pure Pydantic)
# ---------------------------------------------------------------------------

class FtStrModel(SQLModelBase):
    s16: Str16 = ""
    s64: Str64 = ""
    t1k: Text1K = ""


class FtNonEmptyModel(SQLModelBase):
    plain: NonEmptyStr64 = "x"
    stripped: NonEmptyStrippedStr64 = "x"


class FtHashLangModel(SQLModelBase):
    digest: Sha256Hex = "0" * 64
    lang: BCP47LanguageCode = "en"


class FtNumModel(SQLModelBase):
    port: Port = 80
    pct: Percentage = 0
    pos: PositiveInt = 1
    nn: NonNegativeInt = 0
    pos_big: PositiveBigInt = 1
    nn_big: NonNegativeBigInt = 0
    pos_f: PositiveFloat = 1.0
    nn_f: NonNegativeFloat = 0.0


class FtListModel(SQLModelBase):
    one: List1[int] = []
    three: List3[str] = []
    sixteen: List16[int] = []


# ---------------------------------------------------------------------------
# Table model for SA column mapping
# ---------------------------------------------------------------------------

class FtStrNumColumnTable(SQLModelBase, TableBaseMixin, table=True):
    s16: Str16 = ""
    s64: Str64 = ""
    t1k: Text1K = ""
    nes: NonEmptyStrippedStr64 = "x"
    lang: BCP47LanguageCode = "en"
    port: Port = 80
    pos: PositiveInt = 1
    pos_big: PositiveBigInt = 1
    nn_big: NonNegativeBigInt = 0


# ============================================================
# 1. Str<N> / Text<N> length boundaries + null-byte rejection
# ============================================================

@pytest.mark.parametrize("field, limit", [("s16", 16), ("s64", 64), ("t1k", 1000)])
def test_str_alias_accepts_exactly_max_length(field: str, limit: int) -> None:
    m = FtStrModel(**{field: "a" * limit})
    assert getattr(m, field) == "a" * limit


@pytest.mark.parametrize("field, limit", [("s16", 16), ("s64", 64), ("t1k", 1000)])
def test_str_alias_rejects_max_length_plus_one(field: str, limit: int) -> None:
    with pytest.raises(ValidationError):
        FtStrModel(**{field: "a" * (limit + 1)})


def test_str_alias_accepts_empty_string() -> None:
    # Plain Str<N> has no min_length -- empty string is valid
    assert FtStrModel(s64="").s64 == ""


@pytest.mark.parametrize("field", ["s16", "s64", "t1k"])
def test_str_alias_rejects_null_byte(field: str) -> None:
    """PostgreSQL rejects NUL bytes in text columns; the alias must too."""
    with pytest.raises(ValidationError):
        FtStrModel(**{field: "ab\x00cd"})


# ============================================================
# 2. NonEmptyStr* / NonEmptyStrippedStr*
# ============================================================

def test_non_empty_str_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        FtNonEmptyModel(plain="")


def test_non_empty_str_allows_pure_whitespace() -> None:
    """NonEmptyStr* only enforces min_length=1 -- whitespace passes unchanged."""
    m = FtNonEmptyModel(plain="   ")
    assert m.plain == "   "


def test_non_empty_str_boundaries() -> None:
    assert FtNonEmptyModel(plain="a" * 64).plain == "a" * 64
    with pytest.raises(ValidationError):
        FtNonEmptyModel(plain="a" * 65)


def test_non_empty_stripped_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        FtNonEmptyModel(stripped="")


@pytest.mark.parametrize("value", ["   ", "\t", "\n", " \t\r\n "])
def test_non_empty_stripped_rejects_whitespace_only(value: str) -> None:
    with pytest.raises(ValidationError):
        FtNonEmptyModel(stripped=value)


def test_non_empty_stripped_strips_surrounding_whitespace() -> None:
    m = FtNonEmptyModel(stripped="  hello world \t")
    assert m.stripped == "hello world"


def test_non_empty_stripped_keeps_inner_whitespace() -> None:
    assert FtNonEmptyModel(stripped="a b").stripped == "a b"


def test_non_empty_stripped_max_length() -> None:
    assert FtNonEmptyModel(stripped="a" * 64).stripped == "a" * 64
    with pytest.raises(ValidationError):
        FtNonEmptyModel(stripped="a" * 65)


# ============================================================
# 3. Sha256Hex
# ============================================================

def test_sha256hex_accepts_valid_digest() -> None:
    digest = "a3f5" * 16  # 64 lowercase hex chars
    assert len(digest) == 64
    assert FtHashLangModel(digest=digest).digest == digest


@pytest.mark.parametrize("bad", [
    "A" * 64,                # uppercase hex rejected
    "g" * 64,                # non-hex letter
    "0" * 63,                # too short
    "0" * 65,                # too long
    "",                      # empty
    "0" * 63 + "G",          # one invalid char
])
def test_sha256hex_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValidationError):
        FtHashLangModel(digest=bad)


# ============================================================
# 4. BCP47LanguageCode
# ============================================================

@pytest.mark.parametrize("tag", [
    "zh", "en", "jpn", "zh-CN", "en-US", "zh-Hans", "zh-Hans-CN", "ja-JP",
])
def test_bcp47_accepts_valid_tags(tag: str) -> None:
    assert FtHashLangModel(lang=tag).lang == tag


@pytest.mark.parametrize("bad", [
    "a",                      # primary subtag too short
    "abcd",                   # primary subtag too long (>3 letters)
    "zh_CN",                  # underscore illegal
    "zh-",                    # dangling separator
    "zh-C",                   # subtag shorter than 2
    "12",                     # primary subtag must be letters
    "zh-Hans-CN-variant89",   # exceeds max_length=16
    "",
])
def test_bcp47_rejects_invalid_tags(bad: str) -> None:
    with pytest.raises(ValidationError):
        FtHashLangModel(lang=bad)


# ============================================================
# 5. Numeric aliases
# ============================================================

def test_int_constants_have_expected_values() -> None:
    assert INT32_MAX == 2**31 - 1 == 2147483647
    assert INT64_MAX == 2**63 - 1 == 9223372036854775807
    assert JS_MAX_SAFE_INTEGER == 2**53 - 1 == 9007199254740991


@pytest.mark.parametrize("value, ok", [
    (0, False),
    (1, True),
    (65535, True),
    (65536, False),
])
def test_port_boundaries(value: int, ok: bool) -> None:
    if ok:
        assert FtNumModel(port=value).port == value
    else:
        with pytest.raises(ValidationError):
            FtNumModel(port=value)


@pytest.mark.parametrize("value, ok", [
    (-1, False),
    (0, True),
    (100, True),
    (101, False),
])
def test_percentage_boundaries(value: int, ok: bool) -> None:
    if ok:
        assert FtNumModel(pct=value).pct == value
    else:
        with pytest.raises(ValidationError):
            FtNumModel(pct=value)


@pytest.mark.parametrize("value, ok", [
    (0, False),
    (1, True),
    (INT32_MAX, True),
    (INT32_MAX + 1, False),
])
def test_positive_int_boundaries(value: int, ok: bool) -> None:
    if ok:
        assert FtNumModel(pos=value).pos == value
    else:
        with pytest.raises(ValidationError):
            FtNumModel(pos=value)


@pytest.mark.parametrize("value, ok", [
    (-1, False),
    (0, True),
    (INT32_MAX, True),
    (INT32_MAX + 1, False),
])
def test_non_negative_int_boundaries(value: int, ok: bool) -> None:
    if ok:
        assert FtNumModel(nn=value).nn == value
    else:
        with pytest.raises(ValidationError):
            FtNumModel(nn=value)


@pytest.mark.parametrize("value, ok", [
    (0, False),
    (1, True),
    (JS_MAX_SAFE_INTEGER, True),
    (JS_MAX_SAFE_INTEGER + 1, False),  # bound is 2^53-1, not INT64_MAX
    (INT64_MAX, False),
])
def test_positive_big_int_boundaries(value: int, ok: bool) -> None:
    if ok:
        assert FtNumModel(pos_big=value).pos_big == value
    else:
        with pytest.raises(ValidationError):
            FtNumModel(pos_big=value)


@pytest.mark.parametrize("value, ok", [
    (-1, False),
    (0, True),
    (JS_MAX_SAFE_INTEGER, True),
    (JS_MAX_SAFE_INTEGER + 1, False),
])
def test_non_negative_big_int_boundaries(value: int, ok: bool) -> None:
    if ok:
        assert FtNumModel(nn_big=value).nn_big == value
    else:
        with pytest.raises(ValidationError):
            FtNumModel(nn_big=value)


def test_positive_float_excludes_zero() -> None:
    assert FtNumModel(pos_f=0.0001).pos_f == 0.0001
    with pytest.raises(ValidationError):
        FtNumModel(pos_f=0.0)
    with pytest.raises(ValidationError):
        FtNumModel(pos_f=-1.0)


def test_non_negative_float_includes_zero() -> None:
    assert FtNumModel(nn_f=0.0).nn_f == 0.0
    with pytest.raises(ValidationError):
        FtNumModel(nn_f=-0.001)


# ============================================================
# 6. List<N>[T] bounded lists / List[T] alias
# ============================================================

def test_list_alias_is_transparent() -> None:
    """List[T] resolves to plain list[T] at runtime."""
    assert List[int] == list[int]
    assert List[str] == list[str]


@pytest.mark.parametrize("field, limit", [("one", 1), ("three", 3), ("sixteen", 16)])
def test_list_n_accepts_up_to_limit(field: str, limit: int) -> None:
    items = ["x"] * limit if field == "three" else list(range(limit))
    m = FtListModel(**{field: items})
    assert len(getattr(m, field)) == limit


@pytest.mark.parametrize("field, limit", [("one", 1), ("three", 3), ("sixteen", 16)])
def test_list_n_rejects_limit_plus_one(field: str, limit: int) -> None:
    items = ["x"] * (limit + 1) if field == "three" else list(range(limit + 1))
    with pytest.raises(ValidationError):
        FtListModel(**{field: items})


def test_list_n_accepts_empty() -> None:
    m = FtListModel(one=[], three=[], sixteen=[])
    assert m.one == [] and m.three == [] and m.sixteen == []


def test_list_n_validates_item_type() -> None:
    with pytest.raises(ValidationError):
        FtListModel(sixteen=["not-an-int"])


# ============================================================
# 7. SA column type mapping on a table model
# ============================================================

class TestSaColumnMapping:
    @pytest.mark.parametrize("col, length", [
        ("s16", 16),
        ("s64", 64),
        ("t1k", 1000),
        ("nes", 64),
        ("lang", 16),
    ])
    def test_string_aliases_map_to_varchar_with_length(self, col: str, length: int) -> None:
        col_type = FtStrNumColumnTable.__table__.c[col].type
        # AutoString is a TypeDecorator whose impl is VARCHAR; ``length`` drives DDL
        assert col_type.length == length
        assert col_type.compile() == f"VARCHAR({length})"

    def test_int_aliases_map_to_integer(self) -> None:
        assert isinstance(FtStrNumColumnTable.__table__.c.port.type, sa.Integer)
        assert isinstance(FtStrNumColumnTable.__table__.c.pos.type, sa.Integer)

    def test_big_int_aliases_map_to_biginteger(self) -> None:
        assert isinstance(FtStrNumColumnTable.__table__.c.pos_big.type, sa.BigInteger)
        assert isinstance(FtStrNumColumnTable.__table__.c.nn_big.type, sa.BigInteger)
