"""
Invariant tests for the Decimal field-type ladder.

Covers (based on ``NonNegativeDecimal38_18`` and friends from
``sqlmodel_ext.field_types``):

1. **Fixed-point JSON output**: ``model_dump_json()`` never emits scientific
   notation (``0E-18``) and strips redundant trailing zeros.
2. **JSON output is a string, not a number**: protects JS clients from IEEE
   754 double precision loss (decimal.js contract).
3. **dict mode keeps Decimal**: ``model_dump()`` (non-json) returns ``Decimal``
   instances, not ``str``.
4. **Float rejection**: any ``float`` / ``bool`` input raises ValidationError
   (the value already lost precision via IEEE 754).
5. **int / str / Decimal acceptance**: lossless coercion paths stay open.
6. **Optional null parsing**: the nested-Annotated structure of
   ``OptionalNonNegativeDecimal38_18`` must accept JSON ``null`` (regression:
   constraints on the outer layer crash Pydantic on None).
7. **Integer-digit limit**: every alias rejects one integer digit too many
   (regression: ``_REJECT_FLOAT`` before ``Field`` disabled that check).
8. **Write (35-digit) / sum (38-digit) aliases** for NUMERIC(38, 18) columns
   that are aggregated with ``SUM()``.
9. **OpenAPI validation pattern** generated from the digit counts.

Pure Pydantic-layer tests -- no DB, near-zero run cost.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Annotated, Any

import pytest
from pydantic import BeforeValidator, TypeAdapter, ValidationError
from sqlalchemy import Numeric
from sqlmodel import Field

import sqlmodel_ext.field_types as _ft
from sqlmodel_ext import (
    NonNegativeDecimal20_10,
    NonNegativeDecimal38_18,
    OptionalNonNegativeDecimal20_10,
    OptionalNonNegativeDecimal38_18,
    PositiveDecimal38_18,
    SignedDecimal20_10,
    SignedDecimal38_18,
    SQLModelBase,
)


class DecimalModel(SQLModelBase):
    """Test DTO covering every 38_18 alias variant."""
    balance: NonNegativeDecimal38_18
    delta: SignedDecimal38_18
    price: PositiveDecimal38_18
    optional_amount: OptionalNonNegativeDecimal38_18 = None


class RateModel(SQLModelBase):
    """Test DTO covering the 20_10 alias variants."""
    rate: NonNegativeDecimal20_10
    signed_rate: SignedDecimal20_10
    optional_rate: OptionalNonNegativeDecimal20_10 = None


# ============================================================
# 1. Fixed-point output, no scientific notation
# ============================================================

@pytest.mark.parametrize('input_decimal, expected_output', [
    # Every Decimal representation of zero normalizes to '0'
    (Decimal('0').quantize(Decimal('1e-18')), '0'),
    (Decimal('-0').quantize(Decimal('1e-18')), '0'),
    (Decimal('0'), '0'),
    (Decimal('0.0'), '0'),
    (Decimal('0E+5'), '0'),
    # The 18 trailing zeros from a NUMERIC(38, 18) round-trip are stripped
    (Decimal('1200.000000000000000000'), '1200'),
    (Decimal('-50.000000000000000000'), '-50'),
    # Partial trailing zeros stripped down to the first non-zero digit
    (Decimal('0.500000000000000000'), '0.5'),
    (Decimal('100.123450000000000000'), '100.12345'),
    # In-precision fractions preserved as-is
    (Decimal('0.5'), '0.5'),
    (Decimal('-1.5'), '-1.5'),
    # Tiny values must expand to fixed-point, never scientific notation
    (Decimal('0.000000000000000001'), '0.000000000000000001'),
    (Decimal('1E-18'), '0.000000000000000001'),
    # Integral Decimals get no dangling '.'
    (Decimal('100'), '100'),
    (Decimal('1E+5'), '100000'),
    # NUMERIC(38, 18) extreme: 36 significant digits fully preserved
    # (key regression -- ``normalize()`` under the default prec=28 context
    # silently rounds this to ``1E+18``)
    (Decimal('999999999999999999.999999999999999999'),
     '999999999999999999.999999999999999999'),
    (Decimal('123456789.123456789012345678'), '123456789.123456789012345678'),
])
def test_json_output_is_fixed_point_no_scientific(
    input_decimal: Decimal,
    expected_output: str,
) -> None:
    """Every Decimal JSON output is a fixed-point string, trailing zeros stripped, zeros normalized."""
    m = DecimalModel(
        balance=input_decimal if input_decimal >= 0 else Decimal(0),
        delta=input_decimal,  # SignedDecimal38_18 accepts negatives
        price=Decimal('1'),
    )
    parsed = json.loads(m.model_dump_json())
    assert parsed['delta'] == expected_output, (
        f'input {input_decimal!r}: expected {expected_output!r}, got {parsed["delta"]!r}'
    )


@pytest.mark.parametrize('input_decimal', [
    Decimal('0').quantize(Decimal('1e-18')),
    Decimal('-0E-18'),
    Decimal('1200.000000000000000000'),
    Decimal('0.000000000000000001'),
    Decimal('1E-18'),
    Decimal('1E+18'),
    Decimal('999999999999999999.999999999999999999'),
])
def test_json_output_contains_no_scientific_marker(input_decimal: Decimal) -> None:
    """No Decimal field value may ever contain an 'E' / 'e' scientific marker."""
    m = DecimalModel(
        balance=input_decimal if input_decimal >= 0 else Decimal(0),
        delta=input_decimal,
        price=Decimal('1'),
    )
    parsed = json.loads(m.model_dump_json())
    for key in ('balance', 'delta', 'price'):
        value = parsed[key]
        assert isinstance(value, str)
        assert 'E' not in value and 'e' not in value, (
            f'field {key!r} contains scientific notation: {value!r}'
        )


def test_json_roundtrip_preserves_full_precision() -> None:
    """JSON round-trip of an 18-decimal-place value is exact."""
    value = Decimal('123456789.123456789012345678')
    m = DecimalModel(balance=value, delta=value, price=value)
    restored = DecimalModel.model_validate_json(m.model_dump_json())
    assert restored.balance == value
    assert restored.delta == value
    assert restored.price == value


# ============================================================
# 2/3. JSON string vs dict-mode Decimal
# ============================================================

def test_json_output_is_string_not_number() -> None:
    m = DecimalModel(balance=Decimal('1.5'), delta=Decimal('-2'), price=Decimal('3'))
    parsed = json.loads(m.model_dump_json())
    assert isinstance(parsed['balance'], str)
    assert isinstance(parsed['delta'], str)
    assert isinstance(parsed['price'], str)


def test_model_dump_dict_preserves_decimal_type() -> None:
    m = DecimalModel(balance=Decimal('1.5'), delta=Decimal('-2'), price=Decimal('3'))
    dumped = m.model_dump()
    assert isinstance(dumped['balance'], Decimal)
    assert isinstance(dumped['delta'], Decimal)
    assert isinstance(dumped['price'], Decimal)


# ============================================================
# 4. Float / bool rejection
# ============================================================

def test_rejects_float_input() -> None:
    with pytest.raises(ValidationError, match='[Ff]loat'):
        DecimalModel(balance=0.5, delta=Decimal(0), price=Decimal('1'))  # type: ignore[arg-type]


def test_rejects_bool_input() -> None:
    with pytest.raises(ValidationError, match='[Bb]oolean'):
        DecimalModel(balance=True, delta=Decimal(0), price=Decimal('1'))  # type: ignore[arg-type]


def test_rejects_float_via_json_number() -> None:
    """A JSON float number must be rejected at the API boundary."""
    with pytest.raises(ValidationError):
        DecimalModel.model_validate_json(
            '{"balance": 0.5, "delta": "0", "price": "1"}'
        )


# ============================================================
# 5. int / str / Decimal acceptance
# ============================================================

@pytest.mark.parametrize('value', [0, 5, '0.5', '123.456', Decimal('7.25')])
def test_accepts_int_str_decimal(value: int | str | Decimal) -> None:
    m = DecimalModel(balance=value, delta=Decimal(0), price=Decimal('1'))  # type: ignore[arg-type]
    assert isinstance(m.balance, Decimal)
    assert m.balance == Decimal(str(value))


# ============================================================
# 6. Sign constraints + Optional null parsing
# ============================================================

def test_nonnegative_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        DecimalModel(balance=Decimal('-1'), delta=Decimal(0), price=Decimal('1'))


def test_positive_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        DecimalModel(balance=Decimal(0), delta=Decimal(0), price=Decimal(0))


def test_signed_accepts_negative() -> None:
    m = DecimalModel(balance=Decimal(0), delta=Decimal('-99.5'), price=Decimal('1'))
    assert m.delta == Decimal('-99.5')


def test_optional_accepts_none_and_json_null() -> None:
    """Regression (nested-Annotated): JSON ``null`` must parse without crashing.

    If the Ge / max_digits constraints sat on the outer ``Decimal | None``
    layer, Pydantic would raise
    ``TypeError: Unable to apply constraint 'ge' to supplied value None``.
    """
    m = DecimalModel(balance=Decimal(0), delta=Decimal(0), price=Decimal('1'))
    assert m.optional_amount is None

    restored = DecimalModel.model_validate_json(
        '{"balance": "0", "delta": "0", "price": "1", "optional_amount": null}'
    )
    assert restored.optional_amount is None

    with_value = DecimalModel.model_validate_json(
        '{"balance": "0", "delta": "0", "price": "1", "optional_amount": "1.25"}'
    )
    assert with_value.optional_amount == Decimal('1.25')


def test_optional_rejects_negative_when_present() -> None:
    """The inner Ge(0) constraint still applies to non-None values."""
    with pytest.raises(ValidationError):
        DecimalModel.model_validate_json(
            '{"balance": "0", "delta": "0", "price": "1", "optional_amount": "-1"}'
        )


# ============================================================
# 20_10 variants (same machinery, narrower precision)
# ============================================================

def test_20_10_fixed_point_and_null() -> None:
    m = RateModel(rate=Decimal('0.0000000001'), signed_rate=Decimal('-1.5'))
    parsed = json.loads(m.model_dump_json())
    assert parsed['rate'] == '0.0000000001'
    assert parsed['signed_rate'] == '-1.5'
    assert parsed['optional_rate'] is None

    restored = RateModel.model_validate_json(
        '{"rate": "1", "signed_rate": "0", "optional_rate": null}'
    )
    assert restored.optional_rate is None


# ============================================================
# 7. Integer-digit limit is enforced (metadata order regression)
# ============================================================

# (alias name, max integer digits)
_DECIMAL_ALIASES: list[tuple[str, int]] = [
    ('SignedDecimal38_18', 20),
    ('NonNegativeDecimal38_18', 20),
    ('PositiveDecimal38_18', 20),
    ('OptionalNonNegativeDecimal38_18', 20),
    ('OptionalSignedDecimal38_18', 20),
    ('SignedWriteDecimal38_18', 17),
    ('NonNegativeWriteDecimal38_18', 17),
    ('PositiveWriteDecimal38_18', 17),
    ('OptionalNonNegativeWriteDecimal38_18', 17),
    ('OptionalSignedWriteDecimal38_18', 17),
    ('SignedSumDecimal38_18', 20),
    ('SignedDecimal20_10', 10),
    ('NonNegativeDecimal20_10', 10),
    ('OptionalNonNegativeDecimal20_10', 10),
    ('NullableNonNegativeDecimal20_10', 10),
]


@pytest.mark.parametrize('name, whole_digits', _DECIMAL_ALIASES)
def test_integer_digit_limit_enforced(name: str, whole_digits: int) -> None:
    """Every Decimal alias rejects one integer digit too many.

    Regression: with ``_REJECT_FLOAT`` placed before ``Field(max_digits=...)``
    Pydantic falls back to validators that check only total digits and decimal
    places, so e.g. a 15-digit integer passed a NUMERIC(20, 10) alias.
    """
    adapter = TypeAdapter(getattr(_ft, name))
    at_limit = Decimal('9' * whole_digits)
    assert adapter.validate_python(at_limit) == at_limit
    with pytest.raises(ValidationError, match='before the decimal point'):
        adapter.validate_python(Decimal('9' * (whole_digits + 1)))


@pytest.mark.parametrize('name, whole_digits', _DECIMAL_ALIASES)
def test_every_alias_still_rejects_float_and_bool(name: str, whole_digits: int) -> None:
    """Moving ``_REJECT_FLOAT`` after ``Field`` keeps the float / bool rejection."""
    adapter = TypeAdapter(getattr(_ft, name))
    with pytest.raises(ValidationError, match='[Ff]loat'):
        adapter.validate_python(1.5)
    with pytest.raises(ValidationError, match='[Bb]oolean'):
        adapter.validate_python(True)
    assert adapter.validate_python('1.5') == Decimal('1.5')
    assert adapter.validate_python(1) == Decimal(1)


def test_before_validator_first_would_skip_integer_digit_check() -> None:
    """Pins the Pydantic behaviour the ordering rule exists for.

    If Pydantic ever enforces integer digits regardless of order, this test
    fails and the ordering comment in ``field_types`` can be relaxed.
    """
    mutated: Any = Annotated[
        Decimal,
        BeforeValidator(lambda value: value),
        Field(max_digits=20, decimal_places=10),
    ]
    assert TypeAdapter(mutated).validate_python(Decimal('9' * 15)) == Decimal('9' * 15)


# ============================================================
# 8. Write-side (35 digits) / sum-side (38 digits) aliases
# ============================================================

def test_write_digit_constants() -> None:
    assert _ft.DECIMAL_38_18_COLUMN_DIGITS == 38
    assert _ft.DECIMAL_38_18_WRITE_DIGITS == 35
    assert _ft.DECIMAL_38_18_PLACES == 18


def test_write_side_accepts_35_digits_and_18_places() -> None:
    adapter = TypeAdapter(_ft.SignedWriteDecimal38_18)
    extreme = Decimal('99999999999999999.999999999999999999')
    assert adapter.validate_python(extreme) == extreme
    negative = Decimal('-99999999999999999.999999999999999999')  # literal: unary minus rounds to prec=28
    assert adapter.validate_python(negative) == negative
    assert json.loads(adapter.dump_json(extreme)) == '99999999999999999.999999999999999999'


def test_sum_side_accepts_what_write_side_rejects() -> None:
    """1000 maximal write-side rows still sum to a valid ``SignedSumDecimal38_18``."""
    # row * 1000, written as a literal: Decimal arithmetic would round to the
    # default context precision (28 digits)
    total = Decimal('99999999999999999999.999999999999999000')
    with pytest.raises(ValidationError):
        TypeAdapter(_ft.SignedWriteDecimal38_18).validate_python(total)
    assert TypeAdapter(_ft.SignedSumDecimal38_18).validate_python(total) == total


@pytest.mark.parametrize('name', [
    'SignedWriteDecimal38_18',
    'NonNegativeWriteDecimal38_18',
    'PositiveWriteDecimal38_18',
    'OptionalNonNegativeWriteDecimal38_18',
    'OptionalSignedWriteDecimal38_18',
])
def test_write_side_column_stays_numeric_38_18(name: str) -> None:
    """The column is NUMERIC(38, 18) even though Pydantic accepts only 35 digits.

    Without an explicit ``sa_type`` SQLModel would derive NUMERIC(35, 18) from
    ``max_digits``.
    """
    sa_types = [
        m.sa_type for m in getattr(_ft, name).__metadata__
        if isinstance(getattr(m, 'sa_type', None), Numeric)
    ]
    assert len(sa_types) == 1
    assert (sa_types[0].precision, sa_types[0].scale) == (38, 18)


def test_sum_side_has_no_sa_type() -> None:
    assert not any(
        isinstance(getattr(m, 'sa_type', None), Numeric)
        for m in _ft.SignedSumDecimal38_18.__metadata__
    )


def test_optional_write_side_accepts_none_and_bounds_values() -> None:
    signed = TypeAdapter(_ft.OptionalSignedWriteDecimal38_18)
    non_negative = TypeAdapter(_ft.OptionalNonNegativeWriteDecimal38_18)
    assert signed.validate_json('null') is None
    assert signed.validate_json('"-1"') == Decimal('-1')
    assert non_negative.validate_json('null') is None
    with pytest.raises(ValidationError):
        non_negative.validate_json('"-1"')


def test_nullable_20_10_is_required_but_accepts_null() -> None:
    class FtNullableRate(SQLModelBase):
        rate: _ft.NullableNonNegativeDecimal20_10

    with pytest.raises(ValidationError, match='[Ff]ield required'):
        FtNullableRate.model_validate({})
    assert FtNullableRate.model_validate({'rate': None}).rate is None
    assert FtNullableRate.model_validate({'rate': '1.5'}).rate == Decimal('1.5')


# ============================================================
# 9. OpenAPI validation pattern (single source of truth)
# ============================================================

def test_pattern_generator_reproduces_original_literals() -> None:
    assert _ft._decimal_str_pattern(20, 18) == (
        r'^(?!^[-+.]*$)[+-]?0*(?:\d{0,20}|(?=[\d.]{1,39}0*$)\d{0,20}\.\d{0,18}0*$)'
    )
    assert _ft._decimal_str_pattern(10, 10) == (
        r'^(?!^[-+.]*$)[+-]?0*(?:\d{0,10}|(?=[\d.]{1,21}0*$)\d{0,10}\.\d{0,10}0*$)'
    )


@pytest.mark.parametrize('name, whole_digits, places', [
    ('SignedDecimal38_18', 20, 18),
    ('SignedWriteDecimal38_18', 17, 18),
    ('SignedSumDecimal38_18', 20, 18),
    ('SignedDecimal20_10', 10, 10),
])
def test_validation_schema_is_string_with_matching_pattern(name: str, whole_digits: int, places: int) -> None:
    schema = TypeAdapter(getattr(_ft, name)).json_schema(mode='validation')
    assert schema == {'type': 'string', 'pattern': _ft._decimal_str_pattern(whole_digits, places)}
