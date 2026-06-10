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

Pure Pydantic-layer tests -- no DB, near-zero run cost.
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

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
