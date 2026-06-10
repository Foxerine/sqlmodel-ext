"""
sqlmodel_ext.field_types -- Reusable type aliases and custom types for SQLModel.

Provides constrained string/numeric types, path types, URL types, and IP address types,
all compatible with Pydantic validation and SQLAlchemy column mapping.
"""
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Generic, TypeAlias, TypeVar

from annotated_types import Ge, Gt
from pydantic import BeforeValidator, PlainSerializer, StringConstraints
from sqlalchemy import BigInteger, Numeric
from sqlmodel import Field

from ._internal.path import _DirectoryPathHandler, _FilePathHandler
from .ip_address import IPAddress
from .mixins import ModuleNameMixin
from .url import HttpUrl, SafeHttpUrl, Url, WebSocketUrl

# Re-export SSRF utilities
from ._ssrf import UnsafeURLError, validate_not_private_host

# ---------------------------------------------------------------------------
#  Public, Database-Agnostic Types
# ---------------------------------------------------------------------------

DirectoryPathType = Annotated[Path, _DirectoryPathHandler]
"""
A directory path type compatible with Pydantic and SQLModel.

Validates that the path should not contain a file extension,
while behaving as a ``pathlib.Path`` in Python code.
"""

FilePathType = Annotated[Path, _FilePathHandler]
"""
A file path type compatible with Pydantic and SQLModel.

Validates that the path must contain a filename component,
while behaving as a ``pathlib.Path`` in Python code.
"""


# ---------------------------------------------------------------------------
#  Field Constraint Type Aliases (Annotated Style)
# ---------------------------------------------------------------------------

_NO_NULL_BYTE = StringConstraints(pattern=r'^[^\x00]*$')
"""PostgreSQL rejects null bytes in text columns. pydantic-core compiles the regex once with zero Python overhead."""

# String length constraints
Str16: TypeAlias = Annotated[str, Field(max_length=16), _NO_NULL_BYTE]
"""16-character string field (trigger words, short tokens)"""

Str24: TypeAlias = Annotated[str, Field(max_length=24), _NO_NULL_BYTE]
"""24-character string field"""

Str32: TypeAlias = Annotated[str, Field(max_length=32), _NO_NULL_BYTE]
"""32-character string field"""

Str36: TypeAlias = Annotated[str, Field(max_length=36), _NO_NULL_BYTE]
"""36-character string field (UUID standard format length)"""

Str48: TypeAlias = Annotated[str, Field(max_length=48), _NO_NULL_BYTE]
"""48-character string field"""

Str64: TypeAlias = Annotated[str, Field(max_length=64), _NO_NULL_BYTE]
"""64-character string field"""

Str100: TypeAlias = Annotated[str, Field(max_length=100), _NO_NULL_BYTE]
"""100-character string field"""

Str128: TypeAlias = Annotated[str, Field(max_length=128), _NO_NULL_BYTE]
"""128-character string field"""

Str255: TypeAlias = Annotated[str, Field(max_length=255), _NO_NULL_BYTE]
"""255-character string field"""

Str256: TypeAlias = Annotated[str, Field(max_length=256), _NO_NULL_BYTE]
"""256-character string field"""

Str500: TypeAlias = Annotated[str, Field(max_length=500), _NO_NULL_BYTE]
"""500-character string field"""

Str512: TypeAlias = Annotated[str, Field(max_length=512), _NO_NULL_BYTE]
"""512-character string field"""

Str2048: TypeAlias = Annotated[str, Field(max_length=2048), _NO_NULL_BYTE]
"""2048-character string field (URLs etc.)"""

Text1K: TypeAlias = Annotated[str, Field(max_length=1000), _NO_NULL_BYTE]
"""1000-character text field"""

Text1024: TypeAlias = Annotated[str, Field(max_length=1024), _NO_NULL_BYTE]
"""1024-character text field"""

Text2K: TypeAlias = Annotated[str, Field(max_length=2000), _NO_NULL_BYTE]
"""2000-character text field"""

Text2500: TypeAlias = Annotated[str, Field(max_length=2500), _NO_NULL_BYTE]
"""2500-character text field"""

Text3K: TypeAlias = Annotated[str, Field(max_length=3000), _NO_NULL_BYTE]
"""3000-character text field"""

Text5K: TypeAlias = Annotated[str, Field(max_length=5000), _NO_NULL_BYTE]
"""5000-character text field"""

Text8K: TypeAlias = Annotated[str, Field(max_length=8000), _NO_NULL_BYTE]
"""8000-character text field (user-editable long descriptions; a balance
between DoS ceiling and expressiveness — more room than Text5K, stricter
than Text10K)"""

Text10K: TypeAlias = Annotated[str, Field(max_length=10000), _NO_NULL_BYTE]
"""10000-character text field"""

Text16K: TypeAlias = Annotated[str, Field(max_length=16000), _NO_NULL_BYTE]
"""16000-character text field (long-form descriptions, tool docs)"""

Text32K: TypeAlias = Annotated[str, Field(max_length=32000), _NO_NULL_BYTE]
"""32000-character text field"""

Text48K: TypeAlias = Annotated[str, Field(max_length=48000), _NO_NULL_BYTE]
"""48000-character text field (large system prompts etc.)"""

Text60K: TypeAlias = Annotated[str, Field(max_length=60000), _NO_NULL_BYTE]
"""60000-character text field"""

Text64K: TypeAlias = Annotated[str, Field(max_length=65536), _NO_NULL_BYTE]
"""65536-character text field"""

Text100K: TypeAlias = Annotated[str, Field(max_length=100000), _NO_NULL_BYTE]
"""100000-character text field"""

Text128K: TypeAlias = Annotated[str, Field(max_length=131072), _NO_NULL_BYTE]
"""131072-character (128 * 1024) text field (large markdown documents etc.)"""

Text1M: TypeAlias = Annotated[str, Field(max_length=1000000), _NO_NULL_BYTE]
"""1000000-character text field (tool call parameters, tool responses, etc.)"""

# NonEmptyStr* — same as Str* but also requires ``min_length=1``, rejecting empty
# ``""`` strings with a 422 ValidationError. Use for naming fields where the empty
# string is semantically invalid (e.g. UserFolder.name).
NonEmptyStr64: TypeAlias = Annotated[str, Field(min_length=1, max_length=64), _NO_NULL_BYTE]
"""1-64 character non-empty string field"""

NonEmptyStr128: TypeAlias = Annotated[str, Field(min_length=1, max_length=128), _NO_NULL_BYTE]
"""1-128 character non-empty string field"""

NonEmptyStr256: TypeAlias = Annotated[str, Field(min_length=1, max_length=256), _NO_NULL_BYTE]
"""1-256 character non-empty string field"""

# NonEmptyStrippedStr* — rejects both ``""`` and whitespace-only strings
# (``"   "`` / ``"\t"``). Compared with ``NonEmptyStr*`` (which only enforces
# ``min_length=1`` and therefore lets pure whitespace through), these declare
# ``Field(min_length=1)`` (pre-strip) plus
# ``StringConstraints(strip_whitespace=True, min_length=1)`` (post-strip), a
# two-layer guard for user-visible required naming fields where a blank or
# whitespace-only name degrades into an unidentifiable empty block in UIs,
# search results, and share links.

NonEmptyStrippedStr64: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=64),
    StringConstraints(strip_whitespace=True, min_length=1),
    _NO_NULL_BYTE,
]
"""1-64 character string, non-empty after stripping (rejects ``""`` and pure whitespace)"""

NonEmptyStrippedStr128: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=128),
    StringConstraints(strip_whitespace=True, min_length=1),
    _NO_NULL_BYTE,
]
"""1-128 character string, non-empty after stripping (rejects ``""`` and pure whitespace)"""

NonEmptyStrippedStr256: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=256),
    StringConstraints(strip_whitespace=True, min_length=1),
    _NO_NULL_BYTE,
]
"""1-256 character string, non-empty after stripping (rejects ``""`` and pure whitespace)"""

# Sha256Hex — exactly 64 lowercase hex characters, the canonical SHA-256 hex digest form.
# The strict regex also implicitly rejects NUL bytes, so no separate _NO_NULL_BYTE is needed.
Sha256Hex: TypeAlias = Annotated[str, StringConstraints(min_length=64, max_length=64, pattern=r'^[0-9a-f]{64}$')]
"""64-char lowercase-hex SHA-256 digest (e.g. content hashes)"""

BCP47LanguageCode: TypeAlias = Annotated[
    str,
    Field(max_length=16),
    StringConstraints(min_length=2, pattern=r'^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$'),
]
"""BCP-47 (RFC 5646) language tag primitive.

Valid examples: ``'zh'`` / ``'en'`` / ``'zh-CN'`` / ``'en-US'`` / ``'zh-Hans'``
/ ``'zh-Hans-CN'`` / ``'ja-JP'``.
Invalid examples: ``'a'`` (too short) / non-ASCII tags / ``'zh_CN'``
(underscore is illegal) / tags longer than 16 characters.

Simplified validation: a 2-3 letter primary subtag followed by zero or more
``-`` subtags (script / region / variant), each 2-8 alphanumerics. No IANA
registry lookup is performed — this validates syntax, not semantics.
``max_length=16`` exceeds every common locale string (``zh-Hans-CN`` is only
10 characters). The length upper bound lives in ``Field`` so it drives the SA
column type, while pattern + min_length live in ``StringConstraints`` for
Pydantic validation.
"""

# Numeric range constraints
Port: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
"""Port number (1-65535)"""

Percentage: TypeAlias = Annotated[int, Field(ge=0, le=100)]
"""Percentage (0-100)"""

INT32_MAX = 2147483647
"""Maximum value for PostgreSQL INTEGER column (2^31-1)"""

INT64_MAX = 9223372036854775807
"""Maximum value for PostgreSQL BIGINT column (2^63-1)"""

JS_MAX_SAFE_INTEGER = 9007199254740991
"""JavaScript ``Number.MAX_SAFE_INTEGER`` (2^53-1).

Integers larger than this value lose precision when parsed by JavaScript
clients. Use this as the upper bound for BigInt fields that cross the API
boundary into a JSON body consumed by browsers.
"""

PositiveInt: TypeAlias = Annotated[int, Field(ge=1, le=INT32_MAX)]
"""Positive integer (1 to 2147483647, fits PostgreSQL INTEGER)"""

NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0, le=INT32_MAX)]
"""Non-negative integer (0 to 2147483647, fits PostgreSQL INTEGER)"""

PositiveBigInt: TypeAlias = Annotated[int, Field(ge=1, le=JS_MAX_SAFE_INTEGER, sa_type=BigInteger)]
"""Positive big integer (1 to JS_MAX_SAFE_INTEGER, stored as PostgreSQL BIGINT).

The upper bound is JS_MAX_SAFE_INTEGER (2^53-1) rather than INT64_MAX so
values serialized to JSON remain exact in JavaScript clients. Raise the
bound to ``INT64_MAX`` explicitly in a custom alias if the field is never
consumed by a browser.
"""

NonNegativeBigInt: TypeAlias = Annotated[int, Field(ge=0, le=JS_MAX_SAFE_INTEGER, sa_type=BigInteger)]
"""Non-negative big integer (0 to JS_MAX_SAFE_INTEGER, stored as PostgreSQL BIGINT).

See :data:`PositiveBigInt` for the rationale behind the JS_MAX_SAFE_INTEGER bound.
"""

PositiveFloat: TypeAlias = Annotated[float, Field(gt=0.0)]
"""Positive float (>0)"""

NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0.0)]
"""Non-negative float (>=0)"""

# ---------------------------------------------------------------------------
#  Decimal Numeric Constraints (NUMERIC(precision, scale) + sign constraint)
# ---------------------------------------------------------------------------
# Naming convention: ``[Optional][Sign]Decimal<P>_<S>`` i.e.
# ``[Optional][Signed|NonNegative|Positive]Decimal{precision}_{scale}``.
#
# - Consistent with the ``NonNegativeInt`` / ``NonNegativeBigInt`` naming habit
# - The DB column type ``NUMERIC(precision, scale)`` is encoded in the type
#   name (precision = total digits, scale = fractional digits)
# - ``PlainSerializer(when_used='json')`` serializes Decimal to a JSON string,
#   avoiding JS Number precision loss (IEEE 754 double only holds ~15
#   significant digits, which conflicts with high-precision Decimal fields)
# - ``model_dump()`` (dict mode) keeps the Decimal object; only
#   ``model_dump_json()`` takes the string path
# - Business-named aliases (currency amounts, token rates, ...) should live in
#   their own domain modules as ``TypeAlias =`` references to these primitives


def _decimal_to_json_str(v: Decimal | None) -> str | None:
    """Decimal → fixed-point JSON string (no scientific notation, no trailing zeros).

    Solves two problems:

    1. **Scientific notation leaking into the API**: ``str(Decimal('0E-18'))``
       returns ``'0E-18'`` — Python ``Decimal.__str__`` switches to scientific
       notation when the coefficient is zero or very small, which is hostile
       to frontends and third-party consumers.
    2. **Meaningless 18-digit zero tails after a NUMERIC(38, 18) round-trip**:
       ``Decimal('1200')`` comes back from the database as
       ``Decimal('1200.000000000000000000')`` whose ``str()`` output is pure
       visual noise.

    Algorithm: ``format(v, 'f')`` expands the Decimal into a fixed-point
    string (**without** going through the Decimal arithmetic context — a pure
    string expansion with zero precision-loss risk), then manually trims
    trailing zeros and a dangling decimal point.

    Examples:

    - ``Decimal('0E-18')``  → ``'0'``
    - ``Decimal('-0E-18')`` → ``'0'`` (negative zero normalized)
    - ``Decimal('1200.000000000000000000')`` → ``'1200'``
    - ``Decimal('0.500000000000000000')`` → ``'0.5'``
    - ``Decimal('0.000000000000000001')`` → ``'0.000000000000000001'`` (precision kept)
    - ``Decimal('999999999999999999.999999999999999999')`` →
      ``'999999999999999999.999999999999999999'`` (all 36 digits kept)

    Pitfall avoided: ``Decimal.normalize()`` looks like it strips trailing
    zeros, but it is bound by ``getcontext().prec`` (default 28) — extreme
    NUMERIC(38, 18) values with more than 28 significant digits would be
    silently rounded. This implementation is pure string post-processing,
    decoupled from the Decimal arithmetic context.
    """
    if v is None:
        return None
    if v.is_zero():
        # Also normalizes ``Decimal('-0')`` / ``Decimal('0E-N')`` / ``Decimal('0E+N')``
        return '0'
    s = format(v, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


_DECIMAL_TO_JSON_STR = PlainSerializer(_decimal_to_json_str, when_used='json')
"""Decimal → JSON string serializer (json mode only; dict mode keeps the Decimal object).

Prevents JS Number precision loss: a JSON number is a double in JavaScript and
loses precision past ~15 significant digits; the string path lets frontends
parse exactly with decimal.js or similar.
"""


def _reject_float_decimal_input(v: Any) -> Any:
    """BeforeValidator for Decimal fields: reject float / bool input.

    Lets Decimal / int / str through for Pydantic's default Decimal coercion:

    - ``Decimal('0.5')`` → ✓ pass through
    - ``0`` (int) → ✓ coerced to ``Decimal(0)`` (int → Decimal is lossless)
    - ``'0.5'`` (str) → ✓ coerced to ``Decimal('0.5')`` (exact string parse)
    - ``0.5`` (float) → ✗ rejected (precision already lost via IEEE 754)
    - ``True`` (bool) → ✗ rejected (bool is an int subclass, semantically nonsense)

    Note: Pydantic v2's ``Strict()`` annotation is too strict for Decimal
    fields (it rejects even int), defeating "Python code may construct with
    int literals". This validator implements exactly "reject float, allow the
    rest".
    """
    if isinstance(v, bool):
        raise ValueError(
            'Boolean input rejected for Decimal field; '
            'use Decimal/int/str instead'
        )
    if isinstance(v, float):
        raise ValueError(
            f"Float input rejected for Decimal field (got {v!r}); "
            "use string ('0.5') or Decimal(Decimal('0.5')) instead. "
            "Float values have already lost precision via IEEE 754 representation."
        )
    return v


_REJECT_FLOAT = BeforeValidator(_reject_float_decimal_input)
"""BeforeValidator for Decimal fields: reject float / bool, allow Decimal / int / str.

- API boundary convention: JSON strings (``"123.45"``) — ✓ accepted
- Backend Python: ``Decimal(123)`` or int ``123`` — ✓ accepted
- Frontend "forgot toString" JSON number ``123.45`` — ✗ 422 rejected
  (a JS Number is already IEEE 754; precision is gone by the time it arrives)

This promotes "string at the boundary" from a documentation convention into a
hard type-system contract.
"""

# NUMERIC(38, 18) — 20 integer digits + 18 fractional digits (matches the EVM
# wei = 1e-18 ether de-facto standard for high-precision amounts)

SignedDecimal38_18: TypeAlias = Annotated[
    Decimal,
    _REJECT_FLOAT,
    Field(max_digits=38, decimal_places=18, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
]
"""NUMERIC(38, 18) Decimal, positive or negative"""

NonNegativeDecimal38_18: TypeAlias = Annotated[
    Decimal,
    _REJECT_FLOAT,
    Ge(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (the SQLModel stub annotates it
    # ``type[Any]`` but the runtime accepts SA type instances like
    # ``Numeric(38, 18)``); ``ge`` is expressed as ``Ge(Decimal(0))`` via
    # annotated_types so the ignore scope stays minimal.
    Field(max_digits=38, decimal_places=18, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
]
"""NUMERIC(38, 18) Decimal, >= 0"""

PositiveDecimal38_18: TypeAlias = Annotated[
    Decimal,
    _REJECT_FLOAT,
    Gt(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(max_digits=38, decimal_places=18, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
]
"""NUMERIC(38, 18) Decimal, > 0"""

OptionalNonNegativeDecimal38_18: TypeAlias = Annotated[
    # Nested Annotated: every numeric constraint (max_digits/decimal_places)
    # must sit on the inner Decimal, not on the outer ``Decimal | None`` Field
    # — otherwise Pydantic crashes parsing JSON ``null`` (``None`` has no
    # ``max_digits``).
    # See https://docs.pydantic.dev/latest/concepts/types/ "Constraints on optional fields"
    Annotated[Decimal, Ge(Decimal(0)), Field(max_digits=38, decimal_places=18)] | None,
    _REJECT_FLOAT,
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(default=None, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
]
"""NUMERIC(38, 18) Decimal, >= 0 or None"""

# NUMERIC(20, 10) — 10 integer digits + 10 fractional digits (conversion
# factors, ratios, exchange rates and other medium-precision scenarios)

SignedDecimal20_10: TypeAlias = Annotated[
    Decimal,
    _REJECT_FLOAT,
    Field(max_digits=20, decimal_places=10, sa_type=Numeric(20, 10)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
]
"""NUMERIC(20, 10) Decimal, positive or negative"""

NonNegativeDecimal20_10: TypeAlias = Annotated[
    Decimal,
    _REJECT_FLOAT,
    Ge(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(max_digits=20, decimal_places=10, sa_type=Numeric(20, 10)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
]
"""NUMERIC(20, 10) Decimal, >= 0"""

OptionalNonNegativeDecimal20_10: TypeAlias = Annotated[
    # Nested Annotated (see OptionalNonNegativeDecimal38_18)
    Annotated[Decimal, Ge(Decimal(0)), Field(max_digits=20, decimal_places=10)] | None,
    _REJECT_FLOAT,
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(default=None, sa_type=Numeric(20, 10)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
]
"""NUMERIC(20, 10) Decimal, >= 0 or None"""

# ---------------------------------------------------------------------------
#  Bounded-Length List Types (named aliases List<N>[T], consistent with
#  Str64 / Text1K naming)
# ---------------------------------------------------------------------------
#
# ``List<N>[T]`` means "a list of at most N elements of type T", equivalent at
# runtime to ``Annotated[list[T], Field(max_length=N)]``.
#
# Design principles:
# - Pyright / Pydantic view: every ``List<N>[T]`` is equivalent to ``list[T]``
#   — the metadata is transparent and does not trigger
#   ``reportGeneralTypeIssues`` (N is encoded in the identifier, not in the
#   type subscript)
# - Naming habit: like ``Str64`` / ``Text1K`` / ``Sha256Hex``, the max_length
#   is encoded in the type name — single source of truth
# - Difference from a dialect ARRAY column type: ``List`` is just a Pydantic
#   list with a length bound, useful for request DTOs / vendor API protocol
#   layers and other non-database-column scenarios

_ListT = TypeVar('_ListT')


class List(list[_ListT], Generic[_ListT]):
    """
    Unbounded ``list[T]`` type — equivalent to ``list[T]``, Pyright-friendly.

    When a length bound is needed, use the named aliases ``List<N>[T]``:
    ``List16`` / ``List50`` / ``List1024`` etc., consistent with the
    ``Str64`` / ``Text1K`` naming habit.

    Runtime behavior: ``List[T]`` returns ``list[T]``.
    """

    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        return list[item]


List1 = Annotated[list[_ListT], Field(max_length=1)]
List2 = Annotated[list[_ListT], Field(max_length=2)]
List3 = Annotated[list[_ListT], Field(max_length=3)]
List7 = Annotated[list[_ListT], Field(max_length=7)]
List10 = Annotated[list[_ListT], Field(max_length=10)]
List16 = Annotated[list[_ListT], Field(max_length=16)]
List20 = Annotated[list[_ListT], Field(max_length=20)]
List32 = Annotated[list[_ListT], Field(max_length=32)]
List40 = Annotated[list[_ListT], Field(max_length=40)]
List50 = Annotated[list[_ListT], Field(max_length=50)]
List64 = Annotated[list[_ListT], Field(max_length=64)]
List100 = Annotated[list[_ListT], Field(max_length=100)]
List128 = Annotated[list[_ListT], Field(max_length=128)]
List200 = Annotated[list[_ListT], Field(max_length=200)]
List256 = Annotated[list[_ListT], Field(max_length=256)]
List1024 = Annotated[list[_ListT], Field(max_length=1024)]
