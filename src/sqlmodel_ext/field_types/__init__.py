"""
sqlmodel_ext.field_types -- Reusable type aliases and custom types for SQLModel.

Provides constrained string/numeric types, path types, URL types, and IP address types,
all compatible with Pydantic validation and SQLAlchemy column mapping.
"""
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import NoneType, UnionType
from typing import Annotated, Any, Generic, TypeAlias, TypeVar, Union, get_args, get_origin

from annotated_types import Ge, GroupedMetadata, Gt, MaxLen
from pydantic import AllowInfNan, BeforeValidator, PlainSerializer, StringConstraints, WithJsonSchema
from pydantic.fields import FieldInfo
from sqlalchemy import BigInteger, Numeric
from sqlmodel import Field

from ._internal.path import _DirectoryPathHandler, _FilePathHandler
from .dialects.postgresql.array import _ArrayTypeHandler
# Re-exports use the redundant ``X as X`` form: the package ships ``py.typed``, and
# type checkers treat a plain ``from .m import X`` in a typed package as private
# (consumers would get ``reportPrivateImportUsage``).
from .ip_address import ClientIPAddress as ClientIPAddress, IPAddress as IPAddress
from .mixins import ModuleNameMixin as ModuleNameMixin
from .url import (
    HttpUrl as HttpUrl,
    SafeHttpUrl as SafeHttpUrl,
    Url as Url,
    WebSocketUrl as WebSocketUrl,
)

# Re-export SSRF utilities
from ._ssrf import UnsafeURLError as UnsafeURLError, validate_not_private_host as validate_not_private_host

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

HttpHeaderName: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=64),
    StringConstraints(pattern=r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$"),
]
"""HTTP header field name (RFC 9110 ``token``: letters, digits and
``!#$%&'*+-.^_`|~``; no whitespace, no colon).

Use it for configurable header names (e.g. "which request header carries the
client IP") so a misspelled name containing a space, a colon or non-ASCII
characters is rejected with a validation error instead of being stored and
then silently never matching. The pattern is already a strict allowlist, so
``_NO_NULL_BYTE`` is not stacked on top."""

# String length constraints
Str1: TypeAlias = Annotated[str, Field(max_length=1), _NO_NULL_BYTE]
"""1-character string field"""

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

Text3072: TypeAlias = Annotated[str, Field(max_length=3072), _NO_NULL_BYTE]
"""3072-character text field"""

Text4K: TypeAlias = Annotated[str, Field(max_length=4000), _NO_NULL_BYTE]
"""4000-character text field (e.g. presigned URLs: CDN host + long object key +
encoded ``response-content-disposition`` realistically stay below ~1800
characters, leaving 2x headroom)"""

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

NonEmptyStrippedStr32: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=32),
    StringConstraints(strip_whitespace=True, min_length=1),
    _NO_NULL_BYTE,
]
"""1-32 character string, non-empty after stripping (rejects ``""`` and pure whitespace).

Suited to short user-written identifiers such as tags. Using the same alias on
the storage side and the filter side guarantees "a value that can be stored
can be filtered by its original text": both sides strip identically."""

NonEmptyStrippedStr64: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=64),
    StringConstraints(strip_whitespace=True, min_length=1),
    _NO_NULL_BYTE,
]
"""1-64 character string, non-empty after stripping (rejects ``""`` and pure whitespace)"""

_LINE_BREAK_SCAN_BOUND = 0x2100
"""Upper bound of the code-point scan that derives ``_LINE_BREAK_CHARS``.

A full scan (0x110000 code points) costs roughly 100-150 ms at import time
versus under 1 ms for this bound, and a full scan finds **no** line-break
character in ``[0x2100, 0x110000)``. This is a measured fact, not a structural
guarantee: should a future Python add a line-break character above the bound,
the derived set would miss it (the test suite re-derives the set over all code
points to catch that)."""

_LINE_BREAK_CHARS = ''.join(
    chr(code) for code in range(_LINE_BREAK_SCAN_BOUND)
    if len(('a' + chr(code) + 'b').splitlines()) > 1
)
"""Every character that starts a new line, **derived** from ``str.splitlines()``.

A hand-written set almost always stops at ``\\n`` (maybe ``\\r``), but there
are ten: ``\\n \\v \\f \\r \\x1c \\x1d \\x1e \\x85 \\u2028 \\u2029``. The last six
also break lines in renderers, terminals and most parsers, yet all of them pass
``_NO_NULL_BYTE``. ``str.splitlines()`` is CPython's definition of "a line",
so it is used as the single source of truth instead of a copied list.

A pattern (rather than e.g. ``annotated_types.Predicate(str.isprintable)``) is
used because a pattern is emitted into the JSON Schema; a predicate is not, so
the published schema would hide the real accepted input set."""

_SINGLE_LINE = StringConstraints(
    pattern='^[^\\x00' + ''.join(f'\\u{ord(c):04x}' for c in _LINE_BREAK_CHARS) + ']*$',
)
"""Single-line constraint: forbids NUL and every line-break character (see
``_LINE_BREAK_CHARS``).

Values that are rendered line by line into another text (listings, search
results, LLM context) must be rejected at the input boundary: a name containing
a line break can forge an extra, non-existent entry in such a listing. Escaping
at render time is not a substitute -- every render site is another chance to
forget, and forgetting is silent.

Tabs are allowed: ``\\t`` does not break lines."""

SingleLineStr64: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=64),
    StringConstraints(strip_whitespace=True, min_length=1),
    _SINGLE_LINE,
]
"""1-64 characters, non-empty after stripping, and **always a single line**.

Equivalent to ``NonEmptyStrippedStr64`` plus the single-line constraint. Use it
for user-visible *names* that are displayed as one line. Descriptions, which
may legitimately span several lines, should use ``Str500`` / ``Text*``
instead."""

SearchQueryStr64: TypeAlias = Annotated[
    str,
    Field(min_length=2, max_length=64),
    StringConstraints(strip_whitespace=True, min_length=2),
    _NO_NULL_BYTE,
]
"""Fuzzy-search keyword: at most 64 characters and **at least 2 characters after
stripping** (rejects ``""``, whitespace-only and single-character queries).

Same shape as ``NonEmptyStrippedStr64`` with a lower bound of 2: a
one-character trigram query has no selective trigrams and degrades into a
near-full table scan.

Being a type alias (rather than a validator) puts ``minLength`` into the JSON
Schema, so generated clients know the real accepted input set.
``strip_whitespace=True`` normalizes ``" ab "`` to ``"ab"`` instead of rejecting
it, and the post-strip ``min_length=2`` rejects ``" a "`` and whitespace of any
length -- a caller that does not want to filter simply omits the parameter."""

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


# ---------------------------------------------------------------------------
#  Reflecting the length bound of an alias
# ---------------------------------------------------------------------------

def _flatten_metadata(candidate: Any) -> Iterator[Any]:
    """Recursively expand ``GroupedMetadata``, yielding leaf constraints in declaration order.

    A ``GroupedMetadata`` is not a constraint itself but a container that
    iterates into constraints -- and what it yields may be a container again
    (``annotated_types.Len`` is one: ``tuple(Len(0, 7)) == (MaxLen(7),)``).
    Pydantic expands recursively, so this must too: expanding one level only
    would miss the inner bound and report a *wider* limit than the one
    Pydantic enforces.

    There is deliberately no depth limit: a self-referencing
    ``GroupedMetadata`` raises ``RecursionError`` here, exactly as it does in
    Pydantic.
    """
    if isinstance(candidate, GroupedMetadata):
        for inner in candidate:
            yield from _flatten_metadata(inner)
    else:
        yield candidate


def max_length_of(alias: Any) -> int:
    """Reflect the effective ``max_length`` of a type alias.

    Use this whenever code needs "how long may this field be" instead of
    repeating the number: a second constant drifts from the alias, and then
    either accepts values Pydantic / the database will reject, or truncates
    values that are still valid.

    Rules, all aligned with what Pydantic actually enforces:

    1. ``Field(max_length=N)`` does not store the bound as a ``FieldInfo``
       attribute; it lives in ``FieldInfo.metadata`` as
       ``annotated_types.MaxLen``, so ``FieldInfo`` metadata is expanded.
       ``StringConstraints`` carries ``max_length`` directly.
    2. **The last constraint wins**, because Pydantic applies stacked
       constraints in order (``Annotated[str, Field(max_length=10),
       Field(max_length=11)]`` accepts 11 characters; ``Field`` and
       ``StringConstraints`` override each other the same way).
    3. **Only carriers that Pydantic enforces are recognized**: ``MaxLen``,
       ``StringConstraints``, any ``GroupedMetadata`` that expands to them
       (recursively, see :func:`_flatten_metadata`), and the handler behind
       ``Array[T, N]``, which injects ``N`` into the list schema as the element
       count bound. Arbitrary metadata that merely has a ``max_length``
       attribute is ignored, since Pydantic ignores it too.
    4. **``X | None`` is accepted**: Pydantic does not lift the ``Annotated``
       metadata of a union member into ``FieldInfo.metadata``, so the union is
       unwrapped first. It must have exactly one non-``None`` member.

    :param alias: A string alias of the ``Annotated[str, Field(max_length=N), ...]``
        shape, its ``X | None`` form, or an ``Array[T, N]`` alias -- for which
        the **element count** bound (JSON Schema ``maxItems``) is returned.
        Typed as ``Any`` because an ``Annotated`` alias is a
        ``typing._AnnotatedAlias`` at runtime, which has no public type.
    :raises TypeError: If the alias declares no ``max_length`` (including an
        unbounded ``Array[T]``), or a union has other than exactly one
        non-``None`` member -- failing loudly instead of inventing a bound.
    """
    if get_origin(alias) in (Union, UnionType):
        members = [member for member in get_args(alias) if member is not NoneType]
        if len(members) != 1:
            raise TypeError(
                f"{alias!r} has {len(members)} non-None members; expected the 'X | None' "
                "form, so there is no single bound to reflect"
            )
        alias = members[0]
    effective: int | None = None
    for meta in get_args(alias)[1:]:
        # Only FieldInfo hides its constraints in ``.metadata``; branch on
        # isinstance explicitly instead of a getattr fallback.
        candidates = (meta, *meta.metadata) if isinstance(meta, FieldInfo) else (meta,)
        for candidate in candidates:
            for item in _flatten_metadata(candidate):
                # Keep iterating: a later constraint overrides an earlier one.
                if isinstance(item, MaxLen):
                    effective = item.max_length
                elif isinstance(item, StringConstraints) and item.max_length is not None:
                    effective = item.max_length
                elif isinstance(item, _ArrayTypeHandler) and item.max_length is not None:
                    effective = item.max_length
    if effective is None:
        raise TypeError(f"type alias {alias!r} declares no max_length; cannot reflect a bound")
    return effective


# Numeric range constraints
Port: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
"""Port number (1-65535)"""

Percentage: TypeAlias = Annotated[int, Field(ge=0, le=100)]
"""Percentage (0-100)"""

INT32_MAX = 2147483647
"""Maximum value for PostgreSQL INTEGER column (2^31-1)"""

INT32_MIN = -2147483648
"""Minimum value for PostgreSQL INTEGER column (-2^31); the explicit lower bound
for full-range integer columns (e.g. a priority that may be negative)"""

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

SignedBigInt: TypeAlias = Annotated[
    int, Field(ge=-JS_MAX_SAFE_INTEGER, le=JS_MAX_SAFE_INTEGER, sa_type=BigInteger)
]
"""Signed big integer (-JS_MAX_SAFE_INTEGER to JS_MAX_SAFE_INTEGER, stored as PostgreSQL BIGINT).

For **increment / delta** columns, as opposed to :data:`NonNegativeBigInt`
(absolute amounts / counters). Both bounds are JS_MAX_SAFE_INTEGER for the
reason given in :data:`PositiveBigInt`.
"""

PositiveFloat: TypeAlias = Annotated[float, Field(gt=0.0), AllowInfNan(False)]
"""Positive **finite** float (>0; rejects inf and nan).

Pydantic floats default to ``allow_inf_nan=True``, and ``gt=0`` is only a
comparison that ``inf`` satisfies: the JSON number ``1e309`` parses to
``float('inf')`` and would pass, only to blow up later (e.g.
``math.ceil(float('inf'))`` raises ``OverflowError``). ``AllowInfNan(False)``
rejects inf / nan at the boundary.
"""

NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0.0), AllowInfNan(False)]
"""Non-negative **finite** float (>=0; rejects inf and nan) -- see :data:`PositiveFloat`"""

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

# ---------------------------------------------------------------------------
#  Decimal OpenAPI validation-schema fix (drop ``number``, keep ``string`` only)
#
#  Pydantic maps ``Decimal`` to ``anyOf: [number, string]`` in JSON Schema, but
#  ``_reject_float_decimal_input`` rejects float at runtime (a JSON number with a
#  decimal point *is* a float). So the docs would advertise ``0.00001005`` as a
#  valid request value while the server returns 422 — a contract mismatch.
#  ``WithJsonSchema(mode='validation')`` overrides the request-body schema only;
#  response-body schemas are unaffected.
# ---------------------------------------------------------------------------

DECIMAL_38_18_COLUMN_DIGITS: int = 38
"""Total digits of a ``NUMERIC(38, 18)`` column -- the **column width**, which is
also the write limit of the ``*Decimal38_18`` aliases.

The ``*WriteDecimal38_18`` aliases accept fewer digits on purpose (see
``DECIMAL_38_18_WRITE_DIGITS``); do not "unify" the two constants.
"""

DECIMAL_38_18_WRITE_DIGITS: int = 35
"""Total digits accepted by the ``*WriteDecimal38_18`` aliases: **17 integer + 18
fractional digits**, three fewer than the ``NUMERIC(38, 18)`` column.

The three-digit gap is headroom for ``SUM()``: if a single row may be as large
as the column, a sum of rows can overflow it. With rows capped at 35 digits and
sums read through ``SignedSumDecimal38_18`` (the full 38 digits),
``10^(38-18) / 10^(35-18) = 1000`` maximal rows must be added before the sum
reaches the column width.

The factor of 1000 is headroom, not a guarantee: summing far more than 1000
rows that are all close to the per-row limit can still overflow.
"""

DECIMAL_38_18_PLACES: int = 18
"""Fractional digits of ``NUMERIC(38, 18)`` (matches the EVM wei = 1e-18 ether
de-facto standard). Identical on the write and the sum side."""

_WHOLE_DIGITS_38_18: int = DECIMAL_38_18_COLUMN_DIGITS - DECIMAL_38_18_PLACES
"""Integer digits of the full-width ``*Decimal38_18`` and ``SignedSumDecimal38_18`` aliases (20)"""

_WRITE_WHOLE_DIGITS_38_18: int = DECIMAL_38_18_WRITE_DIGITS - DECIMAL_38_18_PLACES
"""Integer digits of the ``*WriteDecimal38_18`` aliases (17)"""


def _decimal_str_pattern(whole_digits: int, places: int) -> str:
    """Build the OpenAPI validation pattern for a string-form Decimal.

    Exists so that the digit counts are written once, at the call site: a
    hand-written pattern next to ``max_digits`` drifts when only one of them is
    changed, and the published schema then misstates the accepted input set.

    Known limitation: the integer-only branch ``\\d{0,N}`` has no end anchor,
    and a JSON Schema ``pattern`` matches partially, so any string starting
    with a digit matches (e.g. an over-long integer, ``"123abc"``,
    ``"1.2.3"``). **The digit limits are enforced by ``max_digits`` /
    ``decimal_places``, not by this pattern**; the pattern's job is to declare
    the schema type as ``string`` rather than ``number``.

    :param whole_digits: Maximum digits before the decimal point
    :param places: Maximum digits after the decimal point
    :returns: A regex matching "optional sign + optional leading zeros + decimal literal"
    """
    max_chars = whole_digits + 1 + places
    return (
        rf'^(?!^[-+.]*$)[+-]?0*(?:\d{{0,{whole_digits}}}'
        rf'|(?=[\d.]{{1,{max_chars}}}0*$)\d{{0,{whole_digits}}}\.\d{{0,{places}}}0*$)'
    )


_DECIMAL_38_18_STR_PATTERN = _decimal_str_pattern(_WHOLE_DIGITS_38_18, DECIMAL_38_18_PLACES)
_DECIMAL_WRITE_38_18_STR_PATTERN = _decimal_str_pattern(_WRITE_WHOLE_DIGITS_38_18, DECIMAL_38_18_PLACES)
_DECIMAL_20_10_STR_PATTERN = _decimal_str_pattern(10, 10)


def _str_schema(pattern: str) -> WithJsonSchema:
    """Validation-mode schema: a pattern-constrained string."""
    return WithJsonSchema({'type': 'string', 'pattern': pattern}, mode='validation')


def _optional_str_schema(pattern: str) -> WithJsonSchema:
    """Validation-mode schema: a pattern-constrained string or null."""
    return WithJsonSchema(
        {'anyOf': [{'type': 'string', 'pattern': pattern}, {'type': 'null'}]},
        mode='validation',
    )


_DECIMAL_38_18_STR_SCHEMA = _str_schema(_DECIMAL_38_18_STR_PATTERN)
_DECIMAL_WRITE_38_18_STR_SCHEMA = _str_schema(_DECIMAL_WRITE_38_18_STR_PATTERN)
_DECIMAL_20_10_STR_SCHEMA = _str_schema(_DECIMAL_20_10_STR_PATTERN)
_OPTIONAL_DECIMAL_38_18_STR_SCHEMA = _optional_str_schema(_DECIMAL_38_18_STR_PATTERN)
_OPTIONAL_DECIMAL_WRITE_38_18_STR_SCHEMA = _optional_str_schema(_DECIMAL_WRITE_38_18_STR_PATTERN)
_OPTIONAL_DECIMAL_20_10_STR_SCHEMA = _optional_str_schema(_DECIMAL_20_10_STR_PATTERN)

# ---------------------------------------------------------------------------
#  Metadata order: ``_REJECT_FLOAT`` must come AFTER ``Field(max_digits=...)``
#
#  ``Annotated`` metadata is applied left to right. With the ``BeforeValidator``
#  first, Pydantic cannot inline the numeric constraints into the ``decimal``
#  core schema and falls back to Python validators that check only the total
#  digits and the decimal places -- **not the integer digits**. For example
#  ``Annotated[Decimal, BeforeValidator(...), Field(max_digits=20,
#  decimal_places=10)]`` accepts a 15-digit integer, leaving the database
#  column as the only guard.
#
#  With ``Field`` first, the constraints are inlined as
#  ``function-before(decimal{max_digits, decimal_places})`` and pydantic-core
#  checks all three (total digits, decimal places, integer digits). The float
#  / bool rejection is unaffected by the order.
#
#  The ``Optional*`` aliases keep ``_REJECT_FLOAT`` on the outer layer: their
#  numeric constraints live on the inner ``Annotated[Decimal, ...]`` and are
#  inlined there regardless.
# ---------------------------------------------------------------------------

# NUMERIC(38, 18) — 20 integer digits + 18 fractional digits (matches the EVM
# wei = 1e-18 ether de-facto standard for high-precision amounts)

SignedDecimal38_18: TypeAlias = Annotated[
    Decimal,
    Field(max_digits=38, decimal_places=18, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) Decimal, positive or negative"""

NonNegativeDecimal38_18: TypeAlias = Annotated[
    Decimal,
    Ge(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (the SQLModel stub annotates it
    # ``type[Any]`` but the runtime accepts SA type instances like
    # ``Numeric(38, 18)``); ``ge`` is expressed as ``Ge(Decimal(0))`` via
    # annotated_types so the ignore scope stays minimal.
    Field(max_digits=38, decimal_places=18, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) Decimal, >= 0"""

PositiveDecimal38_18: TypeAlias = Annotated[
    Decimal,
    Gt(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(max_digits=38, decimal_places=18, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_38_18_STR_SCHEMA,
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
    _OPTIONAL_DECIMAL_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) Decimal, >= 0 or None"""

OptionalSignedDecimal38_18: TypeAlias = Annotated[
    # Nested Annotated (see OptionalNonNegativeDecimal38_18); the only
    # difference is the missing ``Ge(Decimal(0))`` -- negatives are allowed.
    Annotated[Decimal, Field(max_digits=38, decimal_places=18)] | None,
    _REJECT_FLOAT,
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(default=None, sa_type=Numeric(38, 18)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
    _OPTIONAL_DECIMAL_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) Decimal, positive, negative or None"""

# ---------------------------------------------------------------------------
#  NUMERIC(38, 18) columns with SUM() headroom: write 35 digits, sum 38
#
#  If a single row may hold as many digits as the column, a ``SUM()`` over rows
#  can exceed the column width -- and reading that sum back through a
#  38-digit Decimal type fails validation. Values that will be aggregated
#  should therefore be *written* through the ``*WriteDecimal38_18`` aliases
#  (35 digits: 17 integer + 18 fractional; the column stays NUMERIC(38, 18))
#  and the aggregate *read* through ``SignedSumDecimal38_18`` (the full 38
#  digits). See ``DECIMAL_38_18_WRITE_DIGITS`` for the headroom arithmetic.
#
#  ``sa_type=Numeric(38, 18)`` is explicit on purpose: without it SQLModel
#  derives the column precision from ``max_digits`` and the column would
#  shrink to NUMERIC(35, 18).
# ---------------------------------------------------------------------------

SignedWriteDecimal38_18: TypeAlias = Annotated[
    Decimal,
    Field(
        max_digits=DECIMAL_38_18_WRITE_DIGITS,
        decimal_places=DECIMAL_38_18_PLACES,
        sa_type=Numeric(DECIMAL_38_18_COLUMN_DIGITS, DECIMAL_38_18_PLACES),  # pyright: ignore[reportArgumentType]
    ),
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_WRITE_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) column, writes limited to 35 digits (17 integer + 18
fractional), positive or negative. Leaves 1000x headroom for ``SUM()``; read
sums through ``SignedSumDecimal38_18``."""

NonNegativeWriteDecimal38_18: TypeAlias = Annotated[
    Decimal,
    Ge(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(
        max_digits=DECIMAL_38_18_WRITE_DIGITS,
        decimal_places=DECIMAL_38_18_PLACES,
        sa_type=Numeric(DECIMAL_38_18_COLUMN_DIGITS, DECIMAL_38_18_PLACES),  # pyright: ignore[reportArgumentType]
    ),
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_WRITE_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) column, writes limited to 35 digits, >= 0 (SUM() headroom,
see ``SignedWriteDecimal38_18``)"""

PositiveWriteDecimal38_18: TypeAlias = Annotated[
    Decimal,
    Gt(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(
        max_digits=DECIMAL_38_18_WRITE_DIGITS,
        decimal_places=DECIMAL_38_18_PLACES,
        sa_type=Numeric(DECIMAL_38_18_COLUMN_DIGITS, DECIMAL_38_18_PLACES),  # pyright: ignore[reportArgumentType]
    ),
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_WRITE_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) column, writes limited to 35 digits, > 0 (SUM() headroom,
see ``SignedWriteDecimal38_18``)"""

OptionalNonNegativeWriteDecimal38_18: TypeAlias = Annotated[
    # Nested Annotated (see OptionalNonNegativeDecimal38_18)
    Annotated[
        Decimal,
        Ge(Decimal(0)),
        Field(max_digits=DECIMAL_38_18_WRITE_DIGITS, decimal_places=DECIMAL_38_18_PLACES),
    ] | None,
    _REJECT_FLOAT,
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(
        default=None,
        sa_type=Numeric(DECIMAL_38_18_COLUMN_DIGITS, DECIMAL_38_18_PLACES),  # pyright: ignore[reportArgumentType]
    ),
    _DECIMAL_TO_JSON_STR,
    _OPTIONAL_DECIMAL_WRITE_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) column, writes limited to 35 digits, >= 0 or None (SUM()
headroom, see ``SignedWriteDecimal38_18``)"""

OptionalSignedWriteDecimal38_18: TypeAlias = Annotated[
    # Nested Annotated (see OptionalNonNegativeDecimal38_18)
    Annotated[
        Decimal,
        Field(max_digits=DECIMAL_38_18_WRITE_DIGITS, decimal_places=DECIMAL_38_18_PLACES),
    ] | None,
    _REJECT_FLOAT,
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(
        default=None,
        sa_type=Numeric(DECIMAL_38_18_COLUMN_DIGITS, DECIMAL_38_18_PLACES),  # pyright: ignore[reportArgumentType]
    ),
    _DECIMAL_TO_JSON_STR,
    _OPTIONAL_DECIMAL_WRITE_38_18_STR_SCHEMA,
]
"""NUMERIC(38, 18) column, writes limited to 35 digits, positive, negative or
None (SUM() headroom, see ``SignedWriteDecimal38_18``)"""

SignedSumDecimal38_18: TypeAlias = Annotated[
    Decimal,
    Field(max_digits=DECIMAL_38_18_COLUMN_DIGITS, decimal_places=DECIMAL_38_18_PLACES),
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_38_18_STR_SCHEMA,
]
"""Result of ``SUM()`` over ``*WriteDecimal38_18`` values: 38 digits (20 integer
+ 18 fractional), positive or negative, **no ``sa_type``** -- a DTO-only type
that is never stored.

Three digits wider than the write side, so 1000 maximal rows still sum to a
representable value. Do not annotate aggregates with a write-side alias:
"per-row limit == sum limit" is exactly the shape that lets a sum overflow.
Do not use it as a column type either: a pre-aggregated column is storage and
needs its own width decision.
"""

# NUMERIC(20, 10) — 10 integer digits + 10 fractional digits (conversion
# factors, ratios, exchange rates and other medium-precision scenarios)

SignedDecimal20_10: TypeAlias = Annotated[
    Decimal,
    Field(max_digits=20, decimal_places=10, sa_type=Numeric(20, 10)),  # pyright: ignore[reportArgumentType]
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_20_10_STR_SCHEMA,
]
"""NUMERIC(20, 10) Decimal, positive or negative"""

NonNegativeDecimal20_10: TypeAlias = Annotated[
    Decimal,
    Ge(Decimal(0)),
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(max_digits=20, decimal_places=10, sa_type=Numeric(20, 10)),  # pyright: ignore[reportArgumentType]
    _REJECT_FLOAT,
    _DECIMAL_TO_JSON_STR,
    _DECIMAL_20_10_STR_SCHEMA,
]
"""NUMERIC(20, 10) Decimal, >= 0"""

OptionalNonNegativeDecimal20_10: TypeAlias = Annotated[
    # Nested Annotated (see OptionalNonNegativeDecimal38_18)
    Annotated[Decimal, Ge(Decimal(0)), Field(max_digits=20, decimal_places=10)] | None,
    _REJECT_FLOAT,
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(default=None, sa_type=Numeric(20, 10)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
    _OPTIONAL_DECIMAL_20_10_STR_SCHEMA,
]
"""NUMERIC(20, 10) Decimal, >= 0 or None"""

NullableNonNegativeDecimal20_10: TypeAlias = Annotated[
    # Nested Annotated (see OptionalNonNegativeDecimal38_18). The only
    # difference from ``OptionalNonNegativeDecimal20_10`` is the missing
    # default: in a validated model (DTO / request body) the key is required,
    # though its value may be null. ``table=True`` models skip Pydantic
    # validation on construction, so omitting it there stores NULL. Suited to
    # nullable settings where a default would hide a caller forgetting the field.
    Annotated[Decimal, Ge(Decimal(0)), Field(max_digits=20, decimal_places=10)] | None,
    _REJECT_FLOAT,
    # pyright ignore targets ``sa_type`` only (see NonNegativeDecimal38_18)
    Field(sa_type=Numeric(20, 10)),  # pyright: ignore[reportArgumentType]
    _DECIMAL_TO_JSON_STR,
    _OPTIONAL_DECIMAL_20_10_STR_SCHEMA,
]
"""NUMERIC(20, 10) Decimal, >= 0 or None, **without** a default (required key in
validated models; the only difference from ``OptionalNonNegativeDecimal20_10``)"""

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
