# Field types

::: tip
This is reference documentation. To see how to use these types on a model, head to [Getting started](/en/tutorials/01-getting-started). For why type aliases beat scattered `Field(max_length=...)`, see [Design philosophy: a single source of truth](/en/explanation/single-source-of-truth).
:::

Except for the "PostgreSQL-only types" section, every field type can be imported from the top-level `sqlmodel_ext` package. Most are `Annotated` type aliases (`TypeAlias`): **one declaration produces** the Pydantic validation, the SQLAlchemy column type and the OpenAPI schema.

`Str*` / `Text*` / `NonEmpty*` / `SearchQueryStr64` also carry `pattern=r'^[^\x00]*$'` (rejects NUL bytes, which would otherwise break PostgreSQL text columns).

## String constraints

```python
from sqlmodel_ext import Str1, Str16, Str24, Str32, Str36, Str48, Str64, Str100, Str128, Str255, Str256, Str500, Str512, Str2048
```

| Type | `max_length` | Equivalent definition |
|------|--------------|-----------------------|
| `Str1` | 1 | `Annotated[str, Field(max_length=1), _NO_NULL_BYTE]` |
| `Str16` | 16 | Same as above |
| `Str24` | 24 | Same as above |
| `Str32` | 32 | Same as above |
| `Str36` | 36 | Same as above (canonical UUID string length) |
| `Str48` | 48 | Same as above |
| `Str64` | 64 | Same as above |
| `Str100` | 100 | Same as above |
| `Str128` | 128 | Same as above |
| `Str255` | 255 | Same as above |
| `Str256` | 256 | Same as above |
| `Str500` | 500 | Same as above |
| `Str512` | 512 | Same as above |
| `Str2048` | 2048 | Same as above |

### Non-empty and specialized strings

```python
from sqlmodel_ext import (
    NonEmptyStr64, NonEmptyStr128, NonEmptyStr256,
    NonEmptyStrippedStr32, NonEmptyStrippedStr64, NonEmptyStrippedStr128, NonEmptyStrippedStr256,
    SingleLineStr64, SearchQueryStr64, HttpHeaderName,
    Sha256Hex, BCP47LanguageCode,
)
```

| Type | Constraint |
|------|------------|
| `NonEmptyStr64/128/256` | `1 <= len <= N`, rejects the empty string `""` |
| `NonEmptyStrippedStr32/64/128/256` | Same as above + `strip_whitespace`, rejects whitespace-only input (`"   "` / `"\t"`) |
| `SingleLineStr64` | `NonEmptyStrippedStr64` + **single line**: rejects NUL and every line-break character (`\n \v \f \r \x1c \x1d \x1e \x85    `, derived from `str.splitlines()`); tabs are allowed. For names rendered line by line, so a line break cannot forge an extra listing entry |
| `SearchQueryStr64` | `2 <= len <= 64` after stripping: a one-character trigram query has no selective trigrams and degrades into a near-full table scan |
| `HttpHeaderName` | RFC 9110 `token` (letters, digits and ``!#$%&'*+-.^_`|~``), `1 <= len <= 64` |
| `Sha256Hex` | Exactly 64 lowercase hex characters (a SHA-256 digest) |
| `BCP47LanguageCode` | BCP-47 language tag syntax (e.g. `zh-Hans-CN`), `max_length=16` |

## Text constraints

```python
from sqlmodel_ext import Text1K, Text1024, Text2K, Text2500, Text3K, Text3072, Text4K, Text5K, Text8K, Text10K, Text16K, Text32K, Text48K, Text60K, Text64K, Text100K, Text128K, Text1M
```

| Type | `max_length` |
|------|--------------|
| `Text1K` | 1000 |
| `Text1024` | 1024 |
| `Text2K` | 2000 |
| `Text2500` | 2500 |
| `Text3K` | 3000 |
| `Text3072` | 3072 |
| `Text4K` | 4000 |
| `Text5K` | 5000 |
| `Text8K` | 8000 |
| `Text10K` | 10000 |
| `Text16K` | 16000 |
| `Text32K` | 32000 |
| `Text48K` | 48000 |
| `Text60K` | 60000 |
| `Text64K` | 65536 |
| `Text100K` | 100000 |
| `Text128K` | 131072 (= 128 × 1024) |
| `Text1M` | 1000000 |

## Reflecting the length bound: `max_length_of()`

```python
from sqlmodel_ext import max_length_of
```

```python
def max_length_of(alias: Any) -> int
```

When code needs "how long may this field be", reflect it from the alias instead of writing the number again (a second constant eventually drifts from the alias).

- Follows the rules Pydantic actually enforces: the `MaxLen` of `Field(max_length=N)`, `StringConstraints.max_length`, recursively expanded `GroupedMetadata`; with stacked constraints **the last one wins**;
- accepts `X | None` (exactly one non-`None` member);
- for `Array[T, N]` it returns the **element-count** bound `N`;
- raises `TypeError` when the alias declares no `max_length`, instead of inventing a number.

```python
from sqlmodel_ext import Str64, max_length_of
from sqlmodel_ext.field_types.dialects.postgresql import Array

assert max_length_of(Str64) == 64
assert max_length_of(Str64 | None) == 64
assert max_length_of(Array[str, 20]) == 20
```

## Numeric constraints

```python
from sqlmodel_ext import (
    Port, Percentage,
    PositiveInt, NonNegativeInt,
    PositiveBigInt, NonNegativeBigInt, SignedBigInt,
    PositiveFloat, NonNegativeFloat,
)
```

| Type | Range | DB column |
|------|-------|-----------|
| `Port` | `1` – `65535` | `INTEGER` |
| `Percentage` | `0` – `100` | `INTEGER` |
| `PositiveInt` | `1` – `INT32_MAX` | `INTEGER` |
| `NonNegativeInt` | `0` – `INT32_MAX` | `INTEGER` |
| `PositiveBigInt` | `1` – `JS_MAX_SAFE_INTEGER` | `BIGINT` |
| `NonNegativeBigInt` | `0` – `JS_MAX_SAFE_INTEGER` | `BIGINT` |
| `SignedBigInt` | `-JS_MAX_SAFE_INTEGER` – `JS_MAX_SAFE_INTEGER` | `BIGINT` (for increment / delta columns) |
| `PositiveFloat` | `> 0.0`, **finite** | `FLOAT` |
| `NonNegativeFloat` | `>= 0.0`, **finite** | `FLOAT` |

::: info Float types reject `inf` / `nan` (since 0.5.0)
Pydantic floats default to `allow_inf_nan=True`, and `gt=0` is only a comparison that `inf` satisfies — the JSON number `1e309` parses to `float('inf')` and would pass, only to blow up later (e.g. `math.ceil(float('inf'))` raises `OverflowError`). These aliases carry `AllowInfNan(False)` and reject it at the boundary.
:::

::: info Why BigInt caps at JS_MAX_SAFE_INTEGER
The `*BigInt` upper bound is `JS_MAX_SAFE_INTEGER = 2⁵³ − 1`, **not** `INT64_MAX`. Browsers lose precision when parsing JSON numbers beyond that range. If your API is not consumed by a browser, define a custom alias with `INT64_MAX` as the upper bound.
:::

### Constants

```python
from sqlmodel_ext import INT32_MIN, INT32_MAX, INT64_MAX, JS_MAX_SAFE_INTEGER
```

| Constant | Value |
|----------|-------|
| `INT32_MIN` | `-2_147_483_648` (−2³¹) |
| `INT32_MAX` | `2_147_483_647` (2³¹−1) |
| `INT64_MAX` | `9_223_372_036_854_775_807` (2⁶³−1) |
| `JS_MAX_SAFE_INTEGER` | `9_007_199_254_740_991` (2⁵³−1) |

## Decimal constraints

```python
from sqlmodel_ext import (
    SignedDecimal38_18, NonNegativeDecimal38_18, PositiveDecimal38_18,
    OptionalNonNegativeDecimal38_18, OptionalSignedDecimal38_18,
    SignedWriteDecimal38_18, NonNegativeWriteDecimal38_18, PositiveWriteDecimal38_18,
    OptionalNonNegativeWriteDecimal38_18, OptionalSignedWriteDecimal38_18,
    SignedSumDecimal38_18,
    SignedDecimal20_10, NonNegativeDecimal20_10,
    OptionalNonNegativeDecimal20_10, NullableNonNegativeDecimal20_10,
    DECIMAL_38_18_COLUMN_DIGITS, DECIMAL_38_18_WRITE_DIGITS, DECIMAL_38_18_PLACES,
)
```

Naming convention: `[Optional|Nullable][Signed|NonNegative|Positive][Write|Sum]Decimal{precision}_{scale}`.

| Type | Sign | Writable digits (integer + fractional) | DB column |
|------|------|------|-----------|
| `SignedDecimal38_18` | any | 20 + 18 | `NUMERIC(38, 18)` |
| `NonNegativeDecimal38_18` | `>= 0` | 20 + 18 | `NUMERIC(38, 18)` |
| `PositiveDecimal38_18` | `> 0` | 20 + 18 | `NUMERIC(38, 18)` |
| `OptionalNonNegativeDecimal38_18` | `>= 0` or `None`, default `None` | 20 + 18 | `NUMERIC(38, 18)` |
| `OptionalSignedDecimal38_18` | any or `None`, default `None` | 20 + 18 | `NUMERIC(38, 18)` |
| `SignedWriteDecimal38_18` | any | **17 + 18** | `NUMERIC(38, 18)` |
| `NonNegativeWriteDecimal38_18` | `>= 0` | 17 + 18 | `NUMERIC(38, 18)` |
| `PositiveWriteDecimal38_18` | `> 0` | 17 + 18 | `NUMERIC(38, 18)` |
| `OptionalNonNegativeWriteDecimal38_18` | `>= 0` or `None`, default `None` | 17 + 18 | `NUMERIC(38, 18)` |
| `OptionalSignedWriteDecimal38_18` | any or `None`, default `None` | 17 + 18 | `NUMERIC(38, 18)` |
| `SignedSumDecimal38_18` | any | 20 + 18 | none (DTO only, for reading `SUM()` results) |
| `SignedDecimal20_10` | any | 10 + 10 | `NUMERIC(20, 10)` |
| `NonNegativeDecimal20_10` | `>= 0` | 10 + 10 | `NUMERIC(20, 10)` |
| `OptionalNonNegativeDecimal20_10` | `>= 0` or `None`, default `None` | 10 + 10 | `NUMERIC(20, 10)` |
| `NullableNonNegativeDecimal20_10` | `>= 0` or `None`, **no default** (a required key in validated models; the value may be null) | 10 + 10 | `NUMERIC(20, 10)` |

Behavioral contract:

- **Integer digits, fractional digits and total digits are all validated** (fixed in 0.5.0: the old metadata order made Pydantic check only total digits and decimal places, leaving integer digits to the database)
- **Rejects float / bool input** (IEEE 754 has already lost precision) — accepts `Decimal` / `int` / `str`
- **JSON serialization emits a fixed-point string** (`model_dump_json()`), never scientific notation (`0E-18` → `'0'`), with redundant trailing zeros stripped (`1200.000...0` → `'1200'`), avoiding JS Number precision loss
- **Dict mode preserves the `Decimal` object** (`model_dump()`)
- The `Optional*` / `Nullable*` variants nest their numeric constraints in an inner `Annotated`, so JSON `null` parses safely
- **OpenAPI request-body schema is `string` only**: Pydantic maps `Decimal` to `anyOf: [number, string]` by default, but the runtime rejects floats; these aliases use `WithJsonSchema(mode='validation')` to narrow the request-body schema to a `string` with a fixed-point pattern. The pattern only declares the type; digit limits are enforced by `max_digits` / `decimal_places`. Response-body schemas are unaffected.

### Write 35 digits, read 38: headroom for `SUM()`

If a single row may fill the column width, a `SUM()` over rows can exceed it — and reading that sum back through a 38-digit type fails validation. Values that will be aggregated should be **written** through `*WriteDecimal38_18` (35 digits: 17 integer + 18 fractional; the column stays `NUMERIC(38, 18)`) and the aggregate **read** through `SignedSumDecimal38_18` (the full 38 digits). `10^(38-18) / 10^(35-18) = 1000`: 1000 maximal rows must be added before the sum reaches the column width — headroom, not a guarantee.

| Constant | Value | Meaning |
|---|---|---|
| `DECIMAL_38_18_COLUMN_DIGITS` | `38` | Column width, also the write limit of `*Decimal38_18` |
| `DECIMAL_38_18_WRITE_DIGITS` | `35` | Write limit of `*WriteDecimal38_18` |
| `DECIMAL_38_18_PLACES` | `18` | Fractional digits (same on the write and the sum side) |

```python
from decimal import Decimal

from pydantic import ValidationError
from sqlmodel_ext import NonNegativeWriteDecimal38_18, SQLModelBase, SignedDecimal20_10, SignedSumDecimal38_18


class Ledger(SQLModelBase):
    amount: NonNegativeWriteDecimal38_18 = Decimal(0)
    total: SignedSumDecimal38_18 = Decimal(0)
    rate: SignedDecimal20_10 = Decimal(0)


Ledger(amount=Decimal('1' * 17), total=Decimal('1' * 20))
for bad in ({'amount': Decimal('1' * 18)}, {'rate': Decimal('12345678901')}, {'rate': 0.5}):
    try:
        Ledger(**bad)
    except ValidationError:
        pass
    else:
        raise AssertionError(bad)
assert Ledger(rate=Decimal('1200.0000000000')).model_dump_json() == '{"amount":"0","total":"0","rate":"1200"}'
```

## Bounded-length List aliases

```python
from sqlmodel_ext import List, List1, List2, List3, List7, List10, List16, List20, List32, List40, List50, List64, List100, List128, List200, List256, List1024
```

`List<N>[T]` is equivalent to `Annotated[list[T], Field(max_length=N)]` — the maximum length is encoded in the type name (consistent with `Str64` / `Text1K`). `List[T]` (no number) is equivalent to `list[T]`. Intended for request DTOs / protocol layers that aren't database columns; unrelated to the PG `Array[T]` column type.

## Field marker: `EXCLUDE_IF_NONE`

```python
from sqlmodel_ext import EXCLUDE_IF_NONE
```

Put it in the metadata position of `Annotated`: when the value is `None`, the **key** is left out of every serialized output. All three parts are required:

```python
from typing import Annotated

from sqlmodel_ext import EXCLUDE_IF_NONE, SQLModelBase


class Event(SQLModelBase):
    marker: Annotated[bool | None, EXCLUDE_IF_NONE] = None


assert Event().model_dump() == {}
assert Event.model_validate_json(Event().model_dump_json()).marker is None
```

(1) `| None`, (2) the marker, (3) the `= None` default — without the default, dumped JSON cannot be read back. Unlike `exclude_none`, it is a property of the field and does not depend on the caller passing a flag. Typical use: adding a nullable field to a structure that older consumers deserialize strictly with `extra='forbid'`. Use it on **non-table** models only. For how it differs from `Unset` see [Unset](/en/explanation/unset-three-state#exclude-if-none-vs-unset).

## URL types

```python
from sqlmodel_ext import Url, HttpUrl, WebSocketUrl, SafeHttpUrl, UnsafeURLError, validate_not_private_host
```

Four URL types, all subclassing `str`, stored in the database as `VARCHAR`.

| Type | Allowed schemes | SSRF protection |
|------|-----------------|-----------------|
| `Url` | Any (http, ftp, ws, ...) | No |
| `HttpUrl` | `http` / `https` | No |
| `WebSocketUrl` | `ws` / `wss` | No |
| `SafeHttpUrl` | `http` / `https` | **Yes** |

`SafeHttpUrl` rejects:

- Loopback (`localhost`, `127.0.0.1`, `::1`)
- Private IPs (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
- Link-local (`169.254.0.0/16`)
- Reserved addresses

On rejection, it raises `UnsafeURLError`.

`validate_not_private_host(host: str) -> None` is the underlying validator and can be called directly.

## IP addresses

```python
from sqlmodel_ext import IPAddress, ClientIPAddress
```

| Type | Purpose | Python value |
|---|---|---|
| `IPAddress` | **Storage column**: validates IPv4 / IPv6 syntax, stored as `VARCHAR` | a `str` subclass; extra method `is_private() -> bool` |
| `ClientIPAddress` | **Parse time**: validates a client IP taken from **untrusted text** (e.g. a reverse-proxy header) | `IPv4Address \| IPv6Address` (Pydantic `IPvAnyAddress`) |

On top of the structural validation of `IPvAnyAddress`, `ClientIPAddress` rejects IPv6 zone IDs (`fe80::1%eth0`): a zone ID names a local interface, is not part of a network address, and `ipaddress` puts no length limit on it. Use `IPAddress` for the column that stores the result.

## Path types

```python
from sqlmodel_ext import FilePathType, DirectoryPathType
```

| Type | Validation |
|------|------------|
| `FilePathType` | Path must include a filename |
| `DirectoryPathType` | Path must not include a file extension |

Behaves as `pathlib.Path` at runtime, so you can use it as a regular `Path`.

## `ModuleNameMixin`

```python
from sqlmodel_ext import ModuleNameMixin
```

On instantiation, if the target field was not passed, it is set to the `__name__` of the **caller's module**. The target field is `name` by default; rename it through the class variable `_module_name_field`. You declare the field on your model yourself.

## PostgreSQL-only types

::: warning PostgreSQL only
The types in this section use native PostgreSQL column types and do not work on SQLite / MySQL. `JSON100K` / `JSONList100K` need `pip install sqlmodel-ext[postgresql]` (`orjson`), `NumpyVector` needs `[pgvector]`.
:::

### `Array[T]` / `Array[T, N]`

```python
from sqlmodel_ext.field_types.dialects.postgresql import Array
```

A PostgreSQL `ARRAY` column. The optional second parameter is an element-count bound (it becomes `maxItems` in the JSON Schema and can be reflected by `max_length_of()`).

| Python view | DB column |
|-------------|-----------|
| `list[str]` | `VARCHAR[]` |
| `list[int]` | `INTEGER[]` |
| `list[dict]` | `JSONB[]` |
| `list[UUID]` | `UUID[]` |
| `list[SomeEnum]` | `someenum[]` (read-tolerant, see below) |

Any other element type raises `TypeError` at class creation.

**Read tolerance for enum arrays**: an `Array[SomeEnum]` column is wrapped in a read-tolerant `TypeDecorator`. When the database returns an enum value that is not a member of the current process's Python enum (typically the version-skew window of a rolling deployment — a newer instance has written a new value, while an older instance's code does not know it yet), that element is dropped on the **read path** with a warning instead of raising `LookupError` (which would 500 the request). The write path still validates strictly, so no dirty writes occur; the dropped value stays in the database and becomes visible again once code that knows it is deployed.

### `JSON100K` / `JSONList100K`

```python
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K, JSONList100K, ensure_json_within_limits
```

| Type | Python view | DB column | Limit |
|------|-------------|-----------|-------|
| `JSON100K` | `dict[str, Any]` | `JSONB` | canonical JSON encoding ≤ 100K characters |
| `JSONList100K` | `list[dict[str, Any]]` | `JSONB` | canonical JSON encoding ≤ 100K characters |

Contract: **object in, object out**.

- **Inbound**: accepts a JSON object / array (preferred) or a JSON **string** (compatibility form). Both forms must be encodable (nesting within the limits of both serializers, orjson and Pydantic — Pydantic's is platform-dependent, about 98 levels on Windows builds and higher on Linux) with a canonical encoding of at most 100K **characters** (counted in characters, not UTF-8 bytes, so CJK text is not rejected 3x too early).
- **Outbound**: `model_dump()`, `model_dump(mode='json')` and `model_dump_json()` all output the object / array itself (since 0.5.0; it used to be a JSON string). The serialization schema declares a pure object / array; the validation schema honestly declares `anyOf[object, string]`.
- **Table models are checked too**: `table=True` models skip Pydantic validation, so `SQLModelBase.model_post_init` calls `ensure_json_within_limits` on these fields (call `super()` when overriding `model_post_init`). Code paths that bypass models can call `ensure_json_within_limits(value)` directly.
- The 100K limit does not appear in the JSON Schema (JSON Schema cannot express an encoded length for objects); document it in the field docstring.

::: warning Known limitation: `JSON100K | None` as a table field
On Python 3.12, `data: JSON100K | None = None` as a field of a `table=True` model fails at class creation with `has no matching SQLAlchemy type`. Specify the column type explicitly:

```python
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K


class Doc(SQLModelBase, UUIDTableBaseMixin, table=True):
    data: JSON100K | None = Field(default=None, sa_type=JSONB)
```
:::

### `NumpyVector[dims, dtype]`

```python
from sqlmodel_ext.field_types.dialects.postgresql import NumpyVector
```

pgvector + NumPy integration: pgvector's `Vector` in the database, `numpy.ndarray` in Python.

| Parameter | Meaning |
|-----------|---------|
| `dims` | Vector dimension (e.g. `1536`) |
| `dtype` | NumPy dtype (e.g. `numpy.float32`) |

Requires `numpy` + `pgvector`, bundled in the `[pgvector]` extra.

## Exception types

```python
from sqlmodel_ext.field_types.dialects.postgresql import (
    VectorError,
    VectorDimensionError,
    VectorDTypeError,
    VectorDecodeError,
)
```

- `VectorError` — base class
- `VectorDimensionError` — dimension mismatch
- `VectorDTypeError` — dtype mismatch
- `VectorDecodeError` — deserialization failure
