# Field types

::: tip
This is reference documentation. To see how these types appear on a model, head to [Getting started](/en/tutorials/01-getting-started).
:::

All field types are importable from the top-level `sqlmodel_ext` package. They are `Annotated` type aliases (`TypeAlias`) that satisfy both Pydantic validation and SQLAlchemy column-type mapping.

All string-constrained types also implicitly carry `pattern=r'^[^\x00]*$'` (rejects NUL bytes, which PostgreSQL text columns refuse).

## String constraints

```python
from sqlmodel_ext import Str16, Str24, Str32, Str36, Str48, Str64, Str100, Str128, Str255, Str256, Str500, Str512, Str2048
```

| Type | `max_length` | Equivalent definition |
|------|--------------|------------------------|
| `Str16` | 16 | `Annotated[str, Field(max_length=16), _NO_NULL_BYTE]` |
| `Str24` | 24 | same form |
| `Str32` | 32 | same form |
| `Str36` | 36 | same form (UUID standard string length) |
| `Str48` | 48 | same form |
| `Str64` | 64 | same form |
| `Str100` | 100 | same form |
| `Str128` | 128 | same form |
| `Str255` | 255 | same form |
| `Str256` | 256 | same form |
| `Str500` | 500 | same form |
| `Str512` | 512 | same form |
| `Str2048` | 2048 | same form |

### Non-empty & special-purpose strings

```python
from sqlmodel_ext import (
    NonEmptyStr64, NonEmptyStr128, NonEmptyStr256,
    NonEmptyStrippedStr64, NonEmptyStrippedStr128, NonEmptyStrippedStr256,
    Sha256Hex, BCP47LanguageCode,
)
```

| Type | Constraint |
|------|-----------|
| `NonEmptyStr64/128/256` | `1 <= len <= N`, rejects empty `""` |
| `NonEmptyStrippedStr64/128/256` | same + `strip_whitespace`, rejects whitespace-only (`"   "` / `"\t"`) |
| `Sha256Hex` | exactly 64 lowercase hex chars (SHA-256 digest) |
| `BCP47LanguageCode` | BCP-47 language tag syntax (e.g. `zh-Hans-CN`), `max_length=16` |

## Text constraints

```python
from sqlmodel_ext import Text1K, Text1024, Text2K, Text2500, Text3K, Text5K, Text8K, Text10K, Text16K, Text32K, Text48K, Text60K, Text64K, Text100K, Text128K, Text1M
```

| Type | `max_length` |
|------|--------------|
| `Text1K` | 1000 |
| `Text1024` | 1024 |
| `Text2K` | 2000 |
| `Text2500` | 2500 |
| `Text3K` | 3000 |
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

## Numeric constraints

```python
from sqlmodel_ext import (
    Port, Percentage,
    PositiveInt, NonNegativeInt,
    PositiveBigInt, NonNegativeBigInt,
    PositiveFloat, NonNegativeFloat,
)
```

| Type | Range | Database column |
|------|-------|------------------|
| `Port` | `1` ~ `65535` | `INTEGER` |
| `Percentage` | `0` ~ `100` | `INTEGER` |
| `PositiveInt` | `1` ~ `INT32_MAX` | `INTEGER` |
| `NonNegativeInt` | `0` ~ `INT32_MAX` | `INTEGER` |
| `PositiveBigInt` | `1` ~ `JS_MAX_SAFE_INTEGER` | `BIGINT` |
| `NonNegativeBigInt` | `0` ~ `JS_MAX_SAFE_INTEGER` | `BIGINT` |
| `PositiveFloat` | `> 0.0` | `FLOAT` |
| `NonNegativeFloat` | `>= 0.0` | `FLOAT` |

::: info JS_MAX_SAFE_INTEGER upper bound for BigInt
The upper bound of `PositiveBigInt` / `NonNegativeBigInt` is `JS_MAX_SAFE_INTEGER = 2⁵³ − 1`, **not** `INT64_MAX`. The reason is that JavaScript JSON parsers lose precision beyond this value. If your API never serves browsers, you can define a custom alias with `INT64_MAX` as the bound.
:::

### Constants

```python
from sqlmodel_ext import INT32_MAX, INT64_MAX, JS_MAX_SAFE_INTEGER
```

| Constant | Value |
|----------|-------|
| `INT32_MAX` | `2_147_483_647` (2³¹−1) |
| `INT64_MAX` | `9_223_372_036_854_775_807` (2⁶³−1) |
| `JS_MAX_SAFE_INTEGER` | `9_007_199_254_740_991` (2⁵³−1) |

## Decimal constraints (new in 0.4.0)

```python
from sqlmodel_ext import (
    SignedDecimal38_18, NonNegativeDecimal38_18, PositiveDecimal38_18, OptionalNonNegativeDecimal38_18,
    SignedDecimal20_10, NonNegativeDecimal20_10, OptionalNonNegativeDecimal20_10,
)
```

Naming convention: `[Optional][Signed|NonNegative|Positive]Decimal{precision}_{scale}`, mapping to a `NUMERIC(precision, scale)` database column.

| Type | Sign | Database column |
|------|------|------------------|
| `SignedDecimal38_18` | any | `NUMERIC(38, 18)` |
| `NonNegativeDecimal38_18` | `>= 0` | `NUMERIC(38, 18)` |
| `PositiveDecimal38_18` | `> 0` | `NUMERIC(38, 18)` |
| `OptionalNonNegativeDecimal38_18` | `>= 0` or `None` | `NUMERIC(38, 18)` |
| `SignedDecimal20_10` | any | `NUMERIC(20, 10)` |
| `NonNegativeDecimal20_10` | `>= 0` | `NUMERIC(20, 10)` |
| `OptionalNonNegativeDecimal20_10` | `>= 0` or `None` | `NUMERIC(20, 10)` |

Behavioral contract:

- **Rejects float / bool input** (precision already lost via IEEE 754) — accepts `Decimal` / `int` / `str`
- **Serializes to a fixed-point JSON string** (`model_dump_json()`): never scientific notation (`0E-18` → `'0'`), trailing zeros stripped (`1200.000...0` → `'1200'`) — protects JS clients from Number precision loss
- **dict mode keeps the `Decimal` object** (`model_dump()`)
- The `Optional*` variants nest their constraints in the inner `Annotated`, so JSON `null` parses safely

## Bounded-length list aliases (new in 0.4.0)

```python
from sqlmodel_ext import List, List1, List2, List3, List7, List10, List16, List20, List32, List40, List50, List64, List100, List128, List200, List256, List1024
```

`List<N>[T]` is equivalent to `Annotated[list[T], Field(max_length=N)]` — the max length is encoded in the type name (consistent with `Str64` / `Text1K`). `List[T]` (no number) is equivalent to `list[T]`. Intended for request DTOs / protocol layers, unrelated to the PG `Array[T]` column type.

## URL types

```python
from sqlmodel_ext import Url, HttpUrl, WebSocketUrl, SafeHttpUrl, UnsafeURLError, validate_not_private_host
```

Four URL types, all subclassing `str`, stored as `VARCHAR` in the database.

| Type | Allowed schemes | SSRF protection |
|------|-----------------|------------------|
| `Url` | any (http, ftp, ws, ...) | no |
| `HttpUrl` | `http` / `https` | no |
| `WebSocketUrl` | `ws` / `wss` | no |
| `SafeHttpUrl` | `http` / `https` | **yes** |

`SafeHttpUrl` rejects:

- Loopback (`localhost`, `127.0.0.1`, `::1`)
- Private IPs (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
- Link-local (`169.254.0.0/16`)
- Reserved addresses

A rejected URL raises `UnsafeURLError`.

`validate_not_private_host(host: str) -> None` is the underlying check function and can be called directly.

## IP address

```python
from sqlmodel_ext import IPAddress
```

Backed by `IPv4Address | IPv6Address`, automatically validates IPv4/IPv6 format.

**Extra method**:

```python
def is_private(self) -> bool
```

Returns whether the address is private (including loopback, link-local, etc.).

## Path types

```python
from sqlmodel_ext.field_types import FilePathType, DirectoryPathType
```

| Type | Validation |
|------|------------|
| `FilePathType` | Path must include a file name (with extension) |
| `DirectoryPathType` | Path must not include a file extension |

Behaves as `pathlib.Path` and can be used like one directly.

## Path & naming mixin

```python
from sqlmodel_ext import ModuleNameMixin
```

Adds a `module_name: str` field to a model (for dynamic-loading / reflection scenarios). See source `field_types/mixins/`.

## PostgreSQL-only types

::: warning PostgreSQL only
The types in this section use native PostgreSQL column types and do not work on SQLite / MySQL. Install via `pip install sqlmodel-ext[postgresql]` or `[pgvector]`.
:::

### `Array[T]`

```python
from sqlmodel_ext.field_types.dialects.postgresql import Array
```

PostgreSQL `ARRAY` column.

| Python representation | Database column |
|-----------------------|------------------|
| `list[str]` | `TEXT[]` |
| `list[int]` | `INTEGER[]` |
| `list[float]` | `FLOAT[]` |
| `list[UUID]` | `UUID[]` |

### `JSON100K` / `JSONList100K`

```python
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K, JSONList100K
```

| Type | Python representation | Database column | Length cap |
|------|-----------------------|------------------|------------|
| `JSON100K` | `dict` | `JSONB` | 100K characters |
| `JSONList100K` | `list` | `JSONB` | 100K characters |

Requires `orjson` for serialization speed; bundled with the `[postgresql]` extra.

### `NumpyVector[dims, dtype]`

```python
from sqlmodel_ext.field_types.dialects.postgresql import NumpyVector
```

pgvector + NumPy integration.

| Parameter | Meaning |
|-----------|---------|
| `dims` | Vector dimensionality (e.g. `1536`) |
| `dtype` | NumPy dtype (e.g. `numpy.float32`) |

Requires `numpy` + `pgvector`, bundled with the `[pgvector]` extra.

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
- `VectorDimensionError` — dimensionality mismatch
- `VectorDTypeError` — dtype mismatch
- `VectorDecodeError` — deserialization failure
