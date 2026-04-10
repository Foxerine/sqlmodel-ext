# Field Types

sqlmodel-ext provides a set of predefined field types that satisfy both Pydantic data validation and SQLAlchemy column type mapping simultaneously.

## String Constraints

Writing `name: Str64` simultaneously tells Pydantic to validate `max_length=64` and SQLAlchemy to create a `VARCHAR(64)` column.

| Type | Max Length | Typical Use |
|------|-----------|-------------|
| `Str16` | 16 | Trigger words, short tokens |
| `Str24` | 24 | Short codes |
| `Str32` | 32 | Tokens, hashes |
| `Str36` | 36 | UUID string format |
| `Str48` | 48 | Short labels |
| `Str64` | 64 | Names, titles |
| `Str100` | 100 | Short descriptions |
| `Str128` | 128 | Paths, identifiers |
| `Str255` / `Str256` | 255 / 256 | Standard VARCHAR |
| `Str512` | 512 | Long identifiers, long paths |
| `Text1K` ~ `Text100K` | 1,000 ~ 100,000 | Various text lengths (incl. `Text5K`) |

```python
from sqlmodel_ext import SQLModelBase, Str64, Str255, Text1K

class Article(SQLModelBase):
    title: Str64          # VARCHAR(64)
    summary: Str255       # VARCHAR(255)
    body: Text1K          # VARCHAR(1000)
```

## Numeric Constraints

| Type | Range | Typical Use |
|------|-------|-------------|
| `Port` | 1 ~ 65535 | Network ports |
| `Percentage` | 0 ~ 100 | Percentages |
| `PositiveInt` | 1 ~ `INT32_MAX` (2³¹−1) | Counts, quantities (fits PostgreSQL INTEGER) |
| `NonNegativeInt` | 0 ~ `INT32_MAX` | Indices, counters |
| `PositiveFloat` | > 0.0 | Prices, weights |
| `PositiveBigInt` | 1 ~ `JS_MAX_SAFE_INTEGER` (2⁵³−1) | Large integer IDs, timestamps (BigInteger storage) |
| `NonNegativeBigInt` | 0 ~ `JS_MAX_SAFE_INTEGER` | Large integer counters (BigInteger storage) |

::: tip BigInt is capped at `JS_MAX_SAFE_INTEGER`
`PositiveBigInt` / `NonNegativeBigInt` use `JS_MAX_SAFE_INTEGER = 2⁵³ − 1`
as their upper bound instead of `INT64_MAX = 2⁶³ − 1`. Integers larger
than `JS_MAX_SAFE_INTEGER` lose precision when parsed by JavaScript
clients. Define a custom alias with `le=INT64_MAX` if the field is never
consumed by a browser. Both constants are exported from `sqlmodel_ext`.
:::

```python
from sqlmodel_ext import Port, Percentage

class ServerConfig(SQLModelBase):
    port: Port                    # Auto-validates 1~65535
    cpu_threshold: Percentage     # Auto-validates 0~100
```

## URL Types

Four URL types, all inheriting `str`, stored as plain `VARCHAR` in the database:

| Type | Allowed Protocols | SSRF Protection |
|------|-------------------|-----------------|
| `Url` | Any (http, ftp, ws, ...) | No |
| `HttpUrl` | http / https only | No |
| `WebSocketUrl` | ws / wss only | No |
| `SafeHttpUrl` | http / https only | **Yes** |

```python
from sqlmodel_ext import HttpUrl, SafeHttpUrl

class Webhook(SQLModelBase):
    url: HttpUrl             # Validates HTTP format
    callback: SafeHttpUrl    # Validates HTTP format + blocks internal addresses
```

### SafeHttpUrl & SSRF Protection

`SafeHttpUrl` not only validates URL format but also blocks addresses pointing to internal networks, preventing SSRF attacks:

::: danger Danger
- Blocks loopback addresses like `localhost`, `127.0.0.1`, `::1`
- Blocks private IPs like `10.x.x.x`, `192.168.x.x`, `172.16-31.x.x`
- Blocks link-local and reserved addresses

Use for user-submitted callback URLs, webhook addresses, etc.
:::

## IPAddress Type

Validates IPv4/IPv6 format with an additional `is_private()` method:

```python
from sqlmodel_ext import IPAddress

class Device(SQLModelBase):
    ip: IPAddress

device = Device(ip="192.168.1.1")
device.ip.is_private()  # True
```

## Path Types

```python
from sqlmodel_ext.field_types import FilePathType, DirectoryPathType

class Storage(SQLModelBase):
    file: FilePathType           # Requires a file extension
    directory: DirectoryPathType # Requires no extension
```

## PostgreSQL-only Types

::: warning Warning
`Array[T]` uses PostgreSQL's native `ARRAY` column type and is not compatible with SQLite or other databases.
:::

```python
from sqlmodel_ext.field_types.dialects.postgresql import Array

class Tag(SQLModelBase, UUIDTableBaseMixin, table=True):
    labels: Array[str]     # Maps to PostgreSQL TEXT[]
    scores: Array[float]   # Maps to PostgreSQL FLOAT[]
```

`Array[T]` behaves as `list[T]` in Pydantic and maps to `ARRAY` column type in PostgreSQL.
