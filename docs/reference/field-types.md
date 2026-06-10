# 字段类型

::: tip
本页是参考文档。要看怎么在模型上用这些类型，去 [快速上手](/tutorials/01-getting-started)。
:::

所有字段类型均可从 `sqlmodel_ext` 顶层导入。它们都是 `Annotated` 类型别名（`TypeAlias`），同时满足 Pydantic 验证和 SQLAlchemy 列类型映射。

所有字符串约束类型还隐含 `pattern=r'^[^\x00]*$'`（拒绝 NUL 字节，避免 PostgreSQL 文本列报错）。

## 字符串约束

```python
from sqlmodel_ext import Str16, Str24, Str32, Str36, Str48, Str64, Str100, Str128, Str255, Str256, Str500, Str512, Str2048
```

| 类型 | `max_length` | 等价定义 |
|------|--------------|---------|
| `Str16` | 16 | `Annotated[str, Field(max_length=16), _NO_NULL_BYTE]` |
| `Str24` | 24 | 同上 |
| `Str32` | 32 | 同上 |
| `Str36` | 36 | 同上（UUID 标准字符串长度） |
| `Str48` | 48 | 同上 |
| `Str64` | 64 | 同上 |
| `Str100` | 100 | 同上 |
| `Str128` | 128 | 同上 |
| `Str255` | 255 | 同上 |
| `Str256` | 256 | 同上 |
| `Str500` | 500 | 同上 |
| `Str512` | 512 | 同上 |
| `Str2048` | 2048 | 同上 |

### 非空与专用字符串

```python
from sqlmodel_ext import (
    NonEmptyStr64, NonEmptyStr128, NonEmptyStr256,
    NonEmptyStrippedStr64, NonEmptyStrippedStr128, NonEmptyStrippedStr256,
    Sha256Hex, BCP47LanguageCode,
)
```

| 类型 | 约束 |
|------|------|
| `NonEmptyStr64/128/256` | `1 <= len <= N`，拒绝空字符串 `""` |
| `NonEmptyStrippedStr64/128/256` | 同上 + `strip_whitespace`，拒绝纯空白（`"   "` / `"\t"`） |
| `Sha256Hex` | 精确 64 位小写 hex（SHA-256 摘要） |
| `BCP47LanguageCode` | BCP-47 语言代码语法（如 `zh-Hans-CN`），`max_length=16` |

## 文本约束

```python
from sqlmodel_ext import Text1K, Text1024, Text2K, Text2500, Text3K, Text5K, Text8K, Text10K, Text16K, Text32K, Text48K, Text60K, Text64K, Text100K, Text128K, Text1M
```

| 类型 | `max_length` |
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
| `Text128K` | 131072（= 128 × 1024） |
| `Text1M` | 1000000 |

## 数值约束

```python
from sqlmodel_ext import (
    Port, Percentage,
    PositiveInt, NonNegativeInt,
    PositiveBigInt, NonNegativeBigInt,
    PositiveFloat, NonNegativeFloat,
)
```

| 类型 | 范围 | 数据库列 |
|------|------|---------|
| `Port` | `1` ~ `65535` | `INTEGER` |
| `Percentage` | `0` ~ `100` | `INTEGER` |
| `PositiveInt` | `1` ~ `INT32_MAX` | `INTEGER` |
| `NonNegativeInt` | `0` ~ `INT32_MAX` | `INTEGER` |
| `PositiveBigInt` | `1` ~ `JS_MAX_SAFE_INTEGER` | `BIGINT` |
| `NonNegativeBigInt` | `0` ~ `JS_MAX_SAFE_INTEGER` | `BIGINT` |
| `PositiveFloat` | `> 0.0` | `FLOAT` |
| `NonNegativeFloat` | `>= 0.0` | `FLOAT` |

::: info BigInt 的 JS_MAX_SAFE_INTEGER 上界
`PositiveBigInt` / `NonNegativeBigInt` 的上界是 `JS_MAX_SAFE_INTEGER = 2⁵³ − 1`，**不是** `INT64_MAX`。原因是浏览器 JSON 解析超出该范围会丢失精度。如果你的 API 不面向浏览器，可自定义别名将上界改为 `INT64_MAX`。
:::

### 常量

```python
from sqlmodel_ext import INT32_MAX, INT64_MAX, JS_MAX_SAFE_INTEGER
```

| 常量 | 值 |
|------|-----|
| `INT32_MAX` | `2_147_483_647`（2³¹−1） |
| `INT64_MAX` | `9_223_372_036_854_775_807`（2⁶³−1） |
| `JS_MAX_SAFE_INTEGER` | `9_007_199_254_740_991`（2⁵³−1） |

## Decimal 约束（0.4.0 新增）

```python
from sqlmodel_ext import (
    SignedDecimal38_18, NonNegativeDecimal38_18, PositiveDecimal38_18, OptionalNonNegativeDecimal38_18,
    SignedDecimal20_10, NonNegativeDecimal20_10, OptionalNonNegativeDecimal20_10,
)
```

命名约定：`[Optional][Signed|NonNegative|Positive]Decimal{precision}_{scale}`，对应数据库列 `NUMERIC(precision, scale)`。

| 类型 | 符号 | 数据库列 |
|------|------|---------|
| `SignedDecimal38_18` | 任意 | `NUMERIC(38, 18)` |
| `NonNegativeDecimal38_18` | `>= 0` | `NUMERIC(38, 18)` |
| `PositiveDecimal38_18` | `> 0` | `NUMERIC(38, 18)` |
| `OptionalNonNegativeDecimal38_18` | `>= 0` 或 `None` | `NUMERIC(38, 18)` |
| `SignedDecimal20_10` | 任意 | `NUMERIC(20, 10)` |
| `NonNegativeDecimal20_10` | `>= 0` | `NUMERIC(20, 10)` |
| `OptionalNonNegativeDecimal20_10` | `>= 0` 或 `None` | `NUMERIC(20, 10)` |

行为契约：

- **拒绝 float / bool 输入**（IEEE 754 已丢精度）——接受 `Decimal` / `int` / `str`
- **JSON 序列化为定点字符串**（`model_dump_json()`），永不出现科学计数法（`0E-18` → `'0'`）、剔除冗余尾零（`1200.000...0` → `'1200'`），防止 JS Number 精度损失
- **dict 模式保留 `Decimal` 对象**（`model_dump()`）
- `Optional*` 变体的约束嵌套在内层 `Annotated`，JSON `null` 解析安全

## 有界长度 List 别名（0.4.0 新增）

```python
from sqlmodel_ext import List, List1, List2, List3, List7, List10, List16, List20, List32, List40, List50, List64, List100, List128, List200, List256, List1024
```

`List<N>[T]` 等价 `Annotated[list[T], Field(max_length=N)]`——最大长度编码在类型名中（与 `Str64` / `Text1K` 命名一致）。`List[T]`（无数字）等价 `list[T]`。用于请求 DTO / 协议层等非数据库列场景，与 PG `Array[T]` 列类型无关。

## URL 类型

```python
from sqlmodel_ext import Url, HttpUrl, WebSocketUrl, SafeHttpUrl, UnsafeURLError, validate_not_private_host
```

四种 URL 类型，都继承 `str`，数据库中存储为 `VARCHAR`。

| 类型 | 允许的协议 | SSRF 防护 |
|------|-----------|---------|
| `Url` | 任意（http、ftp、ws...） | 否 |
| `HttpUrl` | `http` / `https` | 否 |
| `WebSocketUrl` | `ws` / `wss` | 否 |
| `SafeHttpUrl` | `http` / `https` | **是** |

`SafeHttpUrl` 拒绝以下地址：

- 回环（`localhost`、`127.0.0.1`、`::1`）
- 私有 IP（`10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`）
- 链路本地（`169.254.0.0/16`）
- 保留地址

拒绝时抛出 `UnsafeURLError`。

`validate_not_private_host(host: str) -> None` 是底层校验函数，可直接调用。

## IP 地址

```python
from sqlmodel_ext import IPAddress
```

继承自 `IPv4Address | IPv6Address`，自动做 IPv4/IPv6 格式校验。

**额外方法**：

```python
def is_private(self) -> bool
```

判断是否为私有地址（包括回环、链路本地等）。

## 路径类型

```python
from sqlmodel_ext.field_types import FilePathType, DirectoryPathType
```

| 类型 | 校验 |
|------|------|
| `FilePathType` | 路径必须包含文件名（含扩展名） |
| `DirectoryPathType` | 路径不能包含文件扩展名 |

行为上等价于 `pathlib.Path`，可直接当 `Path` 使用。

## 路径与命名 Mixin

```python
from sqlmodel_ext import ModuleNameMixin
```

为模型添加 `module_name: str` 字段（用于动态加载/反射场景）。详见源码 `field_types/mixins/`。

## PostgreSQL 专属类型

::: warning 仅限 PostgreSQL
本节类型使用 PostgreSQL 原生列类型，不适用于 SQLite / MySQL。需要 `pip install sqlmodel-ext[postgresql]` 或 `[pgvector]`。
:::

### `Array[T]`

```python
from sqlmodel_ext.field_types.dialects.postgresql import Array
```

PostgreSQL `ARRAY` 列。

| Python 表现 | 数据库列 |
|------------|---------|
| `list[str]` | `TEXT[]` |
| `list[int]` | `INTEGER[]` |
| `list[float]` | `FLOAT[]` |
| `list[UUID]` | `UUID[]` |

### `JSON100K` / `JSONList100K`

```python
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K, JSONList100K
```

| 类型 | Python 表现 | 数据库列 | 长度上限 |
|------|-----------|---------|---------|
| `JSON100K` | `dict` | `JSONB` | 100K 字符 |
| `JSONList100K` | `list` | `JSONB` | 100K 字符 |

需要 `orjson` 加速序列化，已包含在 `[postgresql]` extras 中。

### `NumpyVector[dims, dtype]`

```python
from sqlmodel_ext.field_types.dialects.postgresql import NumpyVector
```

pgvector + NumPy 集成。

| 参数 | 含义 |
|------|------|
| `dims` | 向量维度（如 `1536`） |
| `dtype` | NumPy dtype（如 `numpy.float32`） |

需要 `numpy` + `pgvector`，包含在 `[pgvector]` extras 中。

## 异常类型

```python
from sqlmodel_ext.field_types.dialects.postgresql import (
    VectorError,
    VectorDimensionError,
    VectorDTypeError,
    VectorDecodeError,
)
```

- `VectorError` — 基类
- `VectorDimensionError` — 维度不匹配
- `VectorDTypeError` — dtype 不匹配
- `VectorDecodeError` — 反序列化失败
