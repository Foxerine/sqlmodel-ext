# 字段类型

::: tip
本页是参考文档。要看怎么在模型上用这些类型，去 [快速上手](/tutorials/01-getting-started)。为什么要用类型别名而不是散落的 `Field(max_length=...)`，见 [设计哲学：单点真相](/explanation/single-source-of-truth)。
:::

除「PostgreSQL 专属类型」一节外，所有字段类型均可从 `sqlmodel_ext` 顶层导入。它们大多是 `Annotated` 类型别名（`TypeAlias`），**一份声明同时产出** Pydantic 校验、SQLAlchemy 列类型和 OpenAPI schema。

`Str*` / `Text*` / `NonEmpty*` / `SearchQueryStr64` 还隐含 `pattern=r'^[^\x00]*$'`（拒绝 NUL 字节，避免 PostgreSQL 文本列报错）。

## 字符串约束

```python
from sqlmodel_ext import Str1, Str16, Str24, Str32, Str36, Str48, Str64, Str100, Str128, Str255, Str256, Str500, Str512, Str2048
```

| 类型 | `max_length` | 等价定义 |
|------|--------------|---------|
| `Str1` | 1 | `Annotated[str, Field(max_length=1), _NO_NULL_BYTE]` |
| `Str16` | 16 | 同上 |
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
    NonEmptyStrippedStr32, NonEmptyStrippedStr64, NonEmptyStrippedStr128, NonEmptyStrippedStr256,
    SingleLineStr64, SearchQueryStr64, HttpHeaderName,
    Sha256Hex, BCP47LanguageCode,
)
```

| 类型 | 约束 |
|------|------|
| `NonEmptyStr64/128/256` | `1 <= len <= N`，拒绝空字符串 `""` |
| `NonEmptyStrippedStr32/64/128/256` | 同上 + `strip_whitespace`，拒绝纯空白（`"   "` / `"\t"`） |
| `SingleLineStr64` | 等于 `NonEmptyStrippedStr64` + **单行**：拒绝 NUL 与所有换行字符（`\n \v \f \r \x1c \x1d \x1e \x85    `，由 `str.splitlines()` 推导）；制表符允许。用于逐行渲染的名称，防止用换行伪造列表条目 |
| `SearchQueryStr64` | 去空白后 `2 <= len <= 64`：单字符 trigram 查询没有选择性，会退化成近乎全表扫描 |
| `HttpHeaderName` | RFC 9110 `token`（字母、数字与 ``!#$%&'*+-.^_`|~``），`1 <= len <= 64` |
| `Sha256Hex` | 精确 64 位小写 hex（SHA-256 摘要） |
| `BCP47LanguageCode` | BCP-47 语言代码语法（如 `zh-Hans-CN`），`max_length=16` |

## 文本约束

```python
from sqlmodel_ext import Text1K, Text1024, Text2K, Text2500, Text3K, Text3072, Text4K, Text5K, Text8K, Text10K, Text16K, Text32K, Text48K, Text60K, Text64K, Text100K, Text128K, Text1M
```

| 类型 | `max_length` |
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
| `Text128K` | 131072（= 128 × 1024） |
| `Text1M` | 1000000 |

## 反射长度上限：`max_length_of()`

```python
from sqlmodel_ext import max_length_of
```

```python
def max_length_of(alias: Any) -> int
```

代码需要"这个字段最长多少"时，从别名里反射出来，而不是再写一遍数字（第二个常量迟早与别名漂移）。

- 与 Pydantic 实际执行的规则一致：`Field(max_length=N)` 的 `MaxLen`、`StringConstraints.max_length`、递归展开的 `GroupedMetadata`；多个约束叠加时**最后一个生效**；
- 接受 `X | None`（必须恰好一个非 `None` 成员）；
- 对 `Array[T, N]` 返回**元素个数**上限 `N`；
- 别名没有声明 `max_length` 时抛 `TypeError`，而不是编一个数字。

```python
from sqlmodel_ext import Str64, max_length_of
from sqlmodel_ext.field_types.dialects.postgresql import Array

assert max_length_of(Str64) == 64
assert max_length_of(Str64 | None) == 64
assert max_length_of(Array[str, 20]) == 20
```

## 数值约束

```python
from sqlmodel_ext import (
    Port, Percentage,
    PositiveInt, NonNegativeInt,
    PositiveBigInt, NonNegativeBigInt, SignedBigInt,
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
| `SignedBigInt` | `-JS_MAX_SAFE_INTEGER` ~ `JS_MAX_SAFE_INTEGER` | `BIGINT`（用于增量 / 差值列） |
| `PositiveFloat` | `> 0.0`，**有限值** | `FLOAT` |
| `NonNegativeFloat` | `>= 0.0`，**有限值** | `FLOAT` |

::: info 浮点类型拒绝 `inf` / `nan`（0.5.0 起）
Pydantic 的 float 默认 `allow_inf_nan=True`，而 `gt=0` 只是一次比较，`inf` 能通过——JSON 数字 `1e309` 会解析成 `float('inf')`，之后在别处炸掉（例如 `math.ceil(float('inf'))` 抛 `OverflowError`）。这两个别名带 `AllowInfNan(False)`，在边界处拒绝。
:::

::: info BigInt 的 JS_MAX_SAFE_INTEGER 上界
`*BigInt` 的上界是 `JS_MAX_SAFE_INTEGER = 2⁵³ − 1`，**不是** `INT64_MAX`。原因是浏览器 JSON 解析超出该范围会丢失精度。如果你的 API 不面向浏览器，可自定义别名将上界改为 `INT64_MAX`。
:::

### 常量

```python
from sqlmodel_ext import INT32_MIN, INT32_MAX, INT64_MAX, JS_MAX_SAFE_INTEGER
```

| 常量 | 值 |
|------|-----|
| `INT32_MIN` | `-2_147_483_648`（−2³¹） |
| `INT32_MAX` | `2_147_483_647`（2³¹−1） |
| `INT64_MAX` | `9_223_372_036_854_775_807`（2⁶³−1） |
| `JS_MAX_SAFE_INTEGER` | `9_007_199_254_740_991`（2⁵³−1） |

## Decimal 约束

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

命名约定：`[Optional|Nullable][Signed|NonNegative|Positive][Write|Sum]Decimal{precision}_{scale}`。

| 类型 | 符号 | 可写位数（整数 + 小数） | 数据库列 |
|------|------|---------|---------|
| `SignedDecimal38_18` | 任意 | 20 + 18 | `NUMERIC(38, 18)` |
| `NonNegativeDecimal38_18` | `>= 0` | 20 + 18 | `NUMERIC(38, 18)` |
| `PositiveDecimal38_18` | `> 0` | 20 + 18 | `NUMERIC(38, 18)` |
| `OptionalNonNegativeDecimal38_18` | `>= 0` 或 `None`，默认 `None` | 20 + 18 | `NUMERIC(38, 18)` |
| `OptionalSignedDecimal38_18` | 任意或 `None`，默认 `None` | 20 + 18 | `NUMERIC(38, 18)` |
| `SignedWriteDecimal38_18` | 任意 | **17 + 18** | `NUMERIC(38, 18)` |
| `NonNegativeWriteDecimal38_18` | `>= 0` | 17 + 18 | `NUMERIC(38, 18)` |
| `PositiveWriteDecimal38_18` | `> 0` | 17 + 18 | `NUMERIC(38, 18)` |
| `OptionalNonNegativeWriteDecimal38_18` | `>= 0` 或 `None`，默认 `None` | 17 + 18 | `NUMERIC(38, 18)` |
| `OptionalSignedWriteDecimal38_18` | 任意或 `None`，默认 `None` | 17 + 18 | `NUMERIC(38, 18)` |
| `SignedSumDecimal38_18` | 任意 | 20 + 18 | 无（仅 DTO，用来读 `SUM()` 结果） |
| `SignedDecimal20_10` | 任意 | 10 + 10 | `NUMERIC(20, 10)` |
| `NonNegativeDecimal20_10` | `>= 0` | 10 + 10 | `NUMERIC(20, 10)` |
| `OptionalNonNegativeDecimal20_10` | `>= 0` 或 `None`，默认 `None` | 10 + 10 | `NUMERIC(20, 10)` |
| `NullableNonNegativeDecimal20_10` | `>= 0` 或 `None`，**无默认值**（校验模型里是必填键，值可为 null） | 10 + 10 | `NUMERIC(20, 10)` |

行为契约：

- **整数位、小数位、总位数都被校验**（0.5.0 修复：此前元数据顺序让 Pydantic 只校验总位数与小数位，整数位只能靠数据库兜底）
- **拒绝 float / bool 输入**（IEEE 754 已丢精度）——接受 `Decimal` / `int` / `str`
- **JSON 序列化为定点字符串**（`model_dump_json()`），永不出现科学计数法（`0E-18` → `'0'`）、剔除冗余尾零（`1200.000...0` → `'1200'`），防止 JS Number 精度损失
- **dict 模式保留 `Decimal` 对象**（`model_dump()`）
- `Optional*` / `Nullable*` 变体的数值约束嵌套在内层 `Annotated`，JSON `null` 解析安全
- **OpenAPI 请求体 schema 仅 `string`**：Pydantic 默认把 `Decimal` 映射为 `anyOf: [number, string]`，但运行时拒绝 float；这些别名通过 `WithJsonSchema(mode='validation')` 把请求体 schema 收窄为带定点小数 pattern 的 `string`。pattern 只负责声明类型，位数上限由 `max_digits` / `decimal_places` 执行。响应体 schema 不受影响。

### 写 35 位、读 38 位：给 `SUM()` 留余量

如果单行就能写满列宽，多行的 `SUM()` 就可能溢出列宽，而用 38 位类型读回这个和会校验失败。会被汇总的值应该用 `*WriteDecimal38_18` **写**（35 位：17 位整数 + 18 位小数，列仍是 `NUMERIC(38, 18)`），汇总结果用 `SignedSumDecimal38_18` **读**（完整 38 位）。`10^(38-18) / 10^(35-18) = 1000`：要加满 1000 个最大行才会触及列宽——这是余量，不是保证。

| 常量 | 值 | 含义 |
|---|---|---|
| `DECIMAL_38_18_COLUMN_DIGITS` | `38` | 列宽，也是 `*Decimal38_18` 的写入上限 |
| `DECIMAL_38_18_WRITE_DIGITS` | `35` | `*WriteDecimal38_18` 的写入上限 |
| `DECIMAL_38_18_PLACES` | `18` | 小数位（写、和两侧相同） |

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

## 有界长度 List 别名

```python
from sqlmodel_ext import List, List1, List2, List3, List7, List10, List16, List20, List32, List40, List50, List64, List100, List128, List200, List256, List1024
```

`List<N>[T]` 等价 `Annotated[list[T], Field(max_length=N)]`——最大长度编码在类型名中（与 `Str64` / `Text1K` 命名一致）。`List[T]`（无数字）等价 `list[T]`。用于请求 DTO / 协议层等非数据库列场景，与 PG `Array[T]` 列类型无关。

## 字段标记：`EXCLUDE_IF_NONE`

```python
from sqlmodel_ext import EXCLUDE_IF_NONE
```

放在 `Annotated` 的元数据位置：值为 `None` 时，这个**键**不出现在任何序列化输出里。三件套缺一不可：

```python
from typing import Annotated

from sqlmodel_ext import EXCLUDE_IF_NONE, SQLModelBase


class Event(SQLModelBase):
    marker: Annotated[bool | None, EXCLUDE_IF_NONE] = None


assert Event().model_dump() == {}
assert Event.model_validate_json(Event().model_dump_json()).marker is None
```

（1）`| None`，（2）该标记，（3）`= None` 默认值——没有默认值时，dump 出去的 JSON 读不回来。与 `exclude_none` 不同，它是字段自身的性质，不依赖调用方传参。典型场景：给用 `extra='forbid'` 严格反序列化的旧消费方新增可空字段。仅用于**非表**模型。与 `Unset` 的区别见 [Unset 三态](/explanation/unset-three-state#exclude-if-none-与-unset-的区别)。

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
from sqlmodel_ext import IPAddress, ClientIPAddress
```

| 类型 | 用途 | Python 值 |
|---|---|---|
| `IPAddress` | **存储列**：校验 IPv4 / IPv6 格式，以 `VARCHAR` 存储 | `str` 子类；额外方法 `is_private() -> bool` |
| `ClientIPAddress` | **解析时**：校验**不可信文本**（如反向代理请求头）中的客户端 IP | `IPv4Address \| IPv6Address`（Pydantic `IPvAnyAddress`） |

`ClientIPAddress` 在 `IPvAnyAddress` 的结构校验之上额外拒绝 IPv6 zone ID（`fe80::1%eth0`）：zone ID 指的是本地网卡，不属于网络地址，而且 `ipaddress` 对它没有长度限制。把结果存库时用 `IPAddress`。

## 路径类型

```python
from sqlmodel_ext import FilePathType, DirectoryPathType
```

| 类型 | 校验 |
|------|------|
| `FilePathType` | 路径必须包含文件名 |
| `DirectoryPathType` | 路径不能包含文件扩展名 |

行为上等价于 `pathlib.Path`，可直接当 `Path` 使用。

## `ModuleNameMixin`

```python
from sqlmodel_ext import ModuleNameMixin
```

实例化时，如果没有传入目标字段，自动把它设为**调用方所在模块**的 `__name__`。目标字段默认叫 `name`，可通过类变量 `_module_name_field` 改名。字段本身需要你在模型里声明。

## PostgreSQL 专属类型

::: warning 仅限 PostgreSQL
本节类型使用 PostgreSQL 原生列类型，不适用于 SQLite / MySQL。`JSON100K` / `JSONList100K` 需要 `pip install sqlmodel-ext[postgresql]`（`orjson`），`NumpyVector` 需要 `[pgvector]`。
:::

### `Array[T]` / `Array[T, N]`

```python
from sqlmodel_ext.field_types.dialects.postgresql import Array
```

PostgreSQL `ARRAY` 列。第二个参数是可选的元素个数上限（进入 JSON Schema 的 `maxItems`，也能被 `max_length_of()` 反射）。

| Python 表现 | 数据库列 |
|------------|---------|
| `list[str]` | `VARCHAR[]` |
| `list[int]` | `INTEGER[]` |
| `list[dict]` | `JSONB[]` |
| `list[UUID]` | `UUID[]` |
| `list[SomeEnum]` | `someenum[]`（读容忍，见下） |

其他元素类型在类创建时抛 `TypeError`。

**枚举数组读容忍**：`Array[SomeEnum]` 列会被包装为读容忍的 `TypeDecorator`。当数据库返回的枚举值不在当前进程的 Python 枚举中时（典型场景：滚动部署的版本偏差窗口——较新的实例已把新枚举值写入数组列，旧实例的代码尚未认识该值），该元素在**读路径**被丢弃并记一条 warning，而非抛 `LookupError` 导致 500。写路径仍走严格校验，不会脏写；被丢弃的值仍在数据库中，认识它的代码版本部署后恢复可见。

### `JSON100K` / `JSONList100K`

```python
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K, JSONList100K, ensure_json_within_limits
```

| 类型 | Python 表现 | 数据库列 | 上限 |
|------|-----------|---------|---------|
| `JSON100K` | `dict[str, Any]` | `JSONB` | 规范 JSON 编码 ≤ 100K 字符 |
| `JSONList100K` | `list[dict[str, Any]]` | `JSONB` | 规范 JSON 编码 ≤ 100K 字符 |

契约：**对象进，对象出**。

- **入站**：接受 JSON 对象 / 数组（推荐）或 JSON **字符串**（兼容形式）。两种形式都要满足：可编码（嵌套深度在 orjson 与 Pydantic 两个序列化器的上限内——Pydantic 的上限与平台有关，Windows 构建约 98 层，Linux 上更高）且规范编码不超过 100K **字符**（按字符而不是 UTF-8 字节计，中文不会被提前 3 倍拒绝）。
- **出站**：`model_dump()`、`model_dump(mode='json')`、`model_dump_json()` 都输出对象 / 数组本身（0.5.0 起；此前输出 JSON 字符串）。序列化 schema 声明为纯 object / array；校验 schema 如实声明为 `anyOf[object, string]`。
- **表模型也检查**：`table=True` 模型跳过 Pydantic 校验，`SQLModelBase.model_post_init` 会对这些字段调用 `ensure_json_within_limits`（覆写 `model_post_init` 时必须调用 `super()`）。绕过模型的代码路径可以直接调用 `ensure_json_within_limits(value)`。
- 100K 上限不会出现在 JSON Schema 里（JSON Schema 无法对对象表达"编码长度"），请写进字段 docstring。

::: warning 已知限制：`JSON100K | None` 作为表字段
在 Python 3.12 上，`data: JSON100K | None = None` 作为 `table=True` 模型的字段会在类创建时报 `has no matching SQLAlchemy type`。显式指定列类型即可：

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

pgvector + NumPy 集成：数据库中是 pgvector 的 `Vector`，Python 中是 `numpy.ndarray`。

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
