# 字段类型

sqlmodel-ext 提供了一系列预定义的字段类型，同时满足 Pydantic 数据验证和 SQLAlchemy 列类型映射。

## 字符串约束

写 `name: Str64` 等于同时告诉 Pydantic 验证 `max_length=64`，以及告诉 SQLAlchemy 创建 `VARCHAR(64)` 列。

| 类型 | 最大长度 | 典型用途 |
|------|----------|---------|
| `Str24` | 24 | 短编码 |
| `Str32` | 32 | Token、哈希 |
| `Str36` | 36 | UUID 字符串格式 |
| `Str48` | 48 | 短标签 |
| `Str64` | 64 | 名称、标题 |
| `Str100` | 100 | 简短描述 |
| `Str128` | 128 | 路径、标识符 |
| `Str255` / `Str256` | 255 / 256 | 标准 VARCHAR |
| `Str512` | 512 | 长标识符、长路径 |
| `Text1K` ~ `Text100K` | 1,000 ~ 100,000 | 各种长度的文本（含 `Text5K`） |

```python
from sqlmodel_ext import SQLModelBase, Str64, Str255, Text1K

class Article(SQLModelBase):
    title: Str64          # VARCHAR(64)
    summary: Str255       # VARCHAR(255)
    body: Text1K          # VARCHAR(1000)
```

## 数值约束

| 类型 | 范围 | 典型用途 |
|------|------|---------|
| `Port` | 1 ~ 65535 | 网络端口 |
| `Percentage` | 0 ~ 100 | 百分比 |
| `PositiveInt` | >= 1 | 计数、数量 |
| `NonNegativeInt` | >= 0 | 索引、计数器 |
| `PositiveFloat` | > 0.0 | 价格、重量 |
| `PositiveBigInt` | >= 1（BigInteger） | 大整数 ID、时间戳 |
| `NonNegativeBigInt` | >= 0（BigInteger） | 大整数计数器 |

```python
from sqlmodel_ext import Port, Percentage

class ServerConfig(SQLModelBase):
    port: Port                    # 自动验证 1~65535
    cpu_threshold: Percentage     # 自动验证 0~100
```

## URL 类型

四种 URL 类型，都继承 `str`，在数据库中存储为普通 `VARCHAR`：

| 类型 | 允许的协议 | SSRF 防护 |
|------|-----------|----------|
| `Url` | 任意（http, ftp, ws, ...） | 无 |
| `HttpUrl` | 仅 http / https | 无 |
| `WebSocketUrl` | 仅 ws / wss | 无 |
| `SafeHttpUrl` | 仅 http / https | **有** |

```python
from sqlmodel_ext import HttpUrl, SafeHttpUrl

class Webhook(SQLModelBase):
    url: HttpUrl             # 验证 HTTP 格式
    callback: SafeHttpUrl    # 验证 HTTP 格式 + 阻止内网地址
```

### SafeHttpUrl 与 SSRF 防护

`SafeHttpUrl` 在验证 URL 格式之外，还会阻止指向内网的地址，防止 SSRF 攻击：

- 禁止 `localhost`、`127.0.0.1`、`::1` 等回环地址
- 禁止 `10.x.x.x`、`192.168.x.x`、`172.16-31.x.x` 等私有 IP
- 禁止链路本地地址和保留地址

适用于用户提交的回调 URL、Webhook 地址等场景。

## IPAddress 类型

验证 IPv4/IPv6 格式，额外提供 `is_private()` 方法：

```python
from sqlmodel_ext import IPAddress

class Device(SQLModelBase):
    ip: IPAddress

device = Device(ip="192.168.1.1")
device.ip.is_private()  # True
```

## 路径类型

```python
from sqlmodel_ext.field_types import FilePathType, DirectoryPathType

class Storage(SQLModelBase):
    file: FilePathType           # 要求包含文件扩展名
    directory: DirectoryPathType # 要求不包含扩展名
```

## PostgreSQL 专属类型

```python
from sqlmodel_ext.field_types.dialects.postgresql import Array

class Tag(SQLModelBase, UUIDTableBaseMixin, table=True):
    labels: Array[str]     # 映射 PostgreSQL TEXT[]
    scores: Array[float]   # 映射 PostgreSQL FLOAT[]
```

`Array[T]` 在 Pydantic 中表现为 `list[T]`，在 PostgreSQL 中映射为 `ARRAY` 列类型。
