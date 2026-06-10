# 分页类型

::: tip
本页是参考文档。要看怎么把分页接到端点上，去 [给列表端点加分页](/how-to/paginate-a-list-endpoint)。
:::

## `PaginationRequest`

```python
from sqlmodel_ext import PaginationRequest
```

继承自 `SQLModelBase`。承载分页和排序参数的 DTO。

**字段**：

| 字段 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `offset` | `int \| None` | `0` | `ge=0` |
| `limit` | `int \| None` | `50` | `ge=1, le=100` |
| `desc` | `bool \| None` | `True` | — |
| `order` | `Literal["created_at", "updated_at"] \| None` | `"created_at"` | — |

## `TimeFilterRequest`

```python
from sqlmodel_ext import TimeFilterRequest
```

继承自 `SQLModelBase`。承载时间过滤参数的 DTO。

**字段**：

| 字段 | 类型 | 默认值 | 语义 |
|------|------|--------|------|
| `created_after_datetime` | `datetime \| None` | `None` | `created_at >= 此值` |
| `created_before_datetime` | `datetime \| None` | `None` | `created_at < 此值` |
| `updated_after_datetime` | `datetime \| None` | `None` | `updated_at >= 此值` |
| `updated_before_datetime` | `datetime \| None` | `None` | `updated_at < 此值` |

时间区间为左闭右开 `[after, before)`。

**`model_post_init` 校验**：

- `created_after_datetime >= created_before_datetime` → `ValueError`
- `updated_after_datetime >= updated_before_datetime` → `ValueError`
- `created_after_datetime >= updated_before_datetime` → `ValueError`（创建时间不能晚于更新时间）

## `TableViewRequest`

```python
from sqlmodel_ext import TableViewRequest
```

```python
class TableViewRequest(TimeFilterRequest, PaginationRequest):
    pass
```

`TimeFilterRequest` 和 `PaginationRequest` 的合并。同时承载分页 + 排序 + 时间过滤参数。

`TableBaseMixin.get()` / `get_with_count()` 接受 `table_view: TableViewRequest | None` 参数。当 `offset` / `limit` / `order_by` / 时间过滤参数同时提供时，**显式参数优先**，未提供时回退到 `table_view` 的值。

## `ListResponse[T]`

```python
from sqlmodel_ext import ListResponse
```

继承自 `pydantic.BaseModel`（**不是** `SQLModelBase`），泛型类。

::: info 为什么不继承 SQLModelBase
SQLModel 的元类与 `Generic[T]` 的 schema 生成有冲突，参见 sqlmodel#1002。`ListResponse` 故意改用 `BaseModel`。
:::

**字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `count` | `int` | 匹配条件的总记录数 |
| `items` | `list[T]` | 当前页数据 |

**`model_config`**：

```python
model_config = ConfigDict(use_attribute_docstrings=True)
```

**典型返回值类型**：`get_with_count()` 返回 `ListResponse[T]`。

## 信息响应 Mixin（DTO）

```python
from sqlmodel_ext import (
    IntIdInfoMixin,
    UUIDIdInfoMixin,
    DatetimeInfoMixin,
    IntIdDatetimeInfoMixin,
    UUIDIdDatetimeInfoMixin,
)
```

用于响应 DTO 的 Mixin。这些字段在 API 响应中**总是有值**，所以声明为必填（无 `| None`）——区别于 `TableBaseMixin` 中的 `id: int | None`（INSERT 之前为 None）。

| Mixin | 字段 |
|-------|------|
| `IntIdInfoMixin` | `id: int` |
| `UUIDIdInfoMixin` | `id: UUID` |
| `DatetimeInfoMixin` | `created_at: datetime`, `updated_at: datetime` |
| `IntIdDatetimeInfoMixin` | 上面两组合（int id） |
| `UUIDIdDatetimeInfoMixin` | 上面两组合（UUID id） |

所有 Mixin 都继承 `SQLModelBase`。
