# 分页类型

::: tip
本页是参考文档。要看怎么把分页接到端点上，去 [给列表端点加分页](/how-to/paginate-a-list-endpoint) 和 [Keyset 游标分页](/how-to/keyset-pagination)。
:::

类层次：

```
PageWindowRequest            offset / limit / desc
  └── PaginationRequest      + order / after_id（keyset 游标）
TimeFilterRequest            created_* / updated_* 时间边界
TableViewRequest             TimeFilterRequest + PaginationRequest
```

它们都是纯数据类，只承载参数；SQL 子句由 `TableBaseMixin.get()` 构造。

## `PageWindowRequest`

```python
from sqlmodel_ext.pagination import PageWindowRequest
```

最底层：只有窗口与方向。给"排序列由自己的领域语义固定、只需要窗口 + 方向"的消费方用——它刻意**不带** `order` / `after_id`，避免在公开 schema 中暴露一些无人消费、会被静默丢弃的字段。

| 字段 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| `offset` | `int \| None` | `0` | `ge=0, le=MAX_TABLE_VIEW_OFFSET` |
| `limit` | `int \| None` | `DEFAULT_PAGE_SIZE`（`50`） | `ge=1, le=MAX_PAGE_SIZE`（`100`） |
| `desc` | `bool \| None` | `True` | — |

`get(table_view=PageWindowRequest(...))` 只应用 `offset` / `limit`；排序由你自己的 `order_by` 决定。

## `PaginationRequest`

```python
from sqlmodel_ext import PaginationRequest
```

继承 `PageWindowRequest`，加上排序列与 keyset 游标。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order` | `Literal["created_at", "updated_at", "id"] \| None` | `"created_at"` | 排序列。子类可覆盖 `Literal` 加入领域排序列（`get()` 按名字解析列，因此每个值都必须是真实列） |
| `after_id` | `uuid.UUID \| None` | `None` | keyset 游标：只返回排在该记录**之后**的记录（传上一页最后一条的 id） |

`get()` 总会在 `order` 之后追加同方向的 `id` 作为决胜列（组合 `(排序列, id)`），因此共享同一时间戳的行在 offset 与 keyset 分页的页边界上都不会重复或遗漏。

**构造期校验**（`after_id` 上的字段校验器，违反即 `ValidationError`，位置为 `after_id`）：

| 规则 | 原因 |
|------|------|
| `after_id` 只能配 `order` 为 `created_at` / `id`（或 `None`） | 可变排序列（`updated_at` 或领域列）会让锚点在更新后移动，破坏"不重不漏"保证 |
| `after_id` 与非零 `offset` **互斥** | 两者会叠加而不是二选一：keyset 已表示"锚点之后"，再加 `offset=N` 会额外跳过 N 条，这些行将永远无法到达。`offset` 默认为 0，"忘了重置"是最自然的误用，所以在构造时就让它不可表示 |

运行期规则（`get()` 中）：锚点必须对本查询可见（`condition` + `filter` + STI 过滤），否则 `KeysetCursorInvalidError`；与显式 `order_by` 或 `join` 同用 → `KeysetCursorUnsupportedError`；只支持 UUID 主键表。详见 [Keyset 游标分页](/how-to/keyset-pagination)。

## `TimeFilterRequest`

```python
from sqlmodel_ext import TimeFilterRequest
```

| 字段 | 类型 | 默认值 | 语义 |
|------|------|--------|------|
| `created_after_datetime` | `AwareDatetime \| None` | `None` | `created_at >= 此值` |
| `created_before_datetime` | `AwareDatetime \| None` | `None` | `created_at < 此值` |
| `updated_after_datetime` | `AwareDatetime \| None` | `None` | `updated_at >= 此值` |
| `updated_before_datetime` | `AwareDatetime \| None` | `None` | `updated_at < 此值` |

时间区间为左闭右开 `[after, before)`。所有边界都是 **`AwareDatetime`**：不带时区的 datetime 在校验时就被拒绝（它无法与时区感知的数据库值比较，会被静默按数据库时区解释）。

**构造期校验**（字段校验器，违反即 `ValidationError`，位置为下面标出的 `*_before_datetime` 字段）：

- `created_after_datetime >= created_before_datetime` → 位置 `created_before_datetime`
- `updated_after_datetime >= updated_before_datetime` → 位置 `updated_before_datetime`
- `created_after_datetime >= updated_before_datetime` → 位置 `updated_before_datetime`（记录的更新时间不可能早于创建时间）

## `TableViewRequest`

```python
from sqlmodel_ext import TableViewRequest
```

```python
class TableViewRequest(TimeFilterRequest, PaginationRequest):
    pass
```

同时承载分页 + 排序 + keyset 游标 + 时间过滤参数。`get()` / `get_with_count()` 接受 `table_view` 参数；显式传入的 `offset` / `limit` / `order_by` / 时间参数优先，未提供时回退到 `table_view`。

## `query_dependency()`

```python
from sqlmodel_ext import query_dependency

TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]
```

把查询参数 DTO（`TableViewRequest`、`PaginationRequest`、`PageWindowRequest`、`TimeFilterRequest`、`TrgmSearchRequest` 或你自己的子类）变成 FastAPI 依赖。需要 `fastapi` extra（没有 FastAPI 时调用会抛 `ImportError`；`import sqlmodel_ext` 本身不需要它）。

| 行为 | 说明 |
|------|------|
| 查询参数 | 每个模型字段一个，名字取字段别名（没有别名就是字段名），类型、约束、默认值、docstring 说明都来自字段——OpenAPI 与模型一致 |
| 校验 | 由依赖自己构造模型，所有校验器（包括跨字段的）都会运行 |
| 错误 | `ValidationError` 被转成 `fastapi.exceptions.RequestValidationError`，每个位置前加 `'query'`（`["query", "after_id"]`；没有字段位置的模型级错误变成 `["query"]`），由 FastAPI 默认处理器返回 422 |
| 其它查询参数 | 忽略；端点可以在依赖旁边声明自己的查询参数 |
| 缓存 | 每个模型类只生成一个可调用对象，FastAPI 的请求级依赖缓存把重复使用视为同一个依赖 |
| 拒绝的模型（`TypeError`） | `table=True` 模型（它们跳过校验）、带 `default_factory` 的字段、`validation_alias` 不是单个字符串的字段 |

为什么不在类上直接用 `Depends()`：FastAPI 逐个校验参数后调用这个类，那次调用抛出的 `ValidationError` 不是 `RequestValidationError`，跨字段错误就成了 500。见 [给列表端点加分页](/how-to/paginate-a-list-endpoint#为什么要用-query-dependency)。

## 常量

```python
from sqlmodel_ext.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_SHARED_PAGE_WINDOW,
    MAX_TABLE_VIEW_OFFSET,
)
```

| 常量 | 值 | 说明 |
|------|------|------|
| `DEFAULT_PAGE_SIZE` | `50` | `limit` 默认值 |
| `MAX_PAGE_SIZE` | `100` | `limit` 上限 |
| `MAX_SHARED_PAGE_WINDOW` | `1000` | 任何 `PageWindowRequest` 子类允许的最大窗口；子类若把 `limit` 放宽到此值以上，必须同时收紧自己的 `offset` 上限 |
| `MAX_TABLE_VIEW_OFFSET` | `JS_MAX_SAFE_INTEGER - MAX_SHARED_PAGE_WINDOW` | `offset` 上限 |

为什么 `offset` 需要上限：只有 `ge=0` 时 `offset=10**100` 能通过校验，要到数据库驱动才失败（`bigint out of range`）。为什么是 `2**53 - 1` 而不是 int8 最大值：这个上限会作为 OpenAPI `maximum` 成为 API 契约的一部分，而大多数 JS/TS 客户端把 `integer` 映射为 IEEE-754 double。为什么再减一个窗口：调用方常用 `offset + limit` 计算下一页，预留窗口后这个加法不会越界。

## `ListResponse[T]`

```python
from sqlmodel_ext import ListResponse
```

继承自 `pydantic.BaseModel`（**不是** `SQLModelBase`），泛型类。

::: info 为什么不继承 SQLModelBase
作为 FastAPI **response_model** 的泛型容器，SQLModel + Generic 仍会为参数化字段生成错误的 JSON schema（`{"items": {}}` 而不是 `$ref`），参见 sqlmodel#1002。只作为方法返回值、不进入 OpenAPI 的泛型容器（如 `GroupSumRow`）不受影响，照常继承 `SQLModelBase`。
:::

| 字段 | 类型 | 说明 |
|------|------|------|
| `count` | `int` | 匹配条件的总记录数（不受 `after_id` 影响） |
| `items` | `list[T]` | 当前页数据 |

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
