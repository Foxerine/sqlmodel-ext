# 给列表端点加分页

**目标**：让一个列表端点接受 `?offset=`、`?limit=`、`?desc=`、`?order=`、`?after_id=`、`?created_after_datetime=` 等查询参数，并返回 `{count, items}` 形式的响应。

**前置条件**：

- 你已经有一个 FastAPI 端点
- 你的模型继承了 `TableBaseMixin` 或 `UUIDTableBaseMixin`
- 你有一个 `XxxResponse` DTO

## 1. 把请求参数声明为 FastAPI 依赖

```python
from typing import Annotated
from fastapi import Depends
from sqlmodel_ext import TableViewRequest, query_dependency

TableViewDep = Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]
```

`TableViewRequest` 同时包含分页（`offset` / `limit` / `desc` / `order`）、keyset 游标（`after_id`）和时间过滤（`created_after_datetime` / `created_before_datetime` / `updated_after_datetime` / `updated_before_datetime`）。`query_dependency()` 为模型的每个字段声明一个查询参数（类型、约束、默认值、说明与模型一致，OpenAPI 照常生成），并由它自己构造 `TableViewRequest`。

### 为什么要用 `query_dependency()`

单个字段的错误（`limit=0`、不带时区的时间）由 FastAPI 逐参数校验，直接是 422。但**跨字段**规则——`after_id` 与 `offset` 互斥、`after_id` 不能配 `order=updated_at`、时间区间先后——只在构造 `TableViewRequest` 时触发。写成裸 `Depends()` 时是 FastAPI 调用 `TableViewRequest(...)`，抛出的 Pydantic `ValidationError` 不是 `RequestValidationError`，FastAPI 不接，结果是 **500**。

`query_dependency()` 把这个 `ValidationError` 转成 `RequestValidationError`，错误位置以 `query` 开头，由 FastAPI 默认处理器返回 **422**，形状与其它查询参数错误一致，不需要注册任何异常处理器：

```json
{"detail": [{"type": "value_error", "loc": ["query", "after_id"],
             "msg": "Value error, after_id and offset cannot be combined: ...", "input": "..."}]}
```

跨字段错误落在规则所约束的那个字段上：`after_id` 的两条规则报在 `after_id`，时间区间报在 `*_before_datetime`。同样的写法适用于 `PageWindowRequest`、`PaginationRequest`、`TimeFilterRequest`、`TrgmSearchRequest` 和你自己的子类（`query_dependency(MyTableViewRequest)`）。端点仍然可以在它旁边声明别的查询参数，未声明的查询参数（如缓存破坏用的 `?_=123`）被忽略。

::: warning 不要改用 `Annotated[TableViewRequest, Query()]`
Pydantic 查询模型只在它是端点**唯一**的查询参数时可用：端点再多一个普通查询参数（如 `status: str | None = None`），FastAPI 就不再把整个查询串交给模型，请求直接 422 `Field required`。而且 `SQLModelBase` 的 `extra='forbid'` 会让任何未声明的查询参数（如 `?_=123`）变成 422。
:::

## 2. 在端点中调用 `get_with_count()`

```python
from sqlmodel_ext import ListResponse

@router.get("", response_model=ListResponse[ArticleResponse])
async def list_articles(
    session: SessionDep,
    table_view: TableViewDep,
) -> ListResponse[Article]:
    return await Article.get_with_count(
        session,
        Article.is_published == True,
        table_view=table_view,
    )
```

`get_with_count()` 先执行 `SELECT ... LIMIT N OFFSET M`（或 keyset 条件），再执行 `COUNT(*)`，组装成 `ListResponse[T]`。`count` 始终是整个过滤集的大小。

## 3. 客户端怎么调用

```http
GET /articles?offset=0&limit=20&desc=true&order=created_at&created_after_datetime=2026-01-01T00:00:00Z
```

返回：

```json
{
  "count": 142,
  "items": [
    { "id": "...", "title": "...", "created_at": "...", "..." : "..." }
  ]
}
```

顺序遍历（无限滚动、导出）时改用 keyset 游标：把上一页最后一条的 `id` 作为 `after_id` 传回，**不再传 `offset`**。见 [Keyset 游标分页](./keyset-pagination)。

## 默认值

| 参数 | 默认值 | 上限 / 取值 |
|------|--------|------|
| `offset` | `0` | `MAX_TABLE_VIEW_OFFSET`（`2**53 - 1 - 1000`） |
| `limit` | `50` | `100` |
| `desc` | `True` | — |
| `order` | `"created_at"` | `"created_at"` / `"updated_at"` / `"id"` |
| `after_id` | `None` | UUID |

排序总会追加同方向的 `id` 作为决胜列，所以同一批创建、`created_at` 相同的行在页边界上不会重复或遗漏。需要按其他字段排序时，要么在 `PaginationRequest` 子类里覆盖 `order` 的 `Literal` 加入领域列，要么跳过 `table_view` 的排序、自己传 `order_by=`。

## 常见陷阱

- **`response_model` 必须用 `ListResponse[ArticleResponse]`**，不能写成 `list[ArticleResponse]`。
- **时间参数必须带时区**（`...Z` 或 `+08:00`）。所有时间边界都是 `AwareDatetime`，不带时区的值直接 422——否则它会被静默按数据库时区解释。
- **时间区间是左闭右开** `[after, before)`。`created_after_datetime=2026-01-01T00:00:00Z` + `created_before_datetime=2026-02-01T00:00:00Z` 表示"整个 1 月（UTC）"。
- **`order` 只能是 `Literal` 中的值**，其它字符串 FastAPI 返回 `422`。
- **`after_id` 不能和非零 `offset` 一起用**，也不能配 `order=updated_at`——两者都在构造 `TableViewRequest` 时被拒绝（经 `query_dependency()` 是 422，位置 `["query", "after_id"]`）。

## 相关参考

- [`TableViewRequest` / `ListResponse` 完整字段](/reference/pagination-types)
- [`get_with_count()` 完整签名](/reference/crud-methods#get-with-count)
- [Keyset 游标分页](./keyset-pagination)
