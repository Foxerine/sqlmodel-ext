# 分页与列表

sqlmodel-ext 提供了四个数据模型来处理分页请求和响应，以及一组 DTO Mixin 用于 API 响应。

## 请求模型

### `PaginationRequest` — 分页排序

```python
from sqlmodel_ext import PaginationRequest

class PaginationRequest(SQLModelBase):
    offset: int | None = Field(default=0, ge=0)            # 跳过前 N 条
    limit:  int | None = Field(default=50, le=100)          # 每页最多 N 条
    desc:   bool | None = True                              # 降序排列
    order:  Literal["created_at", "updated_at"] | None = "created_at"
```

默认行为：按 `created_at` 降序，每页 50 条，上限 100 条。

### `TimeFilterRequest` — 时间过滤

```python
from sqlmodel_ext import TimeFilterRequest

class TimeFilterRequest(SQLModelBase):
    created_after_datetime:  datetime | None = None   # created_at >= 此值
    created_before_datetime: datetime | None = None   # created_at < 此值
    updated_after_datetime:  datetime | None = None   # updated_at >= 此值
    updated_before_datetime: datetime | None = None   # updated_at < 此值
```

使用左闭右开区间 `[after, before)`。内置验证：`after` 必须小于 `before`。

### `TableViewRequest` — 组合

```python
from sqlmodel_ext import TableViewRequest

class TableViewRequest(TimeFilterRequest, PaginationRequest):
    pass  # 同时携带分页参数和时间过滤参数
```

## 响应模型

### `ListResponse[T]` — 分页响应

```python
from sqlmodel_ext import ListResponse

class ListResponse(BaseModel, Generic[ItemT]):
    count: int           # 匹配条件的总记录数
    items: list[ItemT]   # 当前页的数据列表
```

## 在 FastAPI 中使用

```python
from typing import Annotated
from fastapi import Depends
from sqlmodel_ext import ListResponse, TableViewRequest

TableViewDep = Annotated[TableViewRequest, Depends()] # [!code highlight]

@router.get("", response_model=ListResponse[ArticleResponse])
async def list_articles(
    session: SessionDep, table_view: TableViewDep,
) -> ListResponse[Article]:
    return await Article.get_with_count( # [!code focus]
        session, # [!code focus]
        Article.is_published == True, # [!code focus]
        table_view=table_view, # [!code focus]
    ) # [!code focus]
```

客户端发送：

```
GET /articles?offset=0&limit=10&desc=true&created_after_datetime=2024-01-01T00:00:00
```

返回的 JSON：

```json
{
  "count": 42,
  "items": [
    { "id": "a1b2c3d4-...", "title": "Hello World", "..." : "..." }
  ]
}
```

## 响应 DTO Mixin

用于 API 响应模型的 Mixin，字段为必填（数据已入库，一定有值）：

```python
from sqlmodel_ext import UUIDIdDatetimeInfoMixin

class ArticleBase(SQLModelBase):
    title: Str64
    body: Text10K

# 表模型
class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id')

# 响应 DTO
class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID
```

可用的 DTO Mixin：

| Mixin | 字段 |
|-------|------|
| `IntIdInfoMixin` | `id: int` |
| `UUIDIdInfoMixin` | `id: UUID` |
| `DatetimeInfoMixin` | `created_at: datetime`, `updated_at: datetime` |
| `IntIdDatetimeInfoMixin` | `id: int` + 时间戳 |
| `UUIDIdDatetimeInfoMixin` | `id: UUID` + 时间戳 |

## 数据流全景

```mermaid
flowchart LR
    A["客户端请求<br/><code>GET /articles?<br/>offset=0&limit=10&desc=true</code>"] --> B["TableViewRequest"]
    B --> C["Article.get_with_count()"]
    C --> D["count()"]
    C --> E["get(fetch_mode=&quot;all&quot;)"]
    D --> F[("SELECT COUNT(*)")]
    E --> G[("SELECT ... LIMIT 10")]
    D --> H["ListResponse"]
    E --> H
    H --> I["客户端响应<br/><code>{ count: 42, items: [...] }</code>"]
```
