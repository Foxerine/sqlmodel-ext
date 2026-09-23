# Keyset 游标分页

**目标**：顺序遍历一个会被并发插入 / 删除的列表（无限滚动、导出、批处理扫描）时，不因为前面的行变动而重复或漏掉记录。

**前置条件**：

- 模型继承 `UUIDTableBaseMixin`（keyset 游标只支持 UUID 主键）
- 你已经会用 `table_view` 做 offset 分页（见 [给列表端点加分页](./paginate-a-list-endpoint)）

## 为什么 offset 不够

offset 表示"跳过前 N 条"。读完第 1 页后，如果有人在前面插入或删除了一行，第 2 页的起点就整体平移了一格——要么一条重复，要么一条永远看不到。keyset 游标改为锚定"**我读到的最后一条**"：下一页 = 排在锚点之后的记录，前面的变动不影响它（锚点本身被删除除外，见下）。

offset 仍然适合随机访问（"跳到第 7 页"）；keyset 适合顺序遍历。

## 1. 第一页：不带 `after_id`

```python
from sqlmodel_ext import TableViewRequest

page = await Article.get(
    session,
    fetch_mode='all',
    table_view=TableViewRequest(limit=20, desc=True),   # 默认 order='created_at'
)
```

## 2. 下一页：把上一页最后一条的 `id` 作为 `after_id`

```python
next_page = await Article.get(
    session,
    fetch_mode='all',
    table_view=TableViewRequest(limit=20, desc=True, after_id=page[-1].id),  # [!code highlight]
)
```

一直重复，直到返回的列表为空。`get_with_count()` 同样接受 `after_id`；它返回的 `count` 是整个过滤集的大小，**不受游标影响**。

在 FastAPI 里端点代码不用改：`TableViewRequest` 作为依赖时，客户端直接传 `?after_id=<上一页最后一条的 id>&limit=20`（记得注册把 `ValidationError` 映射为 422 的处理器，见下文约束表）。

## 它怎么保证不重不漏

- 排序永远是组合键 `(排序列, id)`：`id` 做决胜列，所以同一毫秒创建、`created_at` 相同的行会被确定地分到不同页，没有空隙也没有重复。
- 锚点的排序值在服务端按 `after_id` 查出来，客户端**只传 id**，不回传时间戳（避免数据库微秒与传输层毫秒之间的精度损失）。`order='id'` 时 id 本身就是锚点值，不额外查询。
- 生成的条件是行值比较：`desc=True` 时为 `(col < 锚点值) OR (col = 锚点值 AND id < after_id)`。

## 3. 约束：哪些组合会被拒绝

| 组合 | 何时拒绝 | 错误 |
|------|---------|------|
| `after_id` + 非零 `offset` | 构造 `TableViewRequest` 时 | `ValidationError`（在 FastAPI 依赖里需要映射为 422，见 [给列表端点加分页](./paginate-a-list-endpoint#把跨字段校验错误映射为-422)） |
| `after_id` + `order='updated_at'`（或任何可变的领域排序列） | 构造时 | `ValidationError` |
| `after_id` + 显式 `order_by=` | `get()` 时 | `KeysetCursorUnsupportedError` |
| `after_id` + `join=` | `get()` 时 | `KeysetCursorUnsupportedError` |
| 锚点不存在，或不在本查询可见范围（`condition` + `filter` + STI 过滤） | `get()` 时 | `KeysetCursorInvalidError` |
| 非 UUID 主键表 | `get()` 时 | `ValueError`（编程错误，改用 offset） |

为什么这么严格：

- **`offset` 会叠加而不是二选一**。锚点 B 之后的记录是 C D E F，若还残留 `offset=2`，查询返回 E F——C 和 D 被静默跳过，从此无法到达。`offset` 默认是 0，"忘了重置"是最自然的误用，所以这个组合在构造时就不可表示。
- **只允许不可变排序列**（`created_at` / `id`）。按 `updated_at` 排序时，锚点被更新后会移到别处，"锚点之后"的含义随之改变。
- **锚点必须可见**。否则任何拿到你作用域外某行 UUID 的人都可以通过"返回一页"与"报错"的差别，探测那行是否存在、何时创建（UUID 是标识符，不是凭证）。因此"不存在"与"不可见"刻意给出同一个错误。时间过滤不作用于锚点（时间窗口是页边界，不是可见性边界）。
- **锚点被删除或不再满足过滤条件时**，游标失效，`get()` 抛错而不是返回一个看起来像"到底了"的空页——客户端应从第一页重来。

## 4. 把游标错误映射为 HTTP 响应

三个游标异常有共同父类 `KeysetCursorError`（它继承 `ValueError`，`status_code = 422`），`str(e)` 是可以直接返回给客户端的安全消息：

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlmodel_ext.mixins import KeysetCursorError

@app.exception_handler(KeysetCursorError)
async def keyset_cursor_error_handler(request: Request, exc: KeysetCursorError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
```

`KeysetCursorUnsupportedError` 的消息不会透露拒绝的具体原因（`order_by` 还是 `join`），原因只写在 `INFO` 日志里。

## 5. 只要窗口、不要游标：`PageWindowRequest`

如果你的查询自己固定了排序（例如"按价格降序"），只需要 `offset` / `limit` / `desc`，不要让公开 schema 里出现一个无人消费的 `order` / `after_id`：

```python
from sqlmodel import col
from sqlmodel_ext.pagination import PageWindowRequest

items = await Article.get(
    session,
    fetch_mode='all',
    order_by=[col(Article.price).desc()],
    table_view=PageWindowRequest(offset=0, limit=20),
)
```

## 相关参考

- [`PaginationRequest` / `PageWindowRequest` 字段与校验](/reference/pagination-types)
- [`get()` 的游标相关异常](/reference/crud-methods#get)
- 缓存模型上的 keyset 分页：`after_id` 是查询缓存键的一部分，不同游标不会共享同一条缓存（见 [事务内的缓存透明性](/explanation/transactional-cache-transparency)）
