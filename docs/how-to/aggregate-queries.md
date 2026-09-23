# 聚合查询

**目标**：在数据库里完成计数、去重、分组求和，而不是把行加载到 Python 里再算；结果带类型，金额不丢精度。

**前置条件**：模型继承 `TableBaseMixin` / `UUIDTableBaseMixin`。

下面的示例都基于这个模型：

```python
from decimal import Decimal
from uuid import UUID
from sqlmodel import col
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, NonNegativeDecimal38_18

class ArticleBase(SQLModelBase):
    title: Str64
    category: Str64
    author_id: UUID
    price: NonNegativeDecimal38_18 = Decimal('0')

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    pass
```

## 1. 计数：`count()`

```python
total = await Article.count(session)
news = await Article.count(session, Article.category == 'news')

# 去重计数：COUNT(DISTINCT author_id)
n_authors = await Article.count(session, distinct_column=col(Article.author_id))
```

`count()` 也接受时间过滤（`created_after_datetime=` 等，或打包成 `time_filter=TimeFilterRequest(...)`）。

## 2. 某一列的去重值：`distinct_column()`

```python
authors: list[UUID] = await Article.distinct_column(session, col(Article.author_id))

news_authors = await Article.distinct_column(
    session,
    col(Article.author_id),
    Article.category == 'news',
    limit=100,
)
```

返回类型跟随列类型推断（`col(Article.author_id)` → `list[UUID]`），不需要手写 `select(distinct(...))`，也不会把整行加载进内存再去重。

## 3. 求和与分组：`group_sum()`

一次查询同时得到行数和若干列的 `COALESCE(SUM(col), 0)`：

```python
# 全表聚合：恰好返回一个元素，key 为 None
[summary] = await Article.group_sum(session, [col(Article.price)])
summary.count        # 5
summary.totals[0]    # Decimal('12.500000000000000000')

# 按分类分组：每组一行，默认按 group_by 升序
rows = await Article.group_sum(
    session,
    [col(Article.price)],
    group_by=col(Article.category),
    condition=Article.author_id == author_id,
)
for row in rows:
    print(row.key, row.count, row.totals[0])
```

每一行是 `GroupSumRow[GK]`：

| 字段 | 说明 |
|------|------|
| `key` | 分组键（`group_by` 的值；全表聚合时为 `None`） |
| `count` | 组内行数 |
| `totals` | `list[Decimal]`，**按位置**与 `sum_columns` 对齐：`totals[0]` 是第一个求和列 |

`group_by` 可以是任意表达式，例如按天分桶（PostgreSQL）：

```python
from sqlalchemy import func

daily = await Article.group_sum(
    session,
    [col(Article.price)],
    group_by=func.date_trunc('day', col(Article.created_at)),
)
```

多列求和就是多传几列，`totals` 按顺序对应：

```python
rows = await Article.group_sum(session, [col(Article.price), col(Article.price) * 2], group_by=col(Article.category))
```

**条件求和**（`SUM ... FILTER (WHERE ...)`）：每个条件调用一次，再按 `key` 在 Python 中合并。

空的 `sum_columns` 会抛 `ValueError`——纯计数请用 `count()`。

## 4. 把求和结果交给 API：`SignedSumDecimal38_18`

列类型 `*Decimal38_18` 是 `NUMERIC(38, 18)`，但**写入**别名（`*WriteDecimal38_18`）只允许 35 位有效数字，为求和预留了 1000 倍的余量；求和结果要用全宽的 **`SignedSumDecimal38_18`** 承接，否则一次合法的汇总也可能在响应 DTO 上校验失败：

```python
from sqlmodel_ext import SQLModelBase, Str64, SignedSumDecimal38_18

class CategoryTotal(SQLModelBase):
    category: Str64
    count: int
    revenue: SignedSumDecimal38_18

totals = [
    CategoryTotal(category=row.key, count=row.count, revenue=row.totals[0])
    for row in rows
]
totals[0].model_dump_json()   # '{"category":"blog","count":3,"revenue":"7.5"}'
```

Decimal 在 JSON 中序列化为**字符串**，客户端不会因为 IEEE-754 double 丢精度。

## 所有聚合方法共享的行为

- **STI 过滤一致**：在 STI 子类上调用时，`count()` / `distinct_column()` / `group_sum()` 与 `get()` 一样自动加鉴别列过滤，统计范围与列表范围一致。
- **类型化**：返回值都有精确类型（`int` / `list[V]` / `list[GroupSumRow[GK]]`），basedpyright 能检查你对结果的使用。
- **一次往返**：每次调用恰好一条 SQL。

## 相关参考

- [`count()` / `distinct_column()` / `group_sum()` 签名](/reference/crud-methods#count)
- [Decimal 字段类型](/reference/field-types)
