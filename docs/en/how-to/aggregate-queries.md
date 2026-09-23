# Aggregate queries

**Goal**: do counting, de-duplication and grouped sums in the database instead of loading rows into Python and computing there; results are typed and monetary amounts don't lose precision.

**Prerequisites**: the model inherits `TableBaseMixin` / `UUIDTableBaseMixin`.

All examples below are based on this model:

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

## 1. Counting: `count()`

```python
total = await Article.count(session)
news = await Article.count(session, Article.category == 'news')

# Distinct count: COUNT(DISTINCT author_id)
n_authors = await Article.count(session, distinct_column=col(Article.author_id))
```

`count()` also accepts time filters (`created_after_datetime=` etc., or bundled as `time_filter=TimeFilterRequest(...)`).

## 2. Distinct values of a column: `distinct_column()`

```python
authors: list[UUID] = await Article.distinct_column(session, col(Article.author_id))

news_authors = await Article.distinct_column(
    session,
    col(Article.author_id),
    Article.category == 'news',
    limit=100,
)
```

The return type is inferred from the column type (`col(Article.author_id)` → `list[UUID]`); there is no need to hand-write `select(distinct(...))`, and whole rows are never loaded into memory to be de-duplicated.

## 3. Sums and grouping: `group_sum()`

Get the row count and `COALESCE(SUM(col), 0)` for several columns in a single query:

```python
# Whole-table aggregate: returns exactly one element, with key None
[summary] = await Article.group_sum(session, [col(Article.price)])
summary.count        # 5
summary.totals[0]    # Decimal('12.500000000000000000')

# Grouped by category: one row per group, ascending by group_by by default
rows = await Article.group_sum(
    session,
    [col(Article.price)],
    group_by=col(Article.category),
    condition=Article.author_id == author_id,
)
for row in rows:
    print(row.key, row.count, row.totals[0])
```

Each row is a `GroupSumRow[GK]`:

| Field | Description |
|------|------|
| `key` | Group key (the value of `group_by`; `None` for a whole-table aggregate) |
| `count` | Number of rows in the group |
| `totals` | `list[Decimal]`, aligned **by position** with `sum_columns`: `totals[0]` is the first summed column |

`group_by` can be any expression, e.g. bucketing by day (PostgreSQL):

```python
from sqlalchemy import func

daily = await Article.group_sum(
    session,
    [col(Article.price)],
    group_by=func.date_trunc('day', col(Article.created_at)),
)
```

Summing multiple columns is just passing more columns; `totals` corresponds in order:

```python
rows = await Article.group_sum(session, [col(Article.price), col(Article.price) * 2], group_by=col(Article.category))
```

**Conditional sums** (`SUM ... FILTER (WHERE ...)`): call once per condition, then merge by `key` in Python.

An empty `sum_columns` raises `ValueError` — use `count()` for pure counting.

## 4. Hand sum results to the API: `SignedSumDecimal38_18`

The column types `*Decimal38_18` are `NUMERIC(38, 18)`, but the **write** aliases (`*WriteDecimal38_18`) only allow 35 significant digits, leaving a 1000× headroom for sums; sum results must be received with the full-width **`SignedSumDecimal38_18`**, otherwise even a legitimate aggregate may fail validation on the response DTO:

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

Decimals are serialized as **strings** in JSON, so clients don't lose precision to IEEE-754 doubles.

## Behavior shared by all aggregate methods

- **Consistent STI filtering**: when called on an STI subclass, `count()` / `distinct_column()` / `group_sum()` automatically add the discriminator filter just like `get()`, so the statistics scope matches the listing scope.
- **Typed**: return values all have precise types (`int` / `list[V]` / `list[GroupSumRow[GK]]`), so basedpyright can check how you use the results.
- **One round trip**: each call issues exactly one SQL statement.

## Related reference

- [`count()` / `distinct_column()` / `group_sum()` signatures](/en/reference/crud-methods#count)
- [Decimal field types](/en/reference/field-types)
