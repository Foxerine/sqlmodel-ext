# Reference

> **Lookup-oriented.** Reference is **precise, complete, and neutral** API documentation. It doesn't teach, doesn't explain, doesn't recommend "what you should do" — it just describes what something *is* and what arguments it takes.
> This section is for readers who are **already writing code** and need to verify a signature or constant fast.

## Public API

All public symbols of `sqlmodel-ext` are re-exported from the top-level package:

```python
from sqlmodel_ext import (
    SQLModelBase, ExtraIgnoreModelBase,
    TableBaseMixin, UUIDTableBaseMixin,
    CachedTableBaseMixin, OptimisticLockMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
    RelationPreloadMixin, requires_relations, requires_for_update,
    ListResponse, TableViewRequest, PaginationRequest, TimeFilterRequest,
    Str64, Port, HttpUrl, SafeHttpUrl, IPAddress, ...
)
```

## Module index

| Module | Contents |
|--------|----------|
| [Base classes](./base-classes) | `SQLModelBase`, `ExtraIgnoreModelBase`, `TableBaseMixin`, `UUIDTableBaseMixin` |
| [CRUD methods](./crud-methods) | Full signatures for `add` / `save` / `update` / `delete` / `get` / `get_one` / `get_exist_one` / `count` / `get_with_count` |
| [Field types](./field-types) | `Str16`–`Text1M`, `Port`, `Percentage`, `PositiveInt`, `HttpUrl`, `SafeHttpUrl`, `IPAddress`, `Array[T]`, `JSON100K`, `NumpyVector` |
| [Mixins](./mixins) | `CachedTableBaseMixin`, `OptimisticLockMixin`, `PolymorphicBaseMixin`, `AutoPolymorphicIdentityMixin`, `RelationPreloadMixin`, info response mixins |
| [Decorators & helpers](./decorators) | `@requires_relations`, `@requires_for_update`, `rel()`, `cond()`, enhanced `session.reset()` |
| [Pagination types](./pagination-types) | `ListResponse[T]`, `TableViewRequest`, `PaginationRequest`, `TimeFilterRequest` |

## Constants

`sqlmodel-ext` exports three commonly-used integer upper bounds:

| Constant | Value | Description |
|----------|-------|-------------|
| `INT32_MAX` | `2_147_483_647` | Max value of PostgreSQL `INTEGER` (2³¹−1) |
| `INT64_MAX` | `9_223_372_036_854_775_807` | Max value of PostgreSQL `BIGINT` (2⁶³−1) |
| `JS_MAX_SAFE_INTEGER` | `9_007_199_254_740_991` | JavaScript `Number.MAX_SAFE_INTEGER` (2⁵³−1); default upper bound for `PositiveBigInt` / `NonNegativeBigInt` |

## Exceptions

| Exception | Source module | When it's raised |
|-----------|---------------|------------------|
| `RecordNotFoundError` | `sqlmodel_ext._exceptions` | `get_exist_one()` finds no record and FastAPI is not installed |
| `OptimisticLockError` | `sqlmodel_ext.mixins.optimistic_lock` | Optimistic lock version mismatch after retries are exhausted |
| `UnsafeURLError` | `sqlmodel_ext.field_types._ssrf` | `SafeHttpUrl` rejects a URL pointing at a private / loopback address |

## Version

See `sqlmodel_ext.__version__`. This documentation tracks the `0.3.x` series.
