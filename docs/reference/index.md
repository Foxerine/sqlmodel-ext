# 参考

> **查阅导向。** 参考是**精确、完整、中立**的 API 文档。它不教学、不解释、不推荐"应该"怎么做——只描述"是什么"和"接受什么参数"。
> 这部分适合**已经在写代码**、需要快速查证某个签名或常量的读者。

## 公共 API

`sqlmodel-ext` 的所有公开符号都从顶层包导出：

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

## 模块索引

| 模块 | 内容 |
|------|------|
| [基础类](./base-classes) | `SQLModelBase`、`ExtraIgnoreModelBase`、`TableBaseMixin`、`UUIDTableBaseMixin` |
| [CRUD 方法](./crud-methods) | `add` / `save` / `update` / `delete` / `get` / `get_one` / `get_exist_one` / `count` / `get_with_count` 完整签名 |
| [字段类型](./field-types) | `Str16`–`Text1M`、`Port`、`Percentage`、`PositiveInt`、`HttpUrl`、`SafeHttpUrl`、`IPAddress`、`Array[T]`、`JSON100K`、`NumpyVector` |
| [Mixin 类](./mixins) | `CachedTableBaseMixin`、`OptimisticLockMixin`、`PolymorphicBaseMixin`、`AutoPolymorphicIdentityMixin`、`RelationPreloadMixin`、信息响应 Mixin |
| [装饰器与辅助函数](./decorators) | `@requires_relations`、`@requires_for_update`、`rel()`、`cond()`、增强 `session.reset()` |
| [分页类型](./pagination-types) | `ListResponse[T]`、`TableViewRequest`、`PaginationRequest`、`TimeFilterRequest` |

## 常量

`sqlmodel-ext` 导出三个常用上界常量：

| 常量 | 值 | 说明 |
|------|-----|------|
| `INT32_MAX` | `2_147_483_647` | PostgreSQL `INTEGER` 的最大值（2³¹−1） |
| `INT64_MAX` | `9_223_372_036_854_775_807` | PostgreSQL `BIGINT` 的最大值（2⁶³−1） |
| `JS_MAX_SAFE_INTEGER` | `9_007_199_254_740_991` | JavaScript `Number.MAX_SAFE_INTEGER`（2⁵³−1）；`PositiveBigInt` / `NonNegativeBigInt` 的默认上界 |

## 异常

| 异常 | 来源模块 | 触发场景 |
|------|---------|---------|
| `RecordNotFoundError` | `sqlmodel_ext._exceptions` | `get_exist_one()` 找不到记录且未安装 FastAPI |
| `OptimisticLockError` | `sqlmodel_ext.mixins.optimistic_lock` | 乐观锁版本号冲突且重试已耗尽 |
| `UnsafeURLError` | `sqlmodel_ext.field_types._ssrf` | `SafeHttpUrl` 拒绝指向私有 / 内网地址的 URL |

## 版本

参见 `sqlmodel_ext.__version__`。本文档面向 `0.4.x` 系列。
