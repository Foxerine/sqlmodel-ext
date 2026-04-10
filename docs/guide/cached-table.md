# Redis 缓存

`CachedTableBaseMixin` 为模型的 `get()` 查询添加 Redis 缓存层，CRUD 操作时自动失效。

## 基本用法

```python
from sqlmodel_ext import CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin

class CharacterBase(SQLModelBase):
    name: str

class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=1800):
    pass  # 缓存 30 分钟
```

::: warning MRO 顺序
`CachedTableBaseMixin` 必须放在 `UUIDTableBaseMixin` **之前**。
:::

### 配置 Redis

启动时配置一次：

```python
import redis.asyncio as redis

redis_client = redis.from_url("redis://localhost:6379")
CachedTableBaseMixin.configure_redis(redis_client)
```

### 启动检查（推荐）

```python
CachedTableBaseMixin.check_cache_config()
```

验证所有缓存模型的配置正确性：Redis 客户端已设置、`__cache_ttl__` 为正整数、没有禁止的直接调用。

## 缓存行为

### 自动缓存

`get()` 查询结果自动缓存到 Redis，后续相同查询直接读缓存：

```python
# 第一次：查数据库 + 写缓存
char = await Character.get(session, Character.id == char_id) # [!code highlight]

# 第二次：直接读缓存，零 SQL
char = await Character.get(session, Character.id == char_id) # [!code highlight]
```

### 自动失效

CRUD 操作自动清理相关缓存：

| 操作 | 失效策略 |
|------|---------|
| `save()` / `update()` | 删除该记录的 ID 缓存 + 该模型所有查询缓存 |
| `delete(instances)` | 删除每个实例的 ID 缓存 + 所有查询缓存 |
| `delete(condition)` | 删除该模型的所有缓存（ID + 查询） |
| `add()` | 删除所有查询缓存（新对象无旧缓存） |

### 跳过缓存

```python
# 显式跳过缓存
char = await Character.get(session, Character.id == char_id, no_cache=True) # [!code highlight]
```

::: details 自动跳过缓存的场景
- `with_for_update=True`（需要最新数据）
- `populate_existing=True`
- `options` 参数非空
- `join` 参数非空
- 事务内有未提交的待失效数据
:::

## 双层缓存架构

```
1. ID 缓存 (id:{ModelName}:{id_value})
   → 单行 ID 相等查询
   → 行级失效 O(1)

2. 查询缓存 (query:{ModelName}:{hash})
   → 条件/列表查询
   → 模型级失效 SCAN+DEL
```

ID 查询（`Character.id == some_id`）使用精确的 ID 缓存键，失效时只需删除一个键。其他查询使用参数哈希作为缓存键。

## 手动失效

```python
# 失效特定 ID
await Character.invalidate_by_id(char_id)
await Character.invalidate_by_id(id1, id2, id3)  # 多个

# 失效该模型的所有缓存
await Character.invalidate_all()
```

## 级联删除的缓存失效

当父模型被删除时，``CachedTableBaseMixin`` 会按 SQLAlchemy 关系的 ``passive_deletes`` 设置走两条不同的路径，两种都能正确清理子模型的缓存。

| `passive_deletes` | 子行谁删除 | 缓存怎么失效 |
|-------------------|-----------|-------------|
| `False`（默认） | SA 在 flush 时逐行 `DELETE` | `persistent_to_deleted` 事件自动触发 |
| `True` | 数据库 `ON DELETE CASCADE` 静默删除 | `delete()` 在 DELETE 前预查 child ID，DELETE 后显式失效 |

```python
class User(CachedTableBaseMixin, UserBase, UUIDTableBaseMixin, table=True):
    # passive_deletes=True：依赖 DB CASCADE，SA 不会加载子行
    files: list['UserFile'] = Relationship(
        back_populates='user',
        cascade_delete=True,
        passive_deletes=True,
    )
```

``passive_deletes=True`` 在高并发环境下尤其有用，可以避免 SA 把几千行子记录拉进内存。旧版本会拒绝这个组合（要求 ``passive_deletes=False``）；新版本在 ``delete()`` 中预查 child ID 后再交给 DB CASCADE，之后显式 ``_invalidate_id_cache`` + ``_invalidate_query_caches``，保证子模型的 Redis 缓存不会出现陈旧数据。条件删除（``delete(condition=...)``）下无法枚举父 ID，会对子模型执行模型级全量失效。

## 多态继承支持

STI 子类的缓存会自动联动祖先类：子类数据变更时，祖先类的查询缓存也会被清理。

## TTL 配置

::: code-group

```python [关键字参数（推荐）]
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin,
                table=True, cache_ttl=1800):  # [!code highlight]
    pass  # 30 分钟

class Config(CachedTableBaseMixin, ConfigBase, UUIDTableBaseMixin,
             table=True, cache_ttl=86400):  # [!code highlight]
    pass  # 24 小时
```

```python [类变量]
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True):
    __cache_ttl__: ClassVar[int] = 1800  # [!code highlight]
```

:::

默认 TTL 为 3600 秒（1 小时）。

## 优雅降级

::: tip 自动降级
Redis 不可用时自动降级到数据库查询，不会影响业务逻辑：
- 读取失败 → 日志 + 查数据库
- 写入失败 → 日志 + 继续（非关键路径）
- 删除失败 → 日志 + TTL 提供最终一致性
:::

## 原生 SQL 场景

::: warning 绕过 ORM 时需手动管理缓存
如果使用原生 SQL（绕过 ORM 方法），需要手动注册失效：
:::

```python
from sqlmodel_ext.mixins.cached_table import CachedTableBaseMixin

# 在原生 SQL 操作前注册
CachedTableBaseMixin._register_pending_invalidation(session, Character, char_id) # [!code warning]

# 提交并失效
await instance._commit_and_invalidate(session)
```
