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
char = await Character.get(session, Character.id == char_id)

# 第二次：直接读缓存，零 SQL
char = await Character.get(session, Character.id == char_id)
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
char = await Character.get(session, Character.id == char_id, no_cache=True)
```

以下场景自动跳过缓存：
- `with_for_update=True`（需要最新数据）
- `populate_existing=True`
- `options` 参数非空
- `join` 参数非空
- 事务内有未提交的待失效数据

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

## 多态继承支持

STI 子类的缓存会自动联动祖先类：子类数据变更时，祖先类的查询缓存也会被清理。

## TTL 配置

```python
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin,
                table=True, cache_ttl=1800):
    pass  # 30 分钟

class Config(CachedTableBaseMixin, ConfigBase, UUIDTableBaseMixin,
             table=True, cache_ttl=86400):
    pass  # 24 小时
```

也可以通过类变量设置：

```python
class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True):
    __cache_ttl__: ClassVar[int] = 1800
```

默认 TTL 为 3600 秒（1 小时）。

## 优雅降级

Redis 不可用时自动降级到数据库查询：
- 读取失败 → 日志 + 查数据库
- 写入失败 → 日志 + 继续（非关键路径）
- 删除失败 → 日志 + TTL 提供最终一致性

## 原生 SQL 场景

如果使用原生 SQL（绕过 ORM 方法），需要手动注册失效：

```python
from sqlmodel_ext.mixins.cached_table import CachedTableBaseMixin

# 在原生 SQL 操作前注册
CachedTableBaseMixin._register_pending_invalidation(session, Character, char_id)

# 提交并失效
await instance._commit_and_invalidate(session)
```
