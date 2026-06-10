# 给查询加 Redis 缓存

**目标**：为某个频繁读取的模型加 Redis 缓存层，CRUD 操作时自动失效，无需手动清缓存。

**前置条件**：

- 你已经有一个 Redis 实例（开发环境用 `redis://localhost:6379` 即可）
- 你的模型继承了 `UUIDTableBaseMixin` 或 `TableBaseMixin`
- session 工厂使用增强版 session 类型：`async_sessionmaker(engine, class_=sqlmodel_ext.AsyncSession)`——0.4.0 起缓存失效由它的 `commit()` 统一编排，普通 session 会退化为 fire-and-forget 补偿（短暂 stale 窗口）

## 1. 给模型加 `CachedTableBaseMixin`

```python
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    CachedTableBaseMixin,
    NonEmptyStrippedStr64, Text10K,
)

class CharacterBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    system_prompt: Text10K

class Character(
    CachedTableBaseMixin,                     # ← 必须放第一位 // [!code highlight]
    CharacterBase,
    UUIDTableBaseMixin,
    table=True,
    cache_ttl=1800,                            # 30 分钟 // [!code highlight]
):
    pass
```

::: warning MRO 顺序
`CachedTableBaseMixin` **必须**放在 `UUIDTableBaseMixin` / `TableBaseMixin` 之前。这样它的 `get()` / `save()` / `update()` / `delete()` 重写才会生效。
:::

`cache_ttl` 是类关键字参数，由元类转为 `__cache_ttl__: ClassVar[int]`。默认 3600 秒（1 小时）。

## 2. 启动时配置 Redis 客户端

```python
import redis.asyncio as redis
from sqlmodel_ext import CachedTableBaseMixin

# 在应用 lifespan startup 中：
redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)
CachedTableBaseMixin.configure_redis(redis_client)
CachedTableBaseMixin.check_cache_config()  # 验证所有子类配置正确
```

::: danger decode_responses 必须为 False
缓存值是 bytes（来自 `model_dump_json().encode()`），`decode_responses=True` 会破坏序列化。
:::

`check_cache_config()` 检查所有子类的 `__cache_ttl__` 合法性，并注册 SQLAlchemy session 事件钩子（用于 `commit=False` 场景的失效补偿）。

## 3. 直接用，不用改业务代码

```python
# 第一次：查数据库 + 写缓存
char = await Character.get_one(session, char_id)

# 第二次：直接读缓存，零 SQL
char = await Character.get_one(session, char_id) # [!code highlight]
```

```python
# UPDATE 时自动失效
char.name = "新名字"
char = await char.save(session)
# 自动：DEL id:Character:{id} + INCR ver:Character
```

| 操作 | 失效策略 |
|------|---------|
| `save()` / `update()` | `DEL id:Character:{id}` + 查询缓存版本号 `+1` |
| `delete(instance)` | 同上 |
| `delete(condition=...)` | 全模型 ID 清理 + 版本号 `+1` |
| `add()` | 仅版本号 `+1`（新对象无旧缓存） |

## 4. 手动失效（特殊场景）

如果你用原生 SQL 绕过 ORM 修改了数据，需要手动通知缓存层：

```python
await Character.invalidate_by_id(char_id)         # 失效特定 ID
await Character.invalidate_by_id(id1, id2, id3)   # 失效多个
await Character.invalidate_all()                  # 失效该模型的所有缓存
```

## 5. 跳过缓存

```python
# 显式跳过
char = await Character.get_one(session, char_id, no_cache=True)
```

**自动跳过缓存的场景**：

- `with_for_update=True`（行锁需要最新数据）
- `populate_existing=True`
- `options` / `join` 参数非空（无法稳定哈希）
- 当前事务内有待失效数据

## 6. 接入指标系统（可选）

```python
def on_hit(model_name: str) -> None:
    METRIC_CACHE_HIT.labels(model=model_name).inc()

def on_miss(model_name: str) -> None:
    METRIC_CACHE_MISS.labels(model=model_name).inc()

CachedTableBaseMixin.on_cache_hit = on_hit
CachedTableBaseMixin.on_cache_miss = on_miss
```

## 7. 关于 ID 缓存 vs 查询缓存

sqlmodel-ext 用**双层缓存**：

- **ID 缓存**（`id:Character:{uuid}`）— 用于 `cls.id == value` 的精确单行查询，行级失效 O(1)
- **查询缓存**（`query:Character:v3:abcdef0123456789`）— 用于条件 / 列表查询。模型级失效用版本号自增（`INCR ver:Character`），旧版本 key 通过 TTL 自然过期，避免 SCAN+DEL 的开销

这一切对业务代码透明。你只需要写 `Character.get_one(...)`。

## 优雅降级

Redis 挂了？不会影响业务：

| 失败 | 行为 |
|------|------|
| 读取失败 | 日志 + 回退到数据库查询 |
| 写入失败 | 日志 + 继续 |
| 删除失败 | 日志（TTL 提供最终一致性） |

唯一的硬性要求：`configure_redis()` 必须在第一次 `get()` 之前调用，否则会抛 `RuntimeError`。

## 相关参考

- [`CachedTableBaseMixin` 完整 API](/reference/mixins#cachedtablebasemixin)
- [Redis 缓存机制讲解](/explanation/cached-table)（讲为什么这么设计）
