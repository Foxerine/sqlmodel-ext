# 给查询加 Redis 缓存

**目标**：为某个频繁读取的模型加 Redis 缓存层，CRUD 操作时自动失效，无需手动清缓存；事务里读到的永远是正确的值。

**前置条件**：

- 你已经有一个 Redis 实例（开发环境用 `redis://localhost:6379` 即可）
- 你的模型继承了 `UUIDTableBaseMixin` 或 `TableBaseMixin`
- session 工厂使用增强版 session 类型：`SessionFactory(engine, class_=sqlmodel_ext.AsyncSession)`（或 `async_sessionmaker(engine, class_=AsyncSession)`）——缓存失效由它的 `commit()` 统一编排

## 1. 给模型加 `CachedTableBaseMixin`

```python
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    CachedTableBaseMixin,
    NonEmptyStrippedStr64, Text10K,
)
from sqlmodel_ext.mixins import CACHE_TTL_WARM

class CharacterBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    system_prompt: Text10K

class Character(
    CachedTableBaseMixin,                     # ← 必须放第一位 // [!code highlight]
    CharacterBase,
    UUIDTableBaseMixin,
    table=True,
    cache_ttl=CACHE_TTL_WARM,                  # 1800 秒 // [!code highlight]
):
    pass
```

::: warning MRO 顺序
`CachedTableBaseMixin` **必须**放在 `UUIDTableBaseMixin` / `TableBaseMixin` 之前，它的 `get()` / `save()` / `update()` / `delete()` / `add()` 重写才会生效。
:::

`cache_ttl` 是类关键字参数，由元类转为 `__cache_ttl__`（非正整数在类创建时就报 `ValueError`）。默认 3600 秒。语义常量：`CACHE_TTL_HOT`（600）、`CACHE_TTL_WARM`（1800）、`CACHE_TTL_COLD`（3600）。

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
缓存值是 bytes（orjson 输出），`decode_responses=True` 会破坏反序列化。
:::

`check_cache_config()` 检查所有子类的 `__cache_ttl__`、禁止子类方法直接调用失效方法，并注册 SQLAlchemy session 事件钩子。

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
# commit 后同步：DEL id:Character:{id} + INCR ver:Character，再回填最新值
```

| 操作 | 失效策略 |
|------|---------|
| `save()` / `update()` | `DEL id:Character:{id}` + 查询缓存版本号 `+1` |
| `delete(instance)` | 同上 |
| `delete(condition=...)` | 全模型 ID 清理 + 版本号 `+1` |
| `add()` | 版本号 `+1`（显式指定 id 的实例额外清 ID 缓存） |
| 裸 `session.add()` / 改属性 / `session.delete()` 后 `session.commit()` | 同样自动失效（由写出它们的那次 flush 登记，包括事务中更早的 savepoint flush、手动 `flush()` 或 autoflush） |

## 4. 事务里也是对的

在一个还没提交的事务里，只要查询依赖的表被本事务写过（包括 flush 过、原生 DML 写过），这次 `get()` **既不读缓存也不写缓存**：你读到自己的未提交修改，别人永远读不到它。其它没被碰过的表照常走缓存。原理见 [事务内的缓存透明性](/explanation/transactional-cache-transparency)。

## 5. 原生 SQL / 触发器写入：`invalidate_on_commit`

绕过 ORM 修改了数据时，在同一事务里**登记**失效，commit 之后自动执行：

```python
from sqlalchemy import update
from sqlmodel import col

Character.invalidate_on_commit(session, char_id)          # 先登记
await session.execute(
    update(Character).where(col(Character.id) == char_id).values(name='Bob')
)
await session.commit()                                      # 提交后同步失效
```

没有登记就对缓存表执行原生 `UPDATE` / `DELETE`，增强 session 会记一条 warning。

在事务之外（管理脚本、测试）可以立即失效：

```python
await Character.invalidate_by_id(char_id)         # 失效特定 ID（Redis 错误会被吞掉）
await Character.invalidate_by_id(id1, id2, id3)
await Character.invalidate_all()                  # 失效该模型的所有缓存
await Character.invalidate_all(strict=True)       # Redis 出错时重新抛出，调用方能知道失败
```

::: warning 模型方法内部只用 `invalidate_on_commit`
`check_cache_config()` 会拒绝在子类方法里直接调用 `invalidate_by_id` / `invalidate_all`（commit 后访问过期属性会 MissingGreenlet）。
:::

## 6. 跳过缓存

```python
# 普通查询显式跳过（no_cache 只存在于缓存模型的 get() 上）
char = await Character.get(session, Character.id == char_id, no_cache=True)

# 授权读取：必须看到最新已提交行——绕过 Redis 和 identity map
char = await Character.get_one(session, char_id, authoritative=True)
```

`get_one()` / `get_exist_one()` 没有 `no_cache` 参数，需要绕过时用 `authoritative=True` 或 `with_for_update=True`。

**自动跳过缓存的场景**：

- `with_for_update=True`（行锁需要最新数据）
- `populate_existing=True`
- `options` / `join` 非空
- 本事务对查询依赖的表有未提交写入

`session.refresh(obj)` 永远直接读数据库。

## 7. 接入指标系统（可选）

```python
def on_hit(model_name: str) -> None:
    METRIC_CACHE_HIT.labels(model=model_name).inc()

def on_miss(model_name: str) -> None:
    METRIC_CACHE_MISS.labels(model=model_name).inc()

CachedTableBaseMixin.on_cache_hit = on_hit
CachedTableBaseMixin.on_cache_miss = on_miss
```

## 8. 迁移改变了数据含义时

迁移重新缩放了某列的值，旧缓存仍能反序列化却是错的——在迁移文件里声明 `CACHE_INVALIDATIONS`，启动时调用 `run_pending_migration_cache_invalidations()`。见 [更多 Mixin：迁移驱动的缓存失效](./extra-mixins#迁移驱动的缓存失效)。

## 关于 ID 缓存 vs 查询缓存

- **ID 缓存**（`id:Character:{uuid}`）— 用于 `cls.id == value` 的精确单行查询，行级失效
- **查询缓存**（`query:Character:v3:abcdef0123456789`）— 用于条件 / 列表查询。模型级失效用版本号自增（`INCR ver:Character`），旧版本 key 通过 TTL 自然过期

这一切对业务代码透明。

## 优雅降级

| 失败 | 行为 |
|------|------|
| 读取失败 | 日志 + 回退到数据库查询 |
| 写入失败 | 日志 + 继续 |
| 失效失败 | 日志（TTL 提供最终一致性） |

唯一的硬性要求：`configure_redis()` 必须在第一次 `get()` 之前调用，否则抛 `RuntimeError`。

## 相关参考

- [`CachedTableBaseMixin` 完整 API](/reference/mixins#cachedtablebasemixin)
- [Redis 缓存机制讲解](/explanation/cached-table)
- [事务内的缓存透明性](/explanation/transactional-cache-transparency)
