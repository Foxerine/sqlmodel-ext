# Redis 缓存机制

::: tip 源码位置
`src/sqlmodel_ext/mixins/cached_table.py` — `CachedTableBaseMixin`（约 1900 行）
:::

`CachedTableBaseMixin` 为 `get()` 查询提供透明的 Redis 缓存层，CRUD 操作时自动失效。本章解释**它的内部如何工作**；要在自己的项目中接入它，去 [给查询加 Redis 缓存](/how-to/cache-queries)。

## 双层缓存架构

```
1. ID 缓存 (id:{ModelName}:{id_value})
   - 用于 cls.id == value 这种单行精确查询
   - 行级失效 O(1)

2. 查询缓存 (query:{ModelName}:v{version}:{md5_hash})
   - 用于条件查询、列表查询
   - 模型级失效：版本号自增 O(1)（旧 key 靠 TTL 自然过期）
```

::: info 0.3 版本号失效
0.3.0 起，查询缓存的"模型级失效"从 `SCAN+DEL` 改为**版本号自增**：每个模型有一个 `ver:{ModelName}` 计数器，缓存键里嵌入版本号；失效时只需 `INCR` 一次，旧版本的 key 不再被读到，自然过期消失。这把模型级失效的成本从 O(N keys) 降到 O(1)。
:::

### 缓存键生成

ID 缓存键直接拼接：`id:Character:550e8400-...`

查询缓存键对所有参数（条件、分页、排序、过滤、时间）进行规范化后计算 MD5 哈希（前 16 字符），确保语义相同的查询产生相同的键。完整格式：`query:Character:v3:abcdef0123456789`。

## 核心类结构

```python
class CachedTableBaseMixin(TableBaseMixin):
    __cache_ttl__: ClassVar[int] = 3600

    # Redis 客户端（类级别共享）
    _redis_client: ClassVar[Any] = None

    # 监控钩子（可选）
    on_cache_hit: ClassVar[Callable[[str], None] | None] = None
    on_cache_miss: ClassVar[Callable[[str], None] | None] = None

    @classmethod
    def configure_redis(cls, client: Any) -> None: ...

    @classmethod
    def check_cache_config(cls) -> None: ...

    # 缓存原语
    @classmethod
    async def _cache_get(cls, key: str) -> bytes | None: ...
    @classmethod
    async def _cache_set(cls, key: str, value: bytes, ttl: int) -> None: ...
    @classmethod
    async def _cache_delete(cls, key: str) -> None: ...
    @classmethod
    async def _cache_delete_pattern(cls, pattern: str) -> None: ...
```

`on_cache_hit` / `on_cache_miss` 是可选的指标钩子——在启动时设置回调，可以把命中率喂给 Prometheus / Grafana。

## `get()` 重写

重写 `TableBaseMixin.get()`，在数据库查询前后加入缓存逻辑：

```python
@classmethod
async def get(cls, session, condition, *, no_cache=False, ...):
    # 1. 判断是否可以使用缓存
    if no_cache or with_for_update or populate_existing or ...:
        return await super().get(session, condition, ...)

    # 2. 检查事务内是否有待失效数据
    if session.info has pending invalidation for this model:
        return await super().get(session, condition, ...)

    # 3. 检测是否为 ID 查询
    id_value = cls._extract_id_from_condition(condition) # [!code focus]

    # 4. 多 ID 缓存联合查询（load + MANYTOONE 关系）
    if id_value and load contains only cacheable MANYTOONE:
        result = await cls._try_load_from_id_caches(...) # [!code focus]
        if result is not _LOAD_CACHE_MISS:
            return result

    # 5. 构建缓存键 + 尝试读取
    cache_key = cls._build_cache_key(condition, fetch_mode, ...)
    cached = await cls._cache_get(cache_key) # [!code focus]
    if cached:
        return cls._deserialize_result(cached, fetch_mode) # [!code highlight]

    # 6. 缓存未命中，查数据库
    result = await super().get(session, condition, ...) # [!code warning]

    # 7. 写入缓存
    serialized = cls._serialize_result(result)
    await cls._cache_set(cache_key, serialized, cls.__cache_ttl__) # [!code focus]

    return result
```

### ID 查询检测

```python
@classmethod
def _extract_id_from_condition(cls, condition):
    """检测纯 ID 相等查询，返回 ID 值或 None"""
```

检测 `cls.id == value` 形式的条件，使用精确的 ID 缓存键而非查询哈希。

### 多 ID 缓存联合查询

当 `load` 参数指定的关系全部是可缓存的 MANYTOONE 时，尝试从各模型的 ID 缓存中分别读取主对象和关系对象，全部命中则零 SQL 返回。

```python
@classmethod
async def _try_load_from_id_caches(cls, session, id_value, rel_info):
    # 1. 读主模型 ID 缓存
    # 2. 读每个关系目标的 ID 缓存
    # 3. 全部命中 → 组装返回
    # 4. 任何缺失 → 返回 _LOAD_CACHE_MISS
```

## 序列化方案

```python
# 包装格式
{
    "_t": "none|single|list",   # 结果类型
    "_data": {...},             # 单项数据（model_dump_json 的结果）
    "_items": [{...}, ...],     # 列表数据
    "_c": "ClassName"           # 多态安全：记录实际类名
}
```

序列化使用 `model_dump_json()` → JSON → `json.loads()`。反序列化使用 `model_validate()`（不用 `model_validate_json` 以避免 table=True 模型的 UUID 字符串化问题）。

支持 orjson（可选）加速序列化。

## 缓存失效

### CRUD 登记 + commit 单点失效（0.4.0 起）

CRUD 方法重写**不再自行失效**——它们只负责把待失效项登记到 `session.info`，实际失效由增强版 `sqlmodel_ext.AsyncSession.commit()` 在 commit 后统一同步执行：

```python
async def save(self, session, ...):
    # 1. 登记待失效（pending）
    self._register_pending_invalidation(session, model_type, instance_id) # [!code focus]

    # 2. super().save() 内部 await session.commit() ——
    #    增强 AsyncSession 在 commit 后同步失效所有已登记 pending # [!code focus]
    result = await super().save(session, ...)

    # 3. 写穿刷新：将最新数据回填 self + 所有缓存祖先的 ID 缓存
    serialized = cls._serialize_result(result)
    for refill_cls in [model_type, *model_type._cached_ancestors()]: # [!code focus]
        await refill_cls._cache_set(refill_cls._build_id_cache_key(instance_id), serialized, ...)

    return result
```

`commit=False` 时不失效（数据未提交），pending 留到调用方 `await session.commit()` 时统一处理——这正是"多个 `commit=False` 操作 + 一次 commit"场景天然正确的原因。

### 失效粒度

| 操作 | 策略 |
|------|------|
| `save/update` | `DEL id:{cls}:{id}` + `INCR ver:{cls}` |
| `delete(instances)` | 每个实例 `DEL id:*` + `INCR ver:{cls}` |
| `delete(condition)` | 模型级 ID 缓存清理 + `INCR ver:{cls}` |
| `add()` | 仅 `INCR ver:{cls}`（新对象无旧缓存） |

### 多态继承联动

STI 子类变更时，遍历 MRO 失效所有祖先类的缓存：

```python
async def _invalidate_id_cache(cls, instance_id):
    await cls._cache_delete(f"id:{cls.__name__}:{instance_id}")
    # 遍历祖先
    for ancestor in cls._cached_ancestors():
        await ancestor._cache_delete(f"id:{ancestor.__name__}:{instance_id}")
```

`_cached_ancestors()` 缓存了 MRO 中所有同样继承 `CachedTableBaseMixin` 的祖先类。**回填与失效对称**：失效清几个 key，写穿刷新就回填几个 key——否则通过祖先类查询的 caller 会在 race 窗口内拿到 stale 实例。

## 失效补偿机制

### session.info 状态追踪

```python
session.info['_cache_pending']  # 待失效：dict[type, set[id]]
session.info['_cache_synced']   # 已同步：dict[type, set[id]]
```

### 两条路径

1. **同步路径**（增强 `AsyncSession.commit()`）：commit 前自动登记 session 中所有缓存模型变更（含裸 `session.add()` / 改属性 / `session.delete()`）+ 快照 pending，commit 后同步 `await` 失效快照项与级联子项
2. **异步补偿路径**（兜底）：`after_commit` 事件钩子触发补偿函数，失效 synced 未覆盖的部分——覆盖未使用增强 session 的 commit（如 `begin()` / savepoint 路径、普通 `sqlmodel` session），fire-and-forget 语义，存在短暂 stale 窗口

### 哨兵对象

```python
_QUERY_ONLY_INVALIDATION  # add() 场景：只失效查询缓存
_FULL_MODEL_INVALIDATION  # delete(condition) 场景：全模型失效
_LOAD_CACHE_MISS          # 多 ID 缓存联合查询未命中
```

## MissingGreenlet 规避

::: danger 风险点
commit 后 SQLAlchemy 重置对象关联状态，直接访问属性触发同步查询。
:::

解决方案：

- 提交前用 `getattr()` 提取 ID
- 提交后用 `sa_inspect()` 从 identity map 读取（无 DB 查询）
- 外部 SQL 方法使用 `_register_pending_invalidation()` + `await session.commit()`（增强 `AsyncSession` 在 commit 后自动同步失效已登记的 pending）

## `check_cache_config()` 静态检查

```python
@classmethod
def check_cache_config(cls) -> None:
```

验证内容：
1. Redis 客户端已通过 `configure_redis()` 设置
2. 没有子类重写 `_get_client`（会破坏共享客户端）
3. 所有子类的 `__cache_ttl__` 为正整数
4. AST 检查：禁止在非缓存方法中直接调用 `invalidate_by_id()` 等方法（防止 MissingGreenlet）

副作用：注册 SQLAlchemy `after_commit` / `after_rollback` / `persistent_to_deleted` 事件钩子。

## 优雅降级

所有 Redis 操作都包裹在 try/except 中：
- 读取失败 → 返回 None（降级到数据库）
- 写入失败 → 日志记录 + 继续
- 删除失败 → 日志记录（TTL 提供最终一致性）
