# Redis 缓存机制

::: tip 源码位置
`src/sqlmodel_ext/mixins/cached_table.py` — `CachedTableBaseMixin`（约 1700 行）
:::

这是 0.2.0 新增的最大模块，为查询结果提供 Redis 缓存层。

## 双层缓存架构

```
1. ID 缓存 (id:{ModelName}:{id_value})
   - 用于 cls.id == value 这种单行精确查询
   - 行级失效 O(1)

2. 查询缓存 (query:{ModelName}:{md5_hash})
   - 用于条件查询、列表查询
   - 模型级失效 SCAN+DEL
```

### 缓存键生成

ID 缓存键直接拼接：`id:Character:550e8400-...`

查询缓存键对所有参数（条件、分页、排序、过滤、时间）进行规范化后计算 MD5 哈希（前 16 字符），确保语义相同的查询产生相同的键。

## 核心类结构

```python
class CachedTableBaseMixin(TableBaseMixin):
    __cache_ttl__: ClassVar[int] = 3600

    # Redis 客户端（类级别共享）
    _redis_client: ClassVar[Any] = None

    @classmethod
    def configure_redis(cls, client: Any) -> None: ...

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
    id_value = cls._extract_id_from_condition(condition)

    # 4. 多 ID 缓存联合查询（load + MANYTOONE 关系）
    if id_value and load contains only cacheable MANYTOONE:
        result = await cls._try_load_from_id_caches(...)
        if result is not _LOAD_CACHE_MISS:
            return result

    # 5. 构建缓存键 + 尝试读取
    cache_key = cls._build_cache_key(condition, fetch_mode, ...)
    cached = await cls._cache_get(cache_key)
    if cached:
        return cls._deserialize_result(cached, fetch_mode)

    # 6. 缓存未命中，查数据库
    result = await super().get(session, condition, ...)

    # 7. 写入缓存
    serialized = cls._serialize_result(result)
    await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)

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

### CRUD 方法中的失效

每个 CRUD 方法重写后在 commit 前后执行失效：

```python
async def save(self, session, ...):
    result = await super().save(session, ...)

    # commit=True 时立即失效
    await self._invalidate_for_model(instance_id)

    # 写穿刷新：将最新数据写入 ID 缓存
    serialized = cls._serialize_result(result)
    await cls._cache_set(id_cache_key, serialized, cls.__cache_ttl__)

    return result
```

### 失效粒度

| 操作 | 策略 |
|------|------|
| `save/update` | `DEL id:{cls}:{id}` + `SCAN+DEL query:{cls}:*` |
| `delete(instances)` | 每个实例 `DEL id:*` + `SCAN+DEL query:*` |
| `delete(condition)` | `SCAN+DEL id:*` + `SCAN+DEL query:*`（全模型） |
| `add()` | 仅 `SCAN+DEL query:*` |

### 多态继承联动

STI 子类变更时，遍历 MRO 失效所有祖先类的缓存：

```python
async def _invalidate_id_cache(cls, instance_id):
    await cls._cache_delete(f"id:{cls.__name__}:{instance_id}")
    # 遍历祖先
    for ancestor in cls.__mro__:
        if issubclass(ancestor, CachedTableBaseMixin):
            await ancestor._cache_delete(f"id:{ancestor.__name__}:{instance_id}")
```

## 失效补偿机制

处理 `commit=False` 场景（延迟提交）：

### session.info 状态追踪

```python
session.info['_cache_pending']  # 待失效：dict[type, set[id]]
session.info['_cache_synced']   # 已同步：dict[type, set[id]]
```

### 两条路径

1. **同步路径**（CRUD 方法中 `commit=True`）：直接 `await` 失效
2. **异步补偿路径**（`commit=False`）：
   - 在 `session.info` 中记录待失效的类型和 ID
   - 注册 SQLAlchemy `after_commit` 事件钩子
   - commit 时触发补偿函数，失效 synced 未覆盖的部分

### 哨兵对象

```python
_QUERY_ONLY_INVALIDATION  # add() 场景：只失效查询缓存
_FULL_MODEL_INVALIDATION  # delete(condition) 场景：全模型失效
_LOAD_CACHE_MISS          # 多 ID 缓存联合查询未命中
```

## MissingGreenlet 规避

commit 后 SQLAlchemy 重置对象关联状态，直接访问属性触发同步查询。解决方案：

- 提交前用 `getattr()` 提取 ID
- 提交后用 `sa_inspect()` 从 identity map 读取（无 DB 查询）
- 外部 SQL 方法使用 `_register_pending_invalidation()` + `_commit_and_invalidate()`

## `check_cache_config()` 静态检查

```python
@classmethod
def check_cache_config(cls) -> None:
```

验证内容：
1. Redis 客户端已通过 `configure_redis()` 设置
2. 所有子类的 `__cache_ttl__` 为正整数
3. AST 检查：禁止在非缓存方法中直接调用 `invalidate_by_id()` 等方法（防止 MissingGreenlet）

## 优雅降级

所有 Redis 操作都包裹在 try/except 中：
- 读取失败 → 返回 None（降级到数据库）
- 写入失败 → 日志记录 + 继续
- 删除失败 → 日志记录（TTL 提供最终一致性）
