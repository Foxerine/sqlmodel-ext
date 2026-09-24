# Redis 缓存机制

::: tip 源码位置
`src/sqlmodel_ext/mixins/cached_table.py` — `CachedTableBaseMixin`；编排在 `src/sqlmodel_ext/session.py` 的增强 `AsyncSession`
:::

`CachedTableBaseMixin` 为 `get()` 查询提供透明的 Redis 缓存层，写路径自动失效。本章解释**它的内部结构**；事务内的正确性（为什么不会发布未提交数据）单独在 [事务内的缓存透明性](./transactional-cache-transparency) 中讲；要在自己的项目中接入，去 [给查询加 Redis 缓存](/how-to/cache-queries)。

## 双层缓存架构

```
1. ID 缓存 (id:{ModelName}:{id_value})
   - 用于 cls.id == value 这种单行精确查询
   - 行级失效：一次多键 DEL

2. 查询缓存 (query:{ModelName}:v{version}:{md5_hash})
   - 用于条件查询、列表查询
   - 模型级失效：版本号自增 O(1)（旧 key 靠 TTL 自然过期）

3. 版本键 (ver:{ModelName})
   - 查询缓存的命名空间版本；INCR 让旧版本的所有查询键不可达
```

失效查询缓存只需 `INCR ver:{ModelName}` 一次，而不是 `SCAN+DEL` 所有键——成本从 O(N keys) 降到 O(1)。

### 缓存键生成

ID 缓存键直接拼接：`id:Character:0190...`。

查询缓存键对所有会改变结果的参数（条件、分页、排序、`load`、`filter`、时间过滤、keyset 游标 `after_id`）规范化后计算 MD5（取前 16 字符）。`table_view` 先按与 `get()` 相同的规则合并进显式参数，保证语义相同的查询得到同一个键。完整格式：`query:Character:v3:abcdef0123456789`。

## 核心类结构

```python
class CachedTableBaseMixin(TableBaseMixin):
    __cache_ttl__: ClassVar[int] = 3600          # 用 cache_ttl= 类关键字覆盖

    _redis_client: ClassVar[Any] = None          # configure_redis() 设置
    on_cache_hit: ClassVar[Callable[[str], None] | None] = None
    on_cache_miss: ClassVar[Callable[[str], None] | None] = None

    @classmethod
    def configure_redis(cls, client: Any) -> None: ...
    @classmethod
    def check_cache_config(cls) -> None: ...
```

`on_cache_hit` / `on_cache_miss` 是可选的指标钩子，可以把命中率喂给 Prometheus / Grafana。

## `get()` 重写

```python
@classmethod
async def get(cls, session, condition=None, *, no_cache=False, authoritative=False, ...):
    # 1. 显式或结构性跳过
    skip_cache = (no_cache or authoritative or options is not None
                  or with_for_update or populate_existing or join is not None)

    # 2. 事务透明：查询依赖的表在本事务有未提交写入 → 读写都跳过
    if not skip_cache and cls._has_uncommitted_writes(
            session, cls._query_dependency_tables(condition, filter, order_by, load)):
        skip_cache = True

    # 3. load + 全部是可缓存的 MANYTOONE → 多 ID 缓存联合查询（全部命中则零 SQL）
    # 4. 纯 ID 相等查询 → ID 缓存键；否则 → 带版本号的查询缓存键
    # 5. 命中 → 反序列化 → _adopt_cached_instance 并入 identity map（不能安全并入则回源）
    # 6. 未命中 → super().get() 查数据库 → 写入缓存
```

### ID 查询检测

`_extract_id_from_condition()` 识别 `cls.id == value` 形式的条件，并且要求没有分页 / 排序 / `filter` / `table_view` / 时间过滤——这时用精确的 ID 缓存键而不是查询哈希。

### 多 ID 缓存联合查询

`load` 中的关系全部是**多对一**、目标模型也是缓存模型时，从主模型和关系目标各自的 ID 缓存读取并组装，全部命中则零 SQL。链式 `load`（`A.b → B.c`）不走这条路。`fetch_mode='one'` 且缓存里记录的是"不存在"时，照样抛 `NoResultFound`，与 SQL 路径一致。

## 序列化方案

```python
{
    "_t": "none|single|list",   # 结果类型
    "_data": {...},             # 单项数据
    "_items": [{...}, ...],     # 列表数据
    "_c": "ClassName"           # 多态安全：记录实际类名
}
```

序列化：`model_dump()`（只含列）→ orjson（未安装时回退标准库 json）。反序列化：`json.loads` → `model_validate()`（不用 `model_validate_json`，它会把 `table=True` 模型的 UUID 字段留成 `str`），加载出的实例带有合法的 `_sa_instance_state`。反序列化失败（例如 schema 变了）时删掉坏键并回源数据库。

## 缓存失效

### CRUD 登记 + commit 单点失效

CRUD 方法重写**不自行失效**——只把待失效项登记到 `session.info`，实际失效由增强 `AsyncSession.commit()` 在 commit 后统一同步执行：

```python
async def save(self, session, ...):
    # 1. 登记待失效（新建对象登记"只失效查询缓存"哨兵）
    self._register_pending_invalidation(session, model_type, instance_id)

    # 2. super().save(refresh=False)：内部 await session.commit()，
    #    增强 commit 在提交后同步失效所有登记项
    result = await super().save(session, refresh=False, commit=commit, ...)

    # 3. 绕过缓存读数据库（no_cache=True），commit=True 且无 load 时
    #    把最新数据回填到自身 + 所有缓存祖先的 ID 缓存
    ...
```

`commit=False` 时不失效（数据未提交），登记项留到你最终的 `session.commit()`；也不回填（这些行还可能被回滚）。

### 失效粒度

| 操作 | 策略 |
|------|------|
| `save` / `update` | `DEL id:{cls}:{id}`（含缓存祖先）+ `INCR ver:{cls}` |
| `delete(instances)` | 每个实例 `DEL id:...` + `INCR ver:{cls}` |
| `delete(condition)` | 模型级 `SCAN+DEL id:{cls}:*` + `INCR ver:{cls}` |
| `add()` | `INCR ver:{cls}`；显式指定了 id 的实例额外失效其 ID 缓存（防 id 复用） |
| 裸 `session.add()` / 改属性 / `session.delete()` + `commit()` | 由把它们发出去的那次 flush 登记（savepoint 内的 flush、手动 `flush()`、autoflush 或 commit 自身的 flush）：行级（flush 后的 `id`）+ 查询缓存 |

### 级联删除

- `passive_deletes=False`：SQLAlchemy 在 flush 中逐个删除子对象，`persistent_to_deleted` 事件把每个缓存子模型登记进来，commit 后同步失效。
- `passive_deletes=True`：数据库负责级联，ORM 看不到子行。`delete()` 在发出 DELETE **之前**沿整条 `passive_deletes` 链做 BFS，预查出所有缓存子模型的 id 并登记（一层不够：A→B→C 都用数据库 CASCADE 时，B 和 C 都不会被 ORM 加载）。

### 多态继承联动

STI 子类变更时，同时失效所有缓存祖先类的键（`_cached_ancestors()` 沿 MRO 收集）：

```python
async def _invalidate_id_cache(cls, instance_id):
    keys = [f"id:{cls.__name__}:{instance_id}"]
    keys += [f"id:{a.__name__}:{instance_id}" for a in cls._cached_ancestors()]
    await client.delete(*keys)          # 一条命令、一次往返
```

查询缓存用 pipeline 一次往返 `INCR` 自身与所有祖先的版本键。**回填与失效对称**：失效清几个 key，写穿回填就写几个 key——否则通过祖先类查询的调用方会在窗口期内拿到旧实例。

## 失效的两条路径

1. **同步路径**（增强 `AsyncSession.commit()`）：真正 commit 前给 session 打上标记；`after_commit` 事件看到标记，把完整的登记项（包括 commit 自身 flush 期间登记的，如级联子项）移交给 `commit()`，由它在返回前 `await` 失效。每项只失效一次，不记告警。
2. **补偿路径**（没有标记的 `after_commit` 事件）：事件处理器是同步的、无法 `await`，于是调度一个 fire-and-forget 任务失效这些登记项，并记录 `WARNING` "fallback compensation triggered: ..."。它兜住没有经过增强 `commit()` 的提交（普通 SQLAlchemy / SQLModel session、`run_sync(lambda s: s.commit())`），以及在数据库已提交之后失败或被取消的增强 `commit()`；存在极短的 stale 窗口，TTL 提供最终一致性。

哨兵对象：

```python
_QUERY_ONLY_INVALIDATION  # add() / 新建：只失效查询缓存
_FULL_MODEL_INVALIDATION  # delete(condition)：全模型失效
_LOAD_CACHE_MISS          # 多 ID 缓存联合查询未命中
```

## 手动失效的三个入口

| 方法 | 时机 | 用途 |
|------|------|------|
| `invalidate_on_commit(session, *ids)` | 下一次 commit **之后** | 触发器 / 原生 SQL 与 ORM 事务耦合；模型方法内部唯一允许的手动失效方式 |
| `invalidate_by_id(*ids)` | 立即 | 外部调用方（管理脚本、测试）；Redis 错误吞掉 |
| `invalidate_all(*, strict=False)` | 立即 | 整个模型；`strict=True` 重新抛出 Redis 错误（迁移失效依赖它判断成败） |

## MissingGreenlet 规避

::: danger 风险点
commit 后对象过期，直接访问属性会触发同步懒加载。
:::

- 提交前用 `getattr()` 取 ID；提交后用 `sa_inspect()` 从 identity 读取（无 DB 查询）
- `check_cache_config()` 用 AST 检查子类方法体，禁止直接调用 `invalidate_by_id` / `invalidate_all` / 内部失效方法（它们可能在 commit 后访问过期属性）；模型方法里用 `invalidate_on_commit()` + `await session.commit()`

## `check_cache_config()` 静态检查

启动时（`configure_redis()` 之后）调用一次：

1. Redis 客户端已配置
2. 没有子类重写 `_get_client`
3. 所有子类的 `__cache_ttl__` 为正整数
4. AST 检查：子类方法（含 classmethod / staticmethod / property）不直接调用失效方法

副作用：注册 SQLAlchemy `after_commit` / `after_rollback` / `after_flush` / `persistent_to_deleted` 事件钩子（幂等；`configure_redis()` 也会注册），并预建"表名 → 缓存类"索引，让原生 DML 告警的热路径不做懒构建。

## 优雅降级

| 失败 | 行为 |
|------|------|
| Redis 未配置 | `RuntimeError`（配置错误，fail fast） |
| 读取失败 | 记日志，回退数据库 |
| 写入失败 | 记日志，继续 |
| 失效失败 | 同步路径记日志；`invalidate_by_id` / `invalidate_all()` 默认吞掉，`invalidate_all(strict=True)` 重新抛出；TTL 提供最终一致性 |
