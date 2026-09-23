# 更多 Mixin：配额、模糊搜索、跨表扫描、迁移缓存失效

本页收集四个按需混入的能力。它们都从 `sqlmodel_ext.mixins` 导入。

## `ResourceQuotaMixin`

**目标**：限制"每个属主最多拥有 N 个某资源"，并在并发创建时也绝不超额。

### 1. 声明配额规则

宿主类混入 `ResourceQuotaMixin`，实现三个 classmethod：

```python
from uuid import UUID
from sqlalchemy import ColumnElement
from sqlmodel import col
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, NonNegativeInt
from sqlmodel_ext.mixins import ResourceQuotaMixin
from sqlmodel_ext.session import AsyncSession

class Account(SQLModelBase, UUIDTableBaseMixin, table=True):
    owner: Str64
    max_projects: NonNegativeInt = 2

class Project(ResourceQuotaMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    owner_id: UUID
    name: Str64

    @classmethod
    async def _lock_owner(cls, session: AsyncSession, owner_id: UUID, *, with_for_update: bool = True) -> Account | None:
        # 必须把 with_for_update 转发下去：预检路径传 False（无锁、可走缓存）
        return await Account.get(session, Account.id == owner_id, with_for_update=with_for_update)

    @classmethod
    def _quota_condition(cls, owner_id: UUID) -> ColumnElement[bool]:
        return col(cls.owner_id) == owner_id

    @classmethod
    def _quota_max(cls, owner: Account) -> int:
        return owner.max_projects
```

计数默认是 `cls.count(session, _quota_condition(owner_id))`；需要跨多表统计时覆盖 `_count_owner_resources(session, owner_id, owner)`。

### 2. 在 INSERT 前获取槽位

```python
async with Project.acquire_quota_lock(session, owner_id):
    await Project(owner_id=owner_id, name='p1').save(session)   # save 提交 → 释放属主行锁
```

- 进入时 `SELECT ... FOR UPDATE` 锁住属主行，同一属主的并发请求被串行化；`current + count <= max` 与受保护的 INSERT 在同一事务里原子执行。
- 把所有准备工作（校验、查询、内存计算）放在 `async with` **之前**，锁窗口只覆盖"获取 → INSERT → commit"。
- 块本身不 commit。块**正常**退出却没有发生 commit 时抛 `CallerDidNotCommitError`（属主锁还挂着，这是编程错误）。需要由外层更大的事务稍后提交时，传 `defer_commit=True` 关闭这个守卫。
- 一次创建多个资源：`count=3`，只检查一次 `current + 3 <= max`。

超额时：

```python
from sqlmodel_ext.mixins import QuotaExceededError, QuotaOwnerNotFoundError

try:
    async with Project.acquire_quota_lock(session, owner_id):
        await Project(owner_id=owner_id, name='p3').save(session)
except QuotaExceededError as e:          # status_code = 400；e.max_allowed / e.current_count
    await session.rollback()
    raise HTTPException(e.status_code, detail=str(e))
except QuotaOwnerNotFoundError as e:     # status_code = 404
    raise HTTPException(e.status_code, detail="owner not found")
```

### 3. 昂贵步骤之前先无锁预检

```python
await Project.preflight_quota(session, owner_id)        # 无锁、便宜、存在竞态的软门槛
result = await call_paid_external_api(...)               # 已经超额的属主不会产生费用
async with Project.acquire_quota_lock(session, owner_id):   # 原子保证仍来自这里
    await Project(owner_id=owner_id, name=result.name).save(session)
```

比起在昂贵步骤期间一直持有 `FOR UPDATE`，预检不会把同一属主的所有并发请求串行化。

### 4. 并发 get-or-create：`idempotent_check`

同一个"唯一资源"被并发创建时，后到的请求应该拿到先到者的行，而不是误报"超额"或撞唯一约束：

```python
async def find_existing() -> Project | None:
    # 必须读数据库真相（缓存模型请传 no_cache=True）
    return await Project.get(session, (col(Project.owner_id) == owner_id) & (col(Project.name) == 'default'))

async with Project.acquire_quota_lock(session, owner_id, idempotent_check=find_existing) as existing:
    if existing is None:
        existing = await Project(owner_id=owner_id, name='default').save(session)
```

回调在拿到属主锁之后、计数之前执行；返回非 `None` 时跳过配额检查并把它 `yield` 出来。这个分支不注册 commit 守卫（没有东西要提交），但属主锁一直持有到你的事务结束——尽快返回。

## `TrgmSearchableMixin`（仅 PostgreSQL）

**目标**：按名称 / 描述模糊搜索（`?query=`），容忍拼写错误，并且 2 个字符的查询也能走索引。

### 1. 数据库准备（迁移里执行一次）

```python
from sqlmodel_ext.mixins import BIGRAM_FUNCTION_SQL

def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(BIGRAM_FUNCTION_SQL)                       # public.bigrams(text) -> text[]
    op.execute("CREATE INDEX ix_tag_name_trgm ON tag USING gin (name gin_trgm_ops)")
    op.execute("CREATE INDEX ix_tag_name_bigrams ON tag USING gin (public.bigrams(name))")
    op.execute("CREATE INDEX ix_tag_description_bigrams ON tag USING gin (public.bigrams(description))")
```

没有索引搜索照样能用，只是退化为顺序扫描。`BIGRAM_FUNCTION_SQL` 一旦有索引依赖它就**不要再改函数体**——函数索引存的是旧定义算出的键，定义一变就静默漏行，直到 `REINDEX`。

### 2. 声明可搜索列

```python
from typing import ClassVar
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, Str256
from sqlmodel_ext.mixins import TrgmSearchableMixin

class Tag(SQLModelBase, UUIDTableBaseMixin, TrgmSearchableMixin, table=True):
    __trgm_text_columns__: ClassVar[tuple[str, ...]] = ('description',)
    name: Str64                    # __trgm_name_column__ 默认就是 'name'
    description: Str256 = ''
```

- 名称列：`ILIKE '%q%'` 子串匹配 **OR** pg_trgm 相似度运算符 `name % q`（阈值为 `pg_trgm.similarity_threshold`，默认 0.3）
- 文本列：只做 `ILIKE` 子串（长文本对短查询的相似度几乎总是很小，没有信号）
- `ILIKE` 用 `autoescape=True`：用户输入里的 `%` / `_` / 反斜杠不会变成通配符
- 子串匹配先用 bigram 包含（`bigrams(col) @> bigrams(q)`）缩小范围、再用 `ILIKE` 精确复核，所以 2 个字符的查询（常见于中日韩文本）也能走 GIN 索引

### 3. 在端点里使用

```python
from typing import Annotated
from fastapi import Depends
from sqlmodel_ext.mixins import TrgmSearchRequest

@router.get("", response_model=ListResponse[TagResponse])
async def list_tags(
    session: SessionDep,
    table_view: Annotated[TableViewRequest, Depends()],
    search: Annotated[TrgmSearchRequest, Depends()],
) -> ListResponse[Tag]:
    condition = search.apply_condition(Tag, col(Tag.name) != 'hidden')   # AND 进你已有的作用域条件
    return await Tag.get_with_count(session, condition, table_view=table_view)
```

客户端传 `?query=cat`。`TrgmSearchRequest.query` 是 `Str64 | None`（有长度上限、拒绝 NUL 字节）；空白或 `None` 时 `apply_condition` 原样返回你传入的条件。它只产出 `WHERE` 条件、**从不改变排序**——排序仍由 `table_view` 决定。

宿主契约：搜索列使用数据库**默认排序规则**。特殊 collation 下 `ILIKE` 跟随列的排序规则，而 `bigrams(query)` 跟随数据库默认，两边不对称会让 bigram 守卫先把本该匹配的行拒掉。

## `MixinTableScanMixin`

**目标**：同一个字段级 Mixin（例如给 N 张供应商表都加了 `remote_file_key`）挂在多张无关的物理表上，需要"按某条件跨所有挂载表查询 / 更新"。

```python
from sqlmodel import col
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, Str256
from sqlmodel_ext.mixins import MixinTableScanMixin
from sqlmodel_ext.session import AsyncSession

class RemoteFileKeyMixin(MixinTableScanMixin):
    remote_file_key: Str256 | None = None

    @classmethod
    async def find_by_remote_key(cls, session: AsyncSession, key: str) -> list[tuple[type, object]]:
        return [
            pair async for pair in cls._scan_rows_where(
                session,
                lambda table_cls: col(table_cls.remote_file_key) == key,
            )
        ]

class VendorAFile(RemoteFileKeyMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    title: Str64 = ''

class VendorBFile(RemoteFileKeyMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    size: int = 0
```

```python
hits = await RemoteFileKeyMixin.find_by_remote_key(session, 'k1')
# [(VendorAFile, <VendorAFile ...>), (VendorBFile, <VendorBFile ...>)]
```

- 发现是动态的：新表挂上这个 Mixin 就自动被覆盖。从某个具体表类调用时只扫它自己（及其子类）。
- 按 `__table__` 去重：同一 STI 家族共享一张物理表，只保留多态根（不加鉴别列过滤，覆盖所有子类的行）；JTI 子类各有物理表，不受影响。
- 条件由 `condition_factory(table_cls)` 用类型安全表达式构造——**不接受字符串字段名**，本模块不知道任何字段名。

## 迁移驱动的缓存失效

**目标**：迁移改变了已有数据的**含义**（例如把某数值列重新缩放），Redis 里的旧缓存仍能反序列化、却带着错误的值——需要精确地失效受影响的模型，而不是粗暴 `FLUSHDB`（会连带清掉会话、限流等无关键）。

### 1. 在迁移文件里声明

```python
# alembic/versions/<revision>.py 顶部
CACHE_INVALIDATIONS: dict[str, list[str]] = {
    'price-rescale-v1': ['Product', 'OrderLine'],
}
```

- 哨兵名用 `<topic>-v<N>`：同一主题以后再失效时换一个 `N`，不与旧哨兵冲突。
- 列出**真正写缓存的具体子类**：`invalidate_all()` 只沿 MRO **向上**走（自身 + 缓存祖先），父类清不到子类的命名空间（缓存键是 `id:<被查询类名>:<pk>`）。
- 用 Python 类名（`__name__`），不是 SQL 表名。

什么时候需要声明：

| 变更 | 需要？ | 原因 |
|------|------|------|
| `ADD COLUMN` | 否 | 旧缓存条目按字段默认值校验通过 |
| `DROP COLUMN` | 否* | 旧条目里多出的键在 `extra='forbid'` 下反序列化失败，`get()` 删掉坏键并回源数据库（自愈） |
| 改列类型，含义不变 | 否 | 被 Pydantic 转换，或反序列化失败后自愈 |
| 改列类型**并重新缩放 / 重新编码值** | **是** | 旧值能通过校验但是错的 |
| 删列 + 加一个**回填了非默认值**的列 | **是** | 旧条目拿到的是默认值而不是回填值 |
| `UPDATE` 已有行 | **是** | 已有值变了 |

\* 上面"删列 + 回填新列"的情况除外。经验法则："旧缓存条目能成功反序列化但含义变了"必须声明；"反序列化失败"或"含义没变"交给自愈。

### 2. 启动时执行

```python
from sqlmodel_ext import SQLModelBase, CachedTableBaseMixin
from sqlmodel_ext.mixins import run_pending_migration_cache_invalidations

# 迁移完成、configure_redis() 之后：
CachedTableBaseMixin.configure_redis(redis_client)
await run_pending_migration_cache_invalidations(SQLModelBase)          # 从 alembic.ini 收集
```

从 Alembic 收集需要可选依赖 `alembic`（未安装时收集函数抛 `RuntimeError`）。不用 Alembic 或在测试里，直接传任务：

```python
await run_pending_migration_cache_invalidations(
    SQLModelBase,
    tasks={'price-rescale-v1': ['Product', 'OrderLine']},
)
```

- 每个任务检查 Redis 哨兵 `cache_invalidation:<sentinel>`：不存在 → 对每个模型执行 `invalidate_all(strict=True)`，**全部成功**才写哨兵；已存在 → 跳过（幂等）。
- 任一模型失效失败时不写哨兵，下次启动重试；一个任务失败不阻塞其它任务。
- 已不存在的模型名算"没有东西要失效"，不算失败（否则会永远重试）。
- Redis 不可用时记日志并继续启动。
- 多个 worker 同时启动：最坏情况两个都执行了失效——版本号自增 + 删除本身是幂等的。
- `cache_invalidation:*` 命名空间不会被 `invalidate_all()` 触碰，哨兵会一直保留；手动 `FLUSHDB` 后哨兵与旧缓存一起消失，重跑也无害。

## 相关参考

- [Mixin 参考：`ResourceQuotaMixin` / `TrgmSearchableMixin` / `MixinTableScanMixin` / 迁移缓存失效](/reference/mixins#resourcequotamixin)
- [给查询加 Redis 缓存](./cache-queries)
