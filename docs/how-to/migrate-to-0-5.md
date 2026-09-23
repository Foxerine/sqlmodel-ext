# 从 0.4.x 迁移到 0.5.0

::: danger 项目仍处于 WIP / Alpha 阶段
sqlmodel-ext 仍在快速演进中。**任何两个版本之间都可能出现 breaking change**，本项目不提供稳定性或向后兼容保证，使用者自担风险。升级前请固定版本号，并完整阅读本页与 [CHANGELOG](https://github.com/Foxerine/sqlmodel-ext/blob/master/CHANGELOG.md)。
:::

**目标**：把一个基于 sqlmodel-ext 0.4.x 的项目升级到 0.5.0。

每一项的格式：**变化** → **为什么** → **改法**（before / after）→ **数据库迁移**（如需要）。标注「需要数据库迁移」的只有第 3 项。

## 0. 依赖

| 依赖 | 0.4.x | 0.5.0 |
|---|---|---|
| `pydantic` | `>=2.0` | **`>=2.12`**（`Unset` 即 `pydantic.experimental.missing_sentinel.MISSING`，以及 `EXCLUDE_IF_NONE` 背后的 `Field(exclude_if=...)`，都从 2.12 起才有） |
| `typing-extensions` | 传递依赖 | **`>=4.14.1`**（直接导入 `typing_extensions.Sentinel`） |
| `sqlmodel` | `>=0.0.32` | 不变 |
| 新增可选 extra | — | `alembic`：`run_pending_migration_cache_invalidations` 从 alembic 版本脚本发现待办任务 |

类型检查器（可选但强烈推荐）：静态收窄 `Unset | T` 需要 basedpyright ≥ 1.40.1 / pyright ≥ 1.1.414，见 [用 basedpyright 做类型检查](./type-check-with-basedpyright)。

## 1. `all_fields_optional=True` → `partial=True`

**变化**：类关键字 `all_fields_optional` 已删除，使用它会在类创建时抛 `TypeError`（错误信息里附带下面的迁移说明）。替代品 `partial=True` 的语义不同：

| | 0.4.x `all_fields_optional` | 0.5.0 `partial=True` |
|---|---|---|
| 继承字段变成 | `T \| None = None` | `Unset \| T = Unset`（基类可空则 `Unset \| T \| None`） |
| 没传的字段 | `None` | `Unset`（`model_dump()` 里不出现） |
| 不可空字段传 `null` | 校验通过，常常在数据库 NOT NULL 处 500 | **校验时拒绝（422）** |
| `default_factory` | — | 不调用（没传就是 `Unset`） |
| 与 `table=True` 同用 | — | `TypeError` |

**为什么**：`None` 同时表达"没传"和"清空"，两者需要相反的处理；约定（`exclude_unset=True`、`is not None`）总有一处会被忘记。详见 [Unset 三态](/explanation/unset-three-state)。

**改法**：

<!-- skip-run -->
```python
# before (0.4.x)
class ArticleUpdate(ArticleBase, all_fields_optional=True):
    pass

data = body.model_dump(exclude_unset=True)
if body.title is not None:
    article.title = body.title
```

```python
# after (0.5.0)
from sqlmodel_ext import SQLModelBase, Str64, Unset


class ArticleBase(SQLModelBase):
    title: Str64
    subtitle: Str64 | None = None


class ArticleUpdate(ArticleBase, partial=True):
    pass


body = ArticleUpdate.model_validate({'subtitle': None})
data = body.model_dump()                 # 不再需要 exclude_unset=True
assert data == {'subtitle': None}
if body.title is not Unset:              # 用 is not Unset 判断"有没有传"，不要用 is not None
    raise AssertionError("title was not sent")
```

检查清单：

- 全局搜索 `all_fields_optional`，换成 `partial=True`；
- 在所有 partial DTO 的使用处，把"有没有传"的判断从 `is None` / `is not None` 换成 `is Unset` / `is not Unset`；
- `instance.update(session, body)` 不需要改——它本来就只写 `model_fields_set` 里的字段，而 `Unset` 字段不会被 dump；
- 以前靠"`null` 表示不改"的客户端需要改成"不传"；对不可空字段发 `null` 现在是 422；
- 没有数据库迁移。

## 2. 在类体里声明 `oplock_version` 会抛 `TypeError`

**变化**：`oplock_version` 在所有 `SQLModelBase` 子类里都是保留名——不论该类是否启用乐观锁，只要在**自己的类体**里声明它，类创建时就 `TypeError`。只有 `OptimisticLockMixin` 定义它。

**为什么**：如果只对启用了乐观锁的类保留，一个没启用的类就能声明同名的领域字段，而它的某个子类重新启用乐观锁时，这个字段会被静默接成 `version_id_col`。

**改法**：领域上的"版本号"请用别的名字（例如 `version`、`revision`）——0.5 起 `version` 不再被乐观锁占用，可以自由使用。

## 3. 乐观锁列 `version` → `oplock_version`（需要数据库迁移）

**变化**：

| | 0.4.x | 0.5.0 |
|---|---|---|
| 列名 | `version` | `oplock_version` |
| 列类型 | `INTEGER` | `BIGINT`（上界 `JS_MAX_SAFE_INTEGER`） |
| 数据库默认值 | 无 | `server_default 0` |
| `model_dump()` 中 | 出现 | **不出现**（`exclude=True`） |
| `save()` / `update()` 冲突重试 | 默认 0 次 | 默认 **3 次**（见第 4 项） |

**为什么**：

- 改名：`version` 是一个很常见的领域字段名，被乐观锁占用会逼用户给业务字段起别扭的名字；
- `BIGINT`：频繁更新的行消除 `INTEGER` 溢出的现实风险；
- `server_default`：不认识这一列的写入方（裸 SQL、其他服务、测试夹具）INSERT 时会省略它，数据库默认值让这些 INSERT 依然合法。它**不能**让 0.4 → 0.5 的改名变得滚动兼容——见 SQL 下方的警告；
- `exclude=True`：它是 ORM 内部状态，不应泄漏进 `model_dump()`（比如泄漏进 `extra='forbid'` 的响应 DTO）。

**改法**：

- 代码里读 `obj.version` 的地方改成 `obj.oplock_version`；
- 继承顺序照旧：`OptimisticLockMixin` 必须排在 `TableBaseMixin` / `UUIDTableBaseMixin` 之前；
- 如果 API 曾经把 `version` 暴露给客户端，现在它不会再出现在 `model_dump()` 里；由于 `oplock_version` 是保留名，响应 DTO 里需要用另一个字段名显式赋值。

```python
from sqlmodel_ext import OptimisticLockMixin, SQLModelBase, Str64, UUIDTableBaseMixin


class Order(SQLModelBase, OptimisticLockMixin, UUIDTableBaseMixin, table=True):
    status: Str64 = 'pending'


order = Order()
assert order.oplock_version == 0
assert 'oplock_version' not in order.model_dump()
assert Order.__optimistic_retry_default__ == 3
```

**数据库迁移（PostgreSQL，以表 `order` 为例）**：

```sql
ALTER TABLE "order" ALTER COLUMN version TYPE BIGINT;
ALTER TABLE "order" ALTER COLUMN version SET DEFAULT 0;
ALTER TABLE "order" RENAME version TO oplock_version;
```

JTI 层级里只有**根表**带这一列（子表经 mapper 继承共享）；STI 本来就只有一张表。对每个混入了 `OptimisticLockMixin` 的模型，按它的根表各跑一次迁移。

::: warning 这次改名不是滚动兼容的
列一改名，仍在运行 sqlmodel-ext 0.4.x 的应用实例读写的还是已经不存在的 `version` 列，这些模型的每次加载、保存都会失败；反过来，0.5.0 的实例也无法在旧列名上运行。所以对使用了 `OptimisticLockMixin` 的模型：

- **最简单**：停掉所有 0.4.x 实例 → 跑迁移 → 启动 0.5.0 实例（一个短维护窗口）；
- **零停机**：自行设计 expand/contract 序列（例如上线期间用数据库触发器保持两列同步，完成后再删掉 `version`）。本库不提供这种桥接。

没有使用 `OptimisticLockMixin` 的模型不受影响。0.4/0.5 混跑期间共享 Redis 缓存是安全的：对方版本写入的条目会校验失败、被删除，读取回落到数据库——正确性不受影响，只是上线完成前命中率会下降。
:::

对应的 Alembic 操作（上面的 SQL 就是它在 PostgreSQL 方言下生成的）：

<!-- skip-run -->
```python
import sqlalchemy as sa
from alembic import op


def upgrade() -> None:
    op.alter_column(
        'order', 'version',
        new_column_name='oplock_version',
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        server_default=sa.text('0'),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'order', 'oplock_version',
        new_column_name='version',
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        server_default=None,
        existing_nullable=False,
    )
```

::: warning 上线注意
- **改名不能与旧代码混跑**：迁移执行后，仍在运行的 0.4.x 实例找不到 `version` 列；迁移执行前，0.5.0 实例找不到 `oplock_version` 列。请在停机窗口或蓝绿切换中执行，让迁移与代码切换同时完成。
- `INTEGER → BIGINT` 在 PostgreSQL 上会**重写整张表**并持有 `ACCESS EXCLUSIVE` 锁，大表请预估耗时。
- 每一张使用 `OptimisticLockMixin` 的表（STI/JTI 以根表为准）都要执行一遍。
:::

## 4. `OptimisticLockMixin` 默认重试 3 次；`delete()` 冲突抛 `OptimisticLockError`

**变化**：

- `save()` / `update()` 的 `optimistic_retry_count` 默认值从 `0` 变成 `None`，`None` 表示使用模型策略 `__optimistic_retry_default__`：`OptimisticLockMixin` 模型为 **3**，其他模型为 0。`update()` 的重试会重新读取最新行，再把调用方的改动叠加上去。
- `delete()` 遇到乐观锁冲突时，不再抛原始的 `StaleDataError`，而是抛 `OptimisticLockError`（`delete()` 从不重试——"刚被别人改过，我还要不要删"由调用方决定）。

**为什么**：策略应该写在声明能力的地方，而不是每个调用点都记得传参数；冲突异常应该只有一种类型。

**改法**：

- 需要旧行为（冲突立刻失败）的调用点显式传 `optimistic_retry_count=0`；
- 捕获 `StaleDataError` 的 `delete()` 调用点改成捕获 `OptimisticLockError`。

## 5. `delete()` 遇到外键 RESTRICT 抛 `ResourceReferencedError`

**变化**：删除一个仍被外键引用的行时，`delete()` 以前让 `IntegrityError` 原样冒出；现在抛 `ResourceReferencedError`（带 `status_code = 409`、`friendly_message`、`constraint_name`）。它**不是** `IntegrityError` 的子类。

**为什么**：同一个外键违例有两个方向（"引用的目标不存在"是 404 语义，"仍被引用无法删除"是 409 语义），原始异常分不出来；`delete()` 只在错误确实由 DELETE 语句引起时才转换。

**改法**：捕获 `IntegrityError` 来处理"仍被引用"的调用点改成捕获 `ResourceReferencedError`（`from sqlmodel_ext.mixins import ResourceReferencedError`）。友好文案可用 `TableBaseMixin.register_fk_delete_restrict_message(constraint_name, message)` 注册。

## 6. 默认主键改为 UUIDv7

**变化**：`UUIDTableBaseMixin.id` 与 `create_subclass_id_mixin()` 生成的 JTI 子表主键，默认工厂从 `uuid.uuid4` 改为 UUIDv7（`sqlmodel_ext.mixins.uuid7`；Python 3.14+ 上就是 `uuid.uuid7`）。

**为什么**：UUIDv7 的前 48 位是毫秒时间戳，字节序即创建时间序：`ORDER BY id` 近似创建顺序，B-tree 插入集中在右侧，页分裂与随机 I/O 远少于 UUIDv4。

**改法**：

- **既有数据无需迁移**：旧行保留原来的 v4 值；v4/v7 混合时字节序依然是良定义的全序（例如用于锁排序）。列类型不变。
- 注意 ID 会以毫秒精度暴露创建时间；ID 是标识符不是凭证，不要依赖它不可猜测。
- 不要把 ID 顺序当作权威时间顺序（时间戳来自应用时钟，跨进程没有全局单调性）。
- 需要确定性派生主键（例如由幂等键 `uuid5`）时，显式赋值即可——默认工厂只在未提供 id 时生效。

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin


class Note(SQLModelBase, UUIDTableBaseMixin, table=True):
    pass


assert Note().id.version == 7
```

## 7. `TimeFilterRequest` 只接受带时区的 `datetime`

**变化**：`created_after_datetime` / `created_before_datetime` / `updated_after_datetime` / `updated_before_datetime` 的类型从 `datetime` 变为 `AwareDatetime`，不带时区的值校验失败（`TableViewRequest` 继承了这一点）。

**为什么**：不带时区的值无法与数据库里带时区的值比较，它会被静默按数据库时区解释，查询结果错位几个小时而没有任何报错。

**改法**：客户端传 `2026-01-01T00:00:00Z` 或 `2026-01-01T08:00:00+08:00`，不要传 `2026-01-01T00:00:00`。

```python
from datetime import datetime, timezone

from pydantic import ValidationError
from sqlmodel_ext import TimeFilterRequest

TimeFilterRequest(created_after_datetime=datetime(2026, 1, 1, tzinfo=timezone.utc))
try:
    TimeFilterRequest(created_after_datetime=datetime(2026, 1, 1))
except ValidationError:
    pass
else:
    raise AssertionError("naive datetime must be rejected")
```

同时，`PaginationRequest.offset` 现在有上界 `MAX_TABLE_VIEW_OFFSET`（`JS_MAX_SAFE_INTEGER - 1000`），超大的 offset 在校验时就被拒绝，而不是在数据库驱动处报 `bigint out of range`。

## 8. `@requires_for_update` 找不到 session 时报错

**变化**：被装饰方法的调用参数里找不到 `AsyncSession` 时，0.4.x 静默跳过检查；0.5.0 抛 `RuntimeError`（fail-closed）。

**为什么**：找不到 session 恰恰说明守卫失效了（例如内层装饰器没用 `@wraps`，`inspect.signature` 看不到 `session` 参数），它不应该看起来像"检查通过"。

**改法**：确保方法有一个名为 `session` 的参数，且所有内层装饰器都使用 `functools.wraps`。

## 9. 重命名 / 删除的内部方法

| 0.4.x | 0.5.0 |
|---|---|
| `RelationPreloadMixin._ensure_relations_loaded` | `ensure_relations_loaded`（公开；另有批量版 `ensure_relations_loaded_bulk`） |
| `CachedTableBaseMixin._warn_raw_dml_on_cached` | `register_raw_dml_write`（除了告警，还把原生 DML 写过的表登记为"已写未提交"） |
| `CachedTableBaseMixin._refresh_via_cache` | 删除 |
| `CachedTableBaseMixin._has_pending_invalidation` | 删除 |

这些原本是下划线私有方法；如果你的代码调用过它们，按上表替换。

## 10. 会话与读取语义

- **`AsyncSession.refresh()` 不再走缓存**：0.4.x 对缓存模型的整对象 refresh 经 `Model.get()` 走 Redis；0.5.0 一律委托给原生 `session.refresh()`，永远读数据库。refresh 的含义就是"丢弃内存状态、重读数据库"。
- **`with_for_update=True` 强制 `populate_existing`**：加锁读取时，数据库返回的是最新行，但 identity map 可能交还一个已加载、属性陈旧的对象——随后的"读-改-写"就会丢失更新。现在锁读取总是用数据库的值覆盖，没有关闭开关。

一般无需改代码；如果你的逻辑依赖"加锁读取后仍保留内存中未提交的改动"，那本来就是错误的。

## 11. 字段类型的行为变化

| 类型 | 变化 | 改法 |
|---|---|---|
| `JSON100K` / `JSONList100K` | **出站**：`model_dump()`、`model_dump(mode='json')`、`model_dump_json()` 都输出对象 / 数组本身，不再是 JSON **字符串**。**入站**：dict / list 输入现在也受 100K 字符上限与嵌套深度限制（0.4.x 只检查字符串输入）；表模型在构造时也会检查 | 以前把响应里的该字段当字符串再 `JSON.parse` 的客户端要改；入站 JSON 字符串仍被接受。覆写 `model_post_init` 的子类必须调用 `super().model_post_init(context)`，否则检查被静默跳过 |
| `PositiveFloat` / `NonNegativeFloat` | 拒绝 `inf` / `nan`（`AllowInfNan(False)`） | 以前能存进去的 `1e309` 现在是 422 |
| Decimal 别名（`SignedDecimal20_10` 等） | **整数位校验真正生效**：以前校验器的元数据顺序让 Pydantic 只检查总位数和小数位，例如 `SignedDecimal20_10` 能接受 15 位整数，只剩数据库兜底 | 超出整数位的值现在在校验时被拒绝 |

```python
from decimal import Decimal

from pydantic import ValidationError
from sqlmodel_ext import PositiveFloat, SQLModelBase, SignedDecimal20_10
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K


class Sample(SQLModelBase):
    rate: SignedDecimal20_10 = Decimal(0)
    ratio: PositiveFloat = 1.0
    payload: JSON100K = {}


assert Sample(payload={'a': [1, 2]}).model_dump_json().endswith('"payload":{"a":[1,2]}}')
for bad in ({'rate': Decimal('12345678901')}, {'ratio': float('inf')}):
    try:
        Sample(**bad)
    except ValidationError:
        pass
    else:
        raise AssertionError(bad)
```

（`JSON100K` 需要 `orjson`，即 `sqlmodel-ext[postgresql]`。）

## 12. RelationLoadChecker（实验性）

- 检出变多：新增 RLC014（FastAPI `Depends` 在函数体内 commit，导致同一 session 上兄弟依赖注入的 ORM 对象在端点开始时已过期）；session 参数按**子类**识别（包括 `sqlmodel_ext.AsyncSession`），以前按身份识别会漏掉；commit 语义可通过模块级 `conditional_commit_methods` / `explicit_commit_methods` / `dependency_commit_methods` 配置。
- `# noqa: RLCxxx` 现在在**所有**公开入口（`check_app` / `check_model_methods` / `check_project_coroutines` / `check_function`）都生效；端点的 noqa 注释必须放在**第一个装饰器那一行**。

升级后第一次跑检查器可能出现新的告警——那多半是以前漏报的真实问题。

## 升级检查清单

- [ ] `pydantic>=2.12`
- [ ] 搜索 `all_fields_optional` → `partial=True`；partial DTO 上的 `is None` / `is not None` 判断 → `is Unset` / `is not Unset`
- [ ] 类体里没有 `oplock_version`
- [ ] 每张乐观锁表执行列改名 + `BIGINT` + `DEFAULT 0` 迁移；代码里 `.version` → `.oplock_version`
- [ ] 需要"冲突立即失败"的调用点传 `optimistic_retry_count=0`；`delete()` 捕获 `OptimisticLockError` / `ResourceReferencedError`
- [ ] 时间过滤参数的客户端带上时区
- [ ] `@requires_for_update` 方法都有 `session` 参数，内层装饰器都用 `@wraps`
- [ ] 替换对已重命名 / 删除的私有方法的调用
- [ ] 消费 `JSON100K` 字段的客户端改为直接读对象
- [ ] 跑一次 basedpyright（见 [用 basedpyright 做类型检查](./type-check-with-basedpyright)）
