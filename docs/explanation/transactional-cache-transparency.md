# 事务内的缓存透明性

::: tip 源码位置
`src/sqlmodel_ext/mixins/cached_table.py` — `_tables_with_uncommitted_writes`、`_query_dependency_tables`、`_has_uncommitted_writes`、`register_raw_dml_write`、`_adopt_cached_instance`

`src/sqlmodel_ext/session.py` — 增强 `AsyncSession` 的 `commit()` / `refresh()` / `execute()` 等观测点
:::

一个共享缓存要做到"加上它和不加它，业务代码观察到的结果完全一样"，最难的不是失效，而是**事务**。本章解释 `CachedTableBaseMixin` 为什么、以及怎样在事务内保持透明。缓存的整体结构（双层键、版本号、序列化）见 [Redis 缓存机制](./cached-table)。

## 问题：一个事务里的查询结果不是任何已提交状态

Redis 缓存被所有 session 共享；而一个事务内的查询结果包含**本事务自己的未提交写入**（任何访问数据库的查询都会先 autoflush）。如果这样的结果被写进缓存，就会出现两类错误：

| 错误 | 场景 |
|------|------|
| **发布未提交数据** | 事务 A 改了 `name='Alicia'`，在 commit 之前查询了一次；结果被写进 Redis。随后 A 回滚——但别的请求已经从缓存里读到了从未存在过的 `'Alicia'` |
| **脏读自己的旧值** | 事务 A 改了一行，然后查询。缓存命中返回的是修改**之前**的已提交值，A 读不到自己刚写的东西 |

"等到 commit 时再发布"也不对：事务中途的快照可能与最终提交的值不同（commit 前又改了一次、某个 savepoint 回滚了）。唯一正确的处理是**不发布**：只要这个查询依赖的表在本事务里有未提交写入，这次 `get()` 就**既不读缓存，也不写缓存**。

## 判定：依赖表 ∩ 未提交写表

判定在查询**之前**做一次，是"能否使用缓存"的唯一依据：

```python
skip_cache = cls._has_uncommitted_writes(
    session,
    cls._query_dependency_tables(condition, filter, order_by, load),
)
```

### 这个查询依赖哪些表

`_query_dependency_tables` 返回：

- 返回模型**及其所有子类**映射的表（STI 共享一张表，JTI 还有子表）
- `condition` / `filter` / `order_by` 中引用的所有表，**包括子查询、别名、CTE 里的表**
- 每个 `load` 目标（及其子类）的表

范围不能只是"返回模型这一家"：`owner_id IN (SELECT owner.id WHERE name = <未提交的值>)` 这样的条件，会让结果随着 `owner` 表的未提交写入而变化，而返回模型本身根本没被碰过。`load` 目标同理——写侧会把目标的未提交载荷发布进它自己的 ID 缓存。

**fail-closed**：表达式里出现无法追踪的部分（`text()`、`literal_column`、裸 `column()`、没有 metadata 的轻量 `table()`——它们都可以按名字引用任何表），依赖集合就是 metadata 里的**所有表**：任何未提交写入都会让这次查询跳过缓存。

### 本事务写过哪些表

`_tables_with_uncommitted_writes` 合并三个来源，缺一不可：

| 来源 | 覆盖 |
|------|------|
| CRUD 方法登记的待失效项 | `save` / `update` / `delete` / 级联删除 |
| `session.new` / `dirty` / `deleted` | 裸 `session.add()`、直接改属性、`session.delete()`——还没 flush 的部分 |
| "已 flush 未提交"表集合 | 已经发到数据库连接、但尚未 commit 的写：ORM flush（`after_flush` 事件）和原生 DML（`register_raw_dml_write`） |

前两个只覆盖"还没 flush"。一次 autoflush 之后对象看起来是干净的，而事务仍然没有提交——所以第三个来源必不可少。这个集合只在**最外层事务结束**时清空（commit / rollback / `close()` / `reset()` / `invalidate()` 的公共出口），savepoint 回滚后仍保留（过严，但方向安全：本事务剩余时间里这些表上的查询继续跳过缓存）。

### 按表判定，而不是按事务

判定粒度是**表**：改了模型 A 不会让模型 B 的查询跳过缓存（保住命中率），除非 B 的查询条件引用了 A 的表（保住正确性）。

```python
other = await Character.get(session, Character.name == 'Alice')
other.name = 'Alicia'                                  # 未 flush 的修改
await Character.get(session, Character.id == cid)      # 跳过缓存：character 表有未提交写
await Owner.get(session, Owner.id == oid)              # 仍走缓存：owner 表没被碰
```

## 原生 DML 必须登记

`update(...)` / `delete(...)` / `insert(...)` / `text(...)` 不经过 ORM 状态——不进 `new/dirty/deleted`，也不触发 `after_flush`。不登记的话，同一事务里依赖该表的查询会把未提交状态发布进共享缓存。

增强 `AsyncSession` 在**五个**入口自动登记：`execute` / `exec` / `scalar` / `stream` / `stream_scalars`（在 SQLAlchemy 的 `AsyncSession` 里后三个不经过 `execute`，`scalars()` 经过，因此不需要单独钩子）。规则：

- `update` / `delete` / `insert`：登记目标表
- `text()`：首个关键字在 `SELECT` / `SHOW` / `SET` 中视为只读；其余一律登记**所有表**（fail-closed）。`EXPLAIN` 刻意不在只读集合里——`EXPLAIN ANALYZE UPDATE ...` 在 PostgreSQL 里真的会执行写入
- 命中缓存表的 `UPDATE` / `DELETE`，如果该表对应的缓存类都没有登记待失效项，额外记一条 warning（它绕过了缓存失效）

已知盲区：直接在原生连接上执行、绕过 session 的写入不可见；多对多 `secondary` 表不在 `mapper.tables` 里，通过关系集合 `append` 写入的关联行两个 ORM 来源都登记不到；数据库级 `passive_deletes='all'` 级联删除的子表 ORM 不知道。

### 原生 DML 的正确写法：`invalidate_on_commit`

```python
Character.invalidate_on_commit(session, cid)           # 先登记：commit 后失效 id:Character:<cid> + 查询缓存
await session.execute(
    update(Character).where(col(Character.id) == cid).values(name='Bob')
)
await session.commit()                                  # 增强 commit 同步执行登记的失效
```

先登记再执行，warning 就不会触发（该表已有登记）。失效必须在 commit **之后**：之前失效的话，并发读者可能在 commit 落地前把旧值重新回填进缓存。`invalidate_on_commit` 只接受真实主键；rollback / `reset()` 会丢弃登记，重试路径要重新调用。

## 失效只在 commit 之后，且只有一个编排点

CRUD 方法（`save` / `update` / `delete` / `add`）自己**不失效**，只把待失效项登记到 `session.info`。增强 `AsyncSession.commit()` 负责编排：

1. 自动登记 session 里所有缓存模型的变更（覆盖绕过 CRUD 方法的裸 `add` / 改属性 / `delete`）
2. 快照待失效项（`after_commit` 事件会在 commit 中把它们弹出）
3. 真正 commit
4. 同步失效快照项 + 本次 flush 中级联删除的子项
5. 执行 post-commit 回调

`commit=False` 时什么都不失效（数据还没提交），登记项留到你最终的 `session.commit()`——"多个 `commit=False` 操作 + 一次 commit"天然正确。rollback 丢弃登记项。

`after_commit` 事件上还挂着一个 fire-and-forget 补偿任务，覆盖没有经过增强 `commit()` 的提交路径（例如普通 sqlmodel session）；它按 ID 去重，只补同步路径没有覆盖的部分。

### savepoint

savepoint 的 RELEASE 也会触发 `after_commit`，但那时数据只是并入外层事务，对其它 session 仍不可见，外层还可能回滚。所以 savepoint 级别的提交 / 回滚**既不消费也不丢弃**登记项，一切推迟到最外层事务：savepoint 回滚时保留登记项（外层提交时多失效一次——安全，缓存是可重建的副本）。

## 命中时不覆盖 identity map 里的状态

缓存命中后反序列化出的对象要并入 session。无条件 `session.merge()` 会把缓存状态拷贝到 identity map 里已有的对象上，**不检查它是否有未提交修改**——静默丢数据，而且新鲜度取决于"刚好有没有缓存"。`_adopt_cached_instance` 让命中与未命中两条路径对已有对象的处理一致：

| identity map 中已有对象的状态 | 处理 |
|------|------|
| 没有未加载的列 | 直接返回它——与未命中路径的标准 SQLAlchemy 语义相同，不 merge |
| 有未加载的列、且没有已加载的非主键列、且没有未提交历史 | `merge` 补齐（commit 后全部过期的常见状态，省一次查询） |
| 有未加载的列、同时有已加载的非主键列（或有未提交历史） | 放弃这次命中，回源数据库 |

第三种情况不能用缓存载荷去"补"：已加载的列可能比载荷**更新**（例如对象刚被权威读取，然后只有另一列被 expire）。需要最新值的调用方不应依赖这里——用 `authoritative=True`。

## `refresh()` 永不走缓存

`session.refresh(obj)` 的含义是"丢弃内存中的修改，从数据库重读"。增强 session 直接委托原生实现（expire → SELECT → 行不存在时 `ObjectDeletedError`），不经过 Redis。

## 其它绕过缓存的开关

| 条件 | 原因 |
|------|------|
| `authoritative=True` | 授权读取必须看到最新已提交行；同时绕过 identity map |
| `with_for_update=True` | 悲观锁必须读最新行 |
| `populate_existing=True` | 调用方明确要求刷新 identity map |
| `options` 非空 | `ExecutableOption` 可能改变加载行为，无法稳定表示在键里 |
| `join` 非空 | JOIN 目标的变更不会失效主模型，会产生幻读 |
| `no_cache=True` | 调用方显式退出 |

## 缓存键包含所有会改变结果的输入

查询缓存键由条件、分页、排序、`load`、`filter`、时间过滤规范化后哈希得到（`table_view` 先合并进显式参数，语义相同的查询得到同一个键）。keyset 游标 `after_id` **也在键里**：不同游标是不同的页，缺了它两个游标会共享同一条缓存。

## 相关参考

- [Redis 缓存机制](./cached-table)
- [给查询加 Redis 缓存](/how-to/cache-queries)
- [`CachedTableBaseMixin` API](/reference/mixins#cachedtablebasemixin)
