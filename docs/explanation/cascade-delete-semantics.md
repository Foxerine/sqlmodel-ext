# 级联删除语义（cascade × passive_deletes × ondelete）

::: tip 这是讲解（Explanation）类文档
本章回答的是"**为什么**这三个参数的某些组合会产生出人意料的结果"。想直接知道该用哪个组合，去 [操作指南：配置级联删除](/how-to/configure-cascade-delete)。
:::

## 为什么需要这章

SQLModel / SQLAlchemy 的级联删除由三个**正交**原语控制：

- **`cascade_delete`**（Relationship 参数）——ORM 侧是否把"删除"动作传播到子对象
- **`passive_deletes`**（Relationship 参数）——ORM 要不要主动加载子对象，还是信任数据库
- **`ondelete`**（Foreign Key 参数）——数据库侧收到 parent DELETE 时要对子行做什么

三者互动出 18 种可用组合。**多数组合的行为不是你以为的那样**。本章用 53 个受控实验把每种组合的实际结果钉死。

## 三个原语独立分解

### `cascade_delete`：ORM 侧的删除传播

在 `sqlmodel-ext`（以及 SQLModel / SQLAlchemy）里：

- `cascade_delete=False`（默认）→ 底层 cascade 字符串是 `'save-update, merge'`——**不包含** delete 语义
- `cascade_delete=True` → 底层 cascade 字符串是 `'all, delete-orphan'`——**包含** delete 和 delete-orphan

### `passive_deletes`：要不要"被动"让 DB 处理

- `passive_deletes=False`（默认）→ SA **主动**介入：要么发 UPDATE 把子行 FK 置空，要么 SELECT 子行再 DELETE，取决于 cascade
- `passive_deletes=True` → SA **放手**：不加载子对象，直接发 parent DELETE，剩下的全交给 DB
- `passive_deletes='all'` → 比 True 更彻底：SA 甚至不发 UPDATE 置空 FK

### `ondelete`：DB 侧的处理策略

- `'NO ACTION'`（PG 默认）——检测到还有 FK 引用时拒绝删除（`ForeignKeyViolationError`）
- `'CASCADE'`——随 parent 一并删除子行
- `'SET NULL'`——把子行的 FK 置 NULL（子行留下）
- `'RESTRICT'`——同 NO ACTION 但不可延迟

`NO ACTION` / `RESTRICT` 拒绝删除时，`TableBaseMixin.delete()` 在 PostgreSQL 上把这个 `IntegrityError` 翻译成 `ResourceReferencedError`（409 语义：行存在、仍被引用）。外键违反在驱动层有两个方向（"指向不存在的父行"与"父行仍被引用"）且异常完全相同，所以翻译只在"错误在 `delete()` 内捕获**且**失败语句是 `DELETE`"时发生。见 [处理"仍被引用"的删除](/how-to/handle-referenced-deletes)。

---

## 十八种组合的实证矩阵

所有场景配置：parent 一行，一个 child FK 指向 parent，child 的 FK nullable。删除 parent，观察 child 的最终状态。

```
                        cascade_delete=True      cascade_delete=False
ondelete   passive_del=  F    T    'all'       F    T   'all'
─────────────────────────────────────────────────────────────────
NO ACTION               OK   ERR   BAN        UPD   ERR   ERR
CASCADE                 OK   OK    BAN        UPD   OK    OK
SET NULL                OK   ⚠️    BAN        UPD   SN    SN
```

图例：

- **OK**：parent 和 child 都被删除
- **UPD**：SA 先发 UPDATE 把 child.FK 置 NULL，再 DELETE parent——**child 存活**
- **SN**：DB 的 SET NULL 生效——child 存活 FK=NULL
- **ERR**：`IntegrityError`（FK 违反）
- **⚠️**：**语义冲突地雷**——SA 的配置意图是删除 child，但 DB 的 SET NULL 执行了——child 变孤儿
- **BAN**：SA 在 schema 构造时拒绝这个组合（`ArgumentError: can't set passive_deletes='all' in conjunction with 'delete' or 'delete-orphan' cascade`）

## 每一格的详细解读

### `cascade_delete=False` 这一列（右半）

**关键事实：SA 在这里会偷偷发 UPDATE**。很多人以为 `cascade_delete=False` 意味着"SA 什么都不做"，其实不然。

`'save-update, merge'` cascade 不含删除语义，但 SA 的默认行为是"避免数据丢失"——删 parent 前，它会先 UPDATE 所有 child，把 FK 置空。这样 parent 删除不会破坏 DB 的 FK 完整性，child 留下来做孤儿。

| OD | passive_del=False | passive_del=True | passive_del='all' |
|----|-------------------|------------------|-------------------|
| NO ACTION | **UPD**：SA 主动置空 FK，绕开 DB 的 NO ACTION 检查 | **ERR**：SA 放手，DB 拒绝删除有子引用的 parent | **ERR**：同上 |
| CASCADE | **UPD**：SA 抢先置空 FK，DB 的 CASCADE 不触发 | **OK**：DB CASCADE 生效 | **OK**：同 True |
| SET NULL | **UPD**：SA 主动 UPDATE | **SN**：DB 的 SET NULL 生效 | **SN**：同 True |

**想让 SA 不发那个隐式 UPDATE？** 设 `passive_deletes=True`（或 `'all'`）。两者在这一列行为完全一致——`'all'` 仅在涉及 delete cascade 时才和 `True` 有别，但那组合（见右下）已被 SA 禁止。

### `cascade_delete=True` 这一列（左半）

| OD | passive_del=False | passive_del=True | passive_del='all' |
|----|-------------------|------------------|-------------------|
| NO ACTION | **OK**：SA 主动 SELECT child → DELETE child → DELETE parent | **ERR**：SA 放手，DB 拒绝 | **BAN** |
| CASCADE | **OK**：SA 主动删 child，DB CASCADE 冗余也没关系 | **OK**：DB 单条 CASCADE 搞定，最优 ✅ | **BAN** |
| SET NULL | **OK**：SA 主动删（覆盖 DB 的 SET NULL 意图） | **⚠️ 地雷**：SA 不介入，DB 执行 SET NULL，child 变孤儿而非被删 | **BAN** |

**最重要的一格**——`cascade_delete=True, passive_deletes=True, ondelete='SET NULL'` 是个**配置冲突**。ORM 层告诉你"删 parent 时要删 child"，DB 层告诉你"删 parent 时把 child 的 FK 置空"。加了 `passive_deletes=True` 等于让 DB 层赢——child 存活，与你的 ORM 意图相反。

**为什么 `'all'` + delete cascade 被禁**：`'all'` 语义是"SA 一指不动"，delete cascade 语义是"SA 主动删 child"，逻辑矛盾。SA 直接在 schema 构造时 raise `ArgumentError`，这是保护。

---

## `raise_on_sql` 为何（通常）不在级联期间触发

`sqlmodel-ext` 默认给所有 Relationship 设 `lazy='raise_on_sql'`，用来防止异步环境里的隐式懒加载。直觉上，你可能担心级联删除时 SA 会探测 child 的 m2o 关系——比如 `Conversation.project`——然后触发 `raise_on_sql`。

实验证明**级联删除路径不会自动触发 `raise_on_sql`**（在 minimal SA 环境下，跨 53 个场景覆盖了各种 child m2o、FK NULL/非 NULL、identity map 状态、链深度 1/2/3 级、back-ref 是否被加载的组合）。

原因在 SA 源码 `sqlalchemy/orm/dependency.py` 的 `ManyToOneDP.per_state_flush_actions`：

```python
sum_ = state.manager[self.key].impl.get_all_pending(state, dict_)
# get_all_pending 默认 passive=PASSIVE_NO_INITIALIZE
```

`PASSIVE_NO_INITIALIZE` 的 bit flag **不包含** `SQL_OK`。而 `raise_on_sql` 的 raise 条件是：

```python
def _invoke_raise_load(self, state, passive, lazy):
    if not passive & PassiveFlag.SQL_OK:
        return  # 不 raise
    raise sa_exc.InvalidRequestError(...)
```

没有 `SQL_OK` → 直接 return → 不 raise。级联期间的 cycle resolution 用的是"看一眼现有状态，不发 SQL"的模式。**只有你代码里显式访问属性（比如 `for msg in conv.messages`）才会触发 `raise_on_sql`**。

### 那什么情况下级联期间会触发？

在 vanilla SA 我们没复现到。但某些特定基础设施组合可能导致 `SQL_OK` 被意外置上：

- 自定义 metaclass 在 mapper 配置阶段注入了额外的属性访问
- Event listeners（例如 `persistent_to_deleted`、`after_flush`）中访问了 Relationship 属性
- PostgreSQL 触发器回调到 SA 层面
- 异步 greenlet 的 passive flag context 传递有 edge case

如果你在自家项目里遇到级联 `raise_on_sql`，这些都是排查方向。

---

## 设计建议的底层逻辑

给 `cascade_delete=True` 的关系**默认**加 `passive_deletes=True` 有三个好处：

1. **性能**：一条 `DELETE FROM parent WHERE id = :id` 打给 DB，后者用内置 CASCADE 树扫过去；否则 SA 先 `SELECT * FROM child WHERE parent_id = :id` 把子对象加载到 Python 侧，再 N 次 `DELETE FROM child WHERE id = :id`
2. **减少属性访问**：SA 不把 child 对象加载到 session，也就没机会触发它们身上的任何 Relationship 的 `raise_on_sql`
3. **语义一致性**：你强制自己在 DB 层声明 `ondelete='CASCADE'`，避免"ORM 想一套、DB 想另一套"的漂移

前提是 FK 的 `ondelete` **确实是** `'CASCADE'`——如果是 `'SET NULL'`，这条默认就会坑你（参见 `⚠️` 那一格）。

---

## 实证数据来源

本章所有判断基于跨 53 个独立场景的实验。每个场景独立 engine、独立 session、独立 schema，隔离的 `Base(DeclarativeBase)`，覆盖：

- Series A (18)：cascade × passive × ondelete 基础矩阵
- Series B (18)：加 child.external m2o 的 `raise_on_sql` 触发测试
- Series BP (6)：preload child 到 identity map 的变体
- Series C (3)：只有 back-ref m2o 的探针
- Series D (3)：3 级链级联
- Series E (5)：非删除路径的边界情况（直接访问、改 FK、swap m2o 等）

实验代码和原始结果在 `docs/internals/` 下未纳入——如需复现请参考 vanilla SA + DeclarativeBase + `lazy='raise_on_sql'` 的自行搭建。
