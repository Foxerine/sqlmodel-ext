# 配置级联删除

**目标**：为一对多关系选对 `cascade_delete` / `passive_deletes` / `ondelete` 三件套组合，避免孤儿行、避免隐式 UPDATE、避免 SA 拒绝启动。

**前置条件**：

- 你有一个 parent-child 关系（`Character.user_configs`、`Project.conversations` 等）
- 你知道删除 parent 时你**希望** child 怎么处理（删掉？保留但解除关联？禁止删 parent？）

如果你还没想清楚 ORM 侧和 DB 侧两层的语义区别，先读 [讲解：级联删除语义](/explanation/cascade-delete-semantics)。

---

## 1. 决定你的意图

### 意图 A：删 parent 时 child 也删（硬关联，例如 `Conversation.messages`）

```python
class Parent(SQLModelBase, UUIDTableBaseMixin, table=True):
    children: list['Child'] = Relationship(
        back_populates='parent',
        cascade_delete=True,
        passive_deletes=True,  # 关键
    )

class Child(SQLModelBase, UUIDTableBaseMixin, table=True):
    parent_id: UUID = Field(
        foreign_key='parent.id',
        ondelete='CASCADE',  # 必须和 passive_deletes=True 对齐
        index=True,
    )
    parent: 'Parent' = Relationship(back_populates='children')
```

**生效流程**：删 parent 时 SA 只发一条 `DELETE FROM parent WHERE id = :id`，DB 的 CASCADE 自动把所有 child 一起删。

### 意图 B：删 parent 时 child 保留但解除关联（软关联，例如 `Project.user_files`）

```python
class Parent(SQLModelBase, UUIDTableBaseMixin, table=True):
    children: list['Child'] = Relationship(
        back_populates='parent',
        cascade_delete=False,  # 关键：不级联删除
        passive_deletes=True,  # 可选但推荐：避免 SA 发隐式 UPDATE
    )

class Child(SQLModelBase, UUIDTableBaseMixin, table=True):
    parent_id: UUID | None = Field(
        default=None,
        foreign_key='parent.id',
        ondelete='SET NULL',  # 关键：DB 负责置空 FK
        nullable=True,
        index=True,
    )
    parent: 'Parent | None' = Relationship(back_populates='children')
```

**生效流程**：删 parent 时 SA 发 `DELETE FROM parent WHERE id = :id`，DB 把所有 child 的 FK 置 NULL。child 存活。

### 意图 C：禁止删除有 child 的 parent

```python
class Parent(SQLModelBase, UUIDTableBaseMixin, table=True):
    children: list['Child'] = Relationship(
        back_populates='parent',
        cascade_delete=False,
        passive_deletes=True,
    )

class Child(SQLModelBase, UUIDTableBaseMixin, table=True):
    parent_id: UUID = Field(
        foreign_key='parent.id',
        ondelete='RESTRICT',  # 或省略用默认 NO ACTION
        index=True,
    )
```

**生效流程**：删 parent 时如果还有 child，数据库拒绝删除。在 PostgreSQL 上 `Parent.delete(session, parent)` 把它翻译成 `ResourceReferencedError`（`status_code = 409`，消息可以用 `TableBaseMixin.register_fk_delete_restrict_message()` 按约束名注册）。业务代码需要先处理完子数据才能删 parent。详见 [处理"仍被引用"的删除](./handle-referenced-deletes)。

---

## 2. 速查表

```
你的意图                            cascade_delete  passive_deletes  ondelete
──────────────────────────────────────────────────────────────────────────────
删 parent → 删 child（硬关联）         True            True            'CASCADE'
删 parent → child 保留 FK=NULL（软）   False           True            'SET NULL'
有 child 时禁止删 parent              False           True            'RESTRICT'（或默认）
```

只需记这三行。其他所有组合要么有隐式 UPDATE 陷阱，要么被 SA 拒绝（见 [讲解](/explanation/cascade-delete-semantics) 矩阵）。

---

## 3. 常见陷阱

### 陷阱 1：`cascade_delete=True` + `ondelete='SET NULL'`

```python
# ❌ 配置冲突：ORM 说"删 child"，DB 说"置空 child.FK"
children: list['Child'] = Relationship(
    cascade_delete=True,
    passive_deletes=True,  # 让 DB 赢 → 反而不删 child
)
# + FK ondelete='SET NULL'
# → child 存活，FK=NULL，与 cascade_delete 意图相反
```

**症状**：你删了 parent，期望 child 也没了，但 `SELECT * FROM child` 发现 child 还在，FK 是 NULL。

**修法**：对齐两层语义。想硬删：FK 改 `'CASCADE'`。想软解除：`cascade_delete` 改 `False`。

### 陷阱 2：`cascade_delete=True` + `passive_deletes='all'`

```python
# ❌ SA 直接 raise ArgumentError，启动都起不来
children: list['Child'] = Relationship(
    cascade_delete=True,
    passive_deletes='all',
)
```

**症状**：应用启动时 `ArgumentError: can't set passive_deletes='all' in conjunction with 'delete' or 'delete-orphan' cascade`。

**修法**：`passive_deletes='all'` 只在 `cascade_delete=False` 时有意义；通常你想要的是 `True`（基本等价）。

### 陷阱 3：忘了给 FK 加 `ondelete`，默认 NO ACTION

```python
# ❌ 漏了 ondelete
parent_id: UUID = Field(foreign_key='parent.id', index=True)
# + passive_deletes=True
# → 删 parent 时 IntegrityError（DB 拒绝）
```

**症状**：删 parent 时 `IntegrityError: update or delete on "parent" violates foreign key constraint`。

**修法**：明确加 `ondelete='CASCADE'`（或 `'SET NULL'`，看业务意图）。

### 陷阱 4：`cascade_delete=False` + `passive_deletes=False`（默认配置）

表面看"什么都没做"，其实 SA 会**偷偷**发 `UPDATE child SET parent_id = NULL` 给所有 child——无论 FK 是否 nullable。这是 SA 2.x 的保守默认行为，为了避免删 parent 时违反 FK 完整性。

**后果**：

- 性能：每删一次 parent 多出 N 次 UPDATE
- 业务语义：child 可能本不该解除关联，结果 FK 被偷偷清零

**修法**：加 `passive_deletes=True` 让 SA 不插手，让 DB 按 `ondelete` 设置处理。

---

## 4. 迁移现有代码（如果你在升级老项目）

### 检查清单

对每个 `cascade_delete=True` 的 Relationship：

```bash
# 1. grep 找 child FK 的 ondelete
grep -n "foreign_key='parent_table_name'" sqlmodels/
```

2. 对照本文 §2 速查表：
   - FK 是 `'CASCADE'` → 加 `passive_deletes=True`（纯优化）
   - FK 是 `'SET NULL'` → **不要**加 `passive_deletes=True`；先决定业务意图是删还是置空
   - FK 是 `'NO ACTION'` 或默认 → 补上明确 ondelete

3. 改完后**不要批量跑测试就算完**——如果原先在走 `UPD` 路径（SA 隐式 UPDATE child FK），加 `passive_deletes=True` 后会走 `ERR` 或 `SN` 路径，行为可能变化。

### Alembic 迁移提示

修改 `ondelete` 需要 Alembic 迁移：

```python
def upgrade() -> None:
    op.drop_constraint('child_parent_id_fkey', 'child', type_='foreignkey')
    op.create_foreign_key(
        'child_parent_id_fkey', 'child', 'parent',
        ['parent_id'], ['id'],
        ondelete='CASCADE',  # 新值
    )

def downgrade() -> None:
    op.drop_constraint('child_parent_id_fkey', 'child', type_='foreignkey')
    op.create_foreign_key(
        'child_parent_id_fkey', 'child', 'parent',
        ['parent_id'], ['id'],
        ondelete='SET NULL',  # 老值
    )
```

---

## 5. 如何验证你的配置

### 单元测试模板

```python
import pytest
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

@pytest.mark.asyncio
async def test_cascade_delete_removes_children(session: AsyncSession) -> None:
    parent = await Parent().save(session)          # 必须用返回值：commit 后原对象已过期
    parent_id = parent.id
    child = await Child(parent_id=parent_id).save(session)
    child_id = child.id

    await Parent.delete(session, parent)

    # 用原生 SQL 绕过 ORM identity map / Redis 缓存，直接验证 DB
    remaining = (await session.execute(
        text("SELECT COUNT(*) FROM child WHERE id = :id"),
        {'id': str(child_id)},
    )).scalar()
    assert remaining == 0, "child 应已随 parent 删除"
```

对 SET NULL 意图：改 assert 为 `FK IS NULL`；对 RESTRICT（PostgreSQL）：改为 `with pytest.raises(ResourceReferencedError): await Parent.delete(session, parent)`。

---

## 看不到自己的情况？

本指南覆盖一对多（parent→children）的主要形态。如果你要配的是：

- **多对多** → `SecondaryTable` 上设 `ondelete`，Relationship 上 `secondary=table_obj`
- **自引用**（例如 `Conversation.compacted_from`）→ 同样适用但要注意 SA 的 cycle detection 会介入，参考 [讲解](/explanation/cascade-delete-semantics) 关于 cycle 的部分
- **一对一** → 语义上和一对多一样，只是加 `sa_relationship_kwargs={'uselist': False}`

如果情况复杂到清单覆盖不了，去 [讲解](/explanation/cascade-delete-semantics) 读完整矩阵。
