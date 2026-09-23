# Configure Cascade Delete

**Goal**: pick the right combination of `cascade_delete` / `passive_deletes` / `ondelete` for a one-to-many relationship, avoiding orphans, implicit UPDATEs, and SA startup rejections.

**Prerequisites**:

- You have a parent-child relationship (`Character.user_configs`, `Project.conversations`, etc.)
- You've decided **what you want** to happen to children when the parent is deleted (delete them? keep them but detach? forbid parent deletion?)

If you haven't thought through the ORM vs DB semantic distinction, read [Explanation: Cascade Delete Semantics](/en/explanation/cascade-delete-semantics) first.

---

## 1. Decide Your Intent

### Intent A: Delete parent → delete children (hard link, e.g. `Conversation.messages`)

```python
class Parent(SQLModelBase, UUIDTableBaseMixin, table=True):
    children: list['Child'] = Relationship(
        back_populates='parent',
        cascade_delete=True,
        passive_deletes=True,  # key
    )

class Child(SQLModelBase, UUIDTableBaseMixin, table=True):
    parent_id: UUID = Field(
        foreign_key='parent.id',
        ondelete='CASCADE',  # must align with passive_deletes=True
        index=True,
    )
    parent: 'Parent' = Relationship(back_populates='children')
```

**Flow**: deleting parent emits a single `DELETE FROM parent WHERE id = :id`; the DB's CASCADE removes all children automatically.

### Intent B: Delete parent → children survive with FK=NULL (soft link, e.g. `Project.user_files`)

```python
class Parent(SQLModelBase, UUIDTableBaseMixin, table=True):
    children: list['Child'] = Relationship(
        back_populates='parent',
        cascade_delete=False,  # key: don't cascade-delete
        passive_deletes=True,  # optional but recommended: avoid implicit UPDATE
    )

class Child(SQLModelBase, UUIDTableBaseMixin, table=True):
    parent_id: UUID | None = Field(
        default=None,
        foreign_key='parent.id',
        ondelete='SET NULL',  # key: DB nulls the FK
        nullable=True,
        index=True,
    )
    parent: 'Parent | None' = Relationship(back_populates='children')
```

**Flow**: deleting parent emits `DELETE FROM parent WHERE id = :id`; the DB nulls all children's FKs. Children survive.

### Intent C: Forbid deleting parent while children exist

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
        ondelete='RESTRICT',  # or omit for default NO ACTION
        index=True,
    )
```

**Flow**: deleting parent while children still exist is rejected by the database. On PostgreSQL, `Parent.delete(session, parent)` translates this into `ResourceReferencedError` (`status_code = 409`; the message can be registered per constraint name with `TableBaseMixin.register_fk_delete_restrict_message()`). Business code must clean up children first. See [Handle deletes of still-referenced rows](./handle-referenced-deletes).

---

## 2. Cheat Sheet

```
Your intent                              cascade_delete  passive_deletes  ondelete
───────────────────────────────────────────────────────────────────────────────────
Delete parent → delete children (hard)     True            True            'CASCADE'
Delete parent → children survive, FK=NULL  False           True            'SET NULL'
Forbid parent delete while children exist  False           True            'RESTRICT' (or default)
```

Memorize these three rows. Any other combination either has implicit UPDATE traps or is outright banned by SA (see the [Explanation](/en/explanation/cascade-delete-semantics) matrix).

---

## 3. Common Pitfalls

### Pitfall 1: `cascade_delete=True` + `ondelete='SET NULL'`

```python
# ❌ Config conflict: ORM says "delete child", DB says "null child.FK"
children: list['Child'] = Relationship(
    cascade_delete=True,
    passive_deletes=True,  # lets DB win → child NOT deleted
)
# + FK ondelete='SET NULL'
# → child survives with FK=NULL, opposite of your cascade_delete intent
```

**Symptom**: you delete parent, expect child gone, but `SELECT * FROM child` shows the child still there with NULL FK.

**Fix**: align both layers. Want hard delete: change FK to `'CASCADE'`. Want soft detach: change `cascade_delete` to `False`.

### Pitfall 2: `cascade_delete=True` + `passive_deletes='all'`

```python
# ❌ SA raises ArgumentError at startup
children: list['Child'] = Relationship(
    cascade_delete=True,
    passive_deletes='all',
)
```

**Symptom**: app won't start: `ArgumentError: can't set passive_deletes='all' in conjunction with 'delete' or 'delete-orphan' cascade`.

**Fix**: `passive_deletes='all'` only makes sense with `cascade_delete=False`; what you probably want is just `True` (essentially equivalent).

### Pitfall 3: Forgetting `ondelete`, defaulting to NO ACTION

```python
# ❌ Missing ondelete
parent_id: UUID = Field(foreign_key='parent.id', index=True)
# + passive_deletes=True
# → IntegrityError when deleting parent (DB rejects)
```

**Symptom**: `IntegrityError: update or delete on "parent" violates foreign key constraint`.

**Fix**: explicitly add `ondelete='CASCADE'` (or `'SET NULL'`, per intent).

### Pitfall 4: `cascade_delete=False` + `passive_deletes=False` (default config)

On the surface "SA does nothing", but SA will **silently** emit `UPDATE child SET parent_id = NULL` for every child — regardless of FK nullability. This is SA 2.x's conservative default, designed to avoid FK integrity violations when parent is deleted.

**Consequences**:

- Performance: N extra UPDATEs per parent delete
- Business semantics: children may have unexpectedly had their FKs nulled

**Fix**: add `passive_deletes=True` so SA stays out, let DB handle per `ondelete`.

---

## 4. Migrating Existing Code

### Checklist

For every `cascade_delete=True` Relationship:

```bash
# 1. grep child FK ondelete
grep -n "foreign_key='parent_table_name'" sqlmodels/
```

2. Match against §2 cheat sheet:
   - FK is `'CASCADE'` → add `passive_deletes=True` (pure optimization)
   - FK is `'SET NULL'` → **don't** add `passive_deletes=True`; decide business intent first (delete or null)
   - FK is `'NO ACTION'` or default → make `ondelete` explicit

3. **Don't just batch-run tests after changes** — if you were previously on the `UPD` path (implicit UPDATE), adding `passive_deletes=True` moves you to `ERR` or `SN` paths and behavior changes.

### Alembic Migration Template

Changing `ondelete` requires an Alembic migration:

```python
def upgrade() -> None:
    op.drop_constraint('child_parent_id_fkey', 'child', type_='foreignkey')
    op.create_foreign_key(
        'child_parent_id_fkey', 'child', 'parent',
        ['parent_id'], ['id'],
        ondelete='CASCADE',  # new value
    )

def downgrade() -> None:
    op.drop_constraint('child_parent_id_fkey', 'child', type_='foreignkey')
    op.create_foreign_key(
        'child_parent_id_fkey', 'child', 'parent',
        ['parent_id'], ['id'],
        ondelete='SET NULL',  # old value
    )
```

---

## 5. Validating Your Config

### Test Template

```python
import pytest
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

@pytest.mark.asyncio
async def test_cascade_delete_removes_children(session: AsyncSession) -> None:
    parent = await Parent().save(session)          # must use the return value: the original object is expired after commit
    parent_id = parent.id
    child = await Child(parent_id=parent_id).save(session)
    child_id = child.id

    await Parent.delete(session, parent)

    # Use raw SQL to bypass the ORM identity map / Redis cache, verify DB directly
    remaining = (await session.execute(
        text("SELECT COUNT(*) FROM child WHERE id = :id"),
        {'id': str(child_id)},
    )).scalar()
    assert remaining == 0, "child should be deleted with parent"
```

For SET NULL intent: assert `FK IS NULL` instead. For RESTRICT (PostgreSQL): use `with pytest.raises(ResourceReferencedError): await Parent.delete(session, parent)` instead.

---

## Can't Find Your Case?

This guide covers one-to-many (parent→children) in its main forms. If you're configuring:

- **Many-to-many** → put `ondelete` on the SecondaryTable, Relationship gets `secondary=table_obj`
- **Self-referential** (e.g. `Conversation.compacted_from`) → same rules apply but SA cycle detection comes into play — see the cycle discussion in the [Explanation](/en/explanation/cascade-delete-semantics)
- **One-to-one** → semantically same as one-to-many, just add `sa_relationship_kwargs={'uselist': False}`

If your case is complex enough to exceed the cheat sheet, read the full matrix in the [Explanation](/en/explanation/cascade-delete-semantics).
