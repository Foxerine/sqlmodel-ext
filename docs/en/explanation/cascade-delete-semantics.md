# Cascade Delete Semantics (cascade × passive_deletes × ondelete)

::: tip This is Explanation
This chapter answers **why** certain combinations of these three parameters produce surprising results. If you just want to know which combination to use, see [How-to: Configure Cascade Delete](/en/how-to/configure-cascade-delete).
:::

## Why This Chapter Exists

SQLModel / SQLAlchemy cascade delete is controlled by three **orthogonal** primitives:

- **`cascade_delete`** (Relationship param) — does the ORM propagate "delete" to children?
- **`passive_deletes`** (Relationship param) — does the ORM actively load children, or trust the DB?
- **`ondelete`** (Foreign Key param) — what does the DB do to children when the parent is deleted?

These combine into 18 usable cells. **Most of them do not behave the way you think.** This chapter pins down the actual behavior of every cell based on 53 controlled experiments.

## The Three Primitives, One at a Time

### `cascade_delete`: ORM-Side Delete Propagation

In `sqlmodel-ext` (and SQLModel / SQLAlchemy):

- `cascade_delete=False` (default) → underlying cascade string is `'save-update, merge'` — **no** delete semantics
- `cascade_delete=True` → underlying cascade string is `'all, delete-orphan'` — **includes** delete and delete-orphan

### `passive_deletes`: Let the DB Handle It?

- `passive_deletes=False` (default) → SA **actively** intervenes: either emits UPDATE to null child FKs, or SELECTs children then DELETEs them, depending on cascade
- `passive_deletes=True` → SA **hands off**: doesn't load children, emits only parent DELETE, rest is the DB's job
- `passive_deletes='all'` → even more passive: SA won't even emit UPDATE to null FKs

### `ondelete`: DB-Side Strategy

- `'NO ACTION'` (Postgres default) — reject the delete if FK references still exist (`ForeignKeyViolationError`)
- `'CASCADE'` — delete child rows along with the parent
- `'SET NULL'` — set child FKs to NULL (child rows survive)
- `'RESTRICT'` — same as NO ACTION but not deferrable

When `NO ACTION` / `RESTRICT` rejects a delete, `TableBaseMixin.delete()` on PostgreSQL translates that `IntegrityError` into `ResourceReferencedError` (409 semantics: the row exists and is still referenced). A foreign-key violation has two directions at the driver level ("points at a non-existent parent row" and "parent row is still referenced") with exactly the same exception, so the translation only happens when "the error is caught inside `delete()` **and** the failing statement is a `DELETE`". See [Handle deletes of still-referenced rows](/en/how-to/handle-referenced-deletes).

---

## The 18-Cell Matrix

All scenarios use one parent row and one child row with a nullable FK. We delete the parent and check the child.

```
                        cascade_delete=True      cascade_delete=False
ondelete   passive_del=  F    T    'all'       F    T   'all'
─────────────────────────────────────────────────────────────────
NO ACTION               OK   ERR   BAN        UPD   ERR   ERR
CASCADE                 OK   OK    BAN        UPD   OK    OK
SET NULL                OK   ⚠️    BAN        UPD   SN    SN
```

Legend:

- **OK**: parent and child both deleted
- **UPD**: SA emits an UPDATE that nulls the child's FK, then deletes parent — **child survives**
- **SN**: DB's SET NULL kicks in — child survives with FK=NULL
- **ERR**: `IntegrityError` (FK violation)
- **⚠️**: **semantic-conflict landmine** — SA's config intends deletion of the child, but the DB's SET NULL runs first — child becomes orphan
- **BAN**: SA rejects this combination at schema construction (`ArgumentError: can't set passive_deletes='all' in conjunction with 'delete' or 'delete-orphan' cascade`)

## Cell-by-Cell Interpretation

### Right Half: `cascade_delete=False`

**Important fact: SA silently emits UPDATEs here.** Most people think `cascade_delete=False` means "SA does nothing". Not quite.

The `'save-update, merge'` cascade doesn't include delete semantics, but SA's default behavior is "avoid data loss" — before deleting the parent, it UPDATEs all children to null their FKs. That way, parent deletion doesn't break DB FK integrity, and children survive as orphans.

| OD | passive_del=False | passive_del=True | passive_del='all' |
|----|-------------------|------------------|-------------------|
| NO ACTION | **UPD**: SA nulls FKs first, bypassing the DB's NO ACTION check | **ERR**: SA hands off, DB rejects deletion with referencing children | **ERR**: same |
| CASCADE | **UPD**: SA preempts with UPDATE, DB's CASCADE never fires | **OK**: DB CASCADE does it | **OK**: same as True |
| SET NULL | **UPD**: SA actively UPDATEs | **SN**: DB's SET NULL does it | **SN**: same as True |

**To stop SA from emitting that implicit UPDATE?** Set `passive_deletes=True` (or `'all'`). These are equivalent in this column — `'all'` only differs from `True` when a delete cascade is involved, and that combination is banned (see right-bottom).

### Left Half: `cascade_delete=True`

| OD | passive_del=False | passive_del=True | passive_del='all' |
|----|-------------------|------------------|-------------------|
| NO ACTION | **OK**: SA actively SELECTs children → DELETEs them → DELETEs parent | **ERR**: SA hands off, DB rejects | **BAN** |
| CASCADE | **OK**: SA actively deletes, DB CASCADE redundant | **OK**: one DB-level CASCADE — optimal ✅ | **BAN** |
| SET NULL | **OK**: SA actively deletes (overriding DB's SET NULL intent) | **⚠️ landmine**: SA doesn't touch, DB runs SET NULL, child becomes orphan instead of being deleted | **BAN** |

**The most important cell**: `cascade_delete=True, passive_deletes=True, ondelete='SET NULL'` is a **configuration conflict**. The ORM layer says "delete the child when deleting parent"; the DB layer says "null out the child's FK when deleting parent". Adding `passive_deletes=True` lets the DB win — child survives with NULL FK, the opposite of your ORM intent.

**Why `'all'` + delete cascade is banned**: `'all'` means "SA does nothing"; delete cascade means "SA actively deletes children". Logically contradictory. SA raises `ArgumentError` at schema construction — this is a guardrail.

---

## Why `raise_on_sql` (Usually) Doesn't Fire During Cascade

`sqlmodel-ext` sets `lazy='raise_on_sql'` as the default on every Relationship, to prevent implicit lazy loads in async contexts. Intuitively, you might worry that during cascade delete, SA would probe the child's m2o relationships — like `Conversation.project` — and trigger `raise_on_sql`.

Empirically, **cascade delete paths don't automatically trigger `raise_on_sql`** (in minimal SA environments, across 53 scenarios spanning various child m2o, FK NULL/not-null, identity map states, chain depths 1/2/3, back-ref loading states).

Source-level explanation, in `sqlalchemy/orm/dependency.py`, `ManyToOneDP.per_state_flush_actions`:

```python
sum_ = state.manager[self.key].impl.get_all_pending(state, dict_)
# get_all_pending defaults to passive=PASSIVE_NO_INITIALIZE
```

`PASSIVE_NO_INITIALIZE`'s bit flag does **not** include `SQL_OK`. And `raise_on_sql` fires only when:

```python
def _invoke_raise_load(self, state, passive, lazy):
    if not passive & PassiveFlag.SQL_OK:
        return  # don't raise
    raise sa_exc.InvalidRequestError(...)
```

No `SQL_OK` → early return → no raise. Cascade-time cycle resolution uses a "peek at current state without emitting SQL" mode. **`raise_on_sql` only fires when your code explicitly accesses an attribute** (`for msg in conv.messages`).

### When CAN cascade trigger it?

Not reproducible in vanilla SA. But certain infrastructure combinations could cause `SQL_OK` to be set unexpectedly:

- Custom metaclass injecting attribute access during mapper configuration
- Event listeners (`persistent_to_deleted`, `after_flush`) accessing Relationship attributes
- PostgreSQL triggers calling back into SA
- Edge cases in async greenlet passive flag context propagation

If you encounter a cascade-time `raise_on_sql` in your project, these are the investigation leads.

---

## Why the Recommended Config Is What It Is

Adding `passive_deletes=True` to every `cascade_delete=True` relationship has three benefits:

1. **Performance**: one `DELETE FROM parent WHERE id = :id` hits the DB, which walks the built-in CASCADE tree; otherwise SA first `SELECT * FROM child WHERE parent_id = :id` into Python, then emits N `DELETE FROM child WHERE id = :id`
2. **Fewer attribute accesses**: SA doesn't load children into the session, so there's no opportunity to trigger `raise_on_sql` on any of their Relationships
3. **Semantic consistency**: you're forced to explicitly declare `ondelete='CASCADE'` at the DB layer, avoiding "ORM says one thing, DB says another" drift

Prerequisite: the FK's `ondelete` actually **is** `'CASCADE'` — if it's `'SET NULL'`, this default lands you in the `⚠️` cell.

---

## Empirical Data Source

All claims in this chapter come from 53 independent scenarios. Each uses its own engine, session, schema, and isolated `Base(DeclarativeBase)`, covering:

- Series A (18): cascade × passive × ondelete baseline matrix
- Series B (18): add child.external m2o with `raise_on_sql` probes
- Series BP (6): preload child into identity map variants
- Series C (3): back-ref-only m2o probes
- Series D (3): 3-level chain cascade
- Series E (5): non-delete edge cases (direct access, FK modify, m2o swap, etc.)

Experiment code and raw results are not included in `docs/internals/` — reproduce with a hand-rolled vanilla SA + DeclarativeBase + `lazy='raise_on_sql'` setup.
