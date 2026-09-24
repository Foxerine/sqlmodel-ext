# Cache transparency inside transactions

::: tip Source location
`src/sqlmodel_ext/mixins/cached_table.py` — `_tables_with_uncommitted_writes`, `_query_dependency_tables`, `_has_uncommitted_writes`, `register_raw_dml_write`, `_adopt_cached_instance`

`src/sqlmodel_ext/session.py` — observation points of the enhanced `AsyncSession`: `commit()` / `refresh()` / `execute()` etc.
:::

For a shared cache to achieve "business code observes exactly the same results with or without it", the hardest part is not invalidation but **transactions**. This chapter explains why, and how, `CachedTableBaseMixin` stays transparent inside a transaction. The overall cache structure (dual-layer keys, version numbers, serialization) is covered in [Redis cache mechanism](./cached-table).

## The problem: a query result inside a transaction is not any committed state

The Redis cache is shared by all sessions, while a query result inside a transaction includes **the transaction's own uncommitted writes** (every query that hits the database autoflushes first). If such a result were written into the cache, two kinds of errors would appear:

| Error | Scenario |
|-------|----------|
| **Publishing uncommitted data** | Transaction A changes `name='Alicia'` and queries once before committing; the result is written into Redis. A then rolls back — but other requests have already read `'Alicia'`, a value that never existed, from the cache |
| **Dirty-reading its own old value** | Transaction A modifies a row, then queries. A cache hit returns the committed value from **before** the modification, and A can't read what it just wrote |

"Publish at commit time" is not right either: a snapshot taken mid-transaction may differ from the finally committed value (changed again before commit, some savepoint rolled back). The only correct handling is **not to publish**: as long as a table this query depends on has uncommitted writes in this transaction, this `get()` **neither reads nor writes the cache**.

## The decision: dependency tables ∩ uncommitted-write tables

The decision is made once, **before** the query, and is the sole basis for "can the cache be used":

```python
skip_cache = cls._has_uncommitted_writes(
    session,
    cls._query_dependency_tables(condition, filter, order_by, load),
)
```

### Which tables this query depends on

`_query_dependency_tables` returns:

- the tables mapped by the returned model **and all its subclasses** (STI shares one table, JTI also has child tables)
- every table referenced in `condition` / `filter` / `order_by`, **including tables in subqueries, aliases and CTEs**
- the tables of every `load` target (and its subclasses)

The scope can't be just "the returned model's family": a condition like `owner_id IN (SELECT owner.id WHERE name = <uncommitted value>)` makes the result change with uncommitted writes to the `owner` table while the returned model itself was never touched. The same goes for `load` targets — the write side would publish the target's uncommitted payload into its own ID cache.

**fail-closed**: if the expression contains untraceable parts (`text()`, `literal_column`, bare `column()`, lightweight `table()` without metadata — all of which can reference any table by name), the dependency set is **every table** in the metadata: any uncommitted write makes this query skip the cache.

### Which tables this transaction has written

`_tables_with_uncommitted_writes` merges three sources, and none can be omitted:

| Source | Covers |
|--------|--------|
| pending invalidations registered by CRUD methods | `save` / `update` / `delete` / cascade deletes |
| `session.new` / `dirty` / `deleted` | bare `session.add()`, direct attribute mutation, `session.delete()` — the not-yet-flushed part |
| the "flushed but uncommitted" table set | writes already sent over the database connection but not yet committed: ORM flushes (`after_flush` event) and raw DML (`register_raw_dml_write`) |

The first two only cover "not yet flushed". After an autoflush the object looks clean while the transaction is still uncommitted — so the third source is indispensable. This set is cleared only when the **outermost transaction ends** (the common exit of commit / rollback / `close()` / `reset()` / `invalidate()`), and is kept after a savepoint rollback (overly strict, but in the safe direction: for the rest of this transaction, queries on these tables keep skipping the cache).

### Decided per table, not per transaction

The granularity of the decision is the **table**: modifying model A doesn't make model B's queries skip the cache (preserving the hit rate), unless B's query condition references A's table (preserving correctness).

```python
other = await Character.get(session, Character.name == 'Alice')
other.name = 'Alicia'                                  # unflushed modification
await Character.get(session, Character.id == cid)      # skips the cache: the character table has uncommitted writes
await Owner.get(session, Owner.id == oid)              # still uses the cache: the owner table wasn't touched
```

## Raw DML must be registered

`update(...)` / `delete(...)` / `insert(...)` / `text(...)` don't go through ORM state — they don't enter `new/dirty/deleted` and don't trigger `after_flush`. Without registration, queries in the same transaction that depend on that table would publish uncommitted state into the shared cache.

The enhanced `AsyncSession` registers automatically at **five** entry points: `execute` / `exec` / `scalar` / `stream` / `stream_scalars` (in SQLAlchemy's `AsyncSession` the last three don't go through `execute`; `scalars()` does, so it needs no separate hook). Rules:

- `update` / `delete` / `insert`: register the target table
- `text()`: a first keyword among `SELECT` / `SHOW` / `SET` is treated as read-only; everything else registers **all tables** (fail-closed). `EXPLAIN` is deliberately not in the read-only set — `EXPLAIN ANALYZE UPDATE ...` really performs the write in PostgreSQL
- an `INSERT` into a cached table additionally registers a query-level invalidation for every cached class of that table (new rows cannot be in an ID cache, but cached query results may now be incomplete), so after commit no caller action is needed
- an `UPDATE` / `DELETE` hitting a cached table logs an extra warning if none of the cached classes for that table has registered a pending invalidation (it bypassed cache invalidation) — for these, register the affected IDs yourself (below)

Known blind spots: `text()` writes get only the in-transaction registration above, no post-commit invalidation; writes nested inside a `SELECT` (e.g. a writable CTE) are not top-level DML and are not registered at all — after either, call `invalidate_on_commit` / `invalidate_all` yourself; writes executed directly on the raw connection, bypassing the session, are invisible; many-to-many `secondary` tables are not in `mapper.tables`, so association rows written through a relation collection's `append` are registered by neither ORM source; the ORM doesn't know about child tables deleted by a database-level `passive_deletes='all'` cascade.

### The correct way to do raw DML: `invalidate_on_commit`

```python
Character.invalidate_on_commit(session, cid)           # register first: after commit, invalidate id:Character:<cid> + the query cache
await session.execute(
    update(Character).where(col(Character.id) == cid).values(name='Bob')
)
await session.commit()                                  # the enhanced commit synchronously runs the registered invalidation
```

Register first, then execute, and the warning won't fire (the table already has a registration). Invalidation must happen **after** commit: invalidating earlier would let a concurrent reader refill the old value into the cache before the commit lands. `invalidate_on_commit` only accepts real primary keys; rollback / `reset()` discard the registration, so retry paths must call it again.

## Invalidation happens only after commit, at a single orchestration point

CRUD methods (`save` / `update` / `delete` / `add`) **don't invalidate** themselves; they only register pending invalidations into `session.info`. The enhanced `AsyncSession.commit()` orchestrates:

1. Every flush registers the cached models it writes (the `after_flush` event), covering bare `add` / attribute mutation / `delete` that bypass the CRUD methods -- whether they were flushed earlier (inside a savepoint, by a manual `flush()`, by an autoflush) or by this commit
2. Actually commit. The `after_commit` event pops the complete set of registered items — including those registered by this commit's own flush, such as cascade-deleted children — and hands it over to `commit()`
3. Synchronously invalidate the handed-over items, each exactly once
4. Run post-commit callbacks

With `commit=False` nothing is invalidated (the data isn't committed yet); the registered items wait for your final `session.commit()` — "several `commit=False` operations + one commit" is naturally correct. Rollback discards the registered items.

For a commit that doesn't go through the enhanced `commit()` (e.g. a plain sqlmodel session, or `run_sync(lambda s: s.commit())`), the `after_commit` event instead schedules a fire-and-forget compensation task that invalidates the popped items and logs a `WARNING` "fallback compensation triggered: ..." — there is a brief stale window, TTL provides eventual consistency. The same fallback takes over if the enhanced `commit()` fails or is cancelled after the database has committed. A commit through the enhanced session never logs that warning.

### Savepoints

A savepoint RELEASE also triggers `after_commit`, but at that point the data has merely been merged into the outer transaction; it is still invisible to other sessions, and the outer transaction may still roll back. So savepoint-level commits / rollbacks **neither consume nor discard** registered items; everything is deferred to the outermost transaction: registered items are kept on savepoint rollback (the outer commit invalidates one extra time — safe, since the cache is a rebuildable copy).

## A hit does not overwrite state in the identity map

The object deserialized on a cache hit has to be merged into the session. An unconditional `session.merge()` would copy the cached state onto an existing object in the identity map **without checking whether it has uncommitted modifications** — silently losing data, with freshness depending on "whether a cache entry happened to exist". `_adopt_cached_instance` makes the hit and miss paths treat existing objects the same way:

| State of the existing object in the identity map | Handling |
|--------------------------------------------------|----------|
| no unloaded columns | return it directly — the same standard SQLAlchemy semantics as the miss path, no merge |
| has unloaded columns, no loaded non-primary-key columns, and no uncommitted history | `merge` to fill it in (the common state where everything expired after commit; saves a query) |
| has unloaded columns and also loaded non-primary-key columns (or has uncommitted history) | give up this hit and fall back to the database |

The third case can't be "filled in" from the cached payload: the loaded columns may be **newer** than the payload (e.g. the object was just read authoritatively, then only one other column was expired). Callers that need the latest values should not rely on this — use `authoritative=True`.

## `refresh()` never goes through the cache

`session.refresh(obj)` means "discard in-memory modifications and re-read from the database". The enhanced session delegates straight to the native implementation (expire → SELECT → `ObjectDeletedError` when the row doesn't exist), without going through Redis.

## Other switches that bypass the cache

| Condition | Reason |
|-----------|--------|
| `authoritative=True` | an authorization read must see the latest committed row; also bypasses the identity map |
| `with_for_update=True` | a pessimistic lock must read the latest row |
| `populate_existing=True` | the caller explicitly asks to refresh the identity map |
| non-empty `options` | an `ExecutableOption` may change loading behavior and can't be stably represented in the key |
| non-empty `join` | changes to the JOIN target don't invalidate the main model, which would produce phantom reads |
| `no_cache=True` | the caller explicitly opts out |

## The cache key includes every input that changes the result

The query cache key is a hash of the normalized conditions, pagination, sorting, `load`, `filter` and time filters (`table_view` is first merged into the explicit parameters, so semantically identical queries get the same key). The keyset cursor `after_id` is **also in the key**: different cursors are different pages, and without it two cursors would share one cache entry.

## Related references

- [Redis cache mechanism](./cached-table)
- [Cache queries with Redis](/en/how-to/cache-queries)
- [`CachedTableBaseMixin` API](/en/reference/mixins#cachedtablebasemixin)
