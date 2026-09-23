# More mixins: quotas, fuzzy search, cross-table scans, migration cache invalidation

This page collects four capabilities you mix in as needed. All of them are imported from `sqlmodel_ext.mixins`.

## `ResourceQuotaMixin`

**Goal**: limit "each owner may own at most N of some resource", and never exceed it even under concurrent creation.

### 1. Declare the quota rules

The host class mixes in `ResourceQuotaMixin` and implements three classmethods:

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
        # with_for_update must be forwarded: the preflight path passes False (no lock, may use the cache)
        return await Account.get(session, Account.id == owner_id, with_for_update=with_for_update)

    @classmethod
    def _quota_condition(cls, owner_id: UUID) -> ColumnElement[bool]:
        return col(cls.owner_id) == owner_id

    @classmethod
    def _quota_max(cls, owner: Account) -> int:
        return owner.max_projects
```

Counting defaults to `cls.count(session, _quota_condition(owner_id))`; override `_count_owner_resources(session, owner_id, owner)` when you need to count across multiple tables.

### 2. Acquire a slot before the INSERT

```python
async with Project.acquire_quota_lock(session, owner_id):
    await Project(owner_id=owner_id, name='p1').save(session)   # save commits → releases the owner row lock
```

- On entry, `SELECT ... FOR UPDATE` locks the owner row, serializing concurrent requests for the same owner; `current + count <= max` and the protected INSERT run atomically in the same transaction.
- Put all preparation work (validation, queries, in-memory computation) **before** the `async with`, so the lock window only covers "acquire → INSERT → commit".
- The block itself doesn't commit. If the block exits **normally** without a commit having happened, it raises `CallerDidNotCommitError` (the owner lock is still held — a programming error). When a larger outer transaction will commit later, pass `defer_commit=True` to turn this guard off.
- Creating several resources at once: `count=3` checks `current + 3 <= max` only once.

When over quota:

```python
from sqlmodel_ext.mixins import QuotaExceededError, QuotaOwnerNotFoundError

try:
    async with Project.acquire_quota_lock(session, owner_id):
        await Project(owner_id=owner_id, name='p3').save(session)
except QuotaExceededError as e:          # status_code = 400; e.max_allowed / e.current_count
    await session.rollback()
    raise HTTPException(e.status_code, detail=str(e))
except QuotaOwnerNotFoundError as e:     # status_code = 404
    raise HTTPException(e.status_code, detail="owner not found")
```

### 3. Preflight without a lock before expensive steps

```python
await Project.preflight_quota(session, owner_id)        # lock-free, cheap, racy soft gate
result = await call_paid_external_api(...)               # owners already over quota incur no cost
async with Project.acquire_quota_lock(session, owner_id):   # the atomic guarantee still comes from here
    await Project(owner_id=owner_id, name=result.name).save(session)
```

Compared with holding `FOR UPDATE` throughout the expensive step, the preflight doesn't serialize all concurrent requests for the same owner.

### 4. Concurrent get-or-create: `idempotent_check`

When the same "unique resource" is created concurrently, the later request should get the earlier request's row instead of a false "over quota" or a unique-constraint violation:

```python
async def find_existing() -> Project | None:
    # must read the database truth (pass no_cache=True for cached models)
    return await Project.get(session, (col(Project.owner_id) == owner_id) & (col(Project.name) == 'default'))

async with Project.acquire_quota_lock(session, owner_id, idempotent_check=find_existing) as existing:
    if existing is None:
        existing = await Project(owner_id=owner_id, name='default').save(session)
```

The callback runs after the owner lock is acquired and before counting; if it returns non-`None`, the quota check is skipped and the value is `yield`ed. This branch doesn't register the commit guard (there is nothing to commit), but the owner lock is held until your transaction ends — return quickly.

## `TrgmSearchableMixin` (PostgreSQL only)

**Goal**: fuzzy search by name / description (`?query=`), tolerant of typos, and able to use an index even for 2-character queries.

### 1. Database preparation (run once in a migration)

```python
from sqlmodel_ext.mixins import BIGRAM_FUNCTION_SQL

def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(BIGRAM_FUNCTION_SQL)                       # public.bigrams(text) -> text[]
    op.execute("CREATE INDEX ix_tag_name_trgm ON tag USING gin (name gin_trgm_ops)")
    op.execute("CREATE INDEX ix_tag_name_bigrams ON tag USING gin (public.bigrams(name))")
    op.execute("CREATE INDEX ix_tag_description_bigrams ON tag USING gin (public.bigrams(description))")
```

Search works without the indexes too; it just degrades to a sequential scan. Once any index depends on `BIGRAM_FUNCTION_SQL`, **don't change the function body again** — a functional index stores keys computed by the old definition, so changing the definition silently misses rows until `REINDEX`.

### 2. Declare searchable columns

```python
from typing import ClassVar
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin, Str64, Str256
from sqlmodel_ext.mixins import TrgmSearchableMixin

class Tag(SQLModelBase, UUIDTableBaseMixin, TrgmSearchableMixin, table=True):
    __trgm_text_columns__: ClassVar[tuple[str, ...]] = ('description',)
    name: Str64                    # __trgm_name_column__ defaults to 'name'
    description: Str256 = ''
```

- Name column: `ILIKE '%q%'` substring match **OR** the pg_trgm similarity operator `name % q` (threshold is `pg_trgm.similarity_threshold`, default 0.3)
- Text columns: `ILIKE` substring only (the similarity of long text to a short query is almost always tiny, so it carries no signal)
- `ILIKE` uses `autoescape=True`: `%` / `_` / backslashes in user input don't become wildcards
- Substring matching first narrows with bigram containment (`bigrams(col) @> bigrams(q)`) and then re-checks precisely with `ILIKE`, so 2-character queries (common in CJK text) can also use the GIN index

### 3. Use it in an endpoint

```python
from typing import Annotated
from fastapi import Depends
from sqlmodel_ext import query_dependency
from sqlmodel_ext.mixins import TrgmSearchRequest

@router.get("", response_model=ListResponse[TagResponse])
async def list_tags(
    session: SessionDep,
    table_view: Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))],
    search: Annotated[TrgmSearchRequest, Depends(query_dependency(TrgmSearchRequest))],
) -> ListResponse[Tag]:
    condition = search.apply_condition(Tag, col(Tag.name) != 'hidden')   # ANDed into your existing scope condition
    return await Tag.get_with_count(session, condition, table_view=table_view)
```

The client passes `?query=cat`. `TrgmSearchRequest.query` is `Str64 | None` (length-limited, rejects NUL bytes); when it's blank or `None`, `apply_condition` returns the condition you passed in unchanged. It only produces a `WHERE` condition and **never changes the ordering** — ordering is still decided by `table_view`.

Host contract: the searched columns use the database's **default collation**. Under a special collation `ILIKE` follows the column's collation while `bigrams(query)` follows the database default, and this asymmetry can make the bigram guard reject rows that should have matched.

## `MixinTableScanMixin`

**Goal**: the same field-level mixin (e.g. a `remote_file_key` added to N vendor tables) is attached to several unrelated physical tables, and you need to "query / update across all attached tables by some condition".

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

- Discovery is dynamic: a new table that mixes this in is covered automatically. When called from a concrete table class, it scans only that class (and its subclasses).
- De-duplicated by `__table__`: an STI family shares one physical table, so only the polymorphic root is kept (no discriminator filter, covering rows of all subclasses); JTI subclasses each have their own physical table and are unaffected.
- The condition is built by `condition_factory(table_cls)` with a type-safe expression — **string field names are not accepted**; this module knows no field names.

## Migration-driven cache invalidation

**Goal**: a migration changed the **meaning** of existing data (e.g. rescaling a numeric column); old entries in Redis still deserialize but carry wrong values — you need to precisely invalidate the affected models rather than a blunt `FLUSHDB` (which would also wipe unrelated keys such as sessions and rate limits).

### 1. Declare it in the migration file

```python
# at the top of alembic/versions/<revision>.py
CACHE_INVALIDATIONS: dict[str, list[str]] = {
    'price-rescale-v1': ['Product', 'OrderLine'],
}
```

- Name sentinels `<topic>-v<N>`: when the same topic needs invalidating again later, use a different `N` so it doesn't collide with the old sentinel.
- List the **concrete subclasses that actually write to the cache**: `invalidate_all()` only walks the MRO **upward** (itself + cached ancestors), so a parent class can't clear a subclass's namespace (the cache key is `id:<queried class name>:<pk>`).
- Use the Python class name (`__name__`), not the SQL table name.

When a declaration is needed:

| Change | Needed? | Reason |
|------|------|------|
| `ADD COLUMN` | No | Old cache entries pass validation with the field default |
| `DROP COLUMN` | No* | The extra key in old entries fails deserialization under `extra='forbid'`; `get()` deletes the bad key and falls back to the database (self-healing) |
| Change column type, same meaning | No | Converted by Pydantic, or self-heals after failing deserialization |
| Change column type **and rescale / re-encode values** | **Yes** | Old values pass validation but are wrong |
| Drop a column + add a column **backfilled with non-default values** | **Yes** | Old entries get the default instead of the backfilled value |
| `UPDATE` existing rows | **Yes** | Existing values changed |

\* Except for the "drop column + backfilled new column" case above. Rule of thumb: "old cache entries deserialize successfully but their meaning changed" must be declared; "fails to deserialize" or "meaning unchanged" is left to self-healing.

### 2. Run it at startup

```python
from sqlmodel_ext import SQLModelBase, CachedTableBaseMixin
from sqlmodel_ext.mixins import run_pending_migration_cache_invalidations

# after migrations have completed and after configure_redis():
CachedTableBaseMixin.configure_redis(redis_client)
await run_pending_migration_cache_invalidations(SQLModelBase)          # collected from alembic.ini
```

Collecting from Alembic requires the optional dependency `alembic` (when it's not installed the collection function raises `RuntimeError`). Without Alembic, or in tests, pass the tasks directly:

```python
await run_pending_migration_cache_invalidations(
    SQLModelBase,
    tasks={'price-rescale-v1': ['Product', 'OrderLine']},
)
```

- Each task checks the Redis sentinel `cache_invalidation:<sentinel>`: absent → run `invalidate_all(strict=True)` on each model and write the sentinel only if **all succeed**; present → skip (idempotent).
- If invalidating any model fails, the sentinel is not written and it is retried on the next startup; one failing task doesn't block the others.
- A model name that no longer exists counts as "nothing to invalidate", not a failure (otherwise it would be retried forever).
- When Redis is unavailable, it logs and continues starting up.
- Multiple workers starting at the same time: in the worst case both run the invalidation — the version bump + deletion are idempotent anyway.
- The `cache_invalidation:*` namespace is never touched by `invalidate_all()`, so sentinels stay forever; after a manual `FLUSHDB` the sentinels disappear together with the old cache, and rerunning is harmless.

## Related reference

- [Mixins reference: `ResourceQuotaMixin` / `TrgmSearchableMixin` / `MixinTableScanMixin` / migration cache invalidation](/en/reference/mixins#resourcequotamixin)
- [Cache queries with Redis](./cache-queries)
