# Migrate from 0.4.x to 0.5.0

::: danger The project is still WIP / alpha
sqlmodel-ext is still evolving quickly. **Any two versions may contain breaking changes**; the project gives no stability or backward-compatibility guarantee, and you use it at your own risk. Pin the version before upgrading and read this page and the [CHANGELOG](https://github.com/Foxerine/sqlmodel-ext/blob/master/CHANGELOG.md) in full.
:::

**Goal**: upgrade a project built on sqlmodel-ext 0.4.x to 0.5.0.

Every item follows the same format: **change** → **why** → **how to migrate** (before / after) → **database migration** (when needed). Only item 3 needs a database migration.

## 0. Dependencies

| Dependency | 0.4.x | 0.5.0 |
|---|---|---|
| `pydantic` | `>=2.0` | **`>=2.12`** (`Unset` is `pydantic.experimental.missing_sentinel.MISSING`, and `Field(exclude_if=...)` behind `EXCLUDE_IF_NONE`; both first shipped in 2.12) |
| `typing-extensions` | transitive | **`>=4.14.1`** (`typing_extensions.Sentinel` is imported directly) |
| `sqlmodel` | `>=0.0.32` | unchanged |
| New optional extra | — | `alembic`: lets `run_pending_migration_cache_invalidations` discover pending tasks from alembic revision scripts |

Type checker (optional but strongly recommended): narrowing `Unset | T` statically needs basedpyright ≥ 1.40.1 / pyright ≥ 1.1.414, see [Type-check with basedpyright](./type-check-with-basedpyright).

## 1. `all_fields_optional=True` → `partial=True`

**Change**: the class keyword `all_fields_optional` was removed; using it raises `TypeError` at class creation (the message includes the migration notes below). Its replacement `partial=True` has different semantics:

| | 0.4.x `all_fields_optional` | 0.5.0 `partial=True` |
|---|---|---|
| Inherited fields become | `T \| None = None` | `Unset \| T = Unset` (`Unset \| T \| None` when the base is nullable) |
| A field that was not sent | `None` | `Unset` (absent from `model_dump()`) |
| `null` for a non-nullable field | passes validation, often a 500 at the database NOT NULL constraint | **rejected at validation (422)** |
| `default_factory` | — | not called (not sent means `Unset`) |
| Combined with `table=True` | — | `TypeError` |

**Why**: `None` meant both "not sent" and "clear", which need opposite handling, and conventions (`exclude_unset=True`, `is not None`) are always forgotten somewhere. See [Unset](/en/explanation/unset-three-state).

**How to migrate**:

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
data = body.model_dump()                 # exclude_unset=True is no longer needed
assert data == {'subtitle': None}
if body.title is not Unset:              # check "was it sent" with is not Unset, not is not None
    raise AssertionError("title was not sent")
```

Checklist:

- search for `all_fields_optional` and replace it with `partial=True`;
- wherever a partial DTO is used, change "was it sent" checks from `is None` / `is not None` to `is Unset` / `is not Unset`;
- `instance.update(session, body)` needs no change — it already writes only the fields in `model_fields_set`, and `Unset` fields are never dumped;
- clients that relied on "`null` means don't change" must switch to "don't send"; `null` for a non-nullable field is now a 422;
- no database migration.

## 2. Declaring `oplock_version` in a class body raises `TypeError`

**Change**: `oplock_version` is a reserved name in every `SQLModelBase` subclass — whether or not the class enables optimistic locking, declaring it in **its own class body** raises `TypeError` at class creation. Only `OptimisticLockMixin` defines it.

**Why**: if the name were reserved only for classes with locking enabled, a class without the lock could declare a domain field of that name, which a descendant re-enabling the lock would then silently wire up as `version_id_col`.

**How to migrate**: give a domain "version number" another name (e.g. `version`, `revision`) — since 0.5 `version` is no longer taken by optimistic locking and is free to use.

## 3. Optimistic-lock column `version` → `oplock_version` (database migration required)

**Change**:

| | 0.4.x | 0.5.0 |
|---|---|---|
| Column name | `version` | `oplock_version` |
| Column type | `INTEGER` | `BIGINT` (upper bound `JS_MAX_SAFE_INTEGER`) |
| Database default | none | `server_default 0` |
| In `model_dump()` | present | **absent** (`exclude=True`) |
| Conflict retries in `save()` / `update()` | 0 by default | **3** by default (see item 4) |

**Why**:

- the rename: `version` is a very common domain field name, and having it taken by the lock forced awkward names on business fields;
- `BIGINT`: removes any realistic risk of `INTEGER` overflow on frequently updated rows;
- `server_default`: writers that do not know the column (raw SQL, other services, fixtures) omit it on INSERT; a database default keeps those INSERTs valid. It does **not** make the 0.4 → 0.5 rename rolling-compatible — see the deployment notes below;
- `exclude=True`: it is ORM-internal state and must not leak into `model_dump()` (e.g. into response DTOs with `extra='forbid'`).

**How to migrate**:

- replace reads of `obj.version` with `obj.oplock_version`;
- inheritance order is unchanged: `OptimisticLockMixin` must come before `TableBaseMixin` / `UUIDTableBaseMixin`;
- if your API used to expose `version` to clients, it no longer appears in `model_dump()`; since `oplock_version` is reserved, the response DTO needs another field name assigned explicitly.

```python
from sqlmodel_ext import OptimisticLockMixin, SQLModelBase, Str64, UUIDTableBaseMixin


class Order(SQLModelBase, OptimisticLockMixin, UUIDTableBaseMixin, table=True):
    status: Str64 = 'pending'


order = Order()
assert order.oplock_version == 0
assert 'oplock_version' not in order.model_dump()
assert Order.__optimistic_retry_default__ == 3
```

**Database migration (PostgreSQL, table `order` as an example)**:

```sql
ALTER TABLE "order" ALTER COLUMN version TYPE BIGINT;
ALTER TABLE "order" ALTER COLUMN version SET DEFAULT 0;
ALTER TABLE "order" RENAME version TO oplock_version;
```

The equivalent Alembic operation (the SQL above is what it generates for the PostgreSQL dialect):

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

::: warning Deployment notes
- **The rename cannot run alongside old code**: after the migration, still-running 0.4.x instances cannot find the `version` column; before it, 0.5.0 instances cannot find `oplock_version`. Run it in a maintenance window or a blue/green switch so the migration and the code switch happen together.
- `INTEGER → BIGINT` **rewrites the whole table** on PostgreSQL and holds an `ACCESS EXCLUSIVE` lock; estimate the time for large tables.
- Run it for every table using `OptimisticLockMixin` (for STI/JTI, the root table).
- **Clear the Redis cache when you switch** (if you use `CachedTableBaseMixin`): cache keys are the same in 0.4.x and 0.5.0, and an entry written by 0.4.x can validate under 0.5.0 with a different meaning — e.g. a model that now declares a domain `version` field would read the old lock counter into it. Stopping the old instances does not remove their entries, so flush the cache (or call `await Model.invalidate_all(strict=True)` for each cached model) before the 0.5.0 instances serve traffic.
:::

## 4. `OptimisticLockMixin` retries 3 times by default; `delete()` conflicts raise `OptimisticLockError`

**Change**:

- the default of `optimistic_retry_count` in `save()` / `update()` changed from `0` to `None`, meaning "use the model policy `__optimistic_retry_default__`": **3** for `OptimisticLockMixin` models, 0 otherwise. An `update()` retry re-reads the latest row and re-applies the caller's changes on top.
- on an optimistic-lock conflict, `delete()` no longer lets the raw `StaleDataError` escape; it raises `OptimisticLockError` (`delete()` never retries — "someone just changed it, do I still want to delete it?" is the caller's decision).

**Why**: the policy belongs where the capability is declared, not in every call site remembering an argument; a conflict should surface as one exception type.

**How to migrate**:

- call sites that need the old behavior (fail immediately on conflict) pass `optimistic_retry_count=0` explicitly;
- `delete()` call sites catching `StaleDataError` catch `OptimisticLockError` instead.

## 5. `delete()` raises `ResourceReferencedError` on FK RESTRICT

**Change**: deleting a row that is still referenced through a foreign key used to let `IntegrityError` escape from `delete()`; it now raises `ResourceReferencedError` (with `status_code = 409`, `friendly_message`, `constraint_name`). It is **not** a subclass of `IntegrityError`.

**Why**: the same FK violation has two directions ("the referenced target does not exist" is 404 semantics, "still referenced, cannot delete" is 409), which the raw exception cannot tell apart; `delete()` converts only when the error really was caused by the DELETE statement.

**How to migrate**: call sites catching `IntegrityError` to handle "still referenced" catch `ResourceReferencedError` instead (`from sqlmodel_ext.mixins import ResourceReferencedError`). Register a friendly message with `TableBaseMixin.register_fk_delete_restrict_message(constraint_name, message)`.

## 6. Default primary keys are UUIDv7

**Change**: the default factory of `UUIDTableBaseMixin.id` and of the JTI child primary keys generated by `create_subclass_id_mixin()` changed from `uuid.uuid4` to UUIDv7 (`sqlmodel_ext.mixins.uuid7`; on Python 3.14+ this is `uuid.uuid7`).

**Why**: the first 48 bits of a UUIDv7 are a millisecond timestamp, so byte order is creation-time order: `ORDER BY id` approximates creation order and B-tree inserts concentrate on the right edge, with far fewer page splits and random I/O than UUIDv4.

**How to migrate**:

- **existing data needs no migration**: old rows keep their v4 values; mixed v4/v7 values still have a well-defined byte order (e.g. for lock ordering). The column type is unchanged.
- note that the id reveals its creation time with millisecond precision; ids are identifiers, not capabilities — do not rely on them being unguessable.
- do not treat id order as authoritative time order (the timestamp comes from the application clock, with no global monotonicity across processes).
- if a primary key must be derived deterministically (e.g. `uuid5` from an idempotency key), assign it explicitly — the default factory only applies when no id is given.

```python
from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin


class Note(SQLModelBase, UUIDTableBaseMixin, table=True):
    pass


assert Note().id.version == 7
```

## 7. `TimeFilterRequest` accepts only timezone-aware `datetime`s

**Change**: `created_after_datetime` / `created_before_datetime` / `updated_after_datetime` / `updated_before_datetime` changed from `datetime` to `AwareDatetime`; naive values fail validation (`TableViewRequest` inherits this).

**Why**: a naive value cannot be compared with the timezone-aware values in the database; it would be silently interpreted in the database's timezone and the query would be off by hours with no error.

**How to migrate**: clients send `2026-01-01T00:00:00Z` or `2026-01-01T08:00:00+08:00`, not `2026-01-01T00:00:00`.

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

Also, `PaginationRequest.offset` now has an upper bound `MAX_TABLE_VIEW_OFFSET` (`JS_MAX_SAFE_INTEGER - 1000`); a huge offset is rejected at validation instead of failing in the database driver with `bigint out of range`.

## 8. `@requires_for_update` fails when it cannot find the session

**Change**: when no `AsyncSession` can be found in the call arguments of the decorated method, 0.4.x silently skipped the check; 0.5.0 raises `RuntimeError` (fail-closed).

**Why**: not finding the session is exactly the signal that the guard stopped working (e.g. an inner decorator without `@wraps` hides the `session` parameter from `inspect.signature`); it must not look like "check passed".

**How to migrate**: make sure the method has a parameter named `session` and that every inner decorator uses `functools.wraps`.

## 9. Renamed / removed internal methods

| 0.4.x | 0.5.0 |
|---|---|
| `RelationPreloadMixin._ensure_relations_loaded` | `ensure_relations_loaded` (public; plus a batch variant `ensure_relations_loaded_bulk`) |
| `CachedTableBaseMixin._warn_raw_dml_on_cached` | `register_raw_dml_write` (besides warning, it registers the tables written by raw DML as "written but uncommitted") |
| `CachedTableBaseMixin._refresh_via_cache` | removed |
| `CachedTableBaseMixin._has_pending_invalidation` | removed |

These were underscore-private; if your code called them, replace them as above.

## 10. Session and read semantics

- **`AsyncSession.refresh()` no longer goes through the cache**: in 0.4.x a whole-object refresh of a cached model went through `Model.get()` and Redis; in 0.5.0 it always delegates to the native `session.refresh()` and reads the database. A refresh means "discard in-memory state and re-read the database".
- **`with_for_update=True` forces `populate_existing`**: a locking read gets the latest row from the database, but the identity map could hand back an already-loaded object with stale attributes — and the subsequent read-modify-write would lose an update. Locking reads now always overwrite with the database values; there is no opt-out.

Usually no code change is needed; if your logic relied on "a locking read keeps uncommitted in-memory changes", it was already wrong.

## 11. Behavior changes in field types

| Type | Change | How to migrate |
|---|---|---|
| `JSON100K` / `JSONList100K` | **Outbound**: `model_dump()`, `model_dump(mode='json')` and `model_dump_json()` all output the object / array itself, no longer a JSON **string**. **Inbound**: dict / list input is now also subject to the 100K-character limit and the nesting-depth limit (0.4.x only checked string input); table models are checked at construction as well | Clients that treated the field in responses as a string and `JSON.parse`d it must change; an inbound JSON string is still accepted. Subclasses overriding `model_post_init` must call `super().model_post_init(context)`, otherwise the check is silently skipped |
| `PositiveFloat` / `NonNegativeFloat` | reject `inf` / `nan` (`AllowInfNan(False)`) | `1e309`, which used to be stored, is now a 422 |
| Decimal aliases (`SignedDecimal20_10`, …) | **integer-digit validation actually works**: the old metadata order made Pydantic check only total digits and decimal places, e.g. `SignedDecimal20_10` accepted a 15-digit integer, leaving the database as the only guard | values with too many integer digits are now rejected at validation |

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

(`JSON100K` needs `orjson`, i.e. `sqlmodel-ext[postgresql]`.)

## 12. RelationLoadChecker (experimental)

- More findings: the new RLC014 (a FastAPI `Depends` commits in its body, so ORM objects injected by sibling dependencies on the same session are already expired when the endpoint starts); session parameters are recognized by **subclass** (including `sqlmodel_ext.AsyncSession`) — the old identity check missed them; commit semantics are configurable through the module-level `conditional_commit_methods` / `explicit_commit_methods` / `dependency_commit_methods`.
- `# noqa: RLCxxx` now works on **every** public entry point (`check_app` / `check_model_methods` / `check_project_coroutines` / `check_function`); an endpoint's noqa comment must be on its **first decorator line**.

The first run after upgrading may report new warnings — most likely real issues that used to be missed.

## Upgrade checklist

- [ ] `pydantic>=2.12`
- [ ] search `all_fields_optional` → `partial=True`; `is None` / `is not None` checks on partial DTOs → `is Unset` / `is not Unset`
- [ ] no `oplock_version` in class bodies
- [ ] run the rename + `BIGINT` + `DEFAULT 0` migration for every optimistic-lock table; `.version` → `.oplock_version` in code
- [ ] pass `optimistic_retry_count=0` where "fail immediately on conflict" is needed; catch `OptimisticLockError` / `ResourceReferencedError` around `delete()`
- [ ] clients of time-filter parameters send a timezone
- [ ] every `@requires_for_update` method has a `session` parameter and inner decorators use `@wraps`
- [ ] replace calls to renamed / removed private methods
- [ ] clients consuming `JSON100K` fields read the object directly
- [ ] run basedpyright once (see [Type-check with basedpyright](./type-check-with-basedpyright))
