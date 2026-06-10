# Define STI (single table inheritance) models

**Goal**: model "user files" as one parent class `UserFile` plus several subclasses (`PendingFile`, `CompletedFile`), all sharing a single database table, with subclass-specific fields stored as nullable columns.

**When to choose STI**:

- Subclasses add only 1–2 distinct fields each
- You want to avoid JOINs (single-table queries are faster)
- You can accept a few NULL columns in the table

If subclasses differ a lot (5+ exclusive fields), pick [JTI](./define-jti-models).

## 1. Define the parent class

```python
from datetime import datetime
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    NonNegativeBigInt, Sha256Hex, Str256,
)

class UserFile(
    SQLModelBase,
    UUIDTableBaseMixin,
    PolymorphicBaseMixin,
    table=True,
):
    filename: Str256
    user_id: UUID = Field(foreign_key='user.id')
```

Note that an STI parent **does not** inherit `ABC` — it's a real concrete table that subclasses **share**.

## 2. Define subclasses (no FK mixin needed)

```python
class PendingFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    upload_deadline: datetime | None = None  # Auto-added as nullable column // [!code highlight]


class CompletedFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    file_size: NonNegativeBigInt | None = None  # Auto-added as nullable column // [!code highlight]
    sha256: Sha256Hex | None = None
```

::: warning Subclass fields must be nullable
In STI all subclasses share one table. `PendingFile.upload_deadline` is meaningless for `CompletedFile` rows, so the column must be nullable (`| None`). sqlmodel-ext forces the column to `nullable=True`.
:::

`AutoPolymorphicIdentityMixin` automatically sets `_polymorphic_name = 'pendingfile'` / `'completedfile'`.

## 3. Call the two-phase registration functions

STI subclass fields need to be registered to the parent table in two phases. **These two calls must happen after all models are defined**, typically at the end of your application bootstrap or at the bottom of `models/__init__.py`.

```python
from sqlmodel_ext import (
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)
from sqlalchemy.orm import configure_mappers

# After every STI model has been imported:
register_sti_columns_for_all_subclasses()       # Phase 1: add columns to parent table // [!code warning]
configure_mappers()                              # SQLAlchemy mapper configuration
register_sti_column_properties_for_all_subclasses()  # Phase 2: bind columns to mapper // [!code warning]
```

::: danger Order matters
Phase 1 must be **before** `configure_mappers()`, Phase 2 must be **after**. Reason: see [Polymorphic inheritance internals](/en/explanation/polymorphic-internals#sti-column-registration-two-phases).
:::

## 4. Querying: auto-filter by `_polymorphic_name`

```python
# All files (regardless of subclass)
all_files = await UserFile.get(session, fetch_mode='all')

# Only pending files
pending = await PendingFile.get(session, fetch_mode='all')
# Internal SQL: SELECT * FROM userfile WHERE _polymorphic_name = 'pendingfile'
```

::: info STI auto-filter
SQLAlchemy/SQLModel does **not** automatically add `WHERE _polymorphic_name IN (...)` for STI subclass queries. sqlmodel-ext patches it in inside `get()`, using `mapper.self_and_descendants` to include both the class itself and all of its subclasses.
:::

## Verifying the schema

After migration the database has only one table:

```sql
userfile (
    id UUID PRIMARY KEY,
    filename VARCHAR(256) NOT NULL,
    user_id UUID NOT NULL,
    _polymorphic_name VARCHAR NOT NULL,    -- Discriminator ('pendingfile' / 'completedfile')
    upload_deadline TIMESTAMP NULL,         -- PendingFile field, NULL for other subclasses
    file_size BIGINT NULL,                  -- CompletedFile field
    sha256 VARCHAR(64) NULL,                -- CompletedFile field
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
```

## STI subclass field type conflicts

If two subclasses use the same field name with different types (e.g. `Vendor1.duration: int` vs `Vendor2.duration: KlingDurationEnum`), sqlmodel-ext raises `TypeError` in Phase 1. Fix: **use vendor-prefixed field names** (`vendor1_duration` / `kling_25_duration`).

## Polymorphic identity for multi-level subclasses

`AutoPolymorphicIdentityMixin` produces dot-separated identities:

```python
class Generator(SQLModelBase, ..., PolymorphicBaseMixin, table=True): ...
# identity = 'generator'

class FileGenerator(Generator, AutoPolymorphicIdentityMixin, table=True): ...
# identity = 'generator.filegenerator'

class ImageGenerator(FileGenerator, AutoPolymorphicIdentityMixin, table=True): ...
# identity = 'generator.filegenerator.imagegenerator'
```

When filtering by `_polymorphic_name` in a data migration, use `LIKE '%xxx'` to match the suffix.

## Related reference

- [`register_sti_columns_for_all_subclasses` / `register_sti_column_properties_for_all_subclasses`](/en/reference/mixins#register-sti-columns-for-all-subclasses)
- [Polymorphic inheritance internals (two-phase registration rationale)](/en/explanation/polymorphic-internals#sti-column-registration-two-phases)
- [JTI joined table inheritance](./define-jti-models) (the alternative)
