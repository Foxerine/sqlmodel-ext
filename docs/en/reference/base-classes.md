# Base classes

::: tip
This is reference documentation. To see how to use these classes to build models, head to the [tutorials](/en/tutorials/01-getting-started) or [how-to guides](/en/how-to/).
:::

## `SQLModelBase`

```python
from sqlmodel_ext import SQLModelBase
```

Root class for all sqlmodel-ext models. Inherits from `SQLModel` and uses the custom metaclass `__DeclarativeMeta`.

**`model_config`**:

| Key | Value | Description |
|-----|-------|-------------|
| `use_attribute_docstrings` | `True` | The `"""..."""` below a field is automatically used as its description |
| `validate_by_name` | `True` | Allow validation by field name (even when an alias is set) |
| `extra` | `'forbid'` | Passing an undeclared field raises `ValidationError` |

**Class methods**:

```python
@classmethod
def get_computed_field_names(cls) -> set[str]
```

Returns the set of all `@computed_field` field names.

```python
@classmethod
def validate_list(cls, items: Sequence[Any]) -> list[Self]
```

Batch validation: `model_validate`s each ORM instance / dict in the sequence
into the current model type and returns the list. Typical use: converting a
query result list into a response-DTO list.

**Common class-definition keyword arguments** (handled by the metaclass; see
[the metaclass explanation](/en/explanation/metaclass)):

| Keyword | Purpose |
|---------|---------|
| `table_name` | Custom table name (equivalent to `__tablename__`) |
| `table_args` | Table-level constraints/indexes tuple (equivalent to `__table_args__`; `CustomTableArg` subclass instances inside it are intercepted for deferred processing, e.g. `DeferredIndex`) |
| `mapper_args` | SQLAlchemy mapper args dict (equivalent to `__mapper_args__`) |
| `polymorphic_on` / `polymorphic_identity` / `polymorphic_abstract` | STI/JTI polymorphism config (top-level shortcuts) |
| `cache_ttl` | Redis cache TTL in seconds (only effective on `CachedTableBaseMixin` subclasses) |
| `all_fields_optional` | Converts every inherited field to `T \| None = None` (PATCH/UpdateRequest DTOs; constraints are nested and preserved, JSON `null` parses safely) |
| `abstract` | Marks the class abstract (equivalent to `__abstract__`) |

**Inheritance patterns**:

- `class XxxBase(SQLModelBase)` — pure data model (no table), used for API input/output
- `class Xxx(XxxBase, TableBaseMixin, table=True)` — table-backed model

## `ExtraIgnoreModelBase`

```python
from sqlmodel_ext import ExtraIgnoreModelBase
```

Inherits from `SQLModelBase` but with `extra='ignore'`: unknown fields are silently dropped while a WARNING is logged.

**`model_config`**:

| Key | Value | Description |
|-----|-------|-------------|
| `use_attribute_docstrings` | `True` | Same as `SQLModelBase` |
| `validate_by_name` | `True` | Same as `SQLModelBase` |
| `extra` | `'ignore'` | Unknown fields are dropped (no error) |

**Validator**:

```python
@model_validator(mode='before')
@classmethod
def _warn_unknown_fields(cls, data: Any) -> Any
```

If the input is a dict containing fields not declared on the model, logs a WARNING. `alias` and `validation_alias` count as known field names.

**Use cases**: third-party API responses, external WebSocket messages, JSON inputs whose schema may evolve.

## `TableBaseMixin`

```python
from sqlmodel_ext import TableBaseMixin
```

Adds an auto-incrementing integer primary key and CRUD methods to a model.

**Inherits**: `AsyncAttrs` (provides `await obj.awaitable_attrs.xxx` syntax).

**Class marker**:

```python
_has_table_mixin: ClassVar[bool] = True
```

Lets the metaclass identify "this is a table class" and automatically apply `table=True`.

**Fields**:

| Field | Type | Database behavior |
|-------|------|-------------------|
| `id` | `int \| None` | Primary key, auto-generated (`SERIAL` / `INTEGER PRIMARY KEY`) |
| `created_at` | `datetime` | Set on insert via `default_factory=now` |
| `updated_at` | `datetime` | Refreshed on every UPDATE via `onupdate=now` |

**Methods**: full CRUD signatures live in [CRUD methods](./crud-methods).

## `UUIDTableBaseMixin`

```python
from sqlmodel_ext import UUIDTableBaseMixin
```

UUID-keyed variant of `TableBaseMixin`.

**Fields**:

| Field | Type | Database behavior |
|-------|------|-------------------|
| `id` | `uuid.UUID` | Primary key, `default_factory=uuid.uuid4` |
| `created_at` | `datetime` | Same as `TableBaseMixin` |
| `updated_at` | `datetime` | Same as `TableBaseMixin` |

**Type-precise overrides**:

`UUIDTableBaseMixin` overloads `get_one()` / `get_exist_one()` so that the `id` parameter is typed as `uuid.UUID` rather than `int`.

## `RecordNotFoundError`

```python
from sqlmodel_ext import RecordNotFoundError
```

Raised by `get_exist_one()` when no record is found **and** FastAPI is not installed. When FastAPI is installed, `HTTPException(404)` is raised instead.

**Detection logic**: at module import time, `import fastapi` is attempted; the result is cached in `_HAS_FASTAPI`.
