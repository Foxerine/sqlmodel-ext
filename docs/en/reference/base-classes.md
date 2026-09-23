# Base classes

::: tip
This is reference documentation. To see how to use these classes to build models, head to the [tutorials](/en/tutorials/01-getting-started) or [how-to guides](/en/how-to/).
:::

## `SQLModelBase`

```python
from sqlmodel_ext import SQLModelBase
```

Root class for all sqlmodel-ext models. Inherits from `SQLModel` and uses the custom metaclass `__DeclarativeMeta`.

**`model_config`** (of type `SQLModelExtConfig`):

| Key | Value | Description |
|-----|-------|-------------|
| `use_attribute_docstrings` | `True` | The `"""..."""` below a field is automatically used as its description |
| `validate_by_name` | `True` | Allow validation by field name (even when an alias is set) |
| `extra` | `'forbid'` | Passing an undeclared field raises `ValidationError` |
| `omitted_sentinel` | not set (= off) | See `SQLModelExtConfig` below |

**Methods**:

```python
@staticmethod
def annotation_is_omissible(annotation: Any) -> bool
```

Whether the annotation accepts `Unset` (i.e. whether the field may be left out). Searches recursively through `Annotated` and nested unions. The criterion is the annotation, not the default.

```python
@classmethod
def field_is_omissible(cls, field_name: str) -> bool
```

By-field-name variant of `annotation_is_omissible`. Returns `False` (does not raise) for unknown names.

```python
def submitted_fields_among(self, *models: type[SQLModelBase]) -> set[str]
```

Returns `self.model_fields_set` intersected with the union of the given models' field names — "which of these fields did this request explicitly submit". Pure set arithmetic; what a hit means is up to the caller (typical use: permission checks on admin-only fields).

```python
@classmethod
def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]
```

Same as Pydantic; when the model enables `omitted_sentinel`, it additionally adds a `{"const": "__omitted__", "type": "string"}` branch and `"default": "__omitted__"` to every omissible field (and to nested models reachable from the fields).

```python
@classmethod
def get_computed_field_names(cls) -> set[str]
```

Returns the set of all `@computed_field` field names.

```python
@classmethod
def validate_list(cls, items: Sequence[Any]) -> list[Self]
```

Batch validation: `model_validate(..., from_attributes=True)`s each ORM instance / dict in the sequence into the current model type and returns the list. Typical use: converting a query result list into a response-DTO list.

**Construction-time check**: `JSON100K` / `JSONList100K` fields are discovered automatically at class creation (`__orjson_checked_fields__`) and checked for encodability and the 100K limit in `model_post_init` — the check still applies to `table=True` models, which skip Pydantic validation. **Subclasses overriding `model_post_init` must call `super().model_post_init(context)`**, otherwise the check is silently skipped.

**Other fixes**: when a field's type is an enum / nested model, Pydantic sometimes emits a `$ref` property without a `description`; `SQLModelBase` restores it. When a subclass overrides a field without a docstring, the description is inherited from the parent.

**Common class-definition keyword arguments** (handled by the metaclass; see
[the metaclass explanation](/en/explanation/metaclass)):

| Keyword | Purpose |
|---------|---------|
| `table_name` | Custom table name (equivalent to `__tablename__`) |
| `table_args` | Table-level constraints/indexes tuple (equivalent to `__table_args__`; `CustomTableArg` subclass instances inside it are intercepted for deferred processing, e.g. `DeferredIndex`) |
| `mapper_args` | SQLAlchemy mapper args dict (equivalent to `__mapper_args__`) |
| `polymorphic_on` / `polymorphic_identity` / `polymorphic_abstract` / `version_id_col` / `concrete` | Top-level shortcuts for mapper args |
| `cache_ttl` | Redis cache TTL in seconds (positive integer; only effective on `CachedTableBaseMixin` subclasses) |
| `partial` | When `True`, inherited fields become omissible: `T` → `Unset \| T = Unset`, `T \| None` → `Unset \| T \| None = Unset`. Constraints, descriptions and field-level attributes are kept; fields the class declares itself and `Literal` fields are skipped. Cannot be combined with `table=True`. See [Unset](/en/explanation/unset-three-state) |
| `abstract` | Marks the class abstract (equivalent to `__abstract__`) |

::: warning Removed: `all_fields_optional`
Since 0.5.0, using `all_fields_optional` raises `TypeError` at class creation. Use `partial=True` instead; see [Migrate to 0.5](/en/how-to/migrate-to-0-5).
:::

**Reserved field name**: `oplock_version` is reserved for `OptimisticLockMixin`; any `SQLModelBase` subclass declaring it in its own class body raises `TypeError` at class creation.

**Inheritance patterns**:

- `class XxxBase(SQLModelBase)` — pure data model (no table), used for API input/output
- `class Xxx(XxxBase, TableBaseMixin, table=True)` — table-backed model
- `class XxxUpdate(XxxBase, partial=True)` — PATCH request body

`sqlmodel_ext.base.optional_dto_registry: list[type]` records every `partial=True` class in creation order, for tools such as contract tests to enumerate.

## `SQLModelExtConfig`

```python
from sqlmodel_ext import SQLModelExtConfig
```

`SQLModelConfig` (Pydantic `ConfigDict`) plus sqlmodel-ext's own configuration keys. A `total=False` TypedDict: an absent key means "use the default"; all upstream keys remain available.

| Key | Type | Default | Description |
|---|---|---|---|
| `omitted_sentinel` | `bool` | off | Provides a fillable wire sentinel value `'__omitted__'` for omissible fields: every occurrence in an inbound dict, at any depth, is replaced by `Unset` before validation; `model_json_schema()` adds the sentinel branch to omissible fields. For callers that cannot omit keys (e.g. LLM function calling in strict mode). With it on, string fields of the model cannot hold the literal `'__omitted__'` |

```python
from sqlmodel_ext import SQLModelBase, SQLModelExtConfig


class ToolArguments(SQLModelBase):
    model_config = SQLModelExtConfig(omitted_sentinel=True)
```

## `Unset` and `OMITTED_SENTINEL`

```python
from sqlmodel_ext import Unset, OMITTED_SENTINEL
```

- `Unset`: Pydantic's official `pydantic.experimental.missing_sentinel.MISSING`, meaning "this field was not provided". Check with `x is Unset`. Fields whose value is `Unset` never appear in `model_dump()` / `model_dump_json()` output. `copy.deepcopy(Unset) is Unset`.
- `OMITTED_SENTINEL`: `'__omitted__'`, the wire sentinel value; it only exists in the JSON Schema and inbound payloads of models with `omitted_sentinel` enabled and is normalized to `Unset` on entry. Application code should never compare against it.

Full semantics in [Unset](/en/explanation/unset-three-state).

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

If the input is a dict containing fields not declared on the model, logs a WARNING (with the model name, the number of unknown fields and up to 5 sample field names). Field names, `alias` and `validation_alias` (a string, or every string choice of an `AliasChoices`) count as known; `AliasPath` entries are nested paths rather than top-level keys and are not counted.

**Use cases**: third-party API responses, external WebSocket messages, JSON inputs whose schema may evolve.

## `TableBaseMixin`

```python
from sqlmodel_ext import TableBaseMixin
```

Adds an auto-incrementing integer primary key and CRUD methods to a model.

**Inherits**: `AsyncAttrs` (provides `await obj.awaitable_attrs.xxx` syntax).

**Class attributes**:

<!-- skip-run -->
```python
_has_table_mixin: ClassVar[bool] = True
```

Lets the metaclass identify "this is a table class" and automatically apply `table=True`.

<!-- skip-run -->
```python
__optimistic_retry_default__: ClassVar[int] = 0
```

The retry count used by `save()` / `update()` when `optimistic_retry_count` is not passed (`None`). 0 on the base class; `OptimisticLockMixin` overrides it with 3.

**Fields**:

| Field | Type | Database behavior |
|-------|------|-------------------|
| `id` | `int \| None` | Primary key, auto-generated (`SERIAL` / `INTEGER PRIMARY KEY`) |
| `created_at` | `datetime` | `TIMESTAMP WITH TIME ZONE`, set on insert via `default_factory=now` |
| `updated_at` | `datetime` | `TIMESTAMP WITH TIME ZONE`, `onupdate=now`; `save()` / `update()` assign it explicitly when something changed (so it also advances when a joined-table-inheritance update touches only child-table columns) |

**Methods**: full CRUD signatures live in [CRUD methods](./crud-methods).

## `UUIDTableBaseMixin`

```python
from sqlmodel_ext import UUIDTableBaseMixin
```

UUID-keyed variant of `TableBaseMixin`.

**Fields**:

| Field | Type | Database behavior |
|-------|------|-------------------|
| `id` | `uuid.UUID` | Primary key, `default_factory=uuid7` (UUIDv7, time-ordered) |
| `created_at` | `datetime` | Same as `TableBaseMixin` |
| `updated_at` | `datetime` | Same as `TableBaseMixin` |

The first 48 bits of a UUIDv7 are a millisecond Unix timestamp: `ORDER BY id` approximates creation order and B-tree inserts concentrate on the right edge. The generator is `sqlmodel_ext.mixins.uuid7` (the standard library `uuid.uuid7` on Python 3.14+, a built-in RFC 9562 implementation on older versions).

Known limitations: the id reveals its creation time with millisecond precision (ids are identifiers, not capabilities); id order is not authoritative time order; existing rows keep their UUID version, and mixed v4/v7 values are still a well-defined total order; assign deterministically derived primary keys explicitly.

**Type-precise overrides**:

`UUIDTableBaseMixin` overloads `get_one()` / `get_exist_one()` so that the `id` parameter is typed as `uuid.UUID` rather than `int`.

## `RecordNotFoundError`

```python
from sqlmodel_ext import RecordNotFoundError
```

Raised by `get_exist_one()` when no record is found **and** FastAPI is not installed (`status_code = 404`, `detail` is the `detail` argument, default `"Not found"`). When FastAPI is installed, `HTTPException(404)` is raised instead.

**Detection logic**: when `sqlmodel_ext.mixins.table` is imported, it tries `from fastapi import HTTPException` and uses it if available.
