# Metaclass & SQLModelBase

::: tip Source location
`src/sqlmodel_ext/base.py` — `SQLModelBase`, `SQLModelExtConfig` and the `__DeclarativeMeta` metaclass

`src/sqlmodel_ext/_sa_type.py` — extracts SQLAlchemy column types from Annotated metadata

`src/sqlmodel_ext/_compat.py` — Python 3.14 compatibility patches
:::

This is the **cornerstone** of the project, and the place where the "[single source of truth](./single-source-of-truth)" actually holds: you declare a fact once, and the metaclass derives everything SQLAlchemy needs from it **at the moment the class is created**. Every model inherits from `SQLModelBase`, and its metaclass `__DeclarativeMeta` does this work.

::: info About the snippets on this page
The snippets below are **simplified excerpts** of `base.py` meant to show the intent of each step; step numbers match the numbered comments in the source. The source is authoritative.
:::

## What the user writes vs what the metaclass does

```python
class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    email: Str255

class User(UserBase, UUIDTableBaseMixin):   # no table=True
    pass

class UserUpdate(UserBase, partial=True):   # no field re-declared
    pass
```

| Class | Creates a DB table? | Role |
|---|---|---|
| `UserBase` | No | Pure data model — only defines fields |
| `User` | Yes | Inherits fields + CRUD, maps to a DB table |
| `UserUpdate` | No | PATCH body; every field becomes `Unset \| T` |

## `__DeclarativeMeta.__new__` step by step

### Step 1: Auto `table=True`

```python
is_intended_as_table = any(getattr(b, '_has_table_mixin', False) for b in bases)
if is_intended_as_table and 'table' not in kwargs:
    kwargs['table'] = True
```

If a parent carries `_has_table_mixin = True` (defined on `TableBaseMixin`), `table=True` is added automatically.

### Step 1.5: `cache_ttl` keyword

```python
if 'cache_ttl' in kwargs:
    ttl = kwargs.pop('cache_ttl')
    if not isinstance(ttl, int) or ttl <= 0:
        raise ValueError(f"{name}: cache_ttl must be a positive integer, got: {ttl!r}")
    attrs['__cache_ttl__'] = ttl
```

Lets you write `class Foo(..., cache_ttl=1800):`; `CachedTableBaseMixin` reads `__cache_ttl__`.

### Step 2: Detect inheritance type (JTI vs STI)

```python
parent_tablename = None
for base in bases:
    if is_table_model_class(base) and hasattr(base, '__tablename__'):
        parent_tablename = base.__tablename__
        break

# a parent field with a foreign key to the parent table -> JTI characteristic
...
if parent_tablename is not None and will_be_table and not has_own_tablename and not has_fk_to_parent:
    attrs['__tablename__'] = parent_tablename   # STI: share the parent table
```

When a table subclass inherits a table parent: **a foreign key to the parent table** → JTI, the subclass gets its own table; **no foreign key** → STI, it shares the parent table.

### Step 3: Merge `__mapper_args__`

```python
collected_mapper_args = {}
if 'mapper_args' in kwargs:
    collected_mapper_args.update(kwargs.pop('mapper_args'))
for key in cls._KNOWN_MAPPER_KEYS:  # polymorphic_on, polymorphic_identity, ...
    if key in kwargs:
        collected_mapper_args[key] = kwargs.pop(key)
```

This enables the concise syntax:

```python
# sqlmodel-ext (concise)
class Tool(SQLModelBase, polymorphic_on="_polymorphic_name", polymorphic_abstract=True): # [!code ++]
    pass

# Equivalent raw SQLAlchemy (verbose)
class Tool(SQLModel, table=True): # [!code --]
    __mapper_args__ = { # [!code --]
        "polymorphic_on": "_polymorphic_name", # [!code --]
        "polymorphic_abstract": True, # [!code --]
    } # [!code --]
```

`_KNOWN_MAPPER_KEYS`: `polymorphic_on`, `polymorphic_identity`, `polymorphic_abstract`, `version_id_col`, `concrete`.

### Step 3.5: Optimistic-lock wiring

```python
if will_be_table and any(getattr(b, '_has_optimistic_lock', False) for b in bases):
    if not _is_inheriting_table and 'version_id_col' not in attrs.get('__mapper_args__', {}):
        def _mapper_args_with_version_col(target_cls, _static=...):
            merged = dict(_static)
            merged['version_id_col'] = target_cls.__table__.c[OPTIMISTIC_LOCK_VERSION_COLUMN]
            return merged
        attrs['__mapper_args__'] = declared_attr.directive(_mapper_args_with_version_col)
```

A root table class mixing in `OptimisticLockMixin` gets its `oplock_version` column registered as SQLAlchemy's `version_id_col`, so every UPDATE carries `WHERE ... AND oplock_version = :current`. The `Column` object only exists once the table is built, hence the late-evaluated `declared_attr`. STI/JTI children share it through mapper inheritance. — **The policy lives where the capability is declared**: the user only mixes in a mixin.

### Step 3.6: `CustomTableArg` in `table_args`, `table_name`, `abstract`

While processing `table_args`, the metaclass **pulls out** marker objects that inherit `CustomTableArg` and does not pass them to SQLAlchemy:

```python
real_table_args, custom_table_args = [], []
for arg in raw_table_args:
    (custom_table_args if isinstance(arg, CustomTableArg) else real_table_args).append(arg)
attrs['__table_args__'] = tuple(real_table_args)
# appended to the module-level queue classes_with_custom_table_args after super().__new__
```

**Why**: SQLAlchemy's `Table.__init__` consumes every element of `__table_args__` immediately — an `Index` referencing a not-yet-existing column raises on the spot. `CustomTableArg` is a generic "defer this" marker: the metaclass only intercepts and enqueues, knowing nothing about the semantics; the current consumer is `mixins.polymorphic.DeferredIndex` (deferred indexes on STI subclass columns). `table_name=` / `abstract=` become `__tablename__` / `__abstract__`.

### Step 4: Resolve annotations and record "which fields this class declares itself"

```python
annotations, annotation_strings, eval_globals, eval_locals = resolve_annotations(attrs)
_own_annotation_names = frozenset(annotations)
```

This snapshot must be taken **before** any annotation injection: later steps inject inherited fields into `annotations`, after which "declared here" and "inherited and injected" are indistinguishable. Steps 4.5.b and 4.6 rely on it.

### Step 4.5: Recover SQLModel attributes from `Annotated[T, Field(...)]`

When Pydantic v2 processes `Annotated` metadata it replaces `sqlmodel.main.FieldInfo` with `pydantic.fields.FieldInfo`, which knows nothing about SQLModel attributes such as `foreign_key` or `sa_type`. For **table classes** (including `Annotated` fields inherited from parents), `_recover_annotated_sqlmodel_fields()` converts them back into the `= Field(...)` form; non-table classes are left as-is so child table classes can inherit them. When several `FieldInfo`s are merged, an explicit `default=None` is kept as a real value (not "unset") — otherwise the field would silently become required. A field with a right-hand `Field` (`name: Alias = Field(...)`) is rebuilt from Pydantic's resolution of that declaration — `FieldInfo.from_annotated_attribute()` when the table class declares it, the base's resolved field `Base.model_fields[name]` when the table class **inherits** it (without re-declaring it). Both give the same field: its Pydantic attributes are taken as-is (an explicit `alias=None` stays `None`, as in plain SQLModel), and only the column attributes are completed — the `FieldInfoMetadata` carriers of the alias's and the right-hand `Field` are folded into one, since SQLModel reads only the first. Metadata that stays in the class annotation (an `AfterValidator` next to the alias's `Field`, for example) is not copied into the field, so it is applied once.

### Step 4.5.b: `oplock_version` is a reserved name

```python
if OPTIMISTIC_LOCK_VERSION_COLUMN in _own_annotation_names:
    raise TypeError(f"{name}: 'oplock_version' is reserved for OptimisticLockMixin's version_id_col ...")
```

Any class declaring `oplock_version` in **its own body** fails — whether or not it enables optimistic locking. Reserving it only for locked classes is not enough: a class without the lock could declare a domain field of that name, which a descendant re-enabling the lock would then silently wire up as `version_id_col`.

### Step 4.6: `partial=True`

```python
if 'all_fields_optional' in kwargs:
    raise TypeError(f"{name}: the 'all_fields_optional' class keyword was removed in sqlmodel-ext 0.5.0. ...")
is_partial = kwargs.pop('partial', False)
if is_partial:
    if will_be_table:
        raise TypeError(f"{name}: 'partial=True' cannot be combined with 'table=True' ...")
    _apply_partial(annotations, attrs, bases, _own_annotation_names)
```

`_apply_partial()` collects field names from the bases' `model_fields`, fetches the **original annotation** along the MRO (keeping `Annotated` metadata), and then:

- `T` → `Unset | T` with default `Unset` (nullable fields naturally become `Unset | T | None`);
- for fields declared as `field: T = Field(gt=..., le=...)` (non-`Annotated` form) the constraints live on the right-hand side, so they are pulled from the base's `model_fields[name].metadata` and re-wrapped into `Annotated`;
- **field-level** attributes — `exclude` / `alias` / `validation_alias` / `serialization_alias` / `repr` / `frozen` / `deprecated` — would be silently dropped by Pydantic on a union member, so `_hoist_field_metadata()` reads them from the base's resolved field and hoists them outside the union (aliases produced by an `alias_generator` are left to be regenerated, as in plain inheritance), while `_union_member_annotation()` strips them from the member; constraints and `discriminator` stay inside;
- fields the class declares itself (`_own_annotation_names`) and `Literal` fields are skipped.

The generated annotations exist at runtime; static type checkers still see the base annotations. Where static enforcement matters, declare `Unset | T = Unset` explicitly, or use the experimental `python -m sqlmodel_ext.check_derived` (see [Check partial DTOs for misuse](/en/how-to/check-partial-dtos)). See [Unset](./unset-three-state).

### Step 4.7: Extract `sa_type` from type annotations

```python
for field_name, field_type in annotations.items():
    sa_type = extract_sa_type_from_annotation(field_type)
    if sa_type is not None:
        field_value = attrs.get(field_name, Undefined)
        if field_value is Undefined:
            # no "= Field(...)": prefer recovering the user's FieldInfo from inside Annotated,
            # keeping default_factory / max_length etc., and only add sa_type
            annotated_fi = _find_field_info_in_annotated(field_type)
            attrs[field_name] = annotated_fi if annotated_fi is not None else Field(sa_type=sa_type)
        elif isinstance(field_value, FieldInfo):
            _durably_set_sa_type(field_value, sa_type)
        else:
            # bare default (e.g. fpath: FilePathType = Path("a.txt"))
            attrs[field_name] = Field(default=field_value, sa_type=sa_type)
```

`_durably_set_sa_type()` writes `sa_type` into a `FieldInfoMetadata` entry of `FieldInfo.metadata` — the channel SQLModel's own `Field(sa_type=...)` uses and that survives Pydantic's `model_fields` rebuild; a plain `setattr` would be lost before the column is built. An explicitly set `sa_type` is never overwritten.

#### `extract_sa_type_from_annotation()` — three extraction methods

```python
def extract_sa_type_from_annotation(annotation):
    # Method 1: the type itself has a __sqlmodel_sa_type__ attribute
    # Method 2: an Annotated metadata item has __sqlmodel_sa_type__, or the schema returned by
    #           its __get_pydantic_core_schema__ carries 'sa_type' in its metadata
    # Method 3: the schema returned by the type's own __get_pydantic_core_schema__ carries 'sa_type'
    ...
```

Take `Array[str]`: `__class_getitem__` returns `Annotated[list[str], ArrayTypeHandler(str)]`, and the `ArrayTypeHandler` schema carries `metadata={'sa_type': ARRAY(String)}`; the `JSON100K` schema carries `metadata={'sa_type': JSONB}`. **The type declares its own column type**, and the metaclass delivers it to the column builder.

### Steps 5–7: Save SQLModel `FieldInfo`s, call the parent, restore

```python
_saved_sqlmodel_fis = {fn: attrs[fn] for fn in annotations if isinstance(attrs.get(fn), SQLModelFieldInfo)}  # step 5 (table classes only)
result = super().__new__(cls, name, bases, attrs, **kwargs)                                                  # step 6
# step 6.5: append intercepted CustomTableArg markers to the module-level queue
# step 7: Pydantic's model_fields rebuild dropped SQLModel-only attributes (unique / index / foreign_key / sa_type ...);
#         merge the saved SQLModelFieldInfo back and rebuild the Column
```

While merging, boolean flags are never overwritten with `False` (`unique=False` never switches off an inherited `unique=True`), and `FieldInfoMetadata` carriers are folded into one (SQLModel reads only the first; otherwise an all-unset carrier inside a type alias shadows `= Field(primary_key=True)` — which is exactly why `id: NonNegativeInt = Field(primary_key=True)` lost its primary key on sqlmodel ≥ 0.0.32).

### Steps 8–9: Relationship fields under inheritance

```python
# Step 8: JTI subclasses inherit the parent's Relationships
# Step 9: a subclass may not redefine a parent's Relationship
for base in bases:
    for rel_name in getattr(base, '__sqlmodel_relationships__', {}):
        if rel_name in attrs:
            raise TypeError(f"Class {name} cannot redefine parent {base.__name__}'s Relationship field '{rel_name}'. ...")
```

### Step 10: Inherit field descriptions

`use_attribute_docstrings` reads docstrings from the source AST. When a subclass overrides a field without a docstring, or `partial=True` generates annotations programmatically, there is no docstring in the source and the description would be lost. The metaclass restores it from the parents' `model_fields` along the MRO — **the description is written once** and derived DTOs carry it automatically.

### Step 11: Remove Relationship fields from `model_fields`

Relationships are not Pydantic fields; the metaclass removes them from `model_fields` / `__pydantic_fields__` and calls `model_rebuild(force=True)` when needed.

### Step 12: Register partial classes

Classes created with `partial=True` are appended, in order, to `sqlmodel_ext.base.optional_dto_registry` for tools such as contract tests to enumerate.

### PEP 604 nullable relationship annotation normalization

When creating a Relationship, the metaclass does not call SQLModel's `get_relationship_to` directly — it first routes through `_resolve_relationship_target`, which uses `ast` to normalize **flat string / ForwardRef** nullable annotations (such as `'Parent | None'`) into a structured `ForwardRef('Parent')` before handing off upstream.

Root cause: `get_relationship_to` can only strip `None` from an **already-evaluated** `typing.Union`; it cannot parse a PEP 604 annotation that is *one whole string* — it would pass the entire `'Parent | None'` to SQLAlchemy as a class name.

```python
class Child(SQLModelBase, UUIDTableBaseMixin, table=True):
    parent_id: uuid.UUID | None = Field(default=None, foreign_key="parent.id")
    parent: 'Parent | None' = Relationship(back_populates="children")   # no Optional['Parent'] needed
```

Covered forms: `Foo` / `pkg.Foo` / `Foo | None` / `None | Foo` / `Optional[Foo]` / `Union[Foo, None]`, plus nested quotes (`Optional['Foo']`). When an annotation cannot be normalized to a single target (e.g. `Foo | Bar`), it is passed through to upstream unchanged so upstream raises its own error. Normalization only applies to the temporary value passed to `get_relationship_to` and never rewrites `cls.__annotations__`.

Relationships without an explicit `lazy` default to `lazy='raise_on_sql'`: an accidental lazy load in async code raises immediately instead of turning into `MissingGreenlet`.

## `__DeclarativeMeta.__init__` — JTI table creation

After `__new__` creates the class, `__init__` performs follow-up initialization. Its core job: **create JTI child tables**.

```python
def __init__(cls, classname, bases, dict_, **kw):
    if not is_table_model_class(cls):
        ModelMetaclass.__init__(...)
        return

    base_is_table = any(is_table_model_class(base) for base in bases)
    if not base_is_table:
        cls._setup_relationships()
        DeclarativeMeta.__init__(...)
        return

    # parent is also a table -> inheritance
    if is_joined_inheritance:
        # JTI: collect ancestor column names, find the subclass's own fields, rebuild FK columns,
        # drop inherited columns that do not belong to the child table, set up own Relationships
        DeclarativeMeta.__init__(...)
    else:
        # STI: subclass shares the parent table
        ModelMetaclass.__init__(...)
        registry.map_imperatively(...)
```

::: info Why manual handling?
SQLModel's original logic: if the parent is already a table model, the subclass **skips** `DeclarativeMeta.__init__`. But JTI needs the subclass to have its own table! sqlmodel-ext detects JTI and calls it manually to create the child table. For STI it uses `registry.map_imperatively()` to map the subclass onto the parent table.
:::

## `SQLModelBase` itself

```python
class SQLModelBase(SQLModel, metaclass=__DeclarativeMeta):
    model_config = SQLModelExtConfig(
        use_attribute_docstrings=True,  # attribute docstrings become field descriptions
        validate_by_name=True,          # allow validation by field name
        extra='forbid',                 # reject undeclared fields
    )
```

Besides configuration, it carries the tri-state semantics and a few construction-time invariants:

| Member | Purpose |
|---|---|
| `annotation_is_omissible()` / `field_is_omissible()` | "May this field be left out" — decided by whether the annotation contains `Unset` |
| `_normalise_omitted_sentinel` (a `mode='before'` validator) | With `omitted_sentinel` on, replaces `'__omitted__'` at any depth of an inbound dict with `Unset` |
| `model_json_schema()` | With `omitted_sentinel` on, injects the sentinel branch into omissible fields (nested and self-referencing models included). It has to happen at the `model_json_schema()` exit: `__get_pydantic_json_schema__` sees an intermediate product that Pydantic re-assembles later |
| `__pydantic_init_subclass__` + `model_post_init` | Discover `JSON100K` / `JSONList100K` fields at class creation and check encodability and the 100K limit at construction — the check applies even to `table=True` models, which skip Pydantic validation. Overrides of `model_post_init` must call `super()` |
| `__get_pydantic_json_schema__` | Restores `description` dropped from `$ref` properties |
| `submitted_fields_among()` | Explicitly submitted fields ∩ the given models' fields |
| `validate_list()` / `get_computed_field_names()` | Batch conversion / list computed fields |

## `ExtraIgnoreModelBase` — external data base class

```python
class ExtraIgnoreModelBase(SQLModelBase):
    model_config = SQLModelExtConfig(
        use_attribute_docstrings=True, validate_by_name=True, extra='ignore',
    )

    @model_validator(mode='before')
    @classmethod
    def _warn_unknown_fields(cls, data):
        ...  # field names, alias and validation_alias (every string choice of AliasChoices) are known
        if unknown:
            logger.warning("External input contains unknown fields | model=%s ...", cls.__name__, ...)
        return data
```

Unlike `SQLModelBase` (`extra='forbid'`), it silently ignores unknown fields but **logs a WARNING** so developers notice third-party API changes. Use for third-party API responses, client WebSocket messages and external JSON input.

## `_compat.py` — Python 3.14 patches

Python 3.14 introduced PEP 649 (deferred evaluation of annotations), which broke SQLModel internals. `_compat.py` fixes two places:

- **`get_sqlalchemy_type`**: the original function calls `issubclass()` on `ForwardRef`, `ClassVar`, `Literal[StrEnum.MEMBER]` and similar types, raising `TypeError`. The patch intercepts these cases first and respects an explicit `Field(sa_type=...)`.
- **`sqlmodel_table_construct`**: in polymorphic table subclasses, inherited Relationship defaults may be replaced by `InstrumentedAttribute` objects. The patch skips these "polluted" defaults.

Both patches only activate on Python >= 3.14.

## Summary

| Metaclass step | Duplication removed / problem solved |
|----------------|----------------|
| Auto `table=True` | "Is it a table" is expressed only by inheriting `TableBaseMixin` |
| JTI/STI detection | The inheritance style is expressed by "is there an FK to the parent table" |
| Merging `__mapper_args__` | Polymorphism is configured with keywords, not dicts |
| Optimistic-lock wiring | Mix in the mixin; no hand-written `version_id_col` |
| Reserving `oplock_version` | Misuse fails at class creation |
| `partial=True` | Update DTOs derive from the base; no field re-declared |
| Extracting `sa_type` | Custom types declare their own column type |
| Restoring SQLModel `FieldInfo` / inheriting descriptions | Constraints and descriptions are written once and survive inheritance |
| Relationship fixes / JTI child tables | Work around SQLModel/SQLAlchemy inheritance defects |

**Core design idea**: the user writes declarative model definitions only; the metaclass derives every SQLAlchemy detail behind the scenes — each fact written once.
