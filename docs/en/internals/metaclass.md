# Metaclass & SQLModelBase

::: tip Tip
`src/sqlmodel_ext/base.py` — `SQLModelBase` and `__DeclarativeMeta` metaclass

`src/sqlmodel_ext/_sa_type.py` — Extract SQLAlchemy column types from Annotated metadata

`src/sqlmodel_ext/_compat.py` — Python 3.14 compatibility patch
:::

This is the **foundation** of the entire project. All model classes inherit from `SQLModelBase`, and `SQLModelBase`'s metaclass `__DeclarativeMeta` automatically completes a series of configurations during class creation.

## What the User Writes vs What the Metaclass Does

```python
class UserBase(SQLModelBase):
    name: Str64
    email: str

class User(UserBase, UUIDTableBaseMixin, table=True):
    pass
```

| Class | Creates DB Table? | Role |
|-------|-------------------|------|
| `UserBase` | No | Pure data model, only defines fields |
| `User` | Yes | Inherits fields + CRUD capabilities, maps to a database table |

SQLModel uses the `table=True` keyword to decide whether to create a table. **The metaclass is where this parameter is processed.**

## `__DeclarativeMeta.__new__` Step by Step

`__new__` executes at the **very moment** a class object is created.

### Step 1: Auto `table=True`

```python
# base.py:113-116
is_intended_as_table = any(getattr(b, '_has_table_mixin', False) for b in bases)
if is_intended_as_table and 'table' not in kwargs:
    kwargs['table'] = True # [!code focus]
```

Iterates through parent classes — if `_has_table_mixin = True` is found (defined on `TableBaseMixin`), `table=True` is automatically added.

### Step 2: Detect Inheritance Type (JTI vs STI)

```python
# base.py:119-143
parent_tablename = None
for base in bases:
    if is_table_model_class(base) and hasattr(base, '__tablename__'):
        parent_tablename = base.__tablename__
        break

# Check for foreign key pointing to parent table → JTI characteristic
has_fk_to_parent = False
if parent_tablename is not None and will_be_table:
    for base in bases:
        for field_name, field_info in base.model_fields.items():
            fk = getattr(field_info, 'foreign_key', None)
            if fk and parent_tablename in fk:
                has_fk_to_parent = True

# STI: no foreign key to parent, shares parent table
if parent_tablename and will_be_table and not has_own_tablename and not has_fk_to_parent:
    attrs['__tablename__'] = parent_tablename
```

When a table subclass inherits from a table parent:
- **Has foreign key to parent** → JTI, subclass gets its own table
- **No foreign key** → STI, subclass shares parent's `__tablename__`

### Step 3: Merge `__mapper_args__`

```python
# base.py:146-158
collected_mapper_args = {}

if 'mapper_args' in kwargs:
    collected_mapper_args.update(kwargs.pop('mapper_args'))

for key in cls._KNOWN_MAPPER_KEYS:  # polymorphic_on, polymorphic_identity, ...
    if key in kwargs:
        collected_mapper_args[key] = kwargs.pop(key)

if collected_mapper_args:
    existing = attrs.get('__mapper_args__', {}).copy()
    existing.update(collected_mapper_args)
    attrs['__mapper_args__'] = existing
```

Extracts keywords like `polymorphic_on`, `polymorphic_abstract` from `kwargs` and merges them into the `__mapper_args__` dict. This enables a concise syntax:

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

`_KNOWN_MAPPER_KEYS` supports these shortcut keywords: `polymorphic_on`, `polymorphic_identity`, `polymorphic_abstract`, `version_id_col`, `concrete`.

### Step 4: Extract `sa_type` from Type Annotations

This is the **most elegant part** of the metaclass.

```python
# base.py:169-202
annotations, ..., eval_globals, eval_locals = _resolve_annotations(attrs)

for field_name, field_type in annotations.items():
    sa_type = _extract_sa_type_from_annotation(field_type) # [!code focus]

    if sa_type is not None:
        field_value = attrs.get(field_name, Undefined)

        if field_value is Undefined:
            attrs[field_name] = Field(sa_type=sa_type) # [!code focus]
        elif isinstance(field_value, FieldInfo):
            if not hasattr(field_value, 'sa_type') or field_value.sa_type is Undefined:
                field_value.sa_type = sa_type # [!code focus]
```

#### `_extract_sa_type_from_annotation()` — Three Extraction Methods

In `_sa_type.py`, three methods are used to find SQLAlchemy column types from type annotations:

```python
def _extract_sa_type_from_annotation(annotation):
    # Method 1: The type itself has a __sqlmodel_sa_type__ attribute
    if hasattr(annotation, '__sqlmodel_sa_type__'):
        return annotation.__sqlmodel_sa_type__

    # Method 2: Found in Annotated metadata
    if get_origin(annotation) is Annotated:
        for item in get_args(annotation)[1:]:
            if hasattr(item, '__sqlmodel_sa_type__'):
                return item.__sqlmodel_sa_type__
            schema = item.__get_pydantic_core_schema__(...)
            if 'sa_type' in schema.get('metadata', {}):
                return schema['metadata']['sa_type']

    # Method 3: The type's own __get_pydantic_core_schema__ returns metadata
    schema = annotation.__get_pydantic_core_schema__(...)
    return schema.get('metadata', {}).get('sa_type')
```

Example with `Array[str]`: `__class_getitem__` returns `Annotated[list[str], _ArrayTypeHandler(str)]`, and `_ArrayTypeHandler.__get_pydantic_core_schema__` includes `metadata={'sa_type': ARRAY(String)}` in its schema. The metaclass finds this and automatically injects it into `Field(sa_type=ARRAY(String))`.

### Step 5: Call Parent to Create the Class

```python
result = super().__new__(cls, name, bases, attrs, **kwargs)
```

After the first four preprocessing steps, the configured `attrs` and `kwargs` are passed to SQLModel's original metaclass.

### Steps 6-8: Fix Inherited Relationship Fields

```python
# Step 6: JTI subclass inherits parent's Relationships
for base in bases:
    if hasattr(base, '__sqlmodel_relationships__'):
        for rel_name, rel_info in base.__sqlmodel_relationships__.items():
            if rel_name not in result.__sqlmodel_relationships__:
                result.__sqlmodel_relationships__[rel_name] = rel_info

# Step 7: Prevent subclass from redefining parent's Relationships
for base in bases:
    parent_relationships = getattr(base, '__sqlmodel_relationships__', {})
    for rel_name in parent_relationships:
        if rel_name in attrs:
            raise TypeError(f"Cannot redefine parent's Relationship '{rel_name}'")

# Step 8: Remove Relationship fields from model_fields
for rel_name in relationships:
    if rel_name in model_fields:
        del model_fields[rel_name]
if fields_removed:
    result.model_rebuild(force=True)
```

Fixes bugs in SQLModel/SQLAlchemy when handling inheritance + relationships: Relationships being treated as Pydantic fields, JTI subclasses losing parent Relationships, and subclass redefinition causing ambiguity.

## `__DeclarativeMeta.__init__` — JTI Table Creation

After `__new__` creates the class, `__init__` does post-initialization. Core task: **handle JTI sub-table creation**.

```python
def __init__(cls, classname, bases, dict_, **kw):
    if not is_table_model_class(cls):
        ModelMetaclass.__init__(...)
        return

    base_is_table = any(is_table_model_class(base) for base in bases)

    if not base_is_table:
        # First table class, normal flow
        cls._setup_relationships()
        DeclarativeMeta.__init__(...)
        return

    # Parent is also a table → inheritance scenario
    is_joined_inheritance = has_different_tablename and has_fk_to_parent

    if is_joined_inheritance:
        # JTI: create sub-table
        # 1. Collect ancestor table column names
        # 2. Find subclass-owned fields
        # 3. Rebuild foreign key columns
        # 4. Remove columns inherited from ancestors that don't belong in sub-table
        # 5. Set up subclass-owned Relationships
        DeclarativeMeta.__init__(...)

    else:
        # STI: subclass shares parent table
        ModelMetaclass.__init__(...)
        registry.map_imperatively(...)
```

::: info Info
SQLModel's original logic: if the parent is already a table model, the subclass **skips** `DeclarativeMeta.__init__`. But JTI needs the subclass to have its own table! sqlmodel-ext detects JTI scenarios and manually calls it to create the sub-table.

For STI, `registry.map_imperatively()` maps the subclass to the parent table while handling the subclass's Relationships and foreign key resolution.
:::

### Step 1.5: `cache_ttl` Keyword

```python
# base.py:121-126
if 'cache_ttl' in kwargs:
    ttl = kwargs.pop('cache_ttl')
    if not isinstance(ttl, int) or ttl <= 0:
        raise ValueError(f"{name}: cache_ttl must be a positive integer, got: {ttl!r}")
    attrs['__cache_ttl__'] = ttl
```

`CachedTableBaseMixin` uses `__cache_ttl__` to control cache TTL. The metaclass converts the `cache_ttl` keyword argument into a class attribute, enabling the syntax `class Foo(..., table=True, cache_ttl=1800):`.

## `SQLModelBase` Itself

```python
class SQLModelBase(SQLModel, metaclass=__DeclarativeMeta):
    model_config = ConfigDict(
        use_attribute_docstrings=True,  # Attribute docstrings as field descriptions
        validate_by_name=True,          # Allow validation by field name
        extra='forbid',                 # Forbid passing undefined fields
    )

    @classmethod
    def get_computed_field_names(cls) -> set[str]:
        fields = cls.model_computed_fields
        return set(fields.keys()) if fields else set()
```

## `ExtraIgnoreModelBase` — External Data Base Class

```python
class ExtraIgnoreModelBase(SQLModelBase):
    model_config = ConfigDict(
        use_attribute_docstrings=True, validate_by_name=True, extra='ignore',
    )

    @model_validator(mode='before')
    @classmethod
    def _warn_unknown_fields(cls, data):
        if not isinstance(data, dict):
            return data
        accepted = {name for name, fi in cls.model_fields.items()}
        # Also includes alias and validation_alias
        unknown = set(data.keys()) - accepted
        if unknown:
            logger.warning("External input contains unknown fields | model=%s ...", cls.__name__)
        return data
```

Unlike `SQLModelBase` (`extra='forbid'`), `ExtraIgnoreModelBase` uses `extra='ignore'` to silently ignore unknown fields, but **logs a WARNING** to help developers notice third-party API changes.

Use cases: third-party API responses, client WebSocket messages, external JSON inputs.

## `_compat.py` — Python 3.14 Patch

Python 3.14 introduces PEP 649 (deferred annotation evaluation), which causes errors in SQLModel internal functions. `_compat.py` fixes this via monkey patching:

### Patch 1: `get_sqlalchemy_type`

The original function calls `issubclass()` on `ForwardRef`, `ClassVar`, `Literal[StrEnum.MEMBER]` etc., causing `TypeError`. The patch intercepts these special cases before the call.

### Patch 2: `sqlmodel_table_construct`

In polymorphic inheritance table subclasses, inherited Relationship field defaults may be replaced by SQLAlchemy with `InstrumentedAttribute` objects. The patch skips these "polluted" defaults.

::: info Info
Both patches only activate on Python >= 3.14.
:::

## Summary

| Metaclass Step | Problem Solved |
|----------------|---------------|
| Auto `table=True` | Eliminates manual writing |
| Detect JTI/STI | Automatically handles both inheritance modes |
| Merge `__mapper_args__` | Simplifies polymorphic configuration syntax |
| Extract `sa_type` | Custom types auto-map to database columns |
| Fix inherited relation fields | Works around SQLModel/SQLAlchemy bugs |
| JTI sub-table creation | Enables Joined Table Inheritance in SQLModel |

**Core design philosophy**: Users just write declarative model definitions, and the metaclass handles all SQLAlchemy configuration details behind the scenes.
