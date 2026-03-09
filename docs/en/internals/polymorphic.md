# Polymorphic Inheritance

::: tip Tip
`src/sqlmodel_ext/mixins/polymorphic.py` — `PolymorphicBaseMixin`, `AutoPolymorphicIdentityMixin`, `create_subclass_id_mixin`
:::

## `PolymorphicBaseMixin` — Auto-configuring the Parent

```python
class PolymorphicBaseMixin:
    _polymorphic_name: Mapped[str] = mapped_column(String, index=True)
```

`_polymorphic_name` is the **discriminator column**, storing strings like `"emailnotification"` in the database. SQLAlchemy uses this to instantiate the correct subclass.

The single underscore `_` prefix design rationale: it exists in the database (unlike double underscores which trigger name mangling); it's excluded from API serialization (Pydantic skips it by default); it prevents direct external modification.

### `__init_subclass__` Auto-configuration

```python
def __init_subclass__(cls, polymorphic_on=None, polymorphic_abstract=None, **kwargs):
    super().__init_subclass__(**kwargs)

    if '__mapper_args__' not in cls.__dict__:
        cls.__mapper_args__ = {}

    # Auto-set discriminator column
    if 'polymorphic_on' not in cls.__mapper_args__:
        cls.__mapper_args__['polymorphic_on'] = polymorphic_on or '_polymorphic_name'

    # Auto-detect if class is abstract
    if polymorphic_abstract is None:
        has_abc = ABC in cls.__mro__
        has_abstract_methods = bool(getattr(cls, '__abstractmethods__', set()))
        polymorphic_abstract = has_abc and has_abstract_methods

    cls.__mapper_args__['polymorphic_abstract'] = polymorphic_abstract
```

`__init_subclass__` executes when a subclass is defined. Effect: after inheriting `PolymorphicBaseMixin`, you don't need to write `__mapper_args__` manually; if the class inherits `ABC` and has abstract methods, it's automatically marked as `polymorphic_abstract=True`.

### Utility Methods

```python
@classmethod
def _is_joined_table_inheritance(cls) -> bool:
    """Subclass table name differs from parent → JTI"""

@classmethod
def get_concrete_subclasses(cls) -> list[type]:
    """Recursively get all non-abstract subclasses"""

@classmethod
def get_identity_to_class_map(cls) -> dict[str, type]:
    """Identity string to class mapping"""
    # {'emailnotification': EmailNotification, ...}
```

## `create_subclass_id_mixin()` — JTI Foreign Keys

JTI subclasses need a foreign key pointing to the parent table. Dynamically generates a Mixin:

```python
def create_subclass_id_mixin(parent_table_name: str) -> type:
    class SubclassIdMixin(SQLModelBase):
        id: UUID = Field(
            default_factory=uuid.uuid4,
            foreign_key=f'{parent_table_name}.id',
            primary_key=True,
        )
    SubclassIdMixin.__name__ = f'{ParentName}SubclassIdMixin'
    return SubclassIdMixin
```

Why dynamic generation instead of manual writing: different parent table names lead to different foreign key targets; the function parameter handles this.

**MRO order is critical**: The Mixin must be **first** in the inheritance list so its `id` overrides `UUIDTableBaseMixin`'s `id`:

```python
class WebSearchTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
#                   ↑ Must be first
    ...  # ToolSubclassIdMixin's id (with FK) takes priority // [!code highlight]
```

## `AutoPolymorphicIdentityMixin` — Auto Identity

```python
class AutoPolymorphicIdentityMixin:
    def __init_subclass__(cls, polymorphic_identity=None, **kwargs):
        super().__init_subclass__(**kwargs)

        if polymorphic_identity is not None:
            identity = polymorphic_identity        # Explicitly specified
        else:
            class_name = cls.__name__.lower()      # Lowercase class name

            parent_identity = None
            for base in cls.__mro__[1:]:
                if hasattr(base, '__mapper_args__'):
                    parent_identity = base.__mapper_args__.get('polymorphic_identity')
                    if parent_identity:
                        break

            if parent_identity:
                identity = f'{parent_identity}.{class_name}'
            else:
                identity = class_name

        cls.__mapper_args__['polymorphic_identity'] = identity
```

Auto-generated identities use dot-separated hierarchy:

```python
class Function(Tool, ...)     # identity = 'function'
class CodeInterpreter(Function, ...)  # identity = 'function.codeinterpreter'
```

## STI Column Registration (Two Phases)

STI subclass fields need to be added as nullable columns to the parent table. This happens in two phases:

### Phase 1: `_register_sti_columns()`

Called **before** `configure_mappers()`:

```python
@classmethod
def _register_sti_columns(cls):
    parent_table = None
    for base in cls.__mro__[1:]:
        if hasattr(base, '__table__'):
            parent_table = base.__table__
            break

    # JTI detection — skip if subclass has its own table
    if cls.__table__.name != parent_table.name:
        return

    for field_name, field_info in cls.model_fields.items():
        if field_name in parent_fields:   continue
        if field_name in existing_columns: continue

        column = get_column_from_field(field_info)
        column.nullable = True            # STI subclass fields must be nullable // [!code warning]
        parent_table.append_column(column) # [!code focus]
```

### Phase 2: `_register_sti_column_properties()`

Called **after** `configure_mappers()`:

```python
@classmethod
def _register_sti_column_properties(cls):
    child_mapper = inspect(cls).mapper
    parent_mapper = inspect(parent_class).mapper

    for field_name in cls.model_fields:
        if field_name in parent_fields: continue
        column = local_table.columns[field_name]

        child_mapper.add_property(field_name, ColumnProperty(column))
        parent_mapper.add_property(field_name, ColumnProperty(column))
```

### StrEnum Auto-conversion

STI subclass `StrEnum` fields are stored as strings in the database. SQLAlchemy only returns `str` when loading, so event listeners are registered for auto-conversion:

```python
def _register_strenum_coercion_for_subclass(cls):
    strenum_fields = {}  # Find all non-root StrEnum fields

    def _coerce(target):
        for field_name, enum_type in strenum_fields.items():
            raw = target.__dict__.get(field_name)
            if raw is not None and not isinstance(raw, enum_type):
                target.__dict__[field_name] = enum_type(str(raw))

    event.listens_for(cls, 'load')(_on_load)
    event.listens_for(cls, 'refresh')(_on_refresh)
```

## `_fix_polluted_model_fields()` — Fixing Default Value Pollution

During SQLModel inheritance, SQLAlchemy may replace field defaults with `InstrumentedAttribute` or `Column` objects:

```python
def _fix_polluted_model_fields(cls):
    for field_name, current_field in cls.model_fields.items():
        if not isinstance(current_field.default, (InstrumentedAttribute, Column)):
            continue

        # Find the original FieldInfo from MRO
        original = find_original_field_info(field_name)
        cls.model_fields[field_name] = FieldInfo(
            default=original.default,
            default_factory=original.default_factory,
            ...
        )
```

Called in multiple places to ensure Pydantic's model_fields always contains correct defaults.
