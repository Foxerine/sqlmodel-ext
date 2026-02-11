"""
Joined Table Inheritance (JTI) and Single Table Inheritance (STI) utilities.

Provides helper functions and mixins for simplifying polymorphic table design
with SQLModel.

Usage Example::

    from sqlmodel_ext.base import SQLModelBase
    from sqlmodel_ext.mixins import UUIDTableBaseMixin
    from sqlmodel_ext.mixins.polymorphic import (
        PolymorphicBaseMixin,
        create_subclass_id_mixin,
        AutoPolymorphicIdentityMixin,
    )

    # 1. Define Base class (fields only, no table)
    class ASRBase(SQLModelBase):
        name: str
        base_url: str

    # 2. Define abstract parent (with table)
    class ASR(ASRBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
        pass

    # 3. Create subclass ID mixin
    ASRSubclassIdMixin = create_subclass_id_mixin('asr')

    # 4. Create concrete subclass
    class WebSocketASR(ASRSubclassIdMixin, ASR, AutoPolymorphicIdentityMixin, table=True):
        pass
"""
import logging
import types
import uuid
from abc import ABC
from enum import StrEnum
from typing import Annotated, Any, Union, get_args, get_origin
from uuid import UUID

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from sqlalchemy import Column, Enum as SAEnum, Integer, String, event, inspect
from sqlalchemy.orm import ColumnProperty, Mapped, mapped_column
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlmodel import Field
from sqlmodel.main import get_column_from_field

from sqlmodel_ext.base import SQLModelBase

logger = logging.getLogger(__name__)

# Queue for deferred STI subclass column registration
# After all models are loaded, call register_sti_columns_for_all_subclasses()
_sti_subclasses_to_register: list[type] = []


def register_sti_columns_for_all_subclasses() -> None:
    """
    Register columns for all queued STI subclasses (Phase 1: add columns to table).

    Call this before ``configure_mappers()``.
    Adds STI subclass fields to the parent table's metadata and
    fixes model_fields polluted by Column objects.
    """
    for cls in _sti_subclasses_to_register:
        try:
            cls._register_sti_columns()
        except Exception as e:
            logger.warning(f"Error registering STI columns for {cls.__name__}: {e}")

        try:
            _fix_polluted_model_fields(cls)
        except Exception as e:
            logger.warning(f"Error fixing model_fields for STI subclass {cls.__name__}: {e}")


def register_sti_column_properties_for_all_subclasses() -> None:
    """
    Add column properties to mapper for all queued STI subclasses (Phase 2).

    Call this after ``configure_mappers()``.
    Adds STI subclass fields as ColumnProperty to the mapper,
    and registers StrEnum field auto-coercion on load/refresh.
    """
    for cls in _sti_subclasses_to_register:
        try:
            cls._register_sti_column_properties()
        except Exception as e:
            logger.warning(f"Error registering STI column properties for {cls.__name__}: {e}")

    # Register StrEnum field auto-coercion
    for cls in _sti_subclasses_to_register:
        try:
            _register_strenum_coercion_for_subclass(cls)
        except Exception as e:
            logger.warning(f"Error registering StrEnum coercion for {cls.__name__}: {e}")

    _sti_subclasses_to_register.clear()


def _extract_strenum_type(annotation: Any) -> type[StrEnum] | None:
    """
    Extract StrEnum subclass type from a type annotation.

    Handles ``Annotated`` and ``Optional``/``Union`` wrappers.
    Returns ``None`` for non-StrEnum annotations.
    """
    if annotation is None:
        return None

    # Unwrap Annotated[T, ...]
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]

    # Unwrap T | None or Optional[T]
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            annotation = args[0]

    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation

    return None


def _register_strenum_coercion_for_subclass(cls: type) -> None:
    """
    Register SQLAlchemy load/refresh event listeners for StrEnum auto-coercion.

    STI subclass StrEnum columns may be stored as ``String()`` (or ``INTEGER`` etc.)
    in the shared table. SQLAlchemy loads them as raw ``str`` (or ``int``) instead of
    the declared StrEnum type. This registers event listeners that automatically coerce
    raw values to the correct StrEnum type after loading.

    Fields defined on the STI root class (the one that owns ``__tablename__``) use native
    SQLAlchemy Enum columns and are loaded correctly — they are excluded from coercion.
    """
    model_fields = getattr(cls, 'model_fields', None)
    if not model_fields:
        return

    # Find root class fields (use native SAEnum columns, no coercion needed)
    root_fields: set[str] = set()
    for base in cls.__mro__[1:]:
        if '__tablename__' in base.__dict__ and hasattr(base, 'model_fields'):
            root_fields = set(base.model_fields.keys())
            break

    # Collect non-root StrEnum fields
    strenum_fields: dict[str, type[StrEnum]] = {}
    for field_name, field_info in model_fields.items():
        if field_name in root_fields:
            continue
        enum_type = _extract_strenum_type(field_info.annotation)
        if enum_type is not None:
            strenum_fields[field_name] = enum_type

    if not strenum_fields:
        return

    def _coerce(target: Any) -> None:
        """Coerce raw DB values (str/int) to declared StrEnum types."""
        d = target.__dict__
        for field_name, enum_type in strenum_fields.items():
            raw = d.get(field_name)
            if raw is not None and not isinstance(raw, enum_type):
                d[field_name] = enum_type(str(raw))

    @event.listens_for(cls, 'load')
    def _on_load(target, context):
        _coerce(target)

    @event.listens_for(cls, 'refresh')
    def _on_refresh(target, context, attrs):
        _coerce(target)


def _fix_polluted_model_fields(cls: type) -> None:
    """
    Fix model_fields polluted by SQLAlchemy InstrumentedAttribute or Column objects.

    When SQLModel classes inherit from table parents, SQLAlchemy may replace
    original field default values with InstrumentedAttribute or Column objects.
    This function finds the original field definitions from the MRO and restores them.

    :param cls: The class to fix
    """
    if not hasattr(cls, 'model_fields'):
        return

    def find_original_field_info(field_name: str) -> FieldInfo | None:
        for base in cls.__mro__[1:]:
            if hasattr(base, 'model_fields') and field_name in base.model_fields:
                field_info = base.model_fields[field_name]
                if not isinstance(field_info.default, (InstrumentedAttribute, Column)):
                    return field_info
        return None

    for field_name, current_field in cls.model_fields.items():
        if not isinstance(current_field.default, (InstrumentedAttribute, Column)):
            continue

        original = find_original_field_info(field_name)
        if original is None:
            continue

        if original.default_factory:
            new_field = FieldInfo(
                default_factory=original.default_factory,
                annotation=current_field.annotation,
                json_schema_extra=current_field.json_schema_extra,
            )
        elif original.default is not PydanticUndefined:
            new_field = FieldInfo(
                default=original.default,
                annotation=current_field.annotation,
                json_schema_extra=current_field.json_schema_extra,
            )
        else:
            continue

        if hasattr(current_field, 'foreign_key'):
            new_field.foreign_key = current_field.foreign_key
        if hasattr(current_field, 'primary_key'):
            new_field.primary_key = current_field.primary_key

        cls.model_fields[field_name] = new_field


def create_subclass_id_mixin(parent_table_name: str) -> type['SQLModelBase']:
    """
    Dynamically create a SubclassIdMixin class for JTI.

    In joined table inheritance, subclasses need a foreign key pointing to the
    parent table's primary key. This function generates a mixin providing that FK field.

    :param parent_table_name: Parent table name (e.g. 'asr', 'tts', 'tool')
    :returns: A mixin class with an id field (FK + PK + default_factory=uuid.uuid4)

    Example::

        ASRSubclassIdMixin = create_subclass_id_mixin('asr')
        class WebSocketASR(ASRSubclassIdMixin, ASR, table=True):
            pass

    Note:
        The generated mixin should be first in the inheritance list to ensure
        proper MRO resolution over UUIDTableBaseMixin's id field.
    """
    if not parent_table_name:
        raise ValueError("parent_table_name must not be empty")

    class_name_parts = parent_table_name.split('_')
    class_name = ''.join(part.capitalize() for part in class_name_parts) + 'SubclassIdMixin'

    _parent_table_name = parent_table_name

    class SubclassIdMixin(SQLModelBase):
        id: UUID = Field(
            default_factory=uuid.uuid4,
            foreign_key=f'{_parent_table_name}.id',
            primary_key=True,
        )

        @classmethod
        def __pydantic_init_subclass__(cls, **kwargs):
            super().__pydantic_init_subclass__(**kwargs)
            _fix_polluted_model_fields(cls)

    SubclassIdMixin.__name__ = class_name
    SubclassIdMixin.__qualname__ = class_name
    SubclassIdMixin.__doc__ = f"""
    ID Mixin for {parent_table_name} subclasses.

    Provides a foreign key pointing to the {parent_table_name} parent table.
    Place first in MRO to override the inherited id field.
    """

    return SubclassIdMixin


class AutoPolymorphicIdentityMixin:
    """
    Mixin that auto-generates polymorphic_identity from the class name.

    Format: ``{parent_polymorphic_identity}.{classname_lowercase}``

    If no parent polymorphic_identity exists, uses the class name in lowercase.

    Also handles STI subclass column registration.

    Example (JTI)::

        class Tool(UUIDTableBaseMixin, polymorphic_on='_polymorphic_name', polymorphic_abstract=True):
            _polymorphic_name: str

        class Function(Tool, AutoPolymorphicIdentityMixin, polymorphic_abstract=True):
            pass  # identity = 'function'

        class CodeInterpreterFunction(Function, table=True):
            pass  # identity = 'function.codeinterpreterfunction'

    Example (STI)::

        class UserFile(UUIDTableBaseMixin, PolymorphicBaseMixin, table=True, polymorphic_abstract=True):
            user_id: UUID

        class PendingFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
            upload_deadline: datetime | None = None  # auto-added to userfile table
    """

    def __init_subclass__(cls, polymorphic_identity: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        if polymorphic_identity is not None:
            identity = polymorphic_identity
        else:
            class_name = cls.__name__.lower()

            parent_identity = None
            for base in cls.__mro__[1:]:
                if hasattr(base, '__mapper_args__') and isinstance(base.__mapper_args__, dict):
                    parent_identity = base.__mapper_args__.get('polymorphic_identity')
                    if parent_identity:
                        break

            if parent_identity:
                identity = f'{parent_identity}.{class_name}'
            else:
                identity = class_name

        if '__mapper_args__' not in cls.__dict__:
            cls.__mapper_args__ = {}

        if 'polymorphic_identity' not in cls.__mapper_args__:
            cls.__mapper_args__['polymorphic_identity'] = identity

        _sti_subclasses_to_register.append(cls)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        _fix_polluted_model_fields(cls)

    @classmethod
    def _register_sti_columns(cls) -> None:
        """
        Register STI subclass fields to the parent table's column definitions.

        JTI classes are automatically skipped (they have their own tables).
        """
        parent_table = None
        parent_fields: set[str] = set()

        for base in cls.__mro__[1:]:
            if hasattr(base, '__table__') and base.__table__ is not None:
                parent_table = base.__table__
                if hasattr(base, 'model_fields'):
                    parent_fields.update(base.model_fields.keys())
                break

        if parent_table is None:
            return

        # JTI detection
        if hasattr(cls, '__table__') and cls.__table__ is not None:
            if cls.__table__.name != parent_table.name:
                return

        if not hasattr(cls, 'model_fields'):
            return

        existing_columns = {col.name for col in parent_table.columns}

        for field_name, field_info in cls.model_fields.items():
            if field_name in parent_fields:
                continue
            if field_name.startswith('_'):
                continue
            if field_name in existing_columns:
                # Detect type conflicts: Integer vs non-Integer is incompatible
                existing_col = parent_table.columns[field_name]
                try:
                    new_col = get_column_from_field(field_info)
                except Exception:
                    continue
                new_type = new_col.type
                if isinstance(new_type, SAEnum):
                    new_type = String()
                if isinstance(existing_col.type, Integer) != isinstance(new_type, Integer):
                    raise TypeError(
                        f"STI column type conflict: {cls.__name__}.{field_name} type "
                        f"({type(new_type).__name__}) is incompatible with existing "
                        f"{parent_table.name}.{field_name} type "
                        f"({type(existing_col.type).__name__}). "
                        f"Use a different field name."
                    )
                continue

            try:
                column = get_column_from_field(field_info)
                column.name = field_name
                column.key = field_name
                column.nullable = True
                parent_table.append_column(column)
            except Exception as e:
                logger.warning(f"Failed to create column {field_name} for {cls.__name__}: {e}")

    @classmethod
    def _register_sti_column_properties(cls) -> None:
        """
        Add STI subclass columns as ColumnProperty to the mapper.

        Call after ``configure_mappers()``.
        JTI classes are automatically skipped.
        """
        parent_table = None
        parent_class = None
        for base in cls.__mro__[1:]:
            if hasattr(base, '__table__') and base.__table__ is not None:
                parent_table = base.__table__
                parent_class = base
                break

        if parent_table is None:
            return

        if hasattr(cls, '__table__') and cls.__table__ is not None:
            if cls.__table__.name != parent_table.name:
                return

        child_mapper = inspect(cls).mapper
        parent_mapper = inspect(parent_class).mapper
        local_table = child_mapper.local_table

        parent_fields: set[str] = set()
        if hasattr(parent_class, 'model_fields'):
            parent_fields.update(parent_class.model_fields.keys())

        if not hasattr(cls, 'model_fields'):
            return

        child_existing_props = {p.key for p in child_mapper.column_attrs}
        parent_existing_props = {p.key for p in parent_mapper.column_attrs}

        for field_name in cls.model_fields:
            if field_name in parent_fields:
                continue
            if field_name.startswith('_'):
                continue
            if field_name not in local_table.columns:
                continue

            column = local_table.columns[field_name]

            if field_name not in child_existing_props:
                try:
                    prop = ColumnProperty(column)
                    child_mapper.add_property(field_name, prop)
                except Exception as e:
                    logger.warning(f"Failed to add column property {field_name} to {cls.__name__}: {e}")

            if field_name not in parent_existing_props:
                try:
                    prop = ColumnProperty(column)
                    parent_mapper.add_property(field_name, prop)
                except Exception as e:
                    logger.warning(
                        f"Failed to add column property {field_name} from {cls.__name__} to parent {parent_class.__name__}: {e}"
                    )


class PolymorphicBaseMixin:
    """
    Mixin that auto-configures polymorphic settings for inheritance base classes.

    Automatically sets:
    - ``polymorphic_on='_polymorphic_name'``: Uses _polymorphic_name as discriminator
    - ``_polymorphic_name: str``: Defines the discriminator field (with index)
    - ``polymorphic_abstract=True``: When the class inherits ABC and has abstract methods

    Usage::

        from abc import ABC

        class MyTool(UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
            pass  # Auto-configured: polymorphic_on, polymorphic_abstract
    """

    _polymorphic_name: Mapped[str] = mapped_column(String, index=True)
    """
    Polymorphic discriminator field identifying the concrete subclass type.

    Uses single underscore prefix:
    - Stored in database
    - Not included in API serialization
    - Prevents direct external modification
    """

    def __init_subclass__(
        cls,
        polymorphic_on: str | None = None,
        polymorphic_abstract: bool | None = None,
        **kwargs
    ):
        super().__init_subclass__(**kwargs)

        if '__mapper_args__' not in cls.__dict__:
            cls.__mapper_args__ = {}

        if 'polymorphic_on' not in cls.__mapper_args__:
            cls.__mapper_args__['polymorphic_on'] = polymorphic_on or '_polymorphic_name'

        if 'polymorphic_abstract' not in cls.__mapper_args__:
            if polymorphic_abstract is None:
                has_abc = ABC in cls.__mro__
                has_abstract_methods = bool(getattr(cls, '__abstractmethods__', set()))
                polymorphic_abstract = has_abc and has_abstract_methods

            cls.__mapper_args__['polymorphic_abstract'] = polymorphic_abstract

    @classmethod
    def _is_joined_table_inheritance(cls) -> bool:
        """
        Detect whether this class uses Joined Table Inheritance.

        :returns: True for JTI, False for STI or no subclasses
        """
        mapper = inspect(cls)
        base_table_name = mapper.local_table.name

        for subclass in cls.__subclasses__():
            sub_mapper = inspect(subclass)
            if sub_mapper.local_table.name != base_table_name:
                return True

        return False

    @classmethod
    def get_concrete_subclasses(cls) -> list[type['PolymorphicBaseMixin']]:
        """
        Recursively get all concrete (non-abstract) subclasses.

        Used for ``selectin_polymorphic`` loading strategy.

        :returns: List of all concrete subclasses (excluding polymorphic_abstract=True)
        """
        result: list[type[PolymorphicBaseMixin]] = []
        for subclass in cls.__subclasses__():
            mapper = inspect(subclass)
            if not mapper.polymorphic_abstract:
                result.append(subclass)
            if hasattr(subclass, 'get_concrete_subclasses'):
                result.extend(subclass.get_concrete_subclasses())
        return result

    @classmethod
    def get_polymorphic_discriminator(cls) -> str:
        """
        Get the polymorphic discriminator field name.

        :returns: Discriminator field name (e.g. '_polymorphic_name')
        :raises ValueError: If polymorphic_on is not configured
        """
        polymorphic_on = inspect(cls).polymorphic_on
        if polymorphic_on is None:
            raise ValueError(
                f"{cls.__name__} does not have polymorphic_on configured. "
                f"Ensure it correctly inherits PolymorphicBaseMixin."
            )
        return polymorphic_on.key

    @classmethod
    def get_identity_to_class_map(cls) -> dict[str, type['PolymorphicBaseMixin']]:
        """
        Get mapping from polymorphic_identity to concrete subclass.

        Includes all levels of concrete subclasses.

        :returns: Dict mapping identity strings to subclass types
        """
        result: dict[str, type[PolymorphicBaseMixin]] = {}
        for subclass in cls.get_concrete_subclasses():
            identity = inspect(subclass).polymorphic_identity
            if identity:
                result[identity] = subclass
        return result
