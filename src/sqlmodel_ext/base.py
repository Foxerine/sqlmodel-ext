"""
SQLModelBase and __DeclarativeMeta metaclass.

Provides a smart metaclass that handles:
- Automatic ``table=True`` for classes with TableBaseMixin
- Convenient keyword arguments (polymorphic_on, polymorphic_identity, etc.)
- Joined Table Inheritance (JTI) support
- Single Table Inheritance (STI) via registry.map_imperatively()
- Annotated sa_type extraction and injection
- Python 3.14 (PEP 649) compatibility
"""
import sys
import typing
from typing import Any, get_args, get_origin

from pydantic import ConfigDict
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined as Undefined
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Mapped, relationship as sa_relationship
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass, is_table_model_class, get_relationship_to

# Import _compat for side effects (Python 3.14 monkey-patches)
import sqlmodel_ext._compat  # noqa: F401

from sqlmodel_ext._sa_type import (
    _extract_sa_type_from_annotation,
    _resolve_annotations,
    _evaluate_annotation_from_string,
)

# Python 3.14+ support
if sys.version_info >= (3, 14):
    import annotationlib  # noqa: F401
else:
    annotationlib = None


class __DeclarativeMeta(SQLModelMetaclass):
    """
    A smart hybrid metaclass providing flexibility and clarity:

    1.  **Auto table=True**: If a class inherits TableBaseMixin, automatically applies ``table=True``.
    2.  **Explicit dict args**: Supports ``mapper_args={...}``, ``table_args={...}``, ``table_name='...'``.
    3.  **Convenient kwargs**: Supports common mapper args as top-level keywords (e.g. ``polymorphic_on``).
    4.  **Smart merge**: When both dict and kwargs are provided, merges them (kwargs take priority).
    """

    _KNOWN_MAPPER_KEYS = {
        "polymorphic_on",
        "polymorphic_identity",
        "polymorphic_abstract",
        "version_id_col",
        "concrete",
    }

    def __new__(cls, name, bases, attrs, **kwargs):
        # 1. Convention over configuration: auto table=True
        is_intended_as_table = any(getattr(b, '_has_table_mixin', False) for b in bases)
        if is_intended_as_table and 'table' not in kwargs:
            kwargs['table'] = True

        # 2. Detect STI scenario and preprocess
        parent_tablename = None
        for base in bases:
            if is_table_model_class(base) and hasattr(base, '__tablename__'):
                parent_tablename = base.__tablename__
                break

        will_be_table = kwargs.get('table', False)
        has_own_tablename = '__tablename__' in attrs or 'table_name' in kwargs

        # Check for FK to parent (JTI characteristic)
        has_fk_to_parent = False
        if parent_tablename is not None and will_be_table:
            for base in bases:
                if hasattr(base, 'model_fields'):
                    for field_name, field_info in base.model_fields.items():
                        fk = getattr(field_info, 'foreign_key', None)
                        if fk and isinstance(fk, str) and parent_tablename in fk:
                            has_fk_to_parent = True
                            break
                if has_fk_to_parent:
                    break

        # Only STI if no FK to parent
        if parent_tablename is not None and will_be_table and not has_own_tablename and not has_fk_to_parent:
            attrs['__tablename__'] = parent_tablename

        # 3. Smart merge __mapper_args__
        collected_mapper_args = {}

        if 'mapper_args' in kwargs:
            collected_mapper_args.update(kwargs.pop('mapper_args'))

        for key in cls._KNOWN_MAPPER_KEYS:
            if key in kwargs:
                collected_mapper_args[key] = kwargs.pop(key)

        if collected_mapper_args:
            existing = attrs.get('__mapper_args__', {}).copy()
            existing.update(collected_mapper_args)
            attrs['__mapper_args__'] = existing

        # Process other explicit args
        if 'table_args' in kwargs:
            attrs['__table_args__'] = kwargs.pop('table_args')
        if 'table_name' in kwargs:
            attrs['__tablename__'] = kwargs.pop('table_name')
        if 'abstract' in kwargs:
            attrs['__abstract__'] = kwargs.pop('abstract')

        # 4. Extract sa_type from Annotated metadata and inject into Field
        annotations, annotation_strings, eval_globals, eval_locals = _resolve_annotations(attrs)

        if annotations:
            attrs['__annotations__'] = annotations
            if annotationlib is not None:
                attrs['__annotate__'] = None

        for field_name, field_type in annotations.items():
            field_type = _evaluate_annotation_from_string(
                field_name, annotation_strings, field_type, eval_globals, eval_locals,
            )

            if isinstance(field_type, str) or isinstance(field_type, typing.ForwardRef):
                continue

            origin = get_origin(field_type)

            if origin is typing.ClassVar:
                continue

            if origin is Mapped:
                continue

            sa_type = _extract_sa_type_from_annotation(field_type)

            if sa_type is not None:
                field_value = attrs.get(field_name, Undefined)

                if field_value is Undefined:
                    attrs[field_name] = Field(sa_type=sa_type)
                elif isinstance(field_value, FieldInfo):
                    if not hasattr(field_value, 'sa_type') or field_value.sa_type is Undefined:
                        field_value.sa_type = sa_type

        # 5. Call parent __new__
        result = super().__new__(cls, name, bases, attrs, **kwargs)

        # 6. Fix: inherit parent's __sqlmodel_relationships__ for JTI
        if kwargs.get('table', False):
            for base in bases:
                if hasattr(base, '__sqlmodel_relationships__'):
                    for rel_name, rel_info in base.__sqlmodel_relationships__.items():
                        if rel_name not in result.__sqlmodel_relationships__:
                            result.__sqlmodel_relationships__[rel_name] = rel_info
                            if hasattr(base, rel_name):
                                base_attr = getattr(base, rel_name)
                                setattr(result, rel_name, base_attr)

        # 7. Forbid redefining parent's Relationship fields
        for base in bases:
            parent_relationships = getattr(base, '__sqlmodel_relationships__', {})
            for rel_name in parent_relationships:
                if rel_name in attrs:
                    raise TypeError(
                        f"Class {name} cannot redefine parent {base.__name__}'s "
                        f"Relationship field '{rel_name}'. "
                        f"Modify the relationship in the parent class instead."
                    )

        # 8. Fix: remove Relationship fields from model_fields/__pydantic_fields__
        relationships = getattr(result, '__sqlmodel_relationships__', {})
        if relationships:
            model_fields = getattr(result, 'model_fields', {})
            pydantic_fields = getattr(result, '__pydantic_fields__', {})

            fields_removed = False
            for rel_name in relationships:
                if rel_name in model_fields:
                    del model_fields[rel_name]
                    fields_removed = True
                if rel_name in pydantic_fields:
                    del pydantic_fields[rel_name]
                    fields_removed = True

            if fields_removed and hasattr(result, 'model_rebuild'):
                result.model_rebuild(force=True)

        return result

    def __init__(
        cls,
        classname: str,
        bases: tuple[type, ...],
        dict_: dict[str, typing.Any],
        **kw: typing.Any,
    ) -> None:
        """
        Override SQLModel's __init__ to support Joined Table Inheritance.

        SQLModel's original behavior skips DeclarativeMeta.__init__ if any base
        is a table model. This fix detects JTI scenarios and forces the call
        to create the child table.
        """
        from sqlmodel.main import is_table_model_class, DeclarativeMeta, ModelMetaclass

        if not is_table_model_class(cls):
            ModelMetaclass.__init__(cls, classname, bases, dict_, **kw)
            return

        base_is_table = any(is_table_model_class(base) for base in bases)

        if not base_is_table:
            cls._setup_relationships()
            DeclarativeMeta.__init__(cls, classname, bases, dict_, **kw)
            return

        # Detect JTI scenario
        current_tablename = getattr(cls, '__tablename__', None)

        parent_tablename = None
        for base in bases:
            if is_table_model_class(base) and hasattr(base, '__tablename__'):
                parent_tablename = base.__tablename__
                break

        has_different_tablename = (
            current_tablename is not None
            and parent_tablename is not None
            and current_tablename != parent_tablename
        )

        has_fk_to_parent = False

        def _normalize_tablename(name: str) -> str:
            return name.replace('_', '').lower()

        def _fk_matches_parent(fk_str: str, parent_table: str) -> bool:
            if not fk_str or not parent_table:
                return False
            parts = fk_str.split('.')
            if len(parts) >= 2:
                fk_table = parts[-2]
                return _normalize_tablename(fk_table) == _normalize_tablename(parent_table)
            return False

        if has_different_tablename and parent_tablename:
            for field_name, field_info in cls.model_fields.items():
                fk = getattr(field_info, 'foreign_key', None)
                if fk is not None and isinstance(fk, str) and _fk_matches_parent(fk, parent_tablename):
                    has_fk_to_parent = True
                    break

            if not has_fk_to_parent:
                for base in bases:
                    if hasattr(base, 'model_fields'):
                        for field_name, field_info in base.model_fields.items():
                            fk = getattr(field_info, 'foreign_key', None)
                            if fk is not None and isinstance(fk, str) and _fk_matches_parent(fk, parent_tablename):
                                has_fk_to_parent = True
                                break
                    if has_fk_to_parent:
                        break

        is_joined_inheritance = has_different_tablename and has_fk_to_parent

        if is_joined_inheritance:
            # JTI: create child table
            from sqlalchemy import Column, ForeignKey
            from sqlalchemy import Uuid as SA_UUID
            from sqlalchemy.exc import NoInspectionAvailable
            from sqlalchemy.orm.attributes import InstrumentedAttribute

            # Collect all ancestor table column names
            ancestor_column_names: set[str] = set()
            for ancestor in cls.__mro__:
                if ancestor is cls:
                    continue
                if is_table_model_class(ancestor):
                    try:
                        mapper = sa_inspect(ancestor)
                        for col in mapper.local_table.columns:
                            if col.name.startswith('_polymorphic'):
                                continue
                            ancestor_column_names.add(col.name)
                    except NoInspectionAvailable:
                        continue

            # Find child-own fields
            child_own_fields: set[str] = set()
            for field_name in cls.model_fields:
                is_inherited = False
                for base in bases:
                    if hasattr(base, 'model_fields') and field_name in base.model_fields:
                        is_inherited = True
                        break
                if not is_inherited:
                    child_own_fields.add(field_name)

            # Rebuild FK field
            fk_field_name = None
            for base in bases:
                if hasattr(base, 'model_fields'):
                    for field_name, field_info in base.model_fields.items():
                        fk = getattr(field_info, 'foreign_key', None)
                        pk = getattr(field_info, 'primary_key', False)
                        if fk is not None and isinstance(fk, str) and _fk_matches_parent(fk, parent_tablename):
                            fk_field_name = field_name
                            new_col = Column(
                                field_name,
                                SA_UUID(),
                                ForeignKey(fk),
                                primary_key=pk if pk else False
                            )
                            setattr(cls, field_name, new_col)
                            break
                    else:
                        continue
                    break

            # Remove ancestor columns from child class
            for col_name in ancestor_column_names:
                if col_name == fk_field_name:
                    continue
                if col_name == 'id':
                    continue
                if col_name in child_own_fields:
                    continue

                if col_name in cls.__dict__:
                    attr = cls.__dict__[col_name]
                    if isinstance(attr, (Column, InstrumentedAttribute)):
                        try:
                            delattr(cls, col_name)
                        except AttributeError:
                            pass

            # Setup only child-own relationships
            child_own_relationships: set[str] = set()
            for rel_name in cls.__sqlmodel_relationships__:
                is_inherited = False
                for base in bases:
                    if hasattr(base, '__sqlmodel_relationships__') and rel_name in base.__sqlmodel_relationships__:
                        is_inherited = True
                        break
                if not is_inherited:
                    child_own_relationships.add(rel_name)

            if child_own_relationships:
                cls._setup_relationships(only_these=child_own_relationships)

            DeclarativeMeta.__init__(cls, classname, bases, dict_, **kw)
        else:
            # STI: child shares parent table
            ModelMetaclass.__init__(cls, classname, bases, dict_, **kw)

            is_sti_child = (
                current_tablename is not None
                and parent_tablename is not None
                and current_tablename == parent_tablename
            )

            if is_sti_child:
                mapper_args = getattr(cls, '__mapper_args__', {})
                polymorphic_identity = mapper_args.get('polymorphic_identity')

                if polymorphic_identity is not None:
                    parent_cls = None
                    for base in bases:
                        if is_table_model_class(base) and hasattr(base, '__mapper__'):
                            parent_cls = base
                            break

                    if parent_cls is not None:
                        registry = parent_cls._sa_registry

                        rels = getattr(cls, '__sqlmodel_relationships__', {})
                        own_rels = {}
                        for rel_name, rel_info in rels.items():
                            is_inherited = any(
                                hasattr(base, '__sqlmodel_relationships__') and rel_name in base.__sqlmodel_relationships__
                                for base in bases
                            )
                            if not is_inherited:
                                own_rels[rel_name] = rel_info

                        properties = {}
                        if own_rels:
                            for rel_name, rel_info in own_rels.items():
                                if rel_info.sa_relationship:
                                    properties[rel_name] = rel_info.sa_relationship
                                else:
                                    raw_ann = cls.__annotations__.get(rel_name)
                                    if raw_ann:
                                        origin = get_origin(raw_ann)
                                        if origin is Mapped:
                                            ann = raw_ann.__args__[0]
                                        else:
                                            ann = raw_ann
                                        relationship_to = get_relationship_to(
                                            name=rel_name, rel_info=rel_info, annotation=ann
                                        )
                                        rel_kwargs: dict[str, typing.Any] = {}
                                        if rel_info.back_populates:
                                            rel_kwargs["back_populates"] = rel_info.back_populates
                                        if rel_info.cascade_delete:
                                            rel_kwargs["cascade"] = "all, delete-orphan"
                                        if rel_info.passive_deletes:
                                            rel_kwargs["passive_deletes"] = rel_info.passive_deletes
                                        if rel_info.link_model:
                                            ins = sa_inspect(rel_info.link_model)
                                            local_table = getattr(ins, "local_table")
                                            if local_table is None:
                                                raise RuntimeError(
                                                    f"Could not find secondary table for {rel_name}: {rel_info.link_model}"
                                                )
                                            rel_kwargs["secondary"] = local_table

                                        rel_args: list[typing.Any] = []
                                        if rel_info.sa_relationship_args:
                                            rel_args.extend(rel_info.sa_relationship_args)
                                        if rel_info.sa_relationship_kwargs:
                                            rel_kwargs.update(rel_info.sa_relationship_kwargs)

                                        properties[rel_name] = sa_relationship(relationship_to, *rel_args, **rel_kwargs)

                        registry.map_imperatively(
                            cls,
                            parent_cls.__table__,
                            inherits=parent_cls,
                            polymorphic_identity=polymorphic_identity,
                            properties=properties if properties else None,
                        )

    def _setup_relationships(cls, only_these: set[str] | None = None) -> None:
        """
        Set up SQLAlchemy relationship fields.

        :param only_these: If provided, only set up these relationships (for JTI child classes).
                          If None, set up all relationships (default behavior).
        """
        for rel_name, rel_info in cls.__sqlmodel_relationships__.items():
            if only_these is not None and rel_name not in only_these:
                continue
            if rel_info.sa_relationship:
                setattr(cls, rel_name, rel_info.sa_relationship)
                continue

            raw_ann = cls.__annotations__[rel_name]
            origin: typing.Any = get_origin(raw_ann)
            if origin is Mapped:
                ann = raw_ann.__args__[0]
            else:
                ann = raw_ann

            relationship_to = get_relationship_to(
                name=rel_name, rel_info=rel_info, annotation=ann
            )
            rel_kwargs: dict[str, typing.Any] = {}
            if rel_info.back_populates:
                rel_kwargs["back_populates"] = rel_info.back_populates
            if rel_info.cascade_delete:
                rel_kwargs["cascade"] = "all, delete-orphan"
            if rel_info.passive_deletes:
                rel_kwargs["passive_deletes"] = rel_info.passive_deletes
            if rel_info.link_model:
                ins = sa_inspect(rel_info.link_model)
                local_table = getattr(ins, "local_table")
                if local_table is None:
                    raise RuntimeError(
                        f"Couldn't find secondary table for {rel_info.link_model}"
                    )
                rel_kwargs["secondary"] = local_table

            rel_args: list[typing.Any] = []
            if rel_info.sa_relationship_args:
                rel_args.extend(rel_info.sa_relationship_args)
            if rel_info.sa_relationship_kwargs:
                rel_kwargs.update(rel_info.sa_relationship_kwargs)

            rel_value = sa_relationship(relationship_to, *rel_args, **rel_kwargs)
            setattr(cls, rel_name, rel_value)


class SQLModelBase(SQLModel, metaclass=__DeclarativeMeta):
    """
    Base class for all SQLModel models in sqlmodel_ext.

    Must be used together with TableBaseMixin or UUIDTableBaseMixin for table models.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, validate_by_name=True, extra='forbid')

    @classmethod
    def get_computed_field_names(cls) -> set[str]:
        """Get the set of computed_field names for this model class."""
        fields = cls.model_computed_fields
        return set(fields.keys()) if fields else set()
