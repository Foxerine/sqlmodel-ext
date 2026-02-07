"""
Python 3.14+ compatibility patches for SQLModel.

This module applies monkey-patches to fix SQLModel's compatibility issues
with Python 3.14 (PEP 649 deferred evaluation of annotations).

Import this module for side effects before using SQLModelBase.
"""
import sys

if sys.version_info >= (3, 14):
    import annotationlib  # noqa: F401
    import types
    from enum import StrEnum
    from typing import Literal, Union, get_args, get_origin

    from sqlalchemy import String
    from sqlalchemy.orm.attributes import InstrumentedAttribute

    # JSONB with fallback to generic JSON for non-PostgreSQL databases
    try:
        from sqlalchemy.dialects.postgresql import JSONB
    except ImportError:
        from sqlalchemy.types import JSON as JSONB  # type: ignore[assignment]

    # Monkey-patch 1: Fix get_sqlalchemy_type for Python 3.14
    import sqlmodel.main
    _original_get_sqlalchemy_type = sqlmodel.main.get_sqlalchemy_type

    def _patched_get_sqlalchemy_type(field):
        """
        Fix SQLModel's get_sqlalchemy_type for Python 3.14 type issues.

        Handles ForwardRef, ClassVar, Literal, Mapped, and custom types
        that cause issubclass errors under PEP 649.
        """
        # Check field.metadata (Pydantic-processed Annotated types)
        metadata = getattr(field, 'metadata', None)
        if metadata:
            for metadata_item in metadata:
                if hasattr(metadata_item, '__get_pydantic_core_schema__'):
                    try:
                        schema = metadata_item.__get_pydantic_core_schema__(None, None)
                        if isinstance(schema, dict) and 'metadata' in schema:
                            sa_type = schema['metadata'].get('sa_type')
                            if sa_type is not None:
                                return sa_type
                    except (TypeError, AttributeError, KeyError):
                        pass

        # Check InstrumentedAttribute defaults
        default = getattr(field, 'default', None)
        if default is not None:
            if isinstance(default, InstrumentedAttribute):
                return None

        annotation = getattr(field, 'annotation', None)
        if annotation is not None:
            # Handle Union[list/dict/..., None] types -> JSONB
            origin = get_origin(annotation)

            is_union = origin is Union or isinstance(annotation, types.UnionType)
            if is_union:
                args = get_args(annotation)
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    first_arg = non_none_args[0]
                    first_origin = get_origin(first_arg)
                    if first_origin in (list, dict, tuple, set):
                        return JSONB

            # Handle Literal[StrEnum.MEMBER] types -> String
            if origin is Literal:
                literal_args = get_args(annotation)
                if literal_args:
                    if all(isinstance(arg, StrEnum) for arg in literal_args):
                        return String

            # Check __sqlmodel_sa_type__ attribute
            if hasattr(annotation, '__sqlmodel_sa_type__'):
                return annotation.__sqlmodel_sa_type__

            # Check __get_pydantic_core_schema__ for custom types
            if hasattr(annotation, '__get_pydantic_core_schema__'):
                try:
                    schema = annotation.__get_pydantic_core_schema__(annotation, lambda x: None)
                    if isinstance(schema, dict) and 'metadata' in schema:
                        sa_type = schema['metadata'].get('sa_type')
                        if sa_type is not None:
                            return sa_type
                except (TypeError, AttributeError, KeyError):
                    pass

            import typing
            from sqlalchemy.orm import Mapped
            anno_type_name = type(annotation).__name__

            # ForwardRef: Relationship field annotations
            if anno_type_name == 'ForwardRef':
                return None

            # AnnotatedAlias: check for sa_type metadata
            if anno_type_name in ('AnnotatedAlias', '_AnnotatedAlias'):
                args = get_args(annotation)
                for md in args[1:]:
                    if hasattr(md, '__get_pydantic_core_schema__'):
                        try:
                            schema = md.__get_pydantic_core_schema__(None, None)
                            if isinstance(schema, dict) and 'metadata' in schema:
                                sa_type = schema['metadata'].get('sa_type')
                                if sa_type is not None:
                                    return sa_type
                        except (TypeError, AttributeError, KeyError):
                            pass

            # _GenericAlias or GenericAlias: typing generic types
            if anno_type_name in ('_GenericAlias', 'GenericAlias'):
                local_origin = get_origin(annotation)

                if local_origin is typing.ClassVar:
                    return None

                if local_origin in (list, dict, tuple, set):
                    field_info = getattr(field, 'field_info', None)
                    if field_info is None:
                        return None

            # Mapped type handling
            if 'Mapped' in anno_type_name or 'Mapped' in str(annotation):
                return None

            if get_origin(annotation) is Mapped:
                return None
            try:
                if annotation is Mapped or isinstance(annotation, type) and issubclass(annotation, Mapped):
                    return None
            except TypeError:
                pass

        return _original_get_sqlalchemy_type(field)

    sqlmodel.main.get_sqlalchemy_type = _patched_get_sqlalchemy_type

    # Monkey-patch 2: Fix InstrumentedAttribute defaults in inherited table classes
    import sqlmodel._compat as _compat
    from pydantic_core import PydanticUndefined as Undefined
    from sqlalchemy.orm import attributes as _sa_attributes

    _original_sqlmodel_table_construct = _compat.sqlmodel_table_construct

    def _patched_sqlmodel_table_construct(self_instance, values):
        """
        Fix sqlmodel_table_construct to skip InstrumentedAttribute defaults.

        In polymorphic inherited table classes, inherited Relationship fields
        may have InstrumentedAttribute as their default value, causing errors.
        """
        cls = type(self_instance)

        fields_to_set = {}

        for name, field in cls.model_fields.items():
            if name in values:
                fields_to_set[name] = values[name]
                continue

            if isinstance(field.default, _sa_attributes.InstrumentedAttribute):
                continue

            if field.default is not Undefined:
                fields_to_set[name] = field.default
            elif field.default_factory is not None:
                fields_to_set[name] = field.get_default(call_default_factory=True)

        for key, value in fields_to_set.items():
            if not isinstance(value, _sa_attributes.InstrumentedAttribute):
                setattr(self_instance, key, value)

        object.__setattr__(self_instance, '__pydantic_fields_set__', set(values.keys()))
        if not cls.__pydantic_root_model__:
            _extra = None
            if cls.model_config.get('extra') == 'allow':
                _extra = {}
                for k, v in values.items():
                    if k not in cls.model_fields:
                        _extra[k] = v
            object.__setattr__(self_instance, '__pydantic_extra__', _extra)

        if cls.__pydantic_post_init__:
            self_instance.model_post_init(None)
        elif not cls.__pydantic_root_model__:
            object.__setattr__(self_instance, '__pydantic_private__', None)

        for key in self_instance.__sqlmodel_relationships__:
            value = values.get(key, Undefined)
            if value is not Undefined:
                setattr(self_instance, key, value)

        return self_instance

    _compat.sqlmodel_table_construct = _patched_sqlmodel_table_construct
