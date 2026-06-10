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
import copy
import logging
import re
import sys
import types
import typing
from typing import Any, Self, Sequence, get_args, get_origin

import annotated_types as at
from pydantic import ConfigDict, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined as Undefined
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Mapped, declared_attr, relationship as sa_relationship
from sqlmodel import Field, SQLModel
from sqlmodel.main import (
    SQLModelMetaclass,
    is_table_model_class,
    get_relationship_to,
    FieldInfo as SQLModelFieldInfo,  # Internal API: stable since sqlmodel 0.0.22
    FieldInfoMetadata,  # Internal API: pydantic-rebuild-safe sa_type carrier
    get_column_from_field,  # Internal API: stable since sqlmodel 0.0.22
)

# sqlmodel 0.0.32+ FieldInfoMetadata is a @dataclass whose auto-generated
# __eq__ causes __hash__ = None.  Annotated[T, FieldInfoMetadata(...)] then
# becomes unhashable, breaking FastAPI's OpenAPI set-based dedup in
# get_definitions().  Restore identity-based hashing until upstream fixes it.
# Ref: https://github.com/fastapi/sqlmodel/pull/1889
if getattr(FieldInfoMetadata, '__hash__') is None:
    FieldInfoMetadata.__hash__ = object.__hash__  # type: ignore[assignment]

# sqlmodel's FieldInfoMetadata fields default to sqlmodel's own Undefined
# sentinel (distinct from pydantic_core PydanticUndefined imported above);
# capture it to detect "sa_type not yet set" without importing sqlmodel internals.
_FIM_UNSET_SA_TYPE = FieldInfoMetadata().sa_type

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

logger = logging.getLogger(__name__)


def _merge_field_info_attrs(target: SQLModelFieldInfo, source: SQLModelFieldInfo) -> None:
    """
    Merge explicitly-set attributes from ``source`` into ``target``.

    Used when ``Annotated[Str64, Field(unique=True)]`` expands to multiple FieldInfo:
    ``Annotated[str, Field(max_length=64), Field(unique=True)]``.
    Merges attributes dynamically (via ``__slots__`` / ``__annotations__`` / ``__dict__``),
    no hardcoded attribute names — upstream additions are handled automatically.
    """
    attr_names: set[str] = set()
    for klass in type(source).__mro__:
        for slot in getattr(klass, '__slots__', ()):
            if not slot.startswith('_'):
                attr_names.add(slot)
        for ann in getattr(klass, '__annotations__', {}):
            if not ann.startswith('_'):
                attr_names.add(ann)
    for key in (vars(source) if hasattr(source, '__dict__') else ()):
        if not key.startswith('_'):
            attr_names.add(key)

    # metadata lists should be merged, not replaced
    attr_names.discard('metadata')

    for attr_name in attr_names:
        try:
            val = getattr(source, attr_name)
        except AttributeError:
            continue

        if val is None or val is Undefined:
            continue
        if isinstance(val, (list, dict, set)) and not val:
            continue

        # Boolean flags (unique, primary_key, etc.): don't overwrite True with False
        if isinstance(val, bool) and not val:
            current = getattr(target, attr_name, None)
            if isinstance(current, bool) and current:
                continue

        try:
            setattr(target, attr_name, val)
        except (AttributeError, TypeError):
            continue  # read-only slot

    # Merge metadata lists (Pydantic validator metadata like MaxLen, etc.)
    source_meta = getattr(source, 'metadata', None)
    if source_meta:
        target_meta = getattr(target, 'metadata', None) or []
        target.metadata = list(target_meta) + list(source_meta)


def _find_field_info_in_annotated(annotation: Any) -> SQLModelFieldInfo | None:
    """
    Extract SQLModel ``FieldInfo`` embedded in ``Annotated[T, Field(...)]`` metadata.

    Used by the metaclass sa_type injection loop. When a field is declared
    ``Annotated[X, Field(default_factory=list, ...)]`` *without* an explicit
    ``= ...`` assignment, ``attrs[field_name]`` is ``Undefined``. Replacing
    it with a fresh ``Field(sa_type=sa_type)`` would discard the user's
    Field metadata (``default_factory``, ``max_length``, validators, ...)
    that lives inside the Annotated args. This helper recovers that
    FieldInfo so the caller can attach ``sa_type`` to the user's Field
    instead of clobbering it. The bug only surfaces after 2+ levels of
    inheritance: single-class instantiation goes through Pydantic's native
    Annotated path and works, but child classes rebuild ``model_fields``
    from the clobbered ``attrs`` and the field becomes silently
    ``is_required=True``.

    Multiple FieldInfo args (e.g. ``Annotated[Str64, Field(unique=True)]``
    expands to ``Annotated[str, Field(max_length=64), Field(unique=True)]``)
    are merged into a single shallow copy so the shared Annotated metadata
    singletons are never mutated.

    :param annotation: Field type annotation
    :returns: Merged SQLModelFieldInfo (shallow copy, safe to mutate), or None
    """
    if get_origin(annotation) is not typing.Annotated:
        return None
    args = get_args(annotation)
    if len(args) < 2:
        return None
    sqlmodel_fis: list[SQLModelFieldInfo] = [
        arg for arg in args[1:] if isinstance(arg, SQLModelFieldInfo)
    ]
    if not sqlmodel_fis:
        return None
    merged = copy.copy(sqlmodel_fis[0])
    for extra_fi in sqlmodel_fis[1:]:
        _merge_field_info_attrs(merged, extra_fi)
    return merged


def _durably_set_sa_type(field_info: Any, sa_type: Any) -> None:
    """
    Inject ``sa_type`` so it survives Pydantic's model_fields rebuild.

    Root cause this fixes: the metaclass extracts ``sa_type`` from
    ``Array[T]`` / custom Annotated handlers and must hand it to SQLModel's
    column builder. A plain ``setattr(field_info, 'sa_type', sa_type)`` is
    LOST: ``SQLModelMetaclass.__new__`` (invoked from our ``super().__new__``)
    runs ``get_column_from_field`` *before* step-7's SQLModelFieldInfo
    restore, and Pydantic has by then rebuilt ``model_fields`` into fresh
    FieldInfo objects that never saw the post-hoc attribute. The previous
    code only fixed the no-``= Field(...)`` branch (c00696c); the explicit
    ``Array[T] = Field(default_factory=list)`` form still raised
    ``<class 'list'> has no matching SQLAlchemy type``.

    Fix: write ``sa_type`` into a ``FieldInfoMetadata`` entry inside the
    FieldInfo's pydantic ``metadata`` list — the exact channel SQLModel's
    own ``Field(sa_type=...)`` uses and which ``get_sqlalchemy_type``
    (via ``_get_sqlmodel_field_value``) reads *first*. Pydantic preserves
    the ``metadata`` list across rebuilds, so the type survives into the
    column build. The instance attribute is also set as a belt-and-braces
    fallback for any direct ``getattr(field_info, 'sa_type')`` reader.

    :param field_info: target FieldInfo (user's Field or recovered Annotated FI)
    :param sa_type: SQLAlchemy type extracted from the annotation handler
    """
    md = list(getattr(field_info, 'metadata', None) or [])
    existing = next(
        (m for m in md if isinstance(m, FieldInfoMetadata)), None
    )
    if existing is not None:
        if existing.sa_type is _FIM_UNSET_SA_TYPE:
            existing.sa_type = sa_type
    else:
        md.append(FieldInfoMetadata(sa_type=sa_type))
        field_info.metadata = md
    if getattr(field_info, 'sa_type', Undefined) is Undefined:
        try:
            field_info.sa_type = sa_type
        except (AttributeError, TypeError):
            pass


_FIELDINFO_NON_CONSTRAINT_ATTRS = (
    'default', 'default_factory', 'alias', 'serialization_alias', 'validation_alias',
    'title', 'description', 'examples', 'json_schema_extra',
    'exclude', 'repr', 'init', 'init_var', 'kw_only', 'discriminator',
    'frozen', 'validate_default',
    'primary_key', 'foreign_key', 'unique', 'nullable', 'index',
    'sa_type', 'sa_column', 'sa_column_args', 'sa_column_kwargs',
    'ondelete', 'schema_extra',
)
"""Non-constraint attributes on FieldInfo -- sa/orm/default/alias/docs, safe to keep in the outer Annotated.

See the ``_split_metadata_for_optional`` docstring for semantics.
Constraint-related attributes (ge/le/gt/lt/multiple_of/max_digits/decimal_places/
min_length/max_length/pattern/strict) have already been expanded by Pydantic into
``Ge/Le/Gt/Lt/MultipleOf/_PydanticGeneralMetadata/MinLen/MaxLen/Predicate`` markers
inside ``FieldInfo.metadata``, so they need no re-detection on the FieldInfo attrs."""


def _split_metadata_for_optional(
        metadata: tuple[typing.Any, ...],
) -> tuple[list[typing.Any], list[typing.Any]]:
    """Split the metadata of ``Annotated[T, *metadata]`` into (inner_constraints, outer_safe).

    **Why**: Pydantic constraint markers (``Ge/Le/MultipleOf/_PydanticGeneralMetadata``
    etc.) raise ``TypeError: Unable to apply constraint ... to supplied value None`` on
    None -- so the constraint markers of a ``T | None`` field must be wrapped inside the
    inner ``Annotated[T, ...]`` so Pydantic only applies them when the value is not
    None. Schema markers (``PlainSerializer/BeforeValidator`` etc.) and ORM markers
    (``sa_type``/``default`` etc.) are None-safe and may stay in the outer layer. This
    is the essence behind the hand-written nested-Annotated form of optional
    constrained aliases like ``OptionalNonNegativeDecimal38_18``.

    Classification rules:

    - ``annotated_types.BaseMetadata`` subclasses (``Ge/Le/Gt/Lt/MultipleOf/MinLen/
      MaxLen/Predicate`` etc.) -> inner (all constraint validators, crash on None)
    - Pydantic ``FieldInfo`` -> split: ``fi.metadata`` (the constraint marker sequence
      Pydantic already expanded) -> inner; the remaining schema/orm attrs are rebuilt
      into a fresh FieldInfo -> outer
    - objects with ``max_digits`` / ``decimal_places`` / ``pattern`` attributes that are
      not FieldInfo (i.e. ``_PydanticGeneralMetadata``) -> inner
    - everything else (``PlainSerializer/WrapSerializer/BeforeValidator/AfterValidator/
      PlainValidator/WrapValidator`` etc.) -> outer (validators are expected to be
      None-safe, like a reject-float BeforeValidator)
    """
    inner: list[typing.Any] = []
    outer: list[typing.Any] = []
    for m in metadata:
        if isinstance(m, at.BaseMetadata):
            inner.append(m)
        elif isinstance(m, FieldInfo):
            inner.extend(m.metadata)
            outer_fi = _strip_constraint_metadata_from_field_info(m)
            if outer_fi is not None:
                outer.append(outer_fi)
        elif isinstance(m, at.GroupedMetadata):
            # GroupedMetadata protocol (``pydantic.StringConstraints`` etc.):
            # iterating yields a sequence of _PydanticGeneralMetadata /
            # annotated_types markers; all go inner.
            inner.extend(list(m))
        elif hasattr(m, 'max_digits') or hasattr(m, 'decimal_places') or hasattr(m, 'pattern'):
            # _PydanticGeneralMetadata (private, duck-typed): max_digits/decimal_places/pattern
            inner.append(m)
        else:
            outer.append(m)
    return inner, outer


def _strip_constraint_metadata_from_field_info(fi: FieldInfo) -> FieldInfo | None:
    """Copy an outer-layer version of a FieldInfo: drop ``fi.metadata`` (constraint markers,
    moved inner), keep sa_type/default/alias/description and other schema/orm attrs.
    Returns None when no non-constraint attrs remain."""
    new_kwargs: dict[str, typing.Any] = {}
    for attr in _FIELDINFO_NON_CONSTRAINT_ATTRS:
        v = getattr(fi, attr, Undefined)
        if v is None or v is Undefined:
            continue
        new_kwargs[attr] = v
    if not new_kwargs:
        return None
    return Field(**new_kwargs)


def _make_annotation_optional(annotation: typing.Any) -> typing.Any:
    """
    Convert type annotation to optional: ``T → T | None``

    Places ``| None`` inside ``Annotated[]`` to preserve Field metadata.

    **Key constraint** (isomorphic to ``OptionalNonNegativeDecimal38_18``): when ``T``
    carries constraint validators (``Ge/Le/MultipleOf/MaxDigits/DecimalPlaces`` etc.),
    the constraints **must** be nested into the inner ``Annotated[T, *constraints]``,
    otherwise Pydantic crashes parsing JSON ``null`` with
    ``TypeError: Unable to apply constraint 'ge' to supplied value None``.
    Schema/serializer/ORM markers (``PlainSerializer/sa_type/default`` etc.) are
    None-safe and stay outer.

    Examples::

        str → str | None
        Annotated[float, Ge(0)] → Annotated[Annotated[float, Ge(0)] | None]
        Annotated[Decimal, Field(ge=0, sa_type=...), PlainSerializer(...)]
            → Annotated[Annotated[Decimal, Ge(0)] | None, Field(sa_type=...), PlainSerializer(...)]
        str | None → str | None  (already optional, unchanged)
    """
    origin = get_origin(annotation)

    # Annotated[T, metadata...]: split constraint/schema metadata, then build
    # the nested optional annotation.
    if origin is typing.Annotated:
        args = get_args(annotation)
        inner_type = args[0]
        metadata = args[1:]
        inner_meta, outer_meta = _split_metadata_for_optional(metadata)
        # Recurse into inner_type: it may itself be Annotated (nested aliases).
        recursed_inner_type = _maybe_recurse_into_annotated(inner_type)
        if inner_meta:
            inner_annotated = typing.Annotated[tuple([recursed_inner_type, *inner_meta])]
        else:
            inner_annotated = recursed_inner_type
        # ``typing.Union[X, None]`` is equivalent to PEP 604 ``X | None``, but type
        # checkers can correctly infer ``inner_annotated`` is ``Annotated[...]``
        # (a dynamic type without static ``__or__`` support).
        optional_inner = typing.Union[inner_annotated, type(None)]  # pyright: ignore[reportGeneralTypeIssues]
        if outer_meta:
            return typing.Annotated[tuple([optional_inner, *outer_meta])]
        return optional_inner

    # Union / UnionType already contains None → unchanged
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        if type(None) in args:
            return annotation

    # T → T | None
    return annotation | None


def _maybe_recurse_into_annotated(annotation: typing.Any) -> typing.Any:
    """For nested Annotated (``Annotated[Annotated[T, ...], ...]``), split the inner
    layer's constraints too. Non-Annotated values are returned unchanged."""
    if get_origin(annotation) is typing.Annotated:
        args = get_args(annotation)
        inner_type = args[0]
        metadata = args[1:]
        inner_meta, outer_meta = _split_metadata_for_optional(metadata)
        # This layer is already at the outer position or wrapped as inner --
        # keep its structure: the constraints are in inner_meta, reassemble
        # into Annotated without appending | None (the parent caller does that).
        recursed = _maybe_recurse_into_annotated(inner_type)
        all_meta = list(inner_meta) + list(outer_meta)
        if all_meta:
            return typing.Annotated[tuple([recursed, *all_meta])]
        return recursed
    return annotation


def _apply_all_fields_optional(
        annotations: dict[str, typing.Any],
        attrs: dict[str, typing.Any],
        bases: tuple[type, ...],
) -> None:
    """
    Automatically convert inherited fields to optional.

    Two-step strategy (same MRO traversal pattern as ``_recover_annotated_sqlmodel_fields``):

    1. Collect data field names from base ``model_fields`` (filtering ClassVar/Relationship)
    2. Get original type annotations from base MRO ``__annotations__`` (preserving ``Annotated`` metadata)

    Ensures ``Annotated[float, Field(ge=0.0)]`` constraints are not lost.
    """
    # 1. Collect all base class data field names
    field_names: set[str] = set()
    for base in bases:
        base_model_fields = getattr(base, 'model_fields', None)
        if base_model_fields:
            field_names.update(base_model_fields.keys())

    # 2. For each field, get original annotation from MRO and make optional
    for field_name in field_names:
        if field_name in annotations:
            continue
        # Find original annotation from MRO (preserves Annotated metadata)
        original_ann: typing.Any = None
        for base in bases:
            for cls in base.__mro__:
                if cls is object:
                    continue
                cls_ann = getattr(cls, '__annotations__', None)
                if not cls_ann or field_name not in cls_ann:
                    continue
                candidate = cls_ann[field_name]
                if isinstance(candidate, str):
                    continue
                original_ann = candidate
                break
            if original_ann is not None:
                break
        if original_ann is None:
            continue

        # Bug fix: when a field is declared as ``field: T = Field(gt=..., le=...)`` (non-Annotated form),
        # MRO ``__annotations__`` only stores the bare ``T``. The ``Field(...)`` constraint metadata
        # lives on the right-hand-side assignment, so all_fields_optional-derived UpdateRequest classes
        # lose those constraints (e.g. UFT-style "missing range" warnings fire on derived fields that
        # had constraints in the source class). Patch: when ``original_ann`` is not already Annotated,
        # pull constraints from the base class's resolved ``model_fields[name].metadata`` (Pydantic
        # stores them there as ``[Gt(0), Le(600), ...]``) and re-wrap into ``Annotated[T, *metadata]``,
        # letting downstream ``_make_annotation_optional`` preserve the constraints when converting
        # to ``T | None``. The root recommended fix is still to declare fields in Annotated form at
        # the source; this patch is defense-in-depth for legacy ``field: T = Field(...)`` declarations.
        if get_origin(original_ann) is not typing.Annotated:
            for base in bases:
                base_model_fields = getattr(base, 'model_fields', None)
                if not base_model_fields:
                    continue
                base_field_info = base_model_fields.get(field_name)
                if base_field_info is None or not base_field_info.metadata:
                    continue
                original_ann = typing.Annotated[original_ann, *base_field_info.metadata]
                break

        # Skip Literal type fields (e.g. discriminator):
        # Literal['text'] | None breaks Pydantic discriminated union
        raw_type = original_ann
        if get_origin(raw_type) is typing.Annotated:
            raw_type = get_args(raw_type)[0]
        if get_origin(raw_type) is typing.Literal:
            continue

        optional_ann = _make_annotation_optional(original_ann)

        # Replace default_factory with default=None for Annotated[T, Field(default_factory=...)]
        # Unified behavior: all_fields_optional fields all default to None, no factory retained.
        if get_origin(optional_ann) is typing.Annotated:
            ann_args = list(get_args(optional_ann))
            for i, meta in enumerate(ann_args[1:], 1):
                if isinstance(meta, FieldInfo) and meta.default_factory is not None:
                    new_fi = meta._copy() if hasattr(meta, '_copy') else copy.copy(meta)
                    new_fi.default_factory = None
                    new_fi.default = None
                    new_fi._attributes_set = dict(new_fi._attributes_set)
                    new_fi._attributes_set.pop('default_factory', None)
                    new_fi._attributes_set['default'] = None
                    ann_args[i] = new_fi
                    optional_ann = typing.Annotated[tuple(ann_args)]
                    break

        annotations[field_name] = optional_ann
        if field_name not in attrs:
            attrs[field_name] = None


def _recover_annotated_sqlmodel_fields(
    annotations: dict[str, typing.Any],
    attrs: dict[str, typing.Any],
    bases: tuple[type, ...],
    is_table: bool,
) -> None:
    """
    Recover ``Annotated[T, Field(...)]`` back to ``T = Field(default=..., ...)``.

    Pydantic v2 replaces sqlmodel.main.FieldInfo with pydantic.fields.FieldInfo when
    processing Annotated metadata. The latter doesn't support SQLModel-specific attributes
    (foreign_key, sa_type, etc.), causing get_column_from_field() to miss DB constraints.

    This function dynamically discovers all SQLModel FieldInfo in Annotated metadata and
    converts them to ``= Field(...)`` style. Future upstream additions are handled automatically.

    **Only executes for table classes**: non-table classes (e.g. Base classes) keep their
    original Annotated annotations so child table classes can recover inherited constraints.

    :param annotations: Class ``__annotations__`` dict (modified in place)
    :param attrs: Class namespace dict (modified in place)
    :param bases: Base class tuple for checking inherited Annotated fields
    :param is_table: Whether this class is a table class
    """
    # Python 3.14 (PEP 649): when annotations contain unresolvable forward references,
    # get_type_hints() raises NameError and returns empty dict.
    # Recover from __annotate_func__(Format.VALUE) which keeps unresolved refs as str/ForwardRef.
    if not annotations and annotationlib is not None:
        annotate_func = attrs.get('__annotate_func__')
        if annotate_func is not None:
            try:
                annotations.update(annotate_func(annotationlib.Format.VALUE))
            except Exception:
                pass

    # Non-table classes: keep original Annotated annotations unchanged
    if not is_table:
        return

    # Collect all Annotated fields: current class + inherited from parents
    all_annotated: dict[str, typing.Any] = {}

    # Inherited Annotated fields (traverse MRO for multi-level inheritance)
    for base in bases:
        for cls in base.__mro__:
            if cls is object:
                continue
            cls_ann = getattr(cls, '__annotations__', None)
            if not cls_ann:
                continue
            for field_name, field_type in cls_ann.items():
                if field_name not in all_annotated and field_name not in annotations:
                    all_annotated[field_name] = field_type

    # Current class fields (higher priority)
    all_annotated.update(annotations)

    for field_name, field_type in all_annotated.items():
        # Unwrap Optional/Union wrappers: forms like ``Annotated[X, Field(sa_type=BigInteger)] | None``
        # have ``get_origin() == Union`` (or ``types.UnionType`` for PEP 604 ``|``); we must
        # peel back to find the inner Annotated to extract the SQLModel FieldInfo. Otherwise
        # nullable aliases like ``PositiveBigInt | None`` lose ``sa_type`` and the SA column
        # silently degrades (e.g. BigInteger → Integer → asyncpg int32 overflow on large defaults).
        annotated_type = field_type
        union_args: list[typing.Any] | None = None
        union_origin = get_origin(field_type)
        # ``X | None`` (PEP 604) has origin ``types.UnionType``; ``Union[X, None]`` is ``typing.Union``.
        # Match both without depending on ``types`` import inside any version block.
        is_union = union_origin is typing.Union or (
            union_origin is not None and getattr(union_origin, '__name__', '') == 'UnionType'
        )
        if is_union:
            union_args = list(get_args(field_type))
            annotated_in_union = next(
                (arg for arg in union_args if get_origin(arg) is typing.Annotated),
                None,
            )
            if annotated_in_union is None:
                continue
            annotated_type = annotated_in_union

        if get_origin(annotated_type) is not typing.Annotated:
            continue

        args = get_args(annotated_type)
        if len(args) < 2:
            continue

        # Find all SQLModel FieldInfo in Annotated metadata
        sqlmodel_fis: list[SQLModelFieldInfo] = [
            arg for arg in args[1:] if isinstance(arg, SQLModelFieldInfo)
        ]

        if not sqlmodel_fis:
            continue

        # Merge multiple FieldInfo: shallow-copy first then merge to avoid mutating
        # shared Annotated metadata singletons (e.g. Str64 = Annotated[str, Field(max_length=64)])
        sqlmodel_fi = copy.copy(sqlmodel_fis[0])
        for extra_fi in sqlmodel_fis[1:]:
            _merge_field_info_attrs(sqlmodel_fi, extra_fi)

        # Transfer plain defaults from attrs (e.g. = 0, = None) to FieldInfo
        existing_default = attrs.get(field_name, Undefined)
        if existing_default is not Undefined and not isinstance(existing_default, (FieldInfo, SQLModelFieldInfo)):
            sqlmodel_fi.default = existing_default
        elif existing_default is Undefined:
            # Inherit default from parent model_fields, but only when FieldInfo has no default/factory
            if sqlmodel_fi.default is Undefined and sqlmodel_fi.default_factory is None:
                for base in bases:
                    base_fields = getattr(base, 'model_fields', None)
                    if base_fields and field_name in base_fields:
                        base_fi = base_fields[field_name]
                        if base_fi.default is not Undefined:
                            sqlmodel_fi.default = base_fi.default
                        elif base_fi.default_factory is not None:
                            sqlmodel_fi.default_factory = base_fi.default_factory
                        break

        # Inject SQLModel FieldInfo as field default (equivalent to = Field(...) style)
        attrs[field_name] = sqlmodel_fi

        # Update annotations: remove SQLModel FieldInfo from Annotated
        base_type = args[0]
        remaining_metadata = [a for a in args[1:] if not isinstance(a, SQLModelFieldInfo)]
        if remaining_metadata:
            new_inner: typing.Any = typing.Annotated[tuple([base_type] + remaining_metadata)]
        else:
            new_inner = base_type

        if union_args is not None:
            # Rebuild Union: replace the inner Annotated with the stripped version, keeping
            # the other Union members (e.g. ``| None``) intact. Use PEP 604 ``|`` to accumulate
            # so we don't trip the deprecated-typing.Union warning.
            new_union_args = [
                new_inner if get_origin(arg) is typing.Annotated else arg
                for arg in union_args
            ]
            rebuilt: typing.Any = new_union_args[0]
            for extra in new_union_args[1:]:
                rebuilt = rebuilt | extra
            annotations[field_name] = rebuilt
        else:
            annotations[field_name] = new_inner


def _make_sti_fk_resolver(
    fk_string: str,
    sa_registry: typing.Any,
) -> typing.Callable:
    """
    Convert string-format foreign_keys to a callable for deferred resolution in STI.

    STI child columns are added to the parent table via _register_sti_columns(),
    but during configure_mappers() they are not yet registered as ColumnProperty.
    SQLAlchemy's string resolution (_GetColumns.__getattr__) looks up columns via
    mapper.all_orm_descriptors, which fails for unregistered STI columns.

    Solution: convert to callable so configure_mappers() calls it to resolve
    Column objects directly from the table's columns collection (Phase 1 already added them).

    :param fk_string: String-format foreign_keys, e.g. '[NanoBananaFunction.flash_llm_id]'
    :param sa_registry: SQLAlchemy registry for class-name lookup
    :return: callable returning list of Column objects
    """
    inner = fk_string.strip('[]')
    specs = [s.strip() for s in inner.split(',')]

    parsed: list[tuple[str, str]] = []
    for spec in specs:
        m = re.match(r'^(\w+)\.(\w+)$', spec)
        if not m:
            return fk_string  # type: ignore  # cannot parse, return original
        parsed.append((m.group(1), m.group(2)))

    _registry = sa_registry

    def _resolve() -> list:
        columns = []
        for cls_name, col_name in parsed:
            for mapper in _registry.mappers:
                if mapper.class_.__name__ == cls_name:
                    table = mapper.local_table
                    if col_name not in table.c:
                        raise RuntimeError(
                            f"STI FK resolution failed: column '{col_name}' "
                            f"not in table '{table.name}' (class {cls_name})"
                        )
                    columns.append(table.c[col_name])
                    break
            else:
                raise RuntimeError(
                    f"STI FK resolution failed: class '{cls_name}' not in SA registry"
                )
        return columns

    return _resolve


# ==================== Custom table_args element interception ====================
#
# SQLAlchemy's ``Table.__init__`` consumes every ``__table_args__`` element
# **immediately**; an ``Index`` referencing a column that does not exist yet
# raises ``ConstraintColumnNotFoundError`` on the spot.
#
# ``CustomTableArg`` is a generic base class that lets users place
# **non-SQLAlchemy-native** marker objects into ``table_args`` -- the
# metaclass (``__DeclarativeMeta.__new__``) **intercepts** these markers,
# removes them from ``table_args`` (never handing them to SQLAlchemy), and
# pushes (target class, markers) onto the module-level queue
# ``_classes_with_custom_table_args`` for downstream infrastructure (e.g. the
# STI deferred index in ``mixins.polymorphic``) to scan and consume at the
# right moment.
#
# The base layer only knows the generic "defer processing" mechanism and
# **not** the concrete semantics (STI, JTI, ...) -- concrete types (e.g.
# ``DeferredIndex``) are defined by their own submodules as ``CustomTableArg``
# subclasses.


class CustomTableArg:
    """
    Generic base class for non-SQLAlchemy-native ``table_args`` elements.

    Objects inheriting this class, when placed in ``table_args``, are
    **intercepted** by the SQLModel metaclass -- never passed to SQLAlchemy
    ``Table.__init__`` (avoiding immediate-evaluation failures) and instead
    stashed on the ``_classes_with_custom_table_args`` queue for downstream
    infrastructure to consume.

    Concrete subclass example: ``mixins.polymorphic.DeferredIndex``
    (deferred indexes over STI subclass columns).
    """


_classes_with_custom_table_args: list[tuple[type, list[CustomTableArg]]] = []
"""
Queue of classes carrying custom ``table_args`` elements: ``(class, [CustomTableArg, ...])``.

Appended by ``__DeclarativeMeta.__new__`` after class creation completes.
Downstream infrastructure (e.g. ``mixins.polymorphic._create_sti_deferred_indexes()``)
scans this queue and consumes entries as needed (dispatching on ``isinstance``).
"""


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

        # 1.5. CachedTableBaseMixin: cache_ttl class keyword -> __cache_ttl__ attribute
        if 'cache_ttl' in kwargs:
            ttl = kwargs.pop('cache_ttl')
            if not isinstance(ttl, int) or ttl <= 0:
                raise ValueError(f"{name}: cache_ttl must be a positive integer, got: {ttl!r}")
            attrs['__cache_ttl__'] = ttl

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

        # 3.5. OptimisticLockMixin wiring: register the mixin's ``version``
        # column as SQLAlchemy's ``version_id_col`` so every UPDATE emits
        # ``SET version = version + 1 WHERE ... AND version = :current`` and a
        # lost update surfaces as StaleDataError. The Column object only exists
        # after the Table is built, so this must be a ``declared_attr`` that
        # declarative evaluates late. Only applied to the root table class --
        # STI/JTI children inherit version_id_col from the root mapper.
        if will_be_table and any(getattr(b, '_has_optimistic_lock', False) for b in bases):
            _is_inheriting_table = parent_tablename is not None
            if not _is_inheriting_table and 'version_id_col' not in attrs.get('__mapper_args__', {}):
                _static_mapper_args = dict(attrs.get('__mapper_args__', {}))

                def _mapper_args_with_version_col(target_cls, _static=_static_mapper_args):
                    merged = dict(_static)
                    merged['version_id_col'] = target_cls.__table__.c.version
                    return merged

                # ``.directive`` is SQLAlchemy 2.0's spelling for declarative
                # dunder directives like __mapper_args__ (plain declared_attr
                # on a dunder emits a usage warning).
                attrs['__mapper_args__'] = declared_attr.directive(_mapper_args_with_version_col)

        # Process other explicit args
        if 'table_args' in kwargs:
            raw_table_args = kwargs.pop('table_args')
            # Split out CustomTableArg markers -- never handed to SQLAlchemy
            # (avoiding immediate evaluation against not-yet-existing
            # columns); stashed on the module-level
            # ``_classes_with_custom_table_args`` queue for downstream
            # infrastructure (e.g. STI deferred indexes) to consume.
            real_table_args: list[Any] = []
            custom_table_args: list[CustomTableArg] = []
            for arg in raw_table_args:
                if isinstance(arg, CustomTableArg):
                    custom_table_args.append(arg)
                else:
                    real_table_args.append(arg)
            attrs['__table_args__'] = tuple(real_table_args) if real_table_args else ()
            if custom_table_args:
                # Temporarily hung on attrs; appended to the module-level
                # queue after super().__new__ (the result class is fully
                # constructed then, so downstream can read __table__ etc.).
                attrs['__custom_table_args__'] = custom_table_args
        if 'table_name' in kwargs:
            attrs['__tablename__'] = kwargs.pop('table_name')
        if 'abstract' in kwargs:
            attrs['__abstract__'] = kwargs.pop('abstract')

        # 4. Extract sa_type from Annotated metadata and inject into Field
        annotations, annotation_strings, eval_globals, eval_locals = _resolve_annotations(attrs)

        # 4.5. Fix Annotated[T, Field(foreign_key=...)] where SQLModel FieldInfo gets replaced
        # by Pydantic FieldInfo. Must run before super().__new__() because SQLModel calls
        # get_column_from_field() during __new__. Only for table classes; non-table classes
        # keep original Annotated annotations for child table classes to inherit.
        _recover_annotated_sqlmodel_fields(annotations, attrs, bases, will_be_table)

        # 4.6. all_fields_optional: automatically convert inherited fields to optional (T | None = None)
        # Used for UpdateRequest DTOs to avoid manually overriding each field.
        is_all_optional = kwargs.pop('all_fields_optional', False)
        if is_all_optional:
            _apply_all_fields_optional(annotations, attrs, bases)

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
                    # No explicit ``= Field(...)`` assignment. Prefer recovering
                    # FieldInfo from inside Annotated[X, Field(default_factory=..., ...)]
                    # so user-supplied default_factory / max_length / constraints
                    # survive — clobbering with a fresh Field(sa_type=sa_type)
                    # would silently make the field required after multi-level
                    # inheritance.
                    annotated_fi = _find_field_info_in_annotated(field_type)
                    if annotated_fi is not None:
                        _durably_set_sa_type(annotated_fi, sa_type)
                        attrs[field_name] = annotated_fi
                    else:
                        attrs[field_name] = Field(sa_type=sa_type)
                elif isinstance(field_value, FieldInfo):
                    # Explicit ``Array[T] = Field(default_factory=list)`` form.
                    # Must inject sa_type via the pydantic-rebuild-safe
                    # FieldInfoMetadata channel — plain setattr is dropped
                    # before SQLModel's column build (see _durably_set_sa_type).
                    _durably_set_sa_type(field_value, sa_type)

        # 5. Save SQLModel FieldInfo from Annotated fields before super().__new__(),
        # because Pydantic rebuilds model_fields with plain FieldInfo that lacks
        # SQLModel-specific attributes (unique, index, foreign_key, sa_type, etc.).
        _saved_sqlmodel_fis: dict[str, SQLModelFieldInfo] = {}
        if will_be_table:
            for _fn in annotations:
                _fv = attrs.get(_fn)
                if isinstance(_fv, SQLModelFieldInfo):
                    _saved_sqlmodel_fis[_fn] = _fv

        # 6. Call parent __new__
        result = super().__new__(cls, name, bases, attrs, **kwargs)

        # 6.5. Append intercepted CustomTableArg markers to the module-level
        # queue for downstream infrastructure. The result class is fully
        # constructed at this point, so consumers can access __table__ etc.
        _custom_args = attrs.get('__custom_table_args__')
        if _custom_args:
            _classes_with_custom_table_args.append((result, _custom_args))

        # 7. Restore SQLModel FieldInfo attributes discarded by Pydantic and rebuild Columns.
        # Pydantic FieldInfo uses __slots__, so setattr for SQLModel extensions is silently
        # ignored. We replace the Pydantic FieldInfo with our saved SQLModelFieldInfo and
        # rebuild the Column via get_column_from_field().
        if _saved_sqlmodel_fis:
            for _fn, _saved_fi in _saved_sqlmodel_fis.items():
                _current_fi = result.model_fields.get(_fn)
                if _current_fi is None:
                    continue
                _merge_field_info_attrs(_saved_fi, _current_fi)
                if _saved_fi.default is Undefined and _current_fi.default is not Undefined:
                    _saved_fi.default = _current_fi.default
                if _saved_fi.default_factory is None and _current_fi.default_factory is not None:
                    _saved_fi.default_factory = _current_fi.default_factory
                result.model_fields[_fn] = _saved_fi
                _col = get_column_from_field(_saved_fi)
                setattr(result, _fn, _col)

        # 8. Fix: inherit parent's __sqlmodel_relationships__ for JTI
        if kwargs.get('table', False):
            for base in bases:
                if hasattr(base, '__sqlmodel_relationships__'):
                    for rel_name, rel_info in base.__sqlmodel_relationships__.items():
                        if rel_name not in result.__sqlmodel_relationships__:
                            result.__sqlmodel_relationships__[rel_name] = rel_info
                            if hasattr(base, rel_name):
                                base_attr = getattr(base, rel_name)
                                setattr(result, rel_name, base_attr)

        # 9. Forbid redefining parent's Relationship fields
        for base in bases:
            parent_relationships = getattr(base, '__sqlmodel_relationships__', {})
            for rel_name in parent_relationships:
                if rel_name in attrs:
                    raise TypeError(
                        f"Class {name} cannot redefine parent {base.__name__}'s "
                        f"Relationship field '{rel_name}'. "
                        f"Modify the relationship in the parent class instead."
                    )

        # 10. Inherit parent field descriptions (use_attribute_docstrings fix)
        # Pydantic's use_attribute_docstrings parses docstrings from source AST.
        # When a subclass overrides a field (e.g. UpdateRequest changes `name: str`
        # to `name: str | None = None`) or all_fields_optional programmatically
        # generates annotations, there is no docstring in source → description lost.
        # Fix: inherit missing descriptions from parent's model_fields via MRO.
        needs_rebuild = False
        for fname, finfo in result.model_fields.items():
            if finfo.description is not None:
                continue
            for parent in result.__mro__[1:]:
                parent_fields = getattr(parent, 'model_fields', None)
                if parent_fields and fname in parent_fields:
                    parent_desc = parent_fields[fname].description
                    if parent_desc is not None:
                        finfo.description = parent_desc
                        needs_rebuild = True
                        break

        # 11. Fix: remove Relationship fields from model_fields/__pydantic_fields__
        relationships = getattr(result, '__sqlmodel_relationships__', {})
        if relationships:
            model_fields = getattr(result, 'model_fields', {})
            pydantic_fields = getattr(result, '__pydantic_fields__', {})

            for rel_name in relationships:
                if rel_name in model_fields:
                    del model_fields[rel_name]
                    needs_rebuild = True
                if rel_name in pydantic_fields:
                    del pydantic_fields[rel_name]
                    needs_rebuild = True

        # Rebuild Pydantic schema (description inheritance or Relationship removal)
        if needs_rebuild and hasattr(result, 'model_rebuild'):
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
            # JTI FK must also be primary_key (created by SubclassIdMixin).
            # A FK pointing to the parent table that is NOT a PK (e.g. self-referential
            # parent_transaction_id) should NOT be identified as JTI inheritance.
            def _is_jti_fk(fi: typing.Any) -> bool:
                fk = getattr(fi, 'foreign_key', None)
                pk = getattr(fi, 'primary_key', None)
                return (
                    fk is not None
                    and isinstance(fk, str)
                    and pk is True  # PydanticUndefined is truthy, must compare strictly
                    and _fk_matches_parent(fk, parent_tablename)
                )

            for field_name, field_info in cls.model_fields.items():
                if _is_jti_fk(field_info):
                    has_fk_to_parent = True
                    break

            if not has_fk_to_parent:
                for base in bases:
                    if hasattr(base, 'model_fields'):
                        for field_name, field_info in base.model_fields.items():
                            if _is_jti_fk(field_info):
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

                # Support both concrete classes (polymorphic_identity set) and
                # abstract intermediate classes (polymorphic_identity=None, polymorphic_abstract=True)
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

                                    # Default lazy='raise_on_sql' for async safety:
                                    # prevents accidental lazy-loading which causes
                                    # MissingGreenlet errors in async environments.
                                    if 'lazy' not in rel_kwargs:
                                        rel_kwargs['lazy'] = 'raise_on_sql'

                                    # STI foreign_keys deferred resolution:
                                    # STI child columns are not yet registered as ColumnProperty
                                    # during configure_mappers(), so string foreign_keys fail.
                                    # Convert to callable for lazy resolution from table columns.
                                    if 'foreign_keys' in rel_kwargs:
                                        _fk_val = rel_kwargs['foreign_keys']
                                        if isinstance(_fk_val, str):
                                            rel_kwargs['foreign_keys'] = _make_sti_fk_resolver(
                                                _fk_val, registry
                                            )
                                    else:
                                        # Auto-detect FK ambiguity: when the "many" side STI child
                                        # has a {rel_name}_id FK field but foreign_keys is not
                                        # explicitly specified, add a callable to disambiguate.
                                        _fk_field = f'{rel_name}_id'
                                        _model_fields = getattr(cls, 'model_fields', None) or {}
                                        if _fk_field in _model_fields:
                                            _tbl = parent_cls.__table__
                                            _fn = _fk_field
                                            rel_kwargs['foreign_keys'] = (
                                                lambda _t=_tbl, _f=_fn: [_t.c[_f]]
                                            )

                                    properties[rel_name] = sa_relationship(relationship_to, *rel_args, **rel_kwargs)

                    # Build map_imperatively kwargs conditionally
                    map_kwargs: dict[str, typing.Any] = {
                        'inherits': parent_cls,
                        'properties': properties if properties else None,
                    }
                    if polymorphic_identity is not None:
                        map_kwargs['polymorphic_identity'] = polymorphic_identity
                    # Abstract intermediate classes (e.g. TencentCompatibleLLM)
                    # need polymorphic_abstract=True forwarded to map_imperatively
                    if mapper_args.get('polymorphic_abstract'):
                        map_kwargs['polymorphic_abstract'] = True

                    registry.map_imperatively(
                        cls,
                        parent_cls.__table__,
                        **map_kwargs,
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

            # Default lazy='raise_on_sql' for async safety: prevents accidental
            # lazy-loading which causes MissingGreenlet errors in async environments.
            if 'lazy' not in rel_kwargs:
                rel_kwargs['lazy'] = 'raise_on_sql'

            rel_value = sa_relationship(relationship_to, *rel_args, **rel_kwargs)
            setattr(cls, rel_name, rel_value)


class SQLModelBase(SQLModel, metaclass=__DeclarativeMeta):
    """
    Base class for all SQLModel models in sqlmodel_ext.

    Must be used together with TableBaseMixin or UUIDTableBaseMixin for table models.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, validate_by_name=True, extra='forbid')

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: Any,
    ) -> dict[str, Any]:
        """
        Fix Pydantic JSON Schema dropping description for $ref properties.

        When a field type is an enum or nested model, Pydantic sometimes generates
        a bare ``{"$ref": "..."}`` without the ``description`` (even though
        ``model_fields`` has the description correctly set). This method patches
        the generated schema to restore missing descriptions.
        """
        json_schema = handler(core_schema)
        props = json_schema.get('properties')
        if props:
            for fname, prop in props.items():
                if '$ref' in prop and 'description' not in prop:
                    finfo = cls.model_fields.get(fname)
                    if finfo and finfo.description:
                        prop['description'] = finfo.description
        return json_schema

    @classmethod
    def validate_list(cls, items: Sequence[Any]) -> list[Self]:
        """Batch-convert a sequence of ORM instances (or dicts) to this model type."""
        return [cls.model_validate(item, from_attributes=True) for item in items]

    @classmethod
    def get_computed_field_names(cls) -> set[str]:
        """Get the set of computed_field names for this model class."""
        fields = cls.model_computed_fields
        return set(fields.keys()) if fields else set()


class ExtraIgnoreModelBase(SQLModelBase):
    """
    Model base class that ignores unknown fields (extra='ignore').

    Unlike SQLModelBase (extra='forbid'), this class silently ignores undeclared
    fields and logs a WARNING for discoverability.

    Use for:
    - Third-party API responses (where the schema may change without notice)
    - Client WebSocket message envelopes (protocol-level field validation)
    - Any model parsing external JSON input (including nested sub-models)

    Do NOT use for: request models that we construct and send to external services
    (those should keep 'forbid' to catch mistakes).
    """

    model_config = ConfigDict(
        use_attribute_docstrings=True, validate_by_name=True, extra='ignore',
    )

    @model_validator(mode='before')
    @classmethod
    def _warn_unknown_fields(cls, data: Any) -> Any:
        """
        Detect and warn about unknown fields in incoming data.

        Logs a WARNING before Pydantic's extra='ignore' discards unknown fields,
        helping developers notice third-party API changes and add field definitions.
        """
        if not isinstance(data, dict):
            return data
        accepted: set[str] = set()
        for name, field_info in cls.model_fields.items():
            accepted.add(name)
            if field_info.alias:
                accepted.add(field_info.alias)
            if field_info.validation_alias and isinstance(field_info.validation_alias, str):
                accepted.add(field_info.validation_alias)
        unknown = set(data.keys()) - accepted
        if unknown:
            total = len(unknown)
            sample = [name[:64] for name in sorted(unknown)[:5]]
            logger.warning(
                "External input contains unknown fields | model=%s "
                "unknown_count=%d sample_fields=%s",
                cls.__name__, total, sample,
            )
        return data
