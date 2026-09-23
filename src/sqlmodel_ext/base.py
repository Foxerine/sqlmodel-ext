"""
SQLModelBase and __DeclarativeMeta metaclass.

Provides a smart metaclass that handles:
- Automatic ``table=True`` for classes with TableBaseMixin
- Convenient keyword arguments (polymorphic_on, polymorphic_identity, etc.)
- Joined Table Inheritance (JTI) support
- Single Table Inheritance (STI) via registry.map_imperatively()
- Annotated sa_type extraction and injection
- ``partial=True`` PATCH DTOs with tri-state (``Unset | T``) fields
- Opt-in ``omitted_sentinel`` wire value for callers that cannot omit keys
- Python 3.14 (PEP 649) compatibility
"""
import ast
import copy
import dataclasses
import logging
import re
import sys
import inspect
import typing
from collections.abc import Mapping
from typing import Any, Self, Sequence, get_args, get_origin

from pydantic import AliasChoices, BaseModel, model_validator
from pydantic import Field as PydanticField
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema as pydantic_core_schema
from pydantic_core import PydanticUndefined as Undefined
from sqlalchemy import Column, inspect as sa_inspect
from sqlalchemy.orm import Mapped, declared_attr, relationship as sa_relationship
from sqlmodel import Field, SQLModel
from sqlmodel.main import (
    SQLModelConfig,
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
    FieldInfoMetadata.__hash__ = object.__hash__  # pyright: ignore[reportAttributeAccessIssue]  # the dataclass stub declares __hash__ as None -- that is the upstream defect being patched here

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
from sqlmodel_ext.constants import OPTIMISTIC_LOCK_VERSION_COLUMN
from sqlmodel_ext.unset import (
    OMITTED_SENTINEL,
    SCHEMA_ANNOTATION_KEYS,
    SENTINEL_SCHEMA_BRANCH,
    Unset,
)

# JSON100K / JSONList100K need the optional ``orjson`` dependency
# (``sqlmodel-ext[postgresql]``). Without it those types cannot be used at all,
# so there is nothing to check. Only a missing ``orjson`` is tolerated -- any
# other import failure is a bug and must surface.
try:
    from sqlmodel_ext.field_types.dialects.postgresql.jsonb_types import (
        JSON100K,
        JSONList100K,
        ensure_json_within_limits,
    )
except ModuleNotFoundError as _exc:
    if _exc.name != 'orjson':
        raise
    _orjson_checked_types: tuple[type, ...] = ()
    _ensure_json_within_limits: typing.Callable[[Any], None] | None = None
else:
    _orjson_checked_types = (JSON100K, JSONList100K)
    _ensure_json_within_limits = ensure_json_within_limits

# Python 3.14+ support
if sys.version_info >= (3, 14):
    import annotationlib  # noqa: F401
else:
    annotationlib = None

logger = logging.getLogger(__name__)


_ATTRS_WHERE_NONE_IS_A_REAL_VALUE = frozenset({'default'})
"""
FieldInfo attributes on which ``None`` is an explicit, legitimate value rather than "not set".

For ``default``, ``None`` and ``PydanticUndefined`` mean opposite things:
the former is "the default value is None" (the field is optional), the latter
is "there is no default" (the field is required). Any loop that skips
"unset" attributes with ``if value is None: continue`` must exempt the
attributes listed here; otherwise an explicit ``default=None`` is dropped and
the field silently becomes required.

.. warning::

   This set is a registry of attributes **confirmed** to need the exemption,
   not a claim that ``None`` means "unset" for every other attribute. For
   example, in plain Pydantic ``Field(alias=None)`` placed after
   ``Field(alias='old')`` clears the alias. Verify an attribute's ``None``
   semantics (both in plain Pydantic and through this metaclass) before
   adding it here.
"""


def _merge_field_info_attrs(target: SQLModelFieldInfo, source: FieldInfo) -> None:
    """
    Merge explicitly-set attributes from ``source`` into ``target``.

    Used when ``Annotated[Str64, Field(unique=True)]`` expands to multiple FieldInfo:
    ``Annotated[str, Field(max_length=64), Field(unique=True)]``, and when a
    right-hand-side ``= Field(...)`` coexists with a FieldInfo inside the annotation.
    Merges attributes dynamically (via ``__slots__`` / ``__annotations__`` / ``__dict__``),
    no hardcoded attribute names — upstream additions are handled automatically.

    ``source`` is typed as Pydantic's base ``FieldInfo``: it is only read, and
    callers may hand in either flavor. ``target`` is written and must be a
    ``SQLModelFieldInfo``.

    Skipped source values: ``PydanticUndefined``; ``None`` (except for the
    attributes in ``_ATTRS_WHERE_NONE_IS_A_REAL_VALUE``); empty containers; and
    ``False`` for a boolean flag that is already ``True`` on the target (so
    ``unique=False`` never switches off an inherited ``unique=True``).
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

        # ``default=None`` is an explicit value, not "unset" -- skipping it
        # would make ``Annotated[X, Field(...), Field(default=None)]`` required.
        if val is Undefined:
            continue
        if val is None and attr_name not in _ATTRS_WHERE_NONE_IS_A_REAL_VALUE:
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
        raw_target_meta = getattr(target, 'metadata', None)
        target_meta = list(raw_target_meta) if raw_target_meta is not None else []
        target.metadata = _merge_sqlmodel_metadata_carrier(target_meta, list(source_meta))


def _merge_sqlmodel_metadata_carrier(
        target_meta: list[Any],
        source_meta: list[Any],
) -> list[Any]:
    """
    Concatenate two FieldInfo ``metadata`` lists, folding SQLModel's ``FieldInfoMetadata`` carriers into one.

    Since sqlmodel 0.0.32, ``Field(primary_key=..., sa_type=..., ...)`` stores the
    SQLModel-specific attributes in a ``FieldInfoMetadata`` entry of
    ``metadata``, and SQLModel reads only the **first** such entry. Plain
    concatenation would therefore let an earlier, all-unset carrier (e.g. the
    one inside a constrained alias like ``NonNegativeInt``) shadow the values
    set by the later, more specific source (e.g. a right-hand
    ``= Field(primary_key=True)``).

    Set values of the source carrier are folded into a **copy** of the target
    carrier (carriers inside type aliases are shared singletons and must not be
    mutated), using the same rule as ``_merge_field_info_attrs``: ``False`` never
    overwrites ``True``. All other metadata is concatenated unchanged.
    """
    target_fim_index = next(
        (i for i, m in enumerate(target_meta) if isinstance(m, FieldInfoMetadata)), None,
    )
    merged = list(target_meta)
    for item in source_meta:
        if not isinstance(item, FieldInfoMetadata) or target_fim_index is None:
            merged.append(item)
            continue
        folded = copy.copy(merged[target_fim_index])
        for fim_field in dataclasses.fields(item):
            value = getattr(item, fim_field.name)
            if value is _FIM_UNSET_SA_TYPE:
                continue
            current = getattr(folded, fim_field.name)
            if value is False and current is True:
                continue
            setattr(folded, fim_field.name, value)
        merged[target_fim_index] = folded
    return merged


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


def _annotation_contains_orjson_checked_type(annotation: Any) -> bool:
    """
    Whether the annotation tree contains ``JSON100K`` / ``JSONList100K``.

    Looks through ``Annotated`` / unions / generic containers. Used by
    ``SQLModelBase.__pydantic_init_subclass__`` to discover, at class creation,
    the fields that need a construction-time serializability check -- the
    type annotation is the declaration, no manual registration needed.
    Always ``False`` when ``orjson`` is not installed.
    """
    if any(annotation is checked for checked in _orjson_checked_types):
        return True
    return any(_annotation_contains_orjson_checked_type(arg) for arg in get_args(annotation))


optional_dto_registry: list[type] = []
"""
Every class created with ``partial=True``, in creation order.

Public so that tooling (diagnostics, code generators, contract tests) can
enumerate the derived PATCH models -- for example to assert that each one
constructs from an empty payload and dumps to ``{}``.
"""


_UNION_INCOMPATIBLE_FIELD_ATTRS: typing.Final = (
    'exclude', 'alias', 'validation_alias', 'serialization_alias',
    'repr', 'frozen', 'deprecated',
)
"""
``FieldInfo`` attributes that have no effect on a union member and must be hoisted outside the union.

Excluded on purpose:

- ``discriminator``: it already works on the member (Pydantic supports a
  tagged union nested in an outer union, as in
  ``Unset | Annotated[Cat | Dog, Field(discriminator='kind')]``), while
  hoisting it would make Pydantic treat ``Unset`` as a tagged-union variant
  and fail at class creation.

- ``default`` / ``default_factory``: ``partial`` sets the class attribute to
  ``Unset`` (the default); an outer FieldInfo that also carried a
  ``default_factory`` would make Pydantic raise
  ``TypeError: cannot specify both default and default_factory``.
- constraints (``ge`` / ``max_length`` / ...): they keep working on the union
  member, and hoisting them would apply them to the sentinel as well.
"""


def _union_member_annotation(original_ann: typing.Any) -> typing.Any:
    """
    Reduce the ``FieldInfo`` items of an ``Annotated`` annotation to what works on a union member.

    Inside ``Unset | Annotated[T, *meta]``, a plain Pydantic ``FieldInfo`` only
    contributes its constraints (``FieldInfo.metadata``) and its
    ``discriminator``; every field-level attribute on it (``alias`` /
    ``exclude`` / ``frozen`` / ``default`` / ...) is ignored and Pydantic emits
    an ``UnsupportedFieldAttributeWarning`` per attribute. ``partial`` carries
    the field-level attributes on the outer layer instead (see
    ``_hoist_field_metadata``), so each such ``FieldInfo`` is replaced by its
    constraints plus, if set, a ``Field(discriminator=...)``.

    Only items whose type is exactly ``pydantic.fields.FieldInfo`` are
    rewritten -- the same test Pydantic uses for the warning. Other metadata
    (``annotated_types`` constraints, validators, schema handlers, SQLModel's
    ``FieldInfo`` subclass) is kept in place and in order.

    :param original_ann: the annotation that becomes the union member
    :returns: the annotation unchanged if it is not ``Annotated``, otherwise the rewritten ``Annotated``
    """
    if get_origin(original_ann) is not typing.Annotated:
        return original_ann
    inner, *metadata = get_args(original_ann)
    reduced: list[typing.Any] = []
    for meta in metadata:
        if type(meta) is not FieldInfo:
            reduced.append(meta)
            continue
        reduced.extend(meta.metadata)
        if meta.discriminator is not None:
            reduced.append(PydanticField(discriminator=meta.discriminator))
    if not reduced:
        return inner
    return typing.Annotated[inner, *reduced]


def _json_schema_property_key(field_name: str, field: FieldInfo, mode: str, by_alias: bool) -> str:
    """
    The key under which Pydantic lists ``field`` in a JSON schema's ``properties``.

    Mirrors Pydantic's own rule, using the public ``FieldInfo`` attributes: with
    ``by_alias`` the key is the mode's alias -- ``validation_alias`` in
    validation mode, ``serialization_alias`` in serialization mode (``alias``
    fills both) -- and for ``AliasChoices`` the first choice that is a string
    or a single-element string ``AliasPath``; otherwise, and without
    ``by_alias``, the field name. A bare ``AliasPath`` (not inside
    ``AliasChoices``) keeps the field name: Pydantic's rule only picks paths
    out of a list of choices. Checked against Pydantic's output for every
    alias form in both modes.
    """
    if not by_alias:
        return field_name
    alias = field.validation_alias if mode == 'validation' else field.serialization_alias
    if isinstance(alias, str):
        return alias
    if isinstance(alias, AliasChoices):
        for choice in alias.choices:
            if isinstance(choice, str):
                return choice
            if len(choice.path) == 1 and isinstance(choice.path[0], str):
                return choice.path[0]
    return field_name


_ALIAS_FIELD_ATTRS: typing.Final = frozenset({'alias', 'validation_alias', 'serialization_alias'})
"""The members of ``_UNION_INCOMPATIBLE_FIELD_ATTRS`` an ``alias_generator`` can fill in."""

_DEFAULT_FIELD_INFO: typing.Final = FieldInfo()
"""Reference ``FieldInfo`` for "attribute left at its default" (e.g. ``repr`` defaults to ``True``, not ``None``)."""


def _hoist_field_metadata(union_ann: typing.Any, base_field: FieldInfo) -> typing.Any:
    """
    Hoist the base field's field-level attributes to outside the union.

    In ``Unset | Annotated[T, *meta]`` the metadata is attached to one union
    member. ``annotated_types`` constraints (``Ge`` / ``MaxLen`` ...) still work
    there, but Pydantic's *field-level* attributes (``exclude`` / ``alias`` /
    ``repr`` ...) do not -- Pydantic only emits an
    ``UnsupportedFieldAttributeWarning`` and silently drops them.

    This returns ``Annotated[Unset | Annotated[T, *meta], Field(<hoisted>)]``:
    Pydantic picks the field-level attributes up from the outer layer, while
    the inner layer is untouched (constraints stay bound to the real value type).

    The attributes are read from the base class's **resolved** ``FieldInfo``
    (``model_fields[name]``), not from the annotation: the resolved field has
    already merged every place the author could have written them --
    ``Annotated[T, Field(alias=...)]``, ``x: T = Field(alias=...)`` and mixes of
    the two -- so it is the single source of truth. Only attributes that differ
    from a default ``FieldInfo()`` are carried.

    :param union_ann: the already-built ``Unset | <annotation>``
    :param base_field: the base class's resolved ``FieldInfo`` for this field
    :returns: ``union_ann`` unchanged when there is nothing to hoist, otherwise
        ``union_ann`` wrapped in ``Annotated`` with a ``Field`` carrying the hoisted attributes
    """
    # Aliases filled in by a model's ``alias_generator`` carry ``alias_priority``
    # 1; Pydantic regenerates those for every subclass (so a derived class with
    # its own generator gets its own aliases), while author-declared aliases
    # (priority 2) are inherited as-is. Hoisting generated aliases would turn
    # them into explicit ones and freeze them, so the alias family is carried
    # only when priority is 2 -- exactly what plain inheritance keeps.
    aliases_are_declared = base_field.alias_priority is not None and base_field.alias_priority >= 2
    carried: dict[str, typing.Any] = {}
    for attr in _UNION_INCOMPATIBLE_FIELD_ATTRS:
        if attr in _ALIAS_FIELD_ATTRS and not aliases_are_declared:
            continue
        value = getattr(base_field, attr, None)
        if value is not None and value != getattr(_DEFAULT_FIELD_INFO, attr, None):
            carried[attr] = value
    if not carried:
        return union_ann
    # Pydantic's ``Field``, not SQLModel's: the outer layer only carries
    # Pydantic field-level attributes, and ``sqlmodel.Field`` does not accept
    # all of them (e.g. ``frozen``).
    return typing.Annotated[tuple([union_ann, PydanticField(**carried)])]


def _apply_partial(
        annotations: dict[str, typing.Any],
        attrs: dict[str, typing.Any],
        bases: tuple[type, ...],
        own_names: frozenset[str],
) -> None:
    """
    Turn inherited fields into **omissible** fields (``Unset | T = Unset``) -- the PATCH DTO shape.

    The name is ``partial`` (as in TypeScript's ``Partial<T>``) rather than
    "optional": in Python "optional" means ``Optional[T]`` = ``T | None``
    (nullable), whereas this makes fields *omissible*. Nullability is carried
    over unchanged:

    - base ``T``          -> ``Unset | T``        (``null`` rejected)
    - base ``T | None``   -> ``Unset | T | None`` (``null`` is a real value)

    Two-step strategy (same MRO traversal as ``_recover_annotated_sqlmodel_fields``):

    1. Collect data field names from base ``model_fields`` (ClassVar/Relationship excluded)
    2. Take the original annotation from base MRO ``__annotations__`` (keeps ``Annotated`` metadata)

    Constraints are preserved: ``Unset`` is validated by pydantic-core's
    dedicated missing-sentinel branch and never reaches constraint validators,
    so ``Unset | Annotated[int, Field(ge=0)]`` needs no special nesting.

    ``default_factory`` needs no special handling either: the ``Unset`` class
    attribute becomes the default and Pydantic drops the factory, so an omitted
    list field is ``Unset``, not ``[]``.

    Skipped fields:

    - fields the class declares itself (``own_names``): the author's
      declaration wins;
    - ``Literal`` fields (e.g. discriminators): ``Unset | Literal[...]`` would
      break discriminated unions.

    Note that the resulting annotations are created at runtime, so static type
    checkers still see the base class annotations on the derived class.

    :param annotations: class annotations (modified in place)
    :param attrs: class namespace (modified in place)
    :param bases: base classes
    :param own_names: names the class body annotates itself, snapshotted before
        any annotation injection
    """
    field_names: set[str] = set()
    for base in bases:
        base_model_fields = getattr(base, 'model_fields', None)
        if base_model_fields:
            field_names.update(base_model_fields.keys())

    for field_name in field_names:
        # The criterion is "did the author declare it in this class body", not
        # "is it in ``annotations``" -- those are different questions once
        # anything injects inherited annotations.
        if field_name in own_names:
            continue
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

        # The base class's resolved field: the single source of truth for what
        # the author declared, whichever syntax was used.
        base_field_info: FieldInfo | None = None
        for base in bases:
            base_model_fields = getattr(base, 'model_fields', None)
            if base_model_fields and field_name in base_model_fields:
                base_field_info = base_model_fields[field_name]
                break
        if base_field_info is None:
            continue

        # When a field is declared as ``field: T = Field(gt=..., le=...)`` (non-Annotated form),
        # MRO ``__annotations__`` only stores the bare ``T``; the constraints live on the
        # right-hand-side assignment. Pull them from the resolved ``metadata``
        # (``[Gt(0), Le(600), ...]``) and re-wrap into ``Annotated[T, *metadata]``
        # so the derived field keeps them. (Field-level attributes such as
        # ``alias`` / ``exclude`` are not in ``metadata``; they are hoisted below.)
        if get_origin(original_ann) is not typing.Annotated and base_field_info.metadata:
            original_ann = typing.Annotated[original_ann, *base_field_info.metadata]

        # Skip Literal fields (e.g. discriminators): making them omissible breaks
        # Pydantic discriminated unions.
        raw_type = original_ann
        if get_origin(raw_type) is typing.Annotated:
            raw_type = get_args(raw_type)[0]
        if get_origin(raw_type) is typing.Literal:
            continue

        # Field-level attributes (exclude / alias / ...) do not work on a union
        # member and would be silently dropped -- hoist them outside the union.
        annotations[field_name] = _hoist_field_metadata(
            Unset | _union_member_annotation(original_ann), base_field_info,
        )
        if field_name not in attrs:
            attrs[field_name] = Unset


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
        elif isinstance(existing_default, (FieldInfo, SQLModelFieldInfo)):
            # A right-hand-side ``= Field(...)`` coexists with the FieldInfo inside
            # the annotation, e.g.
            # ``value: NonNegativeBigInt | None = Field(default=None, sa_type=BigInteger)``.
            # Without this branch the right-hand FieldInfo would be discarded
            # (overwritten by ``attrs[field_name] = sqlmodel_fi`` below) and an
            # explicit ``default=None`` would vanish, silently making the field
            # required.
            #
            # The right-hand side is more specific than the type alias, so it is
            # merged in as ``source``. Caveat: ``_merge_field_info_attrs`` never
            # lets ``False`` overwrite ``True`` on boolean flags, so
            # ``Annotated[int, Field(unique=True)] = Field(unique=False)`` stays
            # ``unique=True``.
            _merge_field_info_attrs(sqlmodel_fi, existing_default)
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


def _is_none_node(node: ast.expr) -> bool:
    """Whether a type-expression AST node denotes ``None`` / ``NoneType``."""
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    return isinstance(node, ast.Name) and node.id in ('None', 'NoneType')


def _relationship_target_node(node: ast.expr) -> ast.expr | None:
    """
    Extract the single non-``None`` target node from a type-expression AST.

    Covers every shape a relationship annotation can take: ``Foo`` / ``pkg.Foo`` /
    ``Foo | None`` / ``None | Foo`` / ``Optional[Foo]`` / ``Union[Foo, None]``, plus
    an inner requoted form (``Optional['Foo']``). When the expression cannot be
    reduced to a single target (e.g. a multi-member union ``Foo | Bar``), returns
    ``None`` so the caller hands the annotation back to SQLModel's native parser
    untouched, letting it raise its own clear error.
    """
    # Bare class name / dotted (module-qualified) class name.
    if isinstance(node, (ast.Name, ast.Attribute)):
        return node
    # Inner requoted string constant (e.g. ``Optional['Foo']``): recurse into it.
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _relationship_target_node(ast.parse(node.value.strip(), mode='eval').body)
        except SyntaxError:
            return None
    # ``A | None`` / ``None | A``.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left_none, right_none = _is_none_node(node.left), _is_none_node(node.right)
        if left_none ^ right_none:  # exactly one side is None
            return _relationship_target_node(node.right if left_none else node.left)
        return None
    # ``Optional[A]`` / ``Union[A, None]``.
    if isinstance(node, ast.Subscript):
        base = node.value
        base_name = (
            base.id if isinstance(base, ast.Name)
            else base.attr if isinstance(base, ast.Attribute)
            else None
        )
        if base_name == 'Optional':
            return _relationship_target_node(node.slice)
        if base_name == 'Union':
            elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            non_none = [e for e in elts if not _is_none_node(e)]
            if len(non_none) == 1:
                return _relationship_target_node(non_none[0])
    return None


def _resolve_relationship_target(
    name: str,
    rel_info: typing.Any,
    annotation: typing.Any,
) -> typing.Any:
    """
    Wrapper around SQLModel's ``get_relationship_to`` that first normalizes a
    flat-string / ``ForwardRef`` nullable relationship annotation (e.g.
    ``'UserFolder | None'``) into a structured ``ForwardRef('UserFolder')`` via
    ``ast``, then delegates to upstream.

    Root cause: ``get_relationship_to`` can only strip ``None`` from an
    **already-evaluated** ``typing.Union``; it cannot parse a whole-string PEP 604
    annotation -- it would treat the entire ``'UserFolder | None'`` string as the
    class name and hand it to SQLAlchemy. ``Optional['T']`` works only because
    ``Optional[...]`` is evaluated at class-definition time into
    ``Union[ForwardRef('T'), None]``, leaving just the inner ``'T'`` as a ForwardRef.

    This wrapper lets relationship fields be annotated with the PEP 604
    ``'X | None'`` form (pyright-friendly, no ``reportDeprecated`` from
    ``Optional[...]``) and converts it back at runtime into a form
    ``get_relationship_to`` recognizes. Already-evaluated typing objects
    (``list[...]`` / ``Optional[...]`` / concrete classes) pass through unchanged;
    unrecognized expressions are likewise passed through to upstream, preserving
    its original error semantics. Note: only a temporary local value is produced
    for resolution -- ``cls.__annotations__`` is left intact.
    """
    if isinstance(annotation, (str, typing.ForwardRef)):
        expr = (
            annotation.__forward_arg__
            if isinstance(annotation, typing.ForwardRef)
            else annotation
        )
        try:
            target = _relationship_target_node(ast.parse(expr.strip(), mode='eval').body)
        except SyntaxError:
            target = None
        if target is not None:
            annotation = typing.ForwardRef(ast.unparse(target))
    return get_relationship_to(name=name, rel_info=rel_info, annotation=annotation)


def _make_sti_fk_resolver(
    fk_string: str,
    sa_registry: typing.Any,
) -> str | typing.Callable[[], list[Column[Any]]]:
    """
    Convert string-format foreign_keys to a callable for deferred resolution in STI.

    STI child columns are added to the parent table via _register_sti_columns(),
    but during configure_mappers() they are not yet registered as ColumnProperty.
    SQLAlchemy's string resolution (_GetColumns.__getattr__) looks up columns via
    mapper.all_orm_descriptors, which fails for unregistered STI columns.

    Solution: convert to callable so configure_mappers() calls it to resolve
    Column objects directly from the table's columns collection (Phase 1 already added them).

    :param fk_string: String-format foreign_keys, e.g. '[Order.billing_address_id]'
    :param sa_registry: SQLAlchemy registry for class-name lookup
    :return: callable returning list of Column objects, or ``fk_string`` unchanged
        when it cannot be parsed (SQLAlchemy then resolves it as usual)
    """
    inner = fk_string.strip('[]')
    specs = [s.strip() for s in inner.split(',')]

    parsed: list[tuple[str, str]] = []
    for spec in specs:
        m = re.match(r'^(\w+)\.(\w+)$', spec)
        if not m:
            return fk_string  # cannot parse, return original
        parsed.append((m.group(1), m.group(2)))

    _registry = sa_registry

    def _resolve() -> list[Column[Any]]:
        columns: list[Column[Any]] = []
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

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> typing.Any:
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

        # 3.5. OptimisticLockMixin wiring: register the mixin's
        # ``OPTIMISTIC_LOCK_VERSION_COLUMN`` (``oplock_version``) column as
        # SQLAlchemy's ``version_id_col`` so every UPDATE emits
        # ``SET oplock_version = oplock_version + 1 WHERE ... AND oplock_version = :current`` and a
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
                    merged['version_id_col'] = target_cls.__table__.c[OPTIMISTIC_LOCK_VERSION_COLUMN]
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

        # Snapshot the names this class body declares itself. Must be taken
        # before _recover_annotated_sqlmodel_fields, which injects inherited
        # Annotated fields into ``annotations`` (after that, "declared here" and
        # "inherited + injected" are indistinguishable).
        _own_annotation_names = frozenset(annotations)

        # 4.5. Fix Annotated[T, Field(foreign_key=...)] where SQLModel FieldInfo gets replaced
        # by Pydantic FieldInfo. Must run before super().__new__() because SQLModel calls
        # get_column_from_field() during __new__. Only for table classes; non-table classes
        # keep original Annotated annotations for child table classes to inherit.
        _recover_annotated_sqlmodel_fields(annotations, attrs, bases, will_be_table)

        # 4.5.b The optimistic-lock version column name is globally reserved:
        # any class that declares it in its own body fails fast, whether or not
        # it enables optimistic locking. Reserving it only for classes with the
        # lock enabled would let a class without the lock declare a domain
        # column of that name, which a descendant re-enabling the lock would
        # then silently wire up as ``version_id_col``. Classes that merely
        # *inherit* the column from OptimisticLockMixin are not in the own-name
        # snapshot and pass.
        if OPTIMISTIC_LOCK_VERSION_COLUMN in _own_annotation_names:
            raise TypeError(
                f"{name}: '{OPTIMISTIC_LOCK_VERSION_COLUMN}' is reserved for "
                f"OptimisticLockMixin's version_id_col and cannot be declared by a model. "
                f"Use another name (e.g. 'version' or 'revision') for a domain version field."
            )

        # 4.6. partial: turn inherited fields into omissible fields (Unset | T = Unset).
        # Used for PATCH DTOs to avoid re-declaring every field.
        if 'all_fields_optional' in kwargs:
            raise TypeError(
                f"{name}: the 'all_fields_optional' class keyword was removed in "
                f"sqlmodel-ext 0.5.0. Use 'partial=True' instead. The semantics changed: "
                f"omitted fields are now 'Unset' (pydantic's MISSING sentinel), not None, "
                f"and explicit null is only accepted where the base field allows None. "
                f"Replace 'x is not None' / 'x is None' checks on such fields with "
                f"'x is not Unset' / 'x is Unset' (from sqlmodel_ext import Unset); "
                f"Unset fields are already excluded from model_dump()."
            )
        is_partial = kwargs.pop('partial', False)
        if is_partial:
            # ``partial`` and ``table`` are mutually exclusive: the partial
            # default is ``Unset``, a sentinel that cannot be persisted.
            if will_be_table:
                raise TypeError(
                    f"{name}: 'partial=True' cannot be combined with 'table=True' -- the "
                    f"partial default is 'Unset', which cannot be stored. Use a separate "
                    f"non-table DTO class for PATCH payloads."
                )
            _apply_partial(annotations, attrs, bases, _own_annotation_names)

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
                else:
                    # Bare default value (e.g. ``fpath: FilePathType = Path("a.txt")``):
                    # the assignment is neither Undefined nor a FieldInfo. Wrap it in a
                    # Field so the extracted sa_type reaches the column builder while the
                    # default is preserved — without this branch the column silently
                    # falls back to the base-type inference (AutoString for Path),
                    # dropping the type's result_processor.
                    attrs[field_name] = Field(default=field_value, sa_type=sa_type)

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
        # to `name: str | None = None`) or ``partial=True`` programmatically
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

        # 12. Register partial classes (see ``optional_dto_registry``)
        if is_partial:
            optional_dto_registry.append(result)

        return result

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> None:
        """
        Override SQLModel's __init__ to support Joined Table Inheritance.

        SQLModel's original behavior skips DeclarativeMeta.__init__ if any base
        is a table model. This fix detects JTI scenarios and forces the call
        to create the child table.
        """
        from sqlmodel.main import is_table_model_class, DeclarativeMeta, ModelMetaclass

        if not is_table_model_class(cls):
            ModelMetaclass.__init__(cls, name, bases, attrs, **kwargs)
            return

        base_is_table = any(is_table_model_class(base) for base in bases)

        if not base_is_table:
            cls._setup_relationships()
            DeclarativeMeta.__init__(cls, name, bases, attrs, **kwargs)
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

        def _fk_matches_parent(fk_str: str, parent_table: str | None) -> bool:
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

            DeclarativeMeta.__init__(cls, name, bases, attrs, **kwargs)
        else:
            # STI: child shares parent table
            ModelMetaclass.__init__(cls, name, bases, attrs, **kwargs)

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
                                    relationship_to = _resolve_relationship_target(
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
                    # Abstract intermediate classes (polymorphic_abstract=True)
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

            relationship_to = _resolve_relationship_target(
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


class SQLModelExtConfig(SQLModelConfig, total=False):
    """
    ``SQLModelConfig`` plus sqlmodel-ext's own configuration keys.

    A ``total=False`` TypedDict like upstream: an absent key means "use the
    default". All upstream keys remain available::

        class ToolArguments(SQLModelBase):
            model_config = SQLModelExtConfig(omitted_sentinel=True)
    """

    omitted_sentinel: bool
    """
    Whether this model exposes a fillable wire value for **omissible** fields (annotation contains ``Unset``).

    **Off by default.** Pydantic's ``MISSING`` assumes the caller can omit
    keys, so its branch never appears in the JSON Schema. Some schema
    consumers require every key to be present (e.g. LLM function calling in
    strict mode) and would otherwise have to send ``null``, which means
    "clear" rather than "leave alone". When enabled:

    - inbound dict payloads have every occurrence of
      :data:`~sqlmodel_ext.unset.OMITTED_SENTINEL` (``'__omitted__'``) replaced
      by ``Unset`` before validation, at any nesting depth;
    - ``model_json_schema()`` adds a ``{"const": "__omitted__", "type": "string"}``
      branch (and ``"default": "__omitted__"``) to every omissible field of the
      model and of nested models reachable from its fields.

    Whether a caller can omit keys is a property of the caller, and callers are
    distinguished by model, so the switch is per model: the same domain DTO
    keeps a clean schema when a REST model inherits it and gains the sentinel
    branch when a strict-mode model inherits it.

    With the switch on, string fields of the model can no longer hold the
    literal ``'__omitted__'`` itself.
    """


class SQLModelBase(SQLModel, metaclass=__DeclarativeMeta):
    """
    Base class for all SQLModel models in sqlmodel_ext.

    Must be used together with TableBaseMixin or UUIDTableBaseMixin for table models.
    """

    model_config = SQLModelExtConfig(
        use_attribute_docstrings=True, validate_by_name=True, extra='forbid',
    )

    __orjson_checked_fields__: typing.ClassVar[tuple[str, ...]] = ()
    """
    Names of this model's ``JSON100K`` / ``JSONList100K`` fields, filled in by ``__pydantic_init_subclass__``.

    ``table=True`` models skip all Pydantic validators (a known SQLModel
    behavior), so the serializability invariant of these fields is enforced in
    ``model_post_init`` instead -- the field annotation is the declaration, no
    per-model hook to forget. See ``ensure_json_within_limits`` for the exact
    limits.

    Contract: subclasses overriding ``model_post_init`` must call
    ``super().model_post_init(context)``, otherwise the check is silently skipped.
    """

    @staticmethod
    def annotation_is_omissible(annotation: Any) -> bool:
        """
        Whether the annotation accepts ``Unset``, i.e. whether the field can be left unprovided.

        The criterion is the **annotation**, not whether the default happens to
        be ``Unset``: "can this field be omitted" is answered by the type, while
        the default answers a different question ("what do I get when it is
        omitted").

        A ``staticmethod`` because it is also needed where only a bare
        annotation is at hand; use :meth:`field_is_omissible` when you have a
        field name.

        :param annotation: field annotation; ``Annotated[...]`` and nested unions
            are searched recursively (constrained fields are wrapped in
            ``Annotated``, so looking one level deep is not enough)
        """
        pending: list[Any] = [annotation]
        while pending:
            node = pending.pop()
            if node is Unset:
                return True
            pending.extend(get_args(node))
        return False

    @classmethod
    def field_is_omissible(cls, field_name: str) -> bool:
        """
        Whether this model's field can be left unprovided -- orthogonal to whether it may be ``null``.

        Under tri-state semantics these are two independent questions:
        ``Unset`` answers the first, ``| None`` the second.

        :param field_name: field name; returns ``False`` (does not raise) for
            unknown names, so it can be used while scanning fields generically
        """
        info = cls.model_fields.get(field_name)
        return info is not None and cls.annotation_is_omissible(info.annotation)

    @model_validator(mode='before')
    @classmethod
    def _normalise_omitted_sentinel(cls, data: Any) -> Any:
        """
        Inbound normalization: replace the wire value with ``Unset`` (only when ``omitted_sentinel`` is on).

        Only dict input (the JSON path of ``model_validate``) is processed;
        ``from_attributes`` objects and model instances pass through unchanged
        -- the wire value only exists in inbound JSON.

        The whole payload tree is walked, not just the top level: a caller that
        cannot omit keys cannot omit them in nested items either. The wire value
        is a literal that never occurs as a real value, so no per-field
        decision is needed.

        Replacing with ``Unset`` is enough -- no key deletion: ``MISSING`` fields
        are excluded from serialization by Pydantic itself.
        """
        config = typing.cast(Mapping[str, Any], cls.model_config)
        # ``SQLModelExtConfig`` is total=False: an absent key means "off".
        if not config.get('omitted_sentinel', False):
            return data
        if not isinstance(data, dict):
            return data

        # A single pass is cheaper than "check whether it occurs, then replace":
        # the check itself already has to walk the tree.
        def _normalise(value: Any) -> Any:
            if value == OMITTED_SENTINEL:
                return Unset
            if isinstance(value, dict):
                return {k: _normalise(v) for k, v in typing.cast(dict[Any, Any], value).items()}
            if isinstance(value, list):
                return [_normalise(v) for v in typing.cast(list[Any], value)]
            return value

        return _normalise(data)

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Add the sentinel branch to omissible fields (only when ``omitted_sentinel`` is on).

        The injection runs in the JSON-schema generator's ``model_schema``
        hook (a subclass of the caller's ``schema_generator``), not in
        ``__get_pydantic_json_schema__``: the latter sees an intermediate
        product that Pydantic later re-assembles (``$ref`` resolution), and
        edits to ``properties`` made there do not reach the final output,
        whereas ``model_schema``'s return value is the model's final body.

        Which fields are omissible can only be decided at model level: a
        field-level ``Annotated`` hook does not see the host model's config.

        Every model in the generation is processed -- the host and the nested
        models reachable from it, whatever their ``$defs`` keys are -- since a
        caller that cannot omit keys cannot omit them in nested items either.
        The generator exists only for this call, so this never leaks into the
        schema of the nested model itself or of hosts with the switch off.
        """
        config = typing.cast(Mapping[str, Any], cls.model_config)
        if not config.get('omitted_sentinel', False):
            return super().model_json_schema(*args, **kwargs)

        def _inject(props: object, model: type[BaseModel], mode: str, by_alias: bool) -> None:
            """Inject the sentinel branch into the omissible fields of ``model`` found in ``props``."""
            if not isinstance(props, dict):
                return
            typed_props = typing.cast(dict[str, Any], props)
            for field_name, field in model.model_fields.items():
                if not cls.annotation_is_omissible(field.annotation):
                    continue
                key = _json_schema_property_key(field_name, field, mode, by_alias)
                field_schema = typed_props.get(key)
                if not isinstance(field_schema, dict):
                    continue
                # Readability rules (the schema is read by the consumer):
                # 1. annotation keywords (title / description ...) stay on the
                #    outer object instead of moving into one anyOf branch;
                # 2. an existing anyOf is extended rather than nested again.
                typed = typing.cast(dict[str, Any], field_schema)
                notes = {k: v for k, v in typed.items() if k in SCHEMA_ANNOTATION_KEYS}
                constraints = {k: v for k, v in typed.items() if k not in SCHEMA_ANNOTATION_KEYS}
                existing = constraints.pop('anyOf', None)
                branches = list(existing) if isinstance(existing, list) else [constraints]
                typed_props[key] = {
                    **notes,
                    'anyOf': [*branches, SENTINEL_SCHEMA_BRANCH],
                    'default': OMITTED_SENTINEL,
                }

        # Inject per model, inside the generator: ``model_schema`` is called
        # once for every model in this generation (host and nested alike) with
        # the model class in hand, and its output is what lands in the final
        # schema. No pairing of classes with ``$defs`` keys is needed -- those
        # keys are not class names (same-named models get disambiguated keys).
        bound = inspect.signature(BaseModel.model_json_schema).bind(*args, **kwargs)
        bound.apply_defaults()
        base_generator = typing.cast(type[GenerateJsonSchema], bound.arguments['schema_generator'])

        class _SentinelGenerator(base_generator):
            @typing.override
            def model_schema(self, schema: pydantic_core_schema.ModelSchema) -> JsonSchemaValue:
                json_schema = super().model_schema(schema)
                model: type[Any] = schema['cls']
                if issubclass(model, BaseModel):
                    _inject(json_schema.get('properties'), model, self.mode, self.by_alias)
                return json_schema

        bound.arguments['schema_generator'] = _SentinelGenerator
        return super().model_json_schema(*bound.args, **bound.kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Discover, at class creation, the fields that need the construction-time JSON check (see ``__orjson_checked_fields__``)."""
        super().__pydantic_init_subclass__(**kwargs)
        cls.__orjson_checked_fields__ = tuple(
            name for name, field_info in cls.model_fields.items()
            if _annotation_contains_orjson_checked_type(field_info.annotation)
        )

    def model_post_init(self, context: Any, /) -> None:
        """
        Construction-time invariant: ``JSON100K`` / ``JSONList100K`` values must be encodable and within the size limit.

        Deeply nested values (``orjson.loads`` accepts far deeper nesting than
        the serializers do) would otherwise be stored silently and only fail
        later in ``model_dump(mode='json')`` / flush, far from where the input
        entered -- typically when raw external JSON is used to construct a
        table row directly, bypassing DTO validation. See
        ``ensure_json_within_limits``.

        ORM loads from the database do not go through ``__init__`` and pay
        nothing; models without such fields iterate an empty tuple.
        """
        super().model_post_init(context)
        for field_name in self.__orjson_checked_fields__:
            value = getattr(self, field_name)
            # Neither ``Unset`` (not provided) nor ``None`` has content to check.
            # Non-empty ``__orjson_checked_fields__`` implies orjson is installed.
            if value is not Unset and value is not None and _ensure_json_within_limits is not None:
                _ensure_json_within_limits(value)

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

    def submitted_fields_among(self, *models: type['SQLModelBase']) -> set[str]:
        """
        Return the explicitly submitted fields (``model_fields_set``) that belong to any of ``models``.

        Pure set arithmetic: ``self.model_fields_set`` intersected with the union
        of the ``models``' field names. It carries no "forbidden" semantics --
        the caller decides what a hit means. A typical use is privilege checks
        on a shared update body: pass a model declaring the admin-only fields
        and reject the request if a non-admin submitted any of them::

            class ItemAdminOnlyFields(SQLModelBase):
                is_featured: bool

            class ItemAdminUpdate(ItemAdminOnlyFields, ItemUpdate, partial=True):
                pass

            forbidden = body.submitted_fields_among(ItemAdminOnlyFields)
            if forbidden and not user.is_admin:
                raise PermissionError(sorted(forbidden))

        :param models: model classes providing field names (union of their ``model_fields``)
        :returns: the matching field names (empty set = no overlap)
        """
        field_names: set[str] = set()
        for model in models:
            field_names.update(model.model_fields)
        return self.model_fields_set & field_names


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

    model_config = SQLModelExtConfig(
        use_attribute_docstrings=True, validate_by_name=True, extra='ignore',
    )

    @model_validator(mode='before')
    @classmethod
    def _warn_unknown_fields(cls, data: Any) -> Any:
        """
        Detect and warn about unknown fields in incoming data.

        Logs a WARNING before Pydantic's extra='ignore' discards unknown fields,
        helping developers notice third-party API changes and add field definitions.
        Field names, ``alias`` and ``validation_alias`` (a string or every string
        choice of an ``AliasChoices``) count as known keys; ``AliasPath`` entries
        are nested paths, not top-level keys, and are ignored.
        """
        if not isinstance(data, dict):
            return data
        accepted: set[str] = set()
        for name, field_info in cls.model_fields.items():
            accepted.add(name)
            if field_info.alias:
                accepted.add(field_info.alias)
            validation_alias = field_info.validation_alias
            if isinstance(validation_alias, str):
                accepted.add(validation_alias)
            elif isinstance(validation_alias, AliasChoices):
                accepted.update(c for c in validation_alias.choices if isinstance(c, str))
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
