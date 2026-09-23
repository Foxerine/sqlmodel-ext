"""
Regression: ``Annotated[X, Field(default_factory=...)]`` must keep its default
across multi-level inheritance.

Root cause (sqlmodel_ext.base.__DeclarativeMeta.__new__, sa_type injection
loop): when a field is declared ``Annotated[X, Field(default_factory=list,
...)]`` *without* an explicit ``= ...`` assignment, ``attrs[field_name]`` is
``Undefined``. The previous code unconditionally replaced it with a fresh
``Field(sa_type=sa_type)``, dropping the Field metadata that lived inside
the Annotated args. Single-class instantiation went through Pydantic's
native Annotated path and worked, but child classes rebuilt
``model_fields`` from the clobbered ``attrs`` and the field became silently
``is_required=True``.

Fix: ``_find_field_info_in_annotated()`` recovers the embedded FieldInfo
(merging multiple FieldInfo args into a shallow copy so shared Annotated
metadata singletons aren't mutated). The metaclass attaches ``sa_type`` to
that FieldInfo instead of clobbering it.

This regression suite locks in the fix at three abstraction levels:
1. Plain SQLModelBase + plain ``Annotated[list[str], Field(default_factory=...)]``
2. Custom ``__get_pydantic_core_schema__`` provider in the Annotated args
   (the real-world trigger — e.g. PostgreSQL ARRAY types)
3. Multi-FieldInfo in Annotated (``Annotated[Str64Alias, Field(unique=True)]``
   expands to ``Annotated[str, Field(max_length=64), Field(unique=True)]``)
"""
from __future__ import annotations

from typing import Annotated, Any, final

import pytest
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from sqlalchemy import BigInteger, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Field, SQLModel

from sqlmodel_ext import NonNegativeBigInt, NonNegativeInt
from sqlmodel_ext.base import SQLModelBase


# ---------------------------------------------------------------------------
# Helpers: a custom Annotated provider that returns sa_type metadata
# (mirrors what real PostgreSQL Array / pgvector / etc. providers do).
# ---------------------------------------------------------------------------


@final
class _StrArrayProvider:
    """Minimal ``__get_pydantic_core_schema__`` returning ``metadata['sa_type']``.

    Stand-in for the project's PostgreSQL ``Array[str]`` type; matters because
    the metaclass loops on annotations and pulls ``sa_type`` from this
    provider, which is exactly the trigger for the bug.
    """

    def __get_pydantic_core_schema__(
        self, source_type: type[Any], handler: GetCoreSchemaHandler,  # noqa: ARG002
    ) -> CoreSchema:
        list_schema = core_schema.list_schema(core_schema.str_schema())
        return core_schema.json_or_python_schema(
            json_schema=list_schema,
            python_schema=list_schema,
            metadata={'sa_type': ARRAY(String)},
        )


_StrArray = Annotated[list[str], _StrArrayProvider()]


# Module-level Annotated alias used by TestMultiFieldInfoInAnnotated.
# Defined here (not inside the test method) so the SQLModelBase metaclass and
# Pydantic's get_type_hints can resolve the inner Annotated reference cleanly.
_Str64 = Annotated[str, Field(max_length=64)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlainAnnotatedDefaultFactory:
    """Without any custom sa_type provider — should always have worked."""

    def test_default_factory_preserved_through_inheritance(self) -> None:
        class Pub(SQLModelBase):
            tags: Annotated[list[str], Field(default_factory=list, max_length=10)]

        class Mid(Pub):
            pass

        class Leaf(Mid):
            pass

        fi = Leaf.model_fields['tags']
        assert fi.is_required() is False
        assert fi.default_factory is list


class TestSaTypeProviderPlusFieldInfo:
    """The real bug: provider that exposes sa_type co-located with Field()."""

    def test_default_factory_survives_when_paired_with_sa_type(self) -> None:
        class Pub(SQLModelBase):
            arr: Annotated[_StrArray, Field(default_factory=list, max_length=10)]

        class Mid(Pub):
            pass

        class Leaf(Mid):
            pass

        fi = Leaf.model_fields['arr']
        assert fi.is_required() is False, (
            "Annotated[X, Field(default_factory=...)] must NOT be required "
            "after multi-level inheritance when X provides sa_type"
        )
        assert fi.default_factory is list

    def test_no_arg_construction_yields_default(self) -> None:
        class Pub(SQLModelBase):
            arr: Annotated[_StrArray, Field(default_factory=list, max_length=10)]

        class Leaf(Pub):
            pass

        instance = Leaf()
        assert instance.arr == []

    def test_explicit_value_still_respected(self) -> None:
        class Pub(SQLModelBase):
            arr: Annotated[_StrArray, Field(default_factory=list, max_length=10)]

        class Leaf(Pub):
            pass

        instance = Leaf(arr=['x', 'y'])
        assert instance.arr == ['x', 'y']

    def test_max_length_constraint_preserved(self) -> None:
        """The Field(max_length=...) constraint must reach Pydantic too."""
        class Pub(SQLModelBase):
            arr: Annotated[_StrArray, Field(default_factory=list, max_length=2)]

        class Leaf(Pub):
            pass

        with pytest.raises(Exception):  # noqa: B017,PT011 — Pydantic ValidationError or sub
            Leaf(arr=['a', 'b', 'c'])  # over the limit


class TestEqualsFieldFormStillWorks:
    """Sanity: the alternative ``X = Field(...)`` form was never broken
    and must keep working alongside the Annotated-only form."""

    def test_equals_form_remains_optional(self) -> None:
        class Pub(SQLModelBase):
            arr: _StrArray = Field(default_factory=list, max_length=10)

        class Mid(Pub):
            pass

        class Leaf(Mid):
            pass

        fi = Leaf.model_fields['arr']
        assert fi.is_required() is False
        assert fi.default_factory is list


class TestMultiFieldInfoInAnnotated:
    """``Annotated[Str64Alias, Field(unique=True)]`` expands to
    ``Annotated[str, Field(max_length=64), Field(unique=True)]``.

    The helper must merge both FieldInfo args. (Without sa_type provider here
    — we just verify the metadata-only path still works in inheritance.)
    """

    def test_merged_field_info_optional_default_preserved(self) -> None:
        # Two Field() args after Annotated flattening:
        # ``Annotated[_Str64, Field(default='anonymous')]`` ==
        # ``Annotated[str, Field(max_length=64), Field(default='anonymous')]``.
        # The helper must merge both FieldInfo args so the default reaches
        # the leaf class and ``is_required`` is False.
        class Pub(SQLModelBase):
            name: Annotated[_Str64, Field(default='anonymous')]

        class Leaf(Pub):
            pass

        fi = Leaf.model_fields['name']
        assert fi.is_required() is False
        assert fi.default == 'anonymous'
        instance = Leaf()
        assert instance.name == 'anonymous'


# ---------------------------------------------------------------------------
# Explicit ``default=None`` must survive metaclass normalization
# ---------------------------------------------------------------------------
#
# A constrained alias (``NonNegativeBigInt`` = ``Annotated[int, Field(ge=0, ...,
# sa_type=BigInteger)]``) combined with ``| None`` and a right-hand
# ``= Field(default=None, ...)`` used to lose the explicit ``default=None``: the
# right-hand FieldInfo was discarded while the annotation's FieldInfo was
# recovered, and a ``None`` default was treated as "unset" when FieldInfos were
# merged -- the field silently became required. The three declaration shapes
# below must agree on the *real* table-model path.


def _admits_none(annotation: object) -> bool:
    import types
    import typing

    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        return type(None) in typing.get_args(annotation)
    return annotation is type(None)


class ExplicitNoneRhsField(SQLModelBase, table=True):
    """Right-hand ``Field(...)`` coexisting with the alias FieldInfo."""
    id: NonNegativeInt = Field(primary_key=True)
    value: NonNegativeBigInt | None = Field(default=None, sa_type=BigInteger)


class ExplicitNonePlainDefault(SQLModelBase, table=True):
    """Alias with a plain ``= None`` default."""
    id: NonNegativeInt = Field(primary_key=True)
    value: NonNegativeBigInt | None = None


class ExplicitNoneBareType(SQLModelBase, table=True):
    """Control: bare type, no alias FieldInfo involved."""
    id: NonNegativeInt = Field(primary_key=True)
    value: bool | None = None


_EXPLICIT_NONE_MODELS = (ExplicitNoneRhsField, ExplicitNonePlainDefault, ExplicitNoneBareType)
# Reflection only -- keep them out of the shared metadata used by create_all().
for _model in _EXPLICIT_NONE_MODELS:
    SQLModel.metadata.remove(_model.__table__)  # type: ignore[attr-defined]


class TestExplicitNoneDefaultSurvives:
    def test_rhs_field_shape_is_not_required(self) -> None:
        field = _EXPLICIT_NONE_MODELS[0].model_fields['value']
        assert field.is_required() is False
        assert field.default is None

    def test_all_shapes_agree(self) -> None:
        for model in _EXPLICIT_NONE_MODELS:
            field = model.model_fields['value']
            assert field.is_required() is False, model.__name__
            assert field.default is None, model.__name__
            assert _admits_none(field.annotation), model.__name__

    def test_constrained_shapes_keep_column_type_and_constraints(self) -> None:
        for model in _EXPLICIT_NONE_MODELS[:2]:
            column = model.__table__.columns['value']  # type: ignore[attr-defined]
            assert isinstance(column.type, BigInteger), model.__name__
            assert column.nullable is True
            model.model_validate({'id': 1, 'value': 0})
            model.model_validate({'id': 1, 'value': None})
            with pytest.raises(ValueError):
                model.model_validate({'id': 1, 'value': -1})

    def test_merge_keeps_explicit_none_default(self) -> None:
        from pydantic_core import PydanticUndefined

        from sqlmodel_ext.base import _merge_field_info_attrs

        target = Field(sa_type=BigInteger)
        _merge_field_info_attrs(target, Field(default=None))
        assert target.default is None

        # ...while an unset default stays unset.
        target = Field(sa_type=BigInteger)
        _merge_field_info_attrs(target, Field(alias='x'))
        assert target.default is PydanticUndefined
        assert target.alias == 'x'


class RhsPrimaryKeyOnAlias(SQLModelBase, table=True):
    """``= Field(primary_key=True)`` on a constrained alias that carries its own SQLModel FieldInfo."""
    id: NonNegativeInt = Field(primary_key=True)
    label: str = 'x'


SQLModel.metadata.remove(RhsPrimaryKeyOnAlias.__table__)  # type: ignore[attr-defined]


class TestRhsFieldAttributesReachColumn:
    def test_primary_key_from_rhs_field(self) -> None:
        # sqlmodel >= 0.0.32 reads only the first FieldInfoMetadata carrier;
        # the rhs values must be folded into it, not appended after it.
        assert RhsPrimaryKeyOnAlias.__table__.c.id.primary_key is True  # type: ignore[attr-defined]

    def test_alias_singleton_not_mutated(self) -> None:
        import typing

        from sqlmodel.main import FieldInfoMetadata

        alias_fi = typing.get_args(NonNegativeInt)[1]
        carriers = [m for m in alias_fi.metadata if isinstance(m, FieldInfoMetadata)]
        assert carriers and all(c.primary_key is not True for c in carriers)
