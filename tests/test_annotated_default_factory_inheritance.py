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
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Field

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
