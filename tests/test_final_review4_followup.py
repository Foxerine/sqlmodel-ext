"""Final-review 4 regressions for forward-ref partials and late cached models."""

from typing import Annotated, Literal

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import Field

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin, Unset
from sqlmodel_ext.mixins.cached_table import _QUERY_ONLY_INVALIDATION, _SESSION_PENDING_CACHE_KEY


class AnnotatedForwardBase(SQLModelBase):
    child: Annotated['AnnotatedForwardBase | None', Field(description='child')] = None


class AnnotatedForwardPatch(AnnotatedForwardBase, partial=True):
    pass


def test_partial_resolves_nullable_and_container_string_forward_refs() -> None:
    class NodeBase(SQLModelBase):
        name: str
        child: 'NodeBase | None' = None
        kids: 'list[NodeBase]' = Field(default_factory=list)

    class NodePatch(NodeBase, partial=True):
        pass

    empty = NodePatch.model_validate({})
    assert empty.name is Unset
    assert empty.child is Unset
    assert empty.kids is Unset
    assert empty.model_dump() == {}

    explicit_null = NodePatch.model_validate({'child': None})
    assert explicit_null.child is None
    assert explicit_null.model_dump() == {'child': None}

    nested = NodePatch.model_validate({
        'child': {'name': 'child'},
        'kids': [{'name': 'first'}, {'name': 'second'}],
    })
    assert isinstance(nested.child, NodeBase)
    assert [kid.name for kid in nested.kids] == ['first', 'second']


def test_partial_resolves_forward_ref_nested_in_annotated() -> None:
    empty = AnnotatedForwardPatch.model_validate({})
    assert empty.child is Unset
    assert empty.model_dump() == {}
    nested = AnnotatedForwardPatch.model_validate({'child': {'child': None}})
    assert isinstance(nested.child, AnnotatedForwardBase)


def test_partial_string_forward_ref_keeps_non_nullable_field_non_nullable() -> None:
    class NameBase(SQLModelBase):
        name: 'str'

    class NamePatch(NameBase, partial=True):
        pass

    assert NamePatch.model_validate({}).name is Unset
    assert NamePatch.model_dump(NamePatch.model_validate({})) == {}

    try:
        NamePatch.model_validate({'name': None})
    except ValueError:
        pass
    else:
        raise AssertionError('explicit null must remain invalid for a non-nullable forward-ref field')


def test_partial_does_not_reject_string_literal_discriminator() -> None:
    class KindBase(SQLModelBase):
        kind: 'Literal["a", "b"]' = 'a'

    class KindPatch(KindBase, partial=True):
        pass

    assert KindPatch.model_fields['kind'].annotation == Literal['a', 'b']
    assert KindPatch.model_validate({}).kind == 'a'


def test_late_cached_model_rebuilds_warmed_raw_dml_index(engine: AsyncEngine) -> None:
    CachedTableBaseMixin._cached_tablename_index = None
    warmed = CachedTableBaseMixin._build_cached_tablename_index()

    class Review4LateCachedModel(
        SQLModelBase,
        CachedTableBaseMixin,
        UUIDTableBaseMixin,
        table=True,
    ):
        name: str

    assert Review4LateCachedModel.__tablename__ not in warmed
    assert CachedTableBaseMixin._cached_tablename_index is None

    session = AsyncSession(engine)
    try:
        CachedTableBaseMixin.register_raw_dml_write(
            session,
            insert(Review4LateCachedModel.__table__).values(name='late'),
        )
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY, {})
        assert pending.get(Review4LateCachedModel) == {_QUERY_ONLY_INVALIDATION}
    finally:
        CachedTableBaseMixin._cached_tablename_index = None
