"""Round-two adversarial test for raw INSERT invalidation across cached STI classes."""
from uuid import uuid4

import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlmodel_ext import (
    AsyncSession,
    AutoPolymorphicIdentityMixin,
    CachedTableBaseMixin,
    PolymorphicBaseMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
)
from sqlmodel_ext.mixins.cached_table import (
    _QUERY_ONLY_INVALIDATION,
    _SESSION_PENDING_CACHE_KEY,
)


class ReviewCachedStiRoot(
    SQLModelBase,
    CachedTableBaseMixin,
    UUIDTableBaseMixin,
    PolymorphicBaseMixin,
    table=True,
):
    name: str


class ReviewCachedStiChild(
    ReviewCachedStiRoot,
    AutoPolymorphicIdentityMixin,
    table=True,
):
    detail: str | None = None


@pytest.mark.asyncio
async def test_raw_insert_registers_every_cached_class_on_shared_sti_table(
    engine: AsyncEngine,
) -> None:
    CachedTableBaseMixin._cached_tablename_index = None
    async with AsyncSession(engine) as writer:
        await writer.exec(
            insert(ReviewCachedStiRoot.__table__).values(
                id=uuid4(),
                name='raw',
                _polymorphic_name='reviewcachedstichild',
            )
        )
        pending = writer.info[_SESSION_PENDING_CACHE_KEY]
        assert pending[ReviewCachedStiRoot] == {_QUERY_ONLY_INVALIDATION}
        assert pending[ReviewCachedStiChild] == {_QUERY_ONLY_INVALIDATION}
        await writer.rollback()
