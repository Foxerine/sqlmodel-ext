"""Round-two adversarial tests for raw INSERT invalidation edge cases."""
from uuid import uuid4

import fakeredis.aioredis
import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin


class ReviewInsertEdgeItem(
    SQLModelBase,
    CachedTableBaseMixin,
    UUIDTableBaseMixin,
    table=True,
):
    name: str


async def _names(*, engine: AsyncEngine, no_cache: bool = False) -> list[str]:
    async with AsyncSession(engine) as session:
        items = await ReviewInsertEdgeItem.get(
            session,
            fetch_mode='all',
            no_cache=no_cache,
        )
        return [item.name for item in items]


@pytest.mark.asyncio
async def test_raw_insert_commit_false_invalidates_on_later_commit(engine: AsyncEngine) -> None:
    redis = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(redis)
    CachedTableBaseMixin.check_cache_config()
    try:
        assert await _names(engine=engine) == []
        async with AsyncSession(engine) as writer:
            await writer.exec(
                insert(ReviewInsertEdgeItem.__table__).values(id=uuid4(), name='deferred')
            )
            assert await _names(engine=engine) == []
            await writer.commit()
        assert await _names(engine=engine) == ['deferred']
    finally:
        CachedTableBaseMixin._redis_client = None
        await redis.aclose()


@pytest.mark.asyncio
async def test_raw_insert_rolled_back_savepoint_only_overinvalidates(engine: AsyncEngine) -> None:
    redis = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(redis)
    CachedTableBaseMixin.check_cache_config()
    try:
        assert await _names(engine=engine) == []
        async with AsyncSession(engine) as writer:
            transaction = await writer.begin_nested()
            await writer.exec(
                insert(ReviewInsertEdgeItem.__table__).values(id=uuid4(), name='rolled-back')
            )
            await transaction.rollback()
            await writer.commit()
        assert await _names(engine=engine) == []
        assert await _names(engine=engine, no_cache=True) == []
    finally:
        CachedTableBaseMixin._redis_client = None
        await redis.aclose()
