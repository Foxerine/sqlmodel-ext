"""Adversarial release-review test for raw INSERT cache invalidation."""
from uuid import uuid4

import fakeredis.aioredis
import pytest
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin


class ReviewRawInsertItem(
    SQLModelBase,
    CachedTableBaseMixin,
    UUIDTableBaseMixin,
    table=True,
):
    name: str


@pytest.mark.asyncio
async def test_raw_insert_invalidates_existing_query_cache(engine: AsyncEngine) -> None:
    redis = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(redis)
    CachedTableBaseMixin.check_cache_config()
    try:
        async with AsyncSession(engine) as reader:
            cached_before = await ReviewRawInsertItem.get(reader, fetch_mode='all')
            assert cached_before == []

        item_id = uuid4()
        async with AsyncSession(engine) as writer:
            await writer.exec(
                insert(ReviewRawInsertItem.__table__).values(id=item_id, name='raw')
            )
            await writer.commit()

        async with AsyncSession(engine) as reader:
            cached_after = await ReviewRawInsertItem.get(reader, fetch_mode='all')
            database_after = await ReviewRawInsertItem.get(
                reader,
                fetch_mode='all',
                no_cache=True,
            )

        assert [item.name for item in cached_after] == ['raw']
        assert [item.name for item in database_after] == ['raw']
    finally:
        CachedTableBaseMixin._redis_client = None
        await redis.aclose()
