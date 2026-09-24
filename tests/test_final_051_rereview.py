"""Independent re-review probes for the 0.5.1 follow-up."""
from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin


class Final051ExpiredRow(SQLModelBase, CachedTableBaseMixin, UUIDTableBaseMixin, table=True):
    """Cached row used to exercise an expired identity-map instance."""

    value: int


@pytest_asyncio.fixture
async def final_051_rereview_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis()
    CachedTableBaseMixin.configure_redis(client)
    CachedTableBaseMixin.check_cache_config()
    try:
        yield client
    finally:
        CachedTableBaseMixin._redis_client = None
        await client.aclose()


@pytest.mark.asyncio
async def test_expired_instance_bare_mutation_invalidates_id_cache(
    engine: AsyncEngine,
    final_051_rereview_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    async with AsyncSession(engine) as session:
        row = Final051ExpiredRow(value=1)
        session.add(row)
        await session.flush()
        row_id = row.id
        await session.commit()
        cached = await Final051ExpiredRow.get(session, Final051ExpiredRow.id == row_id)
        assert cached is not None
        await session.commit()
        assert "id" not in row.__dict__
        assert await final_051_rereview_redis.exists(f"id:Final051ExpiredRow:{row_id}") == 1

        row.value = 2
        await session.commit()

    assert await final_051_rereview_redis.exists(f"id:Final051ExpiredRow:{row_id}") == 0
