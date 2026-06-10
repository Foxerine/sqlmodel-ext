"""
Enhanced ``AsyncSession``: orchestration-level behavior without Redis.

Full cache invalidation requires a Redis client; these tests cover the parts
that must work standalone:

- ``commit()`` degrades to a plain commit when no cached model is involved.
- ``reset()`` clears the FOR UPDATE tracking key and the three
  cache-invalidation tracking keys from ``session.info``.
- ``refresh()`` falls back to the native refresh for non-cached models.
- ``execute()`` passes statements through unchanged (SELECT path).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlmodel_ext import AsyncSession, SESSION_FOR_UPDATE_KEY
from sqlmodel_ext.mixins.cached_table import (
    _SESSION_CASCADE_DELETED_KEY,
    _SESSION_PENDING_CACHE_KEY,
    _SESSION_SYNCED_CACHE_KEY,
)

from tests._models import FunctionC


@pytest_asyncio.fixture
async def enhanced_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Enhanced session bound to the fresh per-test engine."""
    async with AsyncSession(engine) as s:
        yield s


@pytest.mark.asyncio
async def test_commit_plain_model_degrades_to_plain_commit(
    enhanced_session: AsyncSession,
) -> None:
    """Non-cached models commit normally -- no Redis required."""
    func = FunctionC(name="plain")
    enhanced_session.add(func)
    await enhanced_session.commit()

    result = await enhanced_session.execute(select(FunctionC))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].name == "plain"


@pytest.mark.asyncio
async def test_reset_clears_tracking_state(enhanced_session: AsyncSession) -> None:
    enhanced_session.info[SESSION_FOR_UPDATE_KEY] = {123}
    enhanced_session.info[_SESSION_PENDING_CACHE_KEY] = {}
    enhanced_session.info[_SESSION_SYNCED_CACHE_KEY] = {}
    enhanced_session.info[_SESSION_CASCADE_DELETED_KEY] = {}

    await enhanced_session.reset()

    assert SESSION_FOR_UPDATE_KEY not in enhanced_session.info
    assert _SESSION_PENDING_CACHE_KEY not in enhanced_session.info
    assert _SESSION_SYNCED_CACHE_KEY not in enhanced_session.info
    assert _SESSION_CASCADE_DELETED_KEY not in enhanced_session.info


@pytest.mark.asyncio
async def test_refresh_falls_back_for_non_cached_model(
    enhanced_session: AsyncSession,
) -> None:
    """Non-cached models take the native session.refresh() path."""
    func = FunctionC(name="before")
    enhanced_session.add(func)
    await enhanced_session.commit()

    await enhanced_session.refresh(func)
    assert func.name == "before"


@pytest.mark.asyncio
async def test_crud_save_via_enhanced_session(enhanced_session: AsyncSession) -> None:
    """TableBaseMixin.save() works unchanged on the enhanced session."""
    func = FunctionC(name="crud")
    func = await func.save(enhanced_session)
    assert func.name == "crud"
