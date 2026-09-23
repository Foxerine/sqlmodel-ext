"""Adversarial test for cached models imported after the raw-DML index is built."""

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from sqlmodel_ext import AsyncSession, CachedTableBaseMixin, SQLModelBase, UUIDTableBaseMixin
from sqlmodel_ext.mixins.cached_table import _SESSION_PENDING_CACHE_KEY


def test_startup_warmup_indexes_cached_models_for_raw_dml(engine: AsyncEngine) -> None:
    class ReviewStartupCachedModel(
        SQLModelBase,
        CachedTableBaseMixin,
        UUIDTableBaseMixin,
        table=True,
    ):
        name: str

    CachedTableBaseMixin._cached_tablename_index = None
    session = AsyncSession(engine)
    try:
        CachedTableBaseMixin.register_raw_dml_write(
            session,
            insert(ReviewStartupCachedModel.__table__).values(name='loaded-before-warmup'),
        )
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY, {})
        assert ReviewStartupCachedModel in pending
    finally:
        CachedTableBaseMixin._cached_tablename_index = None
