"""
RelationPreloadMixin / @requires_relations / @requires_for_update tests.

Model chain: PreloadFunction -> generator (PreloadGenerator) -> config (PreloadConfig).

Notes on the runtime safety net: ``SQLModelBase``'s metaclass injects
``lazy='raise_on_sql'`` into every relationship, so touching an unloaded
relationship raises ``sqlalchemy.exc.InvalidRequestError`` instead of doing
sync IO (which would raise MissingGreenlet). The decorator's job is to make
sure that never happens for declared relations.

NOTE: no ``from __future__ import annotations`` here -- PEP 563 string
annotations break SQLAlchemy's resolution of the quoted Relationship
annotations (``list["PreloadGenerator"]`` would reach the registry as the
literal string ``"list['PreloadGenerator']"``).
"""
import uuid

import pytest
from sqlalchemy.exc import InvalidRequestError
from sqlmodel import Field, Relationship
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    SESSION_FOR_UPDATE_KEY,
    RelationPreloadMixin,
    SQLModelBase,
    UUIDTableBaseMixin,
    rel,
    requires_for_update,
    requires_relations,
)


class PreloadConfig(SQLModelBase, UUIDTableBaseMixin, table=True):
    price: int = 0
    generators: list["PreloadGenerator"] = Relationship(back_populates="config")


class PreloadGenerator(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "gen"
    config_id: uuid.UUID | None = Field(default=None, foreign_key="preloadconfig.id")
    config: PreloadConfig | None = Relationship(back_populates="generators")
    functions: list["PreloadFunction"] = Relationship(back_populates="generator")


class PreloadFunction(RelationPreloadMixin, SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "fn"
    generator_id: uuid.UUID | None = Field(default=None, foreign_key="preloadgenerator.id")
    generator: PreloadGenerator | None = Relationship(back_populates="functions")

    @requires_relations('generator')
    async def gen_name(self, session: AsyncSession) -> str:
        return self.generator.name

    @requires_relations('generator', PreloadGenerator.config)
    async def config_price(self, session: AsyncSession) -> int:
        return self.generator.config.price

    @requires_relations('generator')
    async def stream_names(self, session: AsyncSession):
        yield self.generator.name
        yield self.generator.name.upper()

    @requires_relations('generator')
    async def no_session_method(self) -> str:
        """Decorated but receives no session -> decorator cannot auto-load."""
        return self.generator.name

    @requires_for_update
    async def locked_op(self, session: AsyncSession) -> str:
        return "locked-ok"

    @requires_for_update
    async def locked_op_without_session(self) -> str:
        """No session available -> the FOR UPDATE runtime check is skipped."""
        return "no-session-ok"


async def _make_chain(session: AsyncSession) -> uuid.UUID:
    """Persist config -> generator -> function; return the function id."""
    cfg = await PreloadConfig(price=42).save(session)
    gen = await PreloadGenerator(name="gen1", config_id=cfg.id).save(session)
    fn = await PreloadFunction(name="fn1", generator_id=gen.id).save(session)
    return fn.id


@pytest.mark.asyncio
class TestRequiresRelations:
    async def test_unloaded_relation_access_raises_without_decorator(
        self, session: AsyncSession
    ) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None
        assert fn._is_relation_loaded('generator') is False
        with pytest.raises(InvalidRequestError):
            _ = fn.generator  # raise_on_sql safety net, no MissingGreenlet

    async def test_decorated_method_autoloads_relation(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None
        assert fn._is_relation_loaded('generator') is False

        assert await fn.gen_name(session) == "gen1"
        assert fn._is_relation_loaded('generator') is True
        # After auto-load, plain attribute access is safe too.
        assert fn.generator.name == "gen1"

    async def test_already_loaded_relation_is_not_replaced(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None

        await fn.gen_name(session)
        loaded_generator = fn.generator
        await fn.gen_name(session)  # second call: relation already loaded -> skipped
        assert fn.generator is loaded_generator

    async def test_nested_queryable_attribute_spec(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None

        # 'generator' + PreloadGenerator.config nested chain loaded in one shot.
        assert await fn.config_price(session) == 42
        assert fn._is_relation_loaded('generator') is True
        assert fn.generator.config.price == 42

    async def test_async_generator_method_is_supported(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None

        items = [item async for item in fn.stream_names(session)]
        assert items == ["gen1", "GEN1"]

    async def test_decorated_method_without_session_cannot_autoload(
        self, session: AsyncSession
    ) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None

        # No session param -> decorator skips loading -> safety net fires.
        with pytest.raises(InvalidRequestError):
            await fn.no_session_method()

    async def test_decorator_attaches_required_relations_metadata(self) -> None:
        assert PreloadFunction.gen_name._required_relations == ('generator',)
        specs = PreloadFunction.config_price._required_relations
        assert specs[0] == 'generator'
        assert specs[1].key == 'config'

    async def test_invalid_relation_name_rejected_at_class_creation(self) -> None:
        with pytest.raises(AttributeError, match="no such attribute"):
            class BadPreloadDecl(RelationPreloadMixin):
                @requires_relations('nonexistent_rel')
                async def method(self, session: AsyncSession) -> None:
                    ...


@pytest.mark.asyncio
class TestManualPreloadAPI:
    async def test_get_relations_for_method(self) -> None:
        rels = PreloadFunction.get_relations_for_method('gen_name')
        assert [r.key for r in rels] == ['generator']
        # Unknown / undecorated methods yield an empty list.
        assert PreloadFunction.get_relations_for_method('save') == []
        assert PreloadFunction.get_relations_for_method('does_not_exist') == []

    async def test_get_relations_for_methods_deduplicates(self) -> None:
        rels = PreloadFunction.get_relations_for_methods('gen_name', 'config_price')
        assert [r.key for r in rels] == ['generator', 'config']

    async def test_preload_for_loads_declared_relations(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None
        assert fn._is_relation_loaded('generator') is False

        result = await fn.preload_for(session, 'config_price')
        assert result is fn  # chaining contract
        assert fn._is_relation_loaded('generator') is True
        assert fn.generator.config.price == 42


@pytest.mark.asyncio
class TestRequiresForUpdate:
    async def test_unlocked_instance_is_rejected(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None

        with pytest.raises(RuntimeError, match="requires a FOR UPDATE locked instance"):
            await fn.locked_op(session)

    async def test_for_update_get_tracks_instance_and_allows_call(
        self, session: AsyncSession
    ) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(
            session, PreloadFunction.id == fid, with_for_update=True
        )
        assert fn is not None

        # get(with_for_update=True) records id(instance) in session.info.
        assert id(fn) in session.info.get(SESSION_FOR_UPDATE_KEY, set())
        assert await fn.locked_op(session) == "locked-ok"

    async def test_check_skipped_when_no_session_available(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert fn is not None
        assert await fn.locked_op_without_session() == "no-session-ok"

    async def test_static_metadata_flag(self) -> None:
        assert PreloadFunction.locked_op._requires_for_update is True


@pytest.mark.asyncio
class TestRelHelper:
    async def test_rel_returns_queryable_attribute(self) -> None:
        attr = rel(PreloadFunction.generator)
        assert attr.key == 'generator'

    async def test_rel_rejects_non_relationship(self) -> None:
        instance = PreloadFunction(name="x")
        with pytest.raises(AttributeError, match="Expected a Relationship field"):
            rel(instance.name)

    async def test_get_with_rel_load_preloads(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(
            session, PreloadFunction.id == fid, load=rel(PreloadFunction.generator)
        )
        assert fn is not None
        assert fn._is_relation_loaded('generator') is True
        assert fn.generator.name == "gen1"
