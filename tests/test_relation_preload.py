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
from sqlmodel_ext.mixins import (
    SESSION_REPEATABLE_READ_KEY,
    requires_locked_param,
    requires_read_committed,
    requires_repeatable_read,
    validate_locked_instances,
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
        """No session available -> the FOR UPDATE guard fails closed."""
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
            class BadPreloadDecl(RelationPreloadMixin, SQLModelBase):
                @requires_relations('nonexistent_rel')
                async def method(self, session: AsyncSession) -> None:
                    ...

    async def test_plain_mixin_is_validated_by_its_sqlmodel_host(self) -> None:
        # A plain Python mixin cannot own relationships; its declarations are
        # validated when a concrete SQLModel class mixes it in.
        class PlainDeclMixin(RelationPreloadMixin):
            @requires_relations('generator')
            async def method(self, session: AsyncSession) -> None:
                ...

        with pytest.raises(AttributeError, match="no such attribute"):
            class BadHost(PlainDeclMixin, SQLModelBase):
                name: str = "x"


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

    async def test_fails_closed_when_no_session_available(self, session: AsyncSession) -> None:
        # A guard that cannot find the session must raise, not silently pass.
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        assert fn is not None
        with pytest.raises(RuntimeError, match="cannot extract an AsyncSession"):
            await fn.locked_op_without_session()

    async def test_lock_tracking_cleared_on_commit_and_outer_rollback(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        fn = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        assert id(fn) in session.info[SESSION_FOR_UPDATE_KEY]
        await session.commit()
        assert SESSION_FOR_UPDATE_KEY not in session.info

        fn = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        assert id(fn) in session.info[SESSION_FOR_UPDATE_KEY]
        await session.rollback()
        assert SESSION_FOR_UPDATE_KEY not in session.info

    async def test_savepoint_rollback_restores_outer_locks(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        outer = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        gen = await PreloadGenerator.get(session, fetch_mode='one', with_for_update=True)
        outer_id, gen_id = id(outer), id(gen)
        session.info[SESSION_FOR_UPDATE_KEY].discard(gen_id)  # pretend only `outer` was locked before

        nested = await session.begin_nested()
        await PreloadGenerator.get(session, fetch_mode='one', with_for_update=True)
        assert gen_id in session.info[SESSION_FOR_UPDATE_KEY]
        await nested.rollback()
        # Locks taken inside the savepoint are released; the outer lock survives.
        assert session.info[SESSION_FOR_UPDATE_KEY] == {outer_id}
        await session.rollback()

    async def test_savepoint_release_keeps_inner_locks(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        outer = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        async with session.begin_nested():
            gen = await PreloadGenerator.get(session, fetch_mode='one', with_for_update=True)
        assert session.info[SESSION_FOR_UPDATE_KEY] == {id(outer), id(gen)}
        await session.rollback()

    async def test_locking_get_refreshes_identity_map(self, engine, session: AsyncSession) -> None:
        # A FOR UPDATE read must return the latest committed row even if a
        # stale copy of the object is already in the identity map.
        fid = await _make_chain(session)
        stale = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert stale is not None and stale.name == "fn1"
        async with AsyncSession(engine) as other:
            row = await PreloadFunction.get(other, PreloadFunction.id == fid)
            row.name = "changed"
            await row.save(other)
        plain = await PreloadFunction.get(session, PreloadFunction.id == fid)
        assert plain is stale and plain.name == "fn1"  # plain read keeps the identity-map copy
        locked = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        assert locked is stale and locked.name == "changed"

    async def test_authoritative_get_one_refreshes_identity_map(self, engine, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        stale = await PreloadFunction.get_one(session, fid)
        async with AsyncSession(engine) as other:
            row = await PreloadFunction.get_one(other, fid)
            row.name = "changed"
            await row.save(other)
        assert (await PreloadFunction.get_one(session, fid)).name == "fn1"
        fresh = await PreloadFunction.get_one(session, fid, authoritative=True)
        assert fresh is stale and fresh.name == "changed"

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


# ---------------------------------------------------------------------------
# validate_locked_instances / requires_locked_param
# ---------------------------------------------------------------------------

class _LockedBatchOps:
    """Host for the parametrized lock guard (classmethod-like and CM factory shapes)."""

    @staticmethod
    @requires_locked_param('instances')
    async def touch(session: AsyncSession, instances: list | PreloadFunction | None) -> str:
        return "ok"

    @staticmethod
    @requires_locked_param('instances')
    def factory(session: AsyncSession, instances: list) -> str:
        return "cm-built"

    @staticmethod
    @requires_locked_param('instances')
    async def no_session(instances: list) -> str:
        return "never"


@pytest.mark.asyncio
class TestLockedParamGuards:
    async def test_validate_locked_instances(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        locked = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        validate_locked_instances(session, [locked], context="probe")
        with pytest.raises(RuntimeError, match="at least one"):
            validate_locked_instances(session, [], context="probe")
        unlocked = await PreloadGenerator.get(session, fetch_mode='one')
        with pytest.raises(RuntimeError, match="PreloadGenerator instance is not FOR UPDATE locked"):
            validate_locked_instances(session, [locked, unlocked], context="probe")

    async def test_requires_locked_param_async(self, session: AsyncSession) -> None:
        fid = await _make_chain(session)
        plain = await PreloadFunction.get(session, PreloadFunction.id == fid)
        with pytest.raises(RuntimeError, match="not FOR UPDATE locked"):
            await _LockedBatchOps.touch(session, [plain])
        # None skips the check (e.g. a condition-based code path).
        assert await _LockedBatchOps.touch(session, None) == "ok"
        locked = await PreloadFunction.get(session, PreloadFunction.id == fid, with_for_update=True)
        assert await _LockedBatchOps.touch(session, locked) == "ok"   # single instance normalized
        assert await _LockedBatchOps.touch(session=session, instances=[locked]) == "ok"

    async def test_requires_locked_param_sync_factory(self, session: AsyncSession) -> None:
        with pytest.raises(RuntimeError, match="at least one"):
            _LockedBatchOps.factory(session, [])

    async def test_requires_locked_param_without_session_fails_closed(self) -> None:
        with pytest.raises(RuntimeError, match="no `session` parameter"):
            await _LockedBatchOps.no_session([object()])


# ---------------------------------------------------------------------------
# requires_repeatable_read / requires_read_committed
# ---------------------------------------------------------------------------

class _IsolationProbe:
    @requires_repeatable_read
    async def rr_method(self, session: AsyncSession) -> str:
        return "rr-ok"

    @classmethod
    @requires_repeatable_read
    async def rr_classmethod(cls, session: AsyncSession) -> str:
        return "rr-cls-ok"

    @requires_read_committed
    async def rc_method(self, session: AsyncSession) -> str:
        return "rc-ok"


@requires_repeatable_read
async def _rr_function(*, session: AsyncSession) -> str:
    return "rr-fn-ok"


class _FakeConnection:
    def __init__(self, level: str) -> None:
        self.level = level

    async def scalar(self, statement: object) -> str:
        return self.level


@pytest.mark.asyncio
class TestIsolationGuards:
    async def test_repeatable_read_guard_rejects_without_marker(self, session: AsyncSession) -> None:
        with pytest.raises(RuntimeError, match="REPEATABLE READ"):
            await _IsolationProbe().rr_method(session)

    async def test_repeatable_read_guard_accepts_all_call_shapes(self, session: AsyncSession) -> None:
        session.info[SESSION_REPEATABLE_READ_KEY] = True
        assert await _IsolationProbe().rr_method(session) == "rr-ok"
        assert await _IsolationProbe.rr_classmethod(session) == "rr-cls-ok"
        assert await _rr_function(session=session) == "rr-fn-ok"

    async def test_repeatable_read_guard_requires_session_argument(self) -> None:
        with pytest.raises(RuntimeError, match="cannot locate an AsyncSession"):
            await _IsolationProbe().rr_method(object())  # type: ignore[arg-type]

    async def test_read_committed_guard_live_check(
        self, session: AsyncSession, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def rc_connection(self: object, **kwargs: object) -> _FakeConnection:
            return _FakeConnection('read committed')

        monkeypatch.setattr(AsyncSession, "connection", rc_connection)
        assert await _IsolationProbe().rc_method(session) == "rc-ok"

        async def rr_connection(self: object, **kwargs: object) -> _FakeConnection:
            return _FakeConnection('repeatable read')

        monkeypatch.setattr(AsyncSession, "connection", rr_connection)
        with pytest.raises(RuntimeError, match="READ COMMITTED"):
            await _IsolationProbe().rc_method(session)


# ---------------------------------------------------------------------------
# ensure_relations_loaded_bulk / bulk_preload_unsupported_reason
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBulkPreload:
    async def test_unsupported_reasons(self) -> None:
        assert PreloadFunction.bulk_preload_unsupported_reason('generator') is None
        assert "nested" in PreloadFunction.bulk_preload_unsupported_reason(PreloadGenerator.config)
        assert "does not exist" in PreloadFunction.bulk_preload_unsupported_reason('nope')
        assert "not a relationship" in PreloadFunction.bulk_preload_unsupported_reason('name')

    async def test_bulk_loads_many_to_one_in_one_target_query(
        self, session: AsyncSession, engine,
    ) -> None:
        from sqlalchemy import event as sa_event

        cfg_id = (await PreloadConfig(price=1).save(session)).id
        gen_a_id = (await PreloadGenerator(name="a", config_id=cfg_id).save(session)).id
        gen_b_id = (await PreloadGenerator(name="b", config_id=cfg_id).save(session)).id
        await PreloadFunction(name="f1", generator_id=gen_a_id).save(session)
        await PreloadFunction(name="f2", generator_id=gen_b_id).save(session)
        await PreloadFunction(name="f3", generator_id=None).save(session)

        functions = await PreloadFunction.get(session, fetch_mode='all')
        assert all(not f._is_relation_loaded('generator') for f in functions)

        statements: list[str] = []

        def _record(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
            statements.append(statement)

        sa_event.listen(engine.sync_engine, "before_cursor_execute", _record)
        try:
            await PreloadFunction.ensure_relations_loaded_bulk(
                session, functions, {PreloadFunction: ('generator',)},
            )
        finally:
            sa_event.remove(engine.sync_engine, "before_cursor_execute", _record)

        # Fresh owners: no owner refresh, exactly one query for the target root.
        assert len(statements) == 1
        by_name = {f.name: f for f in functions}
        assert all(f._is_relation_loaded('generator') for f in functions)
        assert by_name["f1"].generator.id == gen_a_id
        assert by_name["f2"].generator.id == gen_b_id
        assert by_name["f3"].generator is None

    async def test_bulk_skips_missing_target_rows(self, session: AsyncSession) -> None:
        # A non-NULL FK whose target is missing stays unloaded (fails loudly on access).
        fn = await PreloadFunction(name="orphan", generator_id=uuid.uuid4()).save(session)
        await PreloadFunction.ensure_relations_loaded_bulk(
            session, [fn], {PreloadFunction: ('generator',)},
        )
        assert fn._is_relation_loaded('generator') is False
