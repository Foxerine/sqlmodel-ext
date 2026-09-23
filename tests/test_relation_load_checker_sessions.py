"""
Relation load checker: session identity, commit semantics, detach, RLC014,
pytest fixture resolution, response-model drill-down and RLC012 binding.

The module-level coroutines below are *analyzed*, never executed.

NOTE: no ``from __future__ import annotations`` -- SQLAlchemy must resolve the
``Relationship`` annotations of the models defined here.
"""
import sys
import types
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, Self

import pytest
from fastapi import Depends, FastAPI
from pydantic import Field as PydanticField, PrivateAttr, computed_field
from sqlmodel import Field, Relationship
from sqlmodel.ext.asyncio.session import AsyncSession as UpstreamAsyncSession

import sqlmodel_ext.relation_load_checker as rlc_module
from sqlmodel_ext import (
    RelationLoadChecker,
    SQLModelBase,
    UUIDTableBaseMixin,
)
from sqlmodel_ext._type_unwrap import (
    get_pydantic_generic_args,
    unwrap_generic_to_dto_class,
    unwrap_to_class,
)
from sqlmodel_ext.relation_load_checker import _is_async_session_hint, _session_param_names
from sqlmodel_ext.session import AsyncSession
from tests._models import FunctionA, FunctionB, Tool


# ================================== models ==================================

class RlcsOwner(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "owner"
    pets: list["RlcsPet"] = Relationship(back_populates="owner")


class RlcsPet(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "pet"
    owner_id: uuid.UUID | None = Field(default=None, foreign_key="rlcsowner.id")
    owner: RlcsOwner | None = Relationship(back_populates="pets")

    @computed_field
    @property
    def owner_label(self) -> str:
        # The response field name has nothing to do with the relationship it reads.
        return self.owner.name if self.owner is not None else ""


class RlcsEntry(SQLModelBase, UUIDTableBaseMixin, table=True):
    """Methods annotated with the library's AsyncSession *subclass*."""
    name: str = "entry"

    async def cancel(self, session: AsyncSession) -> bool:
        await self.delete(session)
        return True


class RlcsTicket(SQLModelBase, UUIDTableBaseMixin, table=True):
    """Same method name as RlcsEntry.cancel, but returning a model."""
    name: str = "ticket"

    async def cancel(self, session: AsyncSession) -> Self:
        return await self.save(session)


class RlcsSingleton(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "singleton"

    @classmethod
    async def get_or_create(cls, session: AsyncSession) -> "RlcsSingleton":
        existing = await cls.get(session, fetch_mode="first")
        if existing is None:
            existing = await cls().save(session)
        return existing


class RlcsCacheHolder(SQLModelBase, UUIDTableBaseMixin, table=True):
    """ORM objects kept in an instance-attribute cache filled elsewhere."""
    name: str = "holder"
    _owner_cache: dict[str, RlcsOwner] = PrivateAttr(default_factory=dict)

    async def read_cached(self, session: AsyncSession) -> None:
        owner = self._owner_cache.get("key")
        print(owner.name)  # freshness unknown -> pessimistic RLC007


class RlcsQueue(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "queued"

    @classmethod
    async def enqueue(cls, session: AsyncSession, *, commit: bool = False) -> None:
        session.add(cls())
        if commit:
            await session.commit()

    @classmethod
    async def enqueue_via(cls, session: AsyncSession) -> None:
        await cls.enqueue(session)


# ============================ analysis targets ============================

async def _cancel_then_read(session: AsyncSession, entry: RlcsEntry, other: RlcsOwner) -> None:
    await entry.cancel(session)
    print(other.name)  # expired by the commit inside cancel -> RLC007


async def _scalar_cancel_result(session: AsyncSession, entry: RlcsEntry) -> None:
    result = await entry.cancel(session)
    await session.commit()
    print(result.bit_length())  # a bool, not an expired model


async def _local_session_commit(session: AsyncSession, owner: RlcsOwner, factory: Any) -> None:
    async with factory() as local:
        await local.commit()
    print(owner.name)  # another identity map: not expired


async def _signature_session_commit(session: AsyncSession, owner: RlcsOwner) -> None:
    await session.commit()
    print(owner.name)  # RLC007


async def _two_sessions(a: AsyncSession, b: AsyncSession) -> None:
    owner = await RlcsOwner.get(a, fetch_mode="first")
    await b.commit()
    print(owner.name)  # b's commit does not expire a's object
    await a.commit()
    print(owner.name)  # RLC007


async def _reset_then_commit(session: AsyncSession, owner: RlcsOwner) -> None:
    await session.reset()
    alias = owner
    await session.commit()
    print(owner.name)  # detached: immune to the later commit
    print(alias.name)  # aliases inherit the detached state


async def _commit_then_reset(session: AsyncSession, owner: RlcsOwner) -> None:
    await session.commit()
    await session.reset()
    print(owner.name)  # detached AND expired -> still reported


async def _get_or_create_then_read(session: AsyncSession, owner: RlcsOwner) -> None:
    await RlcsSingleton.get_or_create(session)
    print(owner.name)


async def _enqueue_default(session: AsyncSession, owner: RlcsOwner) -> None:
    await RlcsQueue.enqueue(session)
    print(owner.name)


async def _enqueue_explicit_true(session: AsyncSession, owner: RlcsOwner) -> None:
    await RlcsQueue.enqueue(session, commit=True)
    print(owner.name)


async def _none_narrowing(session: AsyncSession) -> None:
    existing = await RlcsOwner.get(session, fetch_mode="first")
    if existing is None:
        await session.commit()  # only runs when existing is None
    if existing is not None:
        print(existing.name)


async def _conditional_refetch(session: AsyncSession, flag: bool) -> None:
    owner = await RlcsOwner.get(session, fetch_mode="first")
    await session.commit()
    if flag:
        owner = await RlcsOwner.get(session, fetch_mode="first")
    print(owner.name)


async def _conditional_refetch_after_yield(session: AsyncSession, flag: bool) -> AsyncIterator[int]:
    owner = await RlcsOwner.get(session, fetch_mode="first")
    yield 1
    if flag:
        owner = await RlcsOwner.get(session, fetch_mode="first")
    print(owner.name)  # yield-expiry is not healed by a conditional re-fetch -> RLC013


async def _noqa_suppressed(session: AsyncSession, owner: RlcsOwner) -> None:
    await session.commit()
    print(owner.name)  # noqa: RLC007


# ---- RLC014: FastAPI endpoint with a committing sibling dependency ----

async def _get_db() -> AsyncIterator[AsyncSession]:
    raise NotImplementedError  # never executed
    yield  # pragma: no cover


async def _dep_committing_owner(
        session: Annotated[AsyncSession, Depends(_get_db)],
) -> RlcsOwner:
    owner = RlcsOwner()
    owner = await owner.save(session)
    return owner


async def _dep_plain_pet(
        session: Annotated[AsyncSession, Depends(_get_db)],
) -> RlcsPet:
    return await RlcsPet.get(session, fetch_mode="first")


async def _endpoint_with_committing_sibling(
        owner: Annotated[RlcsOwner, Depends(_dep_committing_owner)],
        pet: Annotated[RlcsPet, Depends(_dep_plain_pet)],
) -> None:
    print(owner.name)  # the committing dependency's own result is exempt
    print(pet.name)    # expired by the sibling's commit -> RLC014


# ---- pytest fixture resolution (functions looked up by name in this module) ----

def rlcs_shared_owner(session: AsyncSession) -> RlcsOwner:
    return RlcsOwner()


def rlcs_wrapped_owner(rlcs_shared_owner: RlcsOwner) -> RlcsOwner:
    return rlcs_shared_owner


def rlcs_independent_session() -> AsyncSession:
    raise NotImplementedError


def rlcs_independent_owner(rlcs_independent_session: AsyncSession) -> RlcsOwner:
    return RlcsOwner()


def rlcs_transient_owner() -> RlcsOwner:
    return RlcsOwner()


async def _test_like(
        session: AsyncSession,
        rlcs_shared_owner: RlcsOwner,
        rlcs_wrapped_owner: RlcsOwner,
        rlcs_independent_owner: RlcsOwner,
        rlcs_transient_owner: RlcsOwner,
) -> None:
    await session.commit()
    print(rlcs_shared_owner.name)       # shares the session -> RLC007
    print(rlcs_wrapped_owner.name)      # shares it through another fixture -> RLC007
    print(rlcs_independent_owner.name)  # other session -> detached, clean
    print(rlcs_transient_owner.name)    # transient -> detached, clean


# ================================== tests ==================================

@pytest.fixture
def checker() -> RelationLoadChecker:
    return RelationLoadChecker(SQLModelBase)


def _codes(warnings: list[rlc_module.RelationLoadWarning]) -> list[str]:
    return [w.code for w in warnings]


class TestSessionHintDetection:
    def test_subclass_and_wrappers_are_sessions(self) -> None:
        assert _is_async_session_hint(UpstreamAsyncSession)
        assert _is_async_session_hint(AsyncSession)  # library subclass
        assert _is_async_session_hint(AsyncSession | None)
        assert _is_async_session_hint(Annotated[AsyncSession, "meta"])
        assert _is_async_session_hint('AsyncSession | None')

    def test_unrelated_types_are_not_sessions(self) -> None:
        assert not _is_async_session_hint(int)
        assert not _is_async_session_hint('HttpAsyncSession')
        assert not _is_async_session_hint('requests.AsyncSession')

    def test_session_param_names_excludes_return(self) -> None:
        async def f(a: AsyncSession, b: int, c: UpstreamAsyncSession) -> AsyncSession | None:
            return None

        assert _session_param_names(f) == ['a', 'c']

    def test_subclass_annotated_method_is_discovered_as_commit(self, checker: RelationLoadChecker) -> None:
        assert "cancel" in checker.commit_methods
        assert "cancel" in checker._model_commit_methods["RlcsEntry"]

    def test_commit_via_subclass_annotated_method_expires_objects(self, checker: RelationLoadChecker) -> None:
        assert _codes(checker.check_function(_cancel_then_read)) == ["RLC007"]


class TestPerClassModelReturning:
    def test_same_named_scalar_method_is_not_tracked(self, checker: RelationLoadChecker) -> None:
        assert "cancel" in checker._model_returning_by_class["RlcsTicket"]
        assert "cancel" not in checker._model_returning_by_class["RlcsEntry"]
        assert checker.check_function(_scalar_cancel_result) == []


class TestSessionIdentity:
    def test_local_session_commit_does_not_expire(self, checker: RelationLoadChecker) -> None:
        assert checker.check_function(_local_session_commit) == []

    def test_signature_session_commit_expires(self, checker: RelationLoadChecker) -> None:
        assert _codes(checker.check_function(_signature_session_commit)) == ["RLC007"]

    def test_commit_only_expires_its_own_session(self, checker: RelationLoadChecker) -> None:
        warnings = checker.check_function(_two_sessions)
        assert _codes(warnings) == ["RLC007"]


class TestDetach:
    def test_reset_makes_objects_immune_to_later_commit(self, checker: RelationLoadChecker) -> None:
        assert checker.check_function(_reset_then_commit) == []

    def test_commit_then_reset_is_still_reported(self, checker: RelationLoadChecker) -> None:
        assert _codes(checker.check_function(_commit_then_reset)) == ["RLC007"]


class TestCommitSemanticsConfiguration:
    def test_defaults_contain_only_library_methods(self) -> None:
        assert rlc_module.conditional_commit_methods == frozenset()
        assert rlc_module.explicit_commit_methods == frozenset()
        assert rlc_module.dependency_commit_methods == frozenset({"add", "save", "update", "delete"})

    def test_conditional_commit_method_is_a_commit_by_default(self, checker: RelationLoadChecker) -> None:
        assert "get_or_create" in checker.commit_methods
        assert _codes(checker.check_function(_get_or_create_then_read)) == ["RLC007"]

    def test_conditional_commit_methods_are_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rlc_module, "conditional_commit_methods", frozenset({"get_or_create"}))
        configured = RelationLoadChecker(SQLModelBase)
        assert "get_or_create" not in configured.commit_methods
        assert configured.check_function(_get_or_create_then_read) == []

    def test_explicit_commit_method_default_is_treated_as_commit(self, checker: RelationLoadChecker) -> None:
        assert "enqueue_via" in checker.commit_methods
        assert _codes(checker.check_function(_enqueue_default)) == ["RLC007"]

    def test_explicit_commit_methods_only_commit_with_literal_true(
            self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(rlc_module, "explicit_commit_methods", frozenset({"enqueue"}))
        configured = RelationLoadChecker(SQLModelBase)
        # The method stays a commit method; the closure no longer propagates the default call.
        assert "enqueue" in configured.commit_methods
        assert "enqueue_via" not in configured.commit_methods
        assert configured.check_function(_enqueue_default) == []
        assert _codes(configured.check_function(_enqueue_explicit_true)) == ["RLC007"]


class TestSelfAttributeCaches:
    def test_cache_annotation_is_indexed(self, checker: RelationLoadChecker) -> None:
        assert checker._model_self_attr_models["RlcsCacheHolder"] == {"_owner_cache": "RlcsOwner"}

    def test_object_taken_from_cache_is_tracked_pessimistically(
            self, checker: RelationLoadChecker,
    ) -> None:
        warnings = [
            w for w in checker.check_model_methods()
            if "owner.name" in w.message and w.code == "RLC007"
        ]
        assert len(warnings) == 1


class TestBranchState:
    def test_is_none_narrowing(self, checker: RelationLoadChecker) -> None:
        assert checker.check_function(_none_narrowing) == []

    def test_conditional_refetch_after_commit_is_trusted(self, checker: RelationLoadChecker) -> None:
        assert checker.check_function(_conditional_refetch) == []

    def test_conditional_refetch_does_not_heal_yield_expiry(self, checker: RelationLoadChecker) -> None:
        assert _codes(checker.check_function(_conditional_refetch_after_yield)) == ["RLC013"]


class TestNoqa:
    def test_check_function_filters_noqa(self, checker: RelationLoadChecker) -> None:
        assert checker.check_function(_noqa_suppressed) == []
        # The unsuppressed twin is reported, so the filter (not the analysis) is what silenced it.
        assert _codes(checker.check_function(_signature_session_commit)) == ["RLC007"]

    def test_every_public_check_entry_filters_noqa(self) -> None:
        import inspect

        public_entries = [
            name for name, _ in inspect.getmembers(RelationLoadChecker, inspect.isfunction)
            if name.startswith("check_")
        ]
        assert set(public_entries) >= {
            "check_app", "check_model_methods", "check_project_coroutines", "check_function",
        }
        for name in public_entries:
            source = inspect.getsource(getattr(RelationLoadChecker, name))
            assert "_filter_noqa_suppressions" in source, name


class TestRlc014:
    def _app(self) -> FastAPI:
        app = FastAPI()
        app.get("/rlcs")(_endpoint_with_committing_sibling)
        return app

    def test_sibling_dependency_commit_is_reported(self, checker: RelationLoadChecker) -> None:
        warnings = checker.check_app(self._app())
        assert _codes(warnings) == ["RLC014"]
        assert "pet.name" in warnings[0].message

    def test_committing_dependency_detection(self, checker: RelationLoadChecker) -> None:
        assert checker._get_committing_dep_params(_endpoint_with_committing_sibling) == {"owner"}

    def test_dependency_commit_methods_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rlc_module, "dependency_commit_methods", frozenset({"delete"}))
        configured = RelationLoadChecker(SQLModelBase)
        assert configured._get_committing_dep_params(_endpoint_with_committing_sibling) == set()

    def test_no_fastapi_means_no_dependency_analysis(
            self, checker: RelationLoadChecker, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(rlc_module, "_HAS_FASTAPI", False)
        assert checker._get_committing_dep_params(_endpoint_with_committing_sibling) == set()
        assert checker._analyze_dependencies(_endpoint_with_committing_sibling) == {}


class TestFixtureResolution:
    def test_fixture_session_sharing(self) -> None:
        shares = RelationLoadChecker._fixture_param_shares_session
        assert shares(_test_like, "rlcs_shared_owner") is True
        assert shares(_test_like, "rlcs_wrapped_owner") is True
        assert shares(_test_like, "rlcs_independent_owner") is False
        assert shares(_test_like, "rlcs_transient_owner") is False
        assert shares(_test_like, "does_not_exist") is False

    def test_resolved_fixture_params_are_detached(self, checker: RelationLoadChecker) -> None:
        warnings = checker._check_coroutine(_test_like, label="<test>", resolve_fixture_params=True)
        flagged = sorted(w.message.split("'")[1] for w in warnings)
        assert _codes(warnings) == ["RLC007", "RLC007"]
        assert flagged == ["rlcs_shared_owner.name", "rlcs_wrapped_owner.name"]

    def test_production_mode_treats_all_params_as_shared(self, checker: RelationLoadChecker) -> None:
        assert len(checker.check_function(_test_like)) == 4

    def test_same_name_override_is_not_a_cycle(self) -> None:
        parent = types.ModuleType("rlcs_pkg.conftest")
        child = types.ModuleType("rlcs_pkg.test_mod")

        def rlcs_user(session: AsyncSession) -> RlcsOwner:  # parent definition
            return RlcsOwner()

        def rlcs_user_override(rlcs_user: RlcsOwner) -> RlcsOwner:  # child overrides by name
            return rlcs_user

        async def test_fn(session: AsyncSession, rlcs_user: RlcsOwner) -> None:
            pass

        parent.rlcs_user = rlcs_user  # type: ignore[attr-defined]
        child.rlcs_user = rlcs_user_override  # type: ignore[attr-defined]
        test_fn.__module__ = child.__name__
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, parent.__name__, parent)
            mp.setitem(sys.modules, child.__name__, child)
            assert RelationLoadChecker._fixture_param_shares_session(test_fn, "rlcs_user") is True


class TestResponseModelRelationships:
    def test_property_backed_field_requires_its_relationship(self, checker: RelationLoadChecker) -> None:
        assert checker._get_response_model_relationships(RlcsPet) == {"owner": "RlcsPet"}

    def test_container_and_union_members_are_drilled(self, checker: RelationLoadChecker) -> None:
        class Page(SQLModelBase):
            count: int
            items: list[RlcsPet]

        assert checker._get_response_model_relationships(Page) == {"owner": "RlcsPet"}
        assert checker._get_response_model_relationships(Page | RlcsOwner) == {"owner": "RlcsPet"}

    def test_model_satisfies_is_one_directional(self, checker: RelationLoadChecker) -> None:
        assert checker._model_satisfies("FunctionA", "Tool")
        assert checker._model_satisfies("Tool", "Tool")
        assert not checker._model_satisfies("Tool", "FunctionA")


class _ADto(SQLModelBase):
    kind: Literal["functiona"]
    max_files: int


class _BDto(SQLModelBase):
    kind: Literal["functionb"]
    timeout: int


class _AWrongDto(SQLModelBase):
    kind: Literal["functiona"]
    timeout: int  # FunctionB's field on FunctionA's branch


class _UnboundDto(SQLModelBase):
    kind: Literal["nobody"]
    max_files: int


async def _list_tools() -> list[Tool]:
    raise NotImplementedError


class TestRlc012DiscriminatedUnion:
    def _check(self, checker: RelationLoadChecker, response_model: Any) -> list[str]:
        warnings: list[rlc_module.RelationLoadWarning] = []
        checker._check_rlc012(warnings, response_model, _list_tools, "/tools")
        return [w.message for w in warnings]

    def test_correct_binding_passes(self, checker: RelationLoadChecker) -> None:
        union = Annotated[_ADto | _BDto, PydanticField(discriminator="kind")]
        assert self._check(checker, union) == []

    def test_field_from_another_subclass_is_reported(self, checker: RelationLoadChecker) -> None:
        union = Annotated[_AWrongDto | _BDto, PydanticField(discriminator="kind")]
        messages = self._check(checker, union)
        assert len(messages) == 1
        assert "_AWrongDto" in messages[0] and "timeout" in messages[0]

    def test_unbindable_member_fails_loud(self, checker: RelationLoadChecker) -> None:
        union = Annotated[_UnboundDto | _BDto, PydanticField(discriminator="kind")]
        messages = self._check(checker, union)
        assert len(messages) == 1
        assert "cannot be bound" in messages[0]

    def test_single_dto_still_uses_all_subclasses_rule(self, checker: RelationLoadChecker) -> None:
        messages = self._check(checker, list[_ADto])
        assert any("max_files" in m for m in messages)


class TestTypeUnwrap:
    def test_unwrap_to_class(self) -> None:
        assert unwrap_to_class(Annotated[RlcsOwner | None, "x"]) is RlcsOwner
        assert unwrap_to_class(int) is int
        assert unwrap_to_class(RlcsOwner | RlcsPet) is None

    def test_pydantic_generic_args_and_dto_drilling(self) -> None:
        from sqlmodel_ext import ListResponse

        concrete = ListResponse[_ADto]
        assert _ADto in get_pydantic_generic_args(concrete)
        assert unwrap_generic_to_dto_class(concrete) is _ADto
        assert unwrap_generic_to_dto_class(_ADto | None) is _ADto
        assert get_pydantic_generic_args(int) == ()


def test_models_are_known(checker: RelationLoadChecker) -> None:
    # Guard: the classes above must be mapped, otherwise every test here would be vacuous.
    for name in ("RlcsOwner", "RlcsPet", "RlcsEntry", "RlcsTicket", "RlcsSingleton", "RlcsQueue"):
        assert name in checker.model_classes
    assert FunctionA.__name__ in checker.model_classes and FunctionB.__name__ in checker.model_classes
