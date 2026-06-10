"""
Public-API behavior tests for the relation load checker (static analysis).

Only exercises the public surface: ``RelationLoadChecker`` (knowledge base +
``check_function``), ``RelationLoadWarning``, ``run_model_checks`` gating, and
``RelationLoadCheckMiddleware`` construction / lifespan wiring. The 168KB AST
internals are deliberately treated as a black box.

The module-level coroutines below are *analyzed*, never executed.

NOTE: no ``from __future__ import annotations`` here -- PEP 563 string
annotations break SQLAlchemy's resolution of ``list["RlcPet"]`` Relationship
annotations (the registry would see the literal string ``"list['RlcPet']"``).
"""
import sys
import uuid

import pytest
from sqlmodel import Field, Relationship
from sqlmodel.ext.asyncio.session import AsyncSession

import sqlmodel_ext.relation_load_checker as rlc_module
from sqlmodel_ext import (
    RelationLoadChecker,
    RelationLoadCheckMiddleware,
    RelationLoadWarning,
    SQLModelBase,
    UUIDTableBaseMixin,
    mark_app_check_completed,
    rel,
    run_model_checks,
)


class RlcOwner(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "owner"
    pets: list["RlcPet"] = Relationship(back_populates="owner")


class RlcPet(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "pet"
    owner_id: uuid.UUID | None = Field(default=None, foreign_key="rlcowner.id")
    owner: RlcOwner | None = Relationship(back_populates="pets")


# --------------------- coroutines used as analysis targets ---------------------

async def _bad_unloaded_access(session: AsyncSession) -> None:
    pet = await RlcPet.get(session, fetch_mode="first")
    print(pet.owner.name)  # relationship accessed without load=


async def _good_loaded_access(session: AsyncSession) -> None:
    pet = await RlcPet.get(session, fetch_mode="first", load=rel(RlcPet.owner))
    print(pet.owner.name)


async def _bad_access_after_save(session: AsyncSession) -> None:
    pet = await RlcPet.get(session, fetch_mode="first", load=rel(RlcPet.owner))
    pet = await pet.save(session)  # refreshed without load= -> relation dropped
    print(pet.owner.name)


async def _good_access_after_save_with_load(session: AsyncSession) -> None:
    pet = await RlcPet.get(session, fetch_mode="first", load=rel(RlcPet.owner))
    pet = await pet.save(session, load=rel(RlcPet.owner))
    print(pet.owner.name)


# ------------------------------------ tests ------------------------------------

@pytest.fixture(scope="module")
def checker() -> RelationLoadChecker:
    return RelationLoadChecker(SQLModelBase)


class TestKnowledgeBase:
    def test_models_and_relationships_discovered(self, checker: RelationLoadChecker) -> None:
        assert "RlcPet" in checker.model_classes
        assert "RlcOwner" in checker.model_classes
        assert checker.model_relationships["RlcPet"] == {"owner"}
        assert "pets" in checker.model_relationships["RlcOwner"]
        assert checker.model_rel_targets["RlcPet"]["owner"] == "RlcOwner"
        assert "id" in checker.model_columns["RlcPet"]

    def test_save_update_discovered_as_commit_methods(self, checker: RelationLoadChecker) -> None:
        assert "save" in checker.commit_methods
        assert "update" in checker.commit_methods


class TestCheckFunction:
    def test_unloaded_relation_access_is_flagged(self, checker: RelationLoadChecker) -> None:
        warnings = checker.check_function(_bad_unloaded_access)
        assert warnings, "expected at least one warning for unloaded relation access"
        assert all(isinstance(w, RelationLoadWarning) for w in warnings)
        assert any(w.code == "RLC003" for w in warnings)
        assert any("owner" in w.message for w in warnings)

    def test_loaded_relation_access_is_clean(self, checker: RelationLoadChecker) -> None:
        assert checker.check_function(_good_loaded_access) == []

    def test_relation_access_after_save_without_load_is_flagged(
        self, checker: RelationLoadChecker
    ) -> None:
        warnings = checker.check_function(_bad_access_after_save)
        assert warnings
        assert any("owner" in w.message for w in warnings)

    def test_relation_access_after_save_with_load_is_clean(
        self, checker: RelationLoadChecker
    ) -> None:
        assert checker.check_function(_good_access_after_save_with_load) == []

    def test_check_model_methods_returns_warning_objects(
        self, checker: RelationLoadChecker
    ) -> None:
        # Black-box: must complete without crashing and yield typed warnings.
        warnings = checker.check_model_methods()
        assert isinstance(warnings, list)
        assert all(isinstance(w, RelationLoadWarning) for w in warnings)


class TestRelationLoadWarning:
    def test_str_format(self) -> None:
        warning = RelationLoadWarning(code="RLC003", file="models.py", line=12, message="boom")
        assert str(warning) == "[RLC003] models.py:12 - boom"


class TestRunModelChecks:
    def test_noop_when_check_on_startup_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rlc_module, "check_on_startup", False)
        monkeypatch.setattr(rlc_module, "_model_check_completed", False)
        monkeypatch.setattr(rlc_module, "_base_class", None)

        run_model_checks(SQLModelBase)

        assert rlc_module._model_check_completed is False
        assert rlc_module._base_class is None

    def test_runs_and_completes_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rlc_module, "check_on_startup", True)
        monkeypatch.setattr(rlc_module, "_model_check_completed", False)
        monkeypatch.setattr(rlc_module, "_base_class", None)

        # Under pytest the check is documented to be non-blocking even when
        # warnings are found, so this must not raise.
        assert "pytest" in sys.modules
        run_model_checks(SQLModelBase)

        assert rlc_module._model_check_completed is True
        assert rlc_module._base_class is SQLModelBase

    def test_second_call_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rlc_module, "check_on_startup", True)
        monkeypatch.setattr(rlc_module, "_model_check_completed", True)
        monkeypatch.setattr(rlc_module, "_base_class", None)

        run_model_checks(SQLModelBase)
        # Short-circuited: _base_class was not overwritten.
        assert rlc_module._base_class is None

    def test_mark_app_check_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rlc_module, "_app_check_completed", False)
        mark_app_check_completed()
        assert rlc_module._app_check_completed is True


class TestMiddleware:
    def test_constructor_defaults_and_overrides(self) -> None:
        class DummyApp:
            pass

        mw = RelationLoadCheckMiddleware(DummyApp())
        assert mw.project_root == rlc_module._PROJECT_ROOT
        assert mw.skip_paths is None
        assert mw.skip_third_party_attrs is False
        assert mw._checked is False

        mw2 = RelationLoadCheckMiddleware(
            DummyApp(),
            project_root="C:/custom",
            skip_paths=["/base/"],
            skip_third_party_attrs=True,
        )
        assert mw2.project_root == "C:/custom"
        assert mw2.skip_paths == ["/base/"]
        assert mw2.skip_third_party_attrs is True

    def test_fastapi_add_middleware_assembly(self) -> None:
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(RelationLoadCheckMiddleware, skip_paths=["/internal/"])
        assert any(
            m.cls is RelationLoadCheckMiddleware for m in app.user_middleware
        )

    @pytest.mark.asyncio
    async def test_lifespan_startup_triggers_check_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With check_on_startup disabled, the lifespan hook still marks the
        app check completed and forwards messages unchanged."""
        monkeypatch.setattr(rlc_module, "check_on_startup", False)
        monkeypatch.setattr(rlc_module, "_app_check_completed", False)

        inner_scopes: list[str] = []
        sent: list[dict] = []

        class InnerApp:
            async def __call__(self, scope, receive, send) -> None:
                inner_scopes.append(scope["type"])
                if scope["type"] == "lifespan":
                    await send({"type": "lifespan.startup.complete"})

        async def fake_receive():
            return {"type": "lifespan.startup"}

        async def fake_send(message):
            sent.append(message)

        mw = RelationLoadCheckMiddleware(InnerApp())
        await mw({"type": "lifespan"}, fake_receive, fake_send)

        assert inner_scopes == ["lifespan"]
        assert mw._checked is True
        assert rlc_module._app_check_completed is True
        assert sent == [{"type": "lifespan.startup.complete"}]

        # Non-lifespan scopes pass straight through without re-checking.
        await mw({"type": "http"}, fake_receive, fake_send)
        assert inner_scopes == ["lifespan", "http"]
