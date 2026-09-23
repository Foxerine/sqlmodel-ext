"""
Metaclass / base-class behaviors of ``sqlmodel_ext.base`` not covered by
``test_partial.py``:

- attribute docstrings -> field descriptions (use_attribute_docstrings)
- description inheritance when a subclass re-declares a field without a
  docstring, and when ``partial=True`` regenerates annotations
- ``partial=True`` on plain DTOs: every field (required, defaulted,
  already-optional) becomes omissible (``Unset``)
- SQLModelBase ``extra='forbid'`` vs ExtraIgnoreModelBase ``extra='ignore'``
  + the unknown-field WARNING log (aliases, including ``AliasChoices``, must
  not be flagged)
- ``validate_list``, ``get_computed_field_names`` and
  ``submitted_fields_among`` helpers
- the globally reserved optimistic-lock column name
- construction-time ``JSON100K`` / ``JSONList100K`` checks on table models
- ``EXCLUDE_IF_NONE``
"""
from __future__ import annotations

import logging
from typing import Annotated

import pytest
from pydantic import AliasChoices, AliasPath, Field as PydanticField, ValidationError, computed_field
from sqlmodel import Field, SQLModel

from sqlmodel_ext import (
    EXCLUDE_IF_NONE,
    ExtraIgnoreModelBase,
    SQLModelBase,
    UUIDTableBaseMixin,
    Unset,
)
from sqlmodel_ext.constants import OPTIMISTIC_LOCK_VERSION_COLUMN
from sqlmodel_ext.field_types.dialects.postgresql import JSON100K, JSONList100K


# --------------------------------------------------------------------------
# Module-level models (unique names: Meta* prefix, all non-table DTOs)
# --------------------------------------------------------------------------

class MetaPerson(SQLModelBase):
    """Parent DTO with docstring-described fields."""
    name: str
    """The person's display name."""

    nickname: str | None = None
    """Optional nickname."""

    score: int = 5
    """Reputation score."""


class MetaPersonOverride(MetaPerson):
    """Subclass re-declares ``name`` without a docstring."""
    name: str | None = None


class MetaPersonPatch(MetaPerson, partial=True):
    """Programmatically derived PATCH DTO."""


class MetaEnvelope(ExtraIgnoreModelBase):
    """Extra-ignoring envelope with an aliased field."""
    kind: str
    payload: str | None = PydanticField(default=None, alias="data")


class MetaChoicesEnvelope(ExtraIgnoreModelBase):
    """Extra-ignoring envelope whose field accepts several input keys."""
    message: str | None = PydanticField(
        default=None, validation_alias=AliasChoices("Message", "Msg", AliasPath("wrapped", 0)),
    )


class MetaAdminOnlyFields(SQLModelBase):
    is_featured: bool = False


class MetaItemUpdate(SQLModelBase):
    title: str | None = None
    body: str | None = None


class MetaItemAdminUpdate(MetaAdminOnlyFields, MetaItemUpdate, partial=True):
    """Shared update body: admin-only fields plus regular fields."""


class MetaMarked(SQLModelBase):
    marker: Annotated[bool | None, EXCLUDE_IF_NONE] = None
    plain: bool | None = None


class MetaJsonRow(SQLModelBase, UUIDTableBaseMixin, table=True):
    """Table model: Pydantic validators are skipped, the base post-init check applies."""
    data: JSON100K = Field(default_factory=dict)
    rows: JSONList100K = Field(default_factory=list)
    label: str = "x"


# Only used for reflection/construction; never created in the SQLite test DB
# (JSONB has no SQLite rendering), so detach it from the shared metadata.
SQLModel.metadata.remove(MetaJsonRow.__table__)


def _nested(depth: int) -> dict[str, object]:
    root: dict[str, object] = {}
    current = root
    for _ in range(depth):
        child: dict[str, object] = {}
        current["a"] = child
        current = child
    return root


class MetaComputed(SQLModelBase):
    first: str
    last: str

    @computed_field
    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"


# --------------------------------------------------------------------------
# Attribute docstrings -> descriptions
# --------------------------------------------------------------------------

class TestAttributeDocstrings:
    def test_docstring_becomes_description(self) -> None:
        assert MetaPerson.model_fields["name"].description == "The person's display name."
        assert MetaPerson.model_fields["score"].description == "Reputation score."

    def test_override_without_docstring_inherits_description(self) -> None:
        # The subclass re-declares `name: str | None = None` with no docstring;
        # the metaclass must copy the parent's description via MRO.
        assert (
            MetaPersonOverride.model_fields["name"].description
            == "The person's display name."
        )

    def test_description_lands_in_json_schema(self) -> None:
        schema = MetaPersonOverride.model_json_schema()
        assert schema["properties"]["name"]["description"] == "The person's display name."

    def test_partial_keeps_descriptions(self) -> None:
        # partial regenerates annotations programmatically, so no source
        # docstring exists -- descriptions must still be inherited.
        assert (
            MetaPersonPatch.model_fields["name"].description
            == "The person's display name."
        )
        assert MetaPersonPatch.model_fields["score"].description == "Reputation score."


# --------------------------------------------------------------------------
# partial=True semantics
# --------------------------------------------------------------------------

class TestPartial:
    def test_every_field_defaults_to_unset(self) -> None:
        patch = MetaPersonPatch()
        assert patch.name is Unset
        assert patch.nickname is Unset
        # The original default 5 is NOT retained: omitted means "leave alone".
        assert patch.score is Unset
        assert patch.model_dump() == {}

    def test_parent_unchanged(self) -> None:
        # Deriving the patch class must not mutate the parent's requirements.
        with pytest.raises(ValidationError):
            MetaPerson()  # name still required on the parent
        assert MetaPerson(name="a").score == 5

    def test_dump_yields_partial_payload_without_flags(self) -> None:
        patch = MetaPersonPatch(name="x")
        assert patch.model_dump() == {"name": "x"}
        assert patch.model_dump(exclude_unset=True) == {"name": "x"}

    def test_null_only_where_base_allows_it(self) -> None:
        assert MetaPersonPatch(nickname=None).model_dump() == {"nickname": None}
        with pytest.raises(ValidationError):
            MetaPersonPatch(name=None)

    def test_values_still_validated(self) -> None:
        with pytest.raises(ValidationError):
            MetaPersonPatch(score="not-an-int")

    def test_extra_forbid_inherited(self) -> None:
        with pytest.raises(ValidationError):
            MetaPersonPatch(bogus=1)


# --------------------------------------------------------------------------
# extra='forbid' vs ExtraIgnoreModelBase
# --------------------------------------------------------------------------

class TestExtraHandling:
    def test_sqlmodelbase_forbids_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            MetaPerson.model_validate({"name": "a", "surprise": 1})

    def test_extra_ignore_drops_unknown_fields(self) -> None:
        env = MetaEnvelope.model_validate({"kind": "k", "totally_new_field": 1})
        assert env.kind == "k"
        assert not hasattr(env, "totally_new_field")

    def test_extra_ignore_logs_warning_with_model_and_sample(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="sqlmodel_ext.base"):
            MetaEnvelope.model_validate({"kind": "k", "zzz_unknown": 1, "aaa_unknown": 2})
        warnings = [r for r in caplog.records if "unknown fields" in r.getMessage()]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "MetaEnvelope" in msg
        assert "unknown_count=2" in msg
        assert "aaa_unknown" in msg and "zzz_unknown" in msg

    def test_alias_is_accepted_and_not_warned(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="sqlmodel_ext.base"):
            env = MetaEnvelope.model_validate({"kind": "k", "data": "via-alias"})
        assert env.payload == "via-alias"
        assert not [r for r in caplog.records if "unknown fields" in r.getMessage()]

    @pytest.mark.parametrize("key", ["Message", "Msg"])
    def test_alias_choices_are_not_warned(
        self, key: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="sqlmodel_ext.base"):
            env = MetaChoicesEnvelope.model_validate({key: "hi"})
        assert env.message == "hi"
        assert not [r for r in caplog.records if "unknown fields" in r.getMessage()]

    def test_alias_path_head_is_still_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # AliasPath entries are nested paths, not top-level keys.
        with caplog.at_level(logging.WARNING, logger="sqlmodel_ext.base"):
            MetaChoicesEnvelope.model_validate({"wrapped": ["x"]})
        assert [r for r in caplog.records if "unknown fields" in r.getMessage()]

    def test_field_name_also_accepted_via_validate_by_name(self) -> None:
        env = MetaEnvelope.model_validate({"kind": "k", "payload": "by-name"})
        assert env.payload == "by-name"

    def test_non_dict_input_passes_through_validator(self) -> None:
        # The before-validator must not choke on non-dict input.
        env = MetaEnvelope.model_validate(MetaEnvelope(kind="k"), from_attributes=True)
        assert env.kind == "k"


# --------------------------------------------------------------------------
# Helper classmethods
# --------------------------------------------------------------------------

class TestHelperMethods:
    def test_validate_list_converts_dicts(self) -> None:
        people = MetaPerson.validate_list([{"name": "a"}, {"name": "b", "score": 9}])
        assert [p.name for p in people] == ["a", "b"]
        assert people[1].score == 9
        assert all(isinstance(p, MetaPerson) for p in people)

    def test_validate_list_converts_objects_from_attributes(self) -> None:
        class Source:  # plain object, attribute-based conversion
            name = "obj"
            nickname = None
            score = 7

        [person] = MetaPerson.validate_list([Source()])
        assert person.name == "obj"
        assert person.score == 7

    def test_get_computed_field_names(self) -> None:
        assert MetaComputed.get_computed_field_names() == {"full"}
        assert MetaPerson.get_computed_field_names() == set()

    def test_submitted_fields_among(self) -> None:
        body = MetaItemAdminUpdate(title="t", is_featured=True)
        assert body.submitted_fields_among(MetaAdminOnlyFields) == {"is_featured"}
        assert body.submitted_fields_among(MetaItemUpdate) == {"title"}
        assert body.submitted_fields_among(MetaAdminOnlyFields, MetaItemUpdate) == {
            "title", "is_featured",
        }
        assert body.submitted_fields_among() == set()

    def test_submitted_fields_among_ignores_defaults(self) -> None:
        # Only explicitly submitted fields count, not defaulted ones.
        assert MetaItemUpdate(title="t").submitted_fields_among(MetaItemUpdate) == {"title"}
        assert MetaItemAdminUpdate().submitted_fields_among(MetaAdminOnlyFields) == set()


# --------------------------------------------------------------------------
# Reserved optimistic-lock column name
# --------------------------------------------------------------------------

class TestReservedOptimisticLockColumn:
    def test_constant_value(self) -> None:
        assert OPTIMISTIC_LOCK_VERSION_COLUMN == "oplock_version"

    def test_dto_declaring_reserved_name_fails(self) -> None:
        with pytest.raises(TypeError, match=OPTIMISTIC_LOCK_VERSION_COLUMN):
            class _Bad(SQLModelBase):
                oplock_version: int = 0

    def test_table_without_lock_declaring_reserved_name_fails(self) -> None:
        # Reserved globally, not only for classes that enable the lock.
        with pytest.raises(TypeError, match=OPTIMISTIC_LOCK_VERSION_COLUMN):
            class _BadTable(SQLModelBase, UUIDTableBaseMixin, table=True):
                oplock_version: int = 0

    def test_unrelated_version_name_allowed(self) -> None:
        class _Fine(SQLModelBase):
            version: int = 0

        assert _Fine().version == 0


# --------------------------------------------------------------------------
# JSON100K / JSONList100K construction-time checks on table models
# --------------------------------------------------------------------------

class TestJsonFieldPostInitCheck:
    def test_fields_discovered_from_annotations(self) -> None:
        assert set(MetaJsonRow.__orjson_checked_fields__) == {"data", "rows"}
        assert MetaPerson.__orjson_checked_fields__ == ()

    def test_valid_values_accepted(self) -> None:
        row = MetaJsonRow(data={"a": 1}, rows=[{"b": 2}])
        assert row.data == {"a": 1}
        assert MetaJsonRow().data == {}

    def test_too_deep_value_rejected_at_construction(self) -> None:
        # Table models skip Pydantic validators; the base model_post_init catches it.
        # 400 levels exceed both orjson's (255) and pydantic_core's limit on
        # every platform (the latter is platform-dependent, ~98 on Windows).
        with pytest.raises(ValueError):
            MetaJsonRow(data=_nested(400))

    def test_too_long_value_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            MetaJsonRow(rows=[{"x": "y" * 100_001}])

    def test_unset_is_not_checked(self) -> None:
        class _JsonDto(SQLModelBase):
            data: JSON100K

        class _JsonPatch(_JsonDto, partial=True):
            pass

        assert _JsonPatch.__orjson_checked_fields__ == ("data",)
        assert _JsonPatch().data is Unset


# --------------------------------------------------------------------------
# EXCLUDE_IF_NONE
# --------------------------------------------------------------------------

class TestExcludeIfNone:
    def test_key_absent_without_exclude_none(self) -> None:
        assert "marker" not in MetaMarked().model_dump()
        assert "marker" not in MetaMarked().model_dump_json()

    def test_key_present_when_value_set(self) -> None:
        assert MetaMarked(marker=True).model_dump()["marker"] is True

    def test_unmarked_field_still_emits_null(self) -> None:
        # Control group: without the marker the key is emitted as null.
        assert MetaMarked().model_dump()["plain"] is None

    def test_round_trip_reads_back_when_key_absent(self) -> None:
        assert MetaMarked.model_validate_json(MetaMarked().model_dump_json()).marker is None


def test_import_without_orjson() -> None:
    """orjson is optional: the core package and plain models must work without it."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['orjson'] = None\n"
        "import sqlmodel_ext\n"
        "from sqlmodel_ext import SQLModelBase\n"
        "class A(SQLModelBase):\n"
        "    x: int = 1\n"
        "assert A().x == 1\n"
        "assert A.__orjson_checked_fields__ == ()\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'
