"""
Metaclass / base-class behaviors of ``sqlmodel_ext.base`` not covered by
``test_all_fields_optional_constraints.py``:

- attribute docstrings -> field descriptions (use_attribute_docstrings)
- description inheritance when a subclass re-declares a field without a
  docstring, and when ``all_fields_optional=True`` regenerates annotations
- ``all_fields_optional=True`` on plain DTOs: every field (required,
  defaulted, already-optional) becomes ``T | None = None``
- SQLModelBase ``extra='forbid'`` vs ExtraIgnoreModelBase ``extra='ignore'``
  + the unknown-field WARNING log (aliases must not be flagged)
- ``validate_list`` and ``get_computed_field_names`` helpers
"""
from __future__ import annotations

import logging

import pytest
from pydantic import Field as PydanticField, ValidationError, computed_field

from sqlmodel_ext import ExtraIgnoreModelBase, SQLModelBase


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


class MetaPersonPatch(MetaPerson, all_fields_optional=True):
    """Programmatically derived PATCH DTO."""


class MetaEnvelope(ExtraIgnoreModelBase):
    """Extra-ignoring envelope with an aliased field."""
    kind: str
    payload: str | None = PydanticField(default=None, alias="data")


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

    def test_all_fields_optional_keeps_descriptions(self) -> None:
        # all_fields_optional regenerates annotations programmatically, so no
        # source docstring exists -- descriptions must still be inherited.
        assert (
            MetaPersonPatch.model_fields["name"].description
            == "The person's display name."
        )
        assert MetaPersonPatch.model_fields["score"].description == "Reputation score."


# --------------------------------------------------------------------------
# all_fields_optional=True semantics
# --------------------------------------------------------------------------

class TestAllFieldsOptional:
    def test_every_field_defaults_to_none(self) -> None:
        patch = MetaPersonPatch()
        assert patch.name is None
        assert patch.nickname is None
        # Unified behavior: the original default 5 is NOT retained.
        assert patch.score is None

    def test_parent_unchanged(self) -> None:
        # Deriving the patch class must not mutate the parent's requirements.
        with pytest.raises(ValidationError):
            MetaPerson()  # name still required on the parent
        assert MetaPerson(name="a").score == 5

    def test_exclude_unset_yields_partial_payload(self) -> None:
        patch = MetaPersonPatch(name="x")
        assert patch.model_dump(exclude_unset=True) == {"name": "x"}

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
