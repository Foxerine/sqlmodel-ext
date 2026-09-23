"""
Unit tests for :mod:`sqlmodel_ext.derived_decls` (experimental).

The models below live in this module on purpose: :func:`derived_decls.collect`
reads their source with :mod:`inspect`. They are registered in the global
``optional_dto_registry`` like any partial class, so every validator here is
written correctly for the tri-state (``tests/test_partial.py`` constructs every
registered class from ``{}``). The end-to-end behavior -- basedpyright actually
flagging a wrong guard -- is covered by ``tests/test_check_derived.py``.
"""
import pathlib
from typing import Literal, Self

import pytest
from pydantic import model_validator

from sqlmodel_ext import (
    SQLModelBase,
    Str64,
    Str256,
    Unset,
    check_derived,
    derived_decls,
)
from sqlmodel_ext.check_derived import Diagnostic
from sqlmodel_ext.derived_decls import Span

THIS_FILE = pathlib.Path(__file__).resolve()


class DeclArticleBase(SQLModelBase):
    kind: Literal['article'] = 'article'
    title: Str64
    subtitle: Str256 | None = None
    pinned: Unset | bool = Unset

    @model_validator(mode='after')
    def _check_subtitle(self) -> Self:
        if self.subtitle is not Unset and self.subtitle is not None:
            _ = self.subtitle.strip()
        return self

    @property
    def shouting_title(self) -> str:
        return str(self.title).upper()

    def unrelated(self) -> int:
        return 1


class DeclArticleUpdate(DeclArticleBase, partial=True):
    """PATCH body."""


class DeclArticleAdminUpdate(DeclArticleUpdate):
    """Subclass of a partial class -- not registered itself, but its fields are tri-state too."""


class TestCollect:
    def test_field_declarations_copy_annotation_text_verbatim(self) -> None:
        expansion = derived_decls.collect(DeclArticleUpdate)
        assert dict(expansion.decls) == {'title': 'Str64', 'subtitle': 'Str256 | None'}

    def test_literal_and_hand_written_tri_state_fields_are_not_declared(self) -> None:
        names = dict(derived_decls.collect(DeclArticleUpdate).decls)
        assert 'kind' not in names  # Literal discriminator: the metaclass leaves it alone
        assert 'pinned' not in names  # already Unset | T in source

    def test_members_reading_tri_state_fields_are_expanded(self) -> None:
        members = {member.name: member for member in derived_decls.collect(DeclArticleUpdate).methods}
        assert set(members) == {'_check_subtitle', 'shouting_title'}
        assert members['_check_subtitle'].runs_on_construction
        assert not members['shouting_title'].runs_on_construction

    def test_pydantic_decorators_are_stripped_and_others_kept(self) -> None:
        members = {member.name: member for member in derived_decls.collect(DeclArticleUpdate).methods}
        assert 'model_validator' not in members['_check_subtitle'].source
        assert members['_check_subtitle'].source.startswith('def _check_subtitle')
        assert members['shouting_title'].source.startswith('@property')

    def test_base_class_itself_has_nothing_to_expand(self) -> None:
        assert derived_decls.collect(DeclArticleBase).is_empty

    def test_subclasses_of_partial_classes_are_in_scope(self) -> None:
        expansions, errors = derived_decls.collect_all(THIS_FILE.parent)
        assert not errors
        models = {expansion.model for expansion in expansions}
        assert {DeclArticleUpdate, DeclArticleAdminUpdate} <= models
        assert DeclArticleBase not in models

    def test_classes_outside_the_root_are_skipped(self, tmp_path: pathlib.Path) -> None:
        expansions, errors = derived_decls.collect_all(tmp_path)
        assert (expansions, errors) == ([], [])


class TestHelpers:
    def test_free_names_excludes_parameters_locals_and_own_name(self) -> None:
        code = '''
            def check(self, session: 'AsyncSession') -> Self:
                items = [x for x in self.values]
                total: Decimal = sum(items)
                return helper(total)
        '''
        assert derived_decls._free_names(code) == {  # pyright: ignore[reportPrivateUsage]
            'AsyncSession', 'Self', 'Decimal', 'sum', 'helper',
        }

    def test_accepts_unset_descends_into_annotated(self) -> None:
        field = DeclArticleUpdate.model_fields['title']
        assert derived_decls._accepts_unset(field.annotation)  # pyright: ignore[reportPrivateUsage]
        assert not derived_decls._accepts_unset(DeclArticleBase.model_fields['title'].annotation)  # pyright: ignore[reportPrivateUsage]

    def test_accepts_unset_in_text_uses_names_not_substrings(self) -> None:
        assert derived_decls._accepts_unset_in_text('Unset | int')  # pyright: ignore[reportPrivateUsage]
        assert derived_decls._accepts_unset_in_text("'Unset | Foo'")  # pyright: ignore[reportPrivateUsage]
        assert not derived_decls._accepts_unset_in_text('UnsetLike | int')  # pyright: ignore[reportPrivateUsage]

    def test_reads_any_field_matches_whole_identifiers(self) -> None:
        assert derived_decls._reads_any_field('return self.title', {'title'})  # pyright: ignore[reportPrivateUsage]
        assert not derived_decls._reads_any_field('return self.title_length', {'title'})  # pyright: ignore[reportPrivateUsage]


class TestWriteInto:
    def test_refuses_a_destination_inside_the_project(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match='refusing to write into the project'):
            _ = derived_decls.write_into(tmp_path / 'copy', tmp_path, [])

    def test_imports_go_after_docstring_and_future_imports(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / 'host.py'
        _ = source.write_text('"""Doc."""\nfrom __future__ import annotations\nx = 1\n', encoding='utf-8')
        assert derived_decls._import_anchor(str(source)) == 2  # pyright: ignore[reportPrivateUsage]


class TestBaseline:
    def test_introduced_ignores_line_numbers_and_keeps_extra_occurrences(self) -> None:
        old = Diagnostic('a.py', 'rule', 'msg', 3)
        found = [Diagnostic('a.py', 'rule', 'msg', 10), Diagnostic('a.py', 'rule', 'msg', 20)]
        introduced = check_derived.introduced_by(found, [old])
        assert [item.identity for item in introduced] == [old.identity]  # one extra occurrence, not zero or two
        assert check_derived.introduced_by(found[:1], [old]) == []  # moved line is not "new"

    def test_classify_grades_by_generated_range(self) -> None:
        spans = [
            Span('m.py', 10, 12, 'U <- check', True),
            Span('m.py', 20, 22, 'U <- method', False),
        ]
        consumer = Diagnostic('h.py', 'reportAttributeAccessIssue', 'x', 5)
        in_validator = Diagnostic('m.py', 'reportAttributeAccessIssue', 'x', 11)
        in_method = Diagnostic('m.py', 'reportAttributeAccessIssue', 'x', 21)
        artifact = Diagnostic('m.py', 'reportIncompatibleVariableOverride', 'x', 11)
        blocking, latent = check_derived.classify([consumer, in_validator, in_method, artifact], spans)
        assert blocking == [(consumer, 'consumer'), (in_validator, 'U <- check')]
        assert latent == [(in_method, 'U <- method')]

    def test_jsonc_comments_and_trailing_commas(self) -> None:
        text = '{\n  // comment\n  "a": "http://x/*y*/", /* block */\n  "b": [1, 2,],\n}\n'
        assert check_derived._strip_jsonc(text).replace(' ', '').replace('\n', '') == (  # pyright: ignore[reportPrivateUsage]
            '{"a":"http://x/*y*/","b":[1,2]}'
        )
