"""Reviewer regression probe for the check_derived no-project-write invariant."""

from pathlib import Path

import pytest

from sqlmodel_ext import check_derived


def test_check_derived_does_not_create_holder_before_rejecting_temp_inside_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'sample.py').write_text('VALUE = 1\n', encoding='utf-8')

    created = project / 'check-derived-review-probe'

    def fake_mkdtemp(*, prefix: str, dir: str | Path | None = None) -> str:  # noqa: A002 - mirrors tempfile.mkdtemp
        assert prefix == 'check-derived-'
        # Ignores ``dir`` on purpose: simulates a temp location that resolves
        # into the project whatever parent the tool picked.
        created.mkdir()
        return str(created)

    monkeypatch.setattr(check_derived.tempfile, 'mkdtemp', fake_mkdtemp)
    monkeypatch.chdir(project)

    with pytest.raises(SystemExit, match='temporary directory .* is inside the project'):
        # No --basedpyright: the tool's own lookup (current interpreter's
        # scripts dir, then PATH) works on every platform the suite runs on.
        check_derived.main([
            'sample',
            '--root',
            str(project),
            '--keep',
        ])

    assert not created.exists(), 'the tool wrote into the project before enforcing its invariant'
