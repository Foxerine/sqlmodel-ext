"""
End-to-end tests for ``python -m sqlmodel_ext.check_derived`` (experimental).

Each test writes a small project into ``tmp_path`` and runs the command in a
fresh interpreter: the metaclass registry is process-global, and the models
here are deliberately wrong (their validators would fail on construction), so
they must never be imported into the test session itself.

Every test that expects a failure has been checked against a mutation of the
mechanism it relies on (see the individual docstrings).
"""
import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MODELS = '''
"""Models for the check_derived tests."""
from typing import Self

from pydantic import model_validator

from sqlmodel_ext import SQLModelBase, Str64, Str256, Unset


class ArticleBase(SQLModelBase):
    title: Str64
    subtitle: Str256 | None = None

    {validator}

    def headline(self) -> str:
        return self.title.upper()


class ArticleUpdate(ArticleBase, partial=True):
    """PATCH body."""
'''

GUARD_TOO_WEAK = '''@model_validator(mode='after')
    def _check_subtitle(self) -> Self:
        if self.subtitle is not None:
            _ = self.subtitle.strip()
        return self'''

GUARD_TRI_STATE = '''@model_validator(mode='after')
    def _check_subtitle(self) -> Self:
        if self.subtitle is not Unset and self.subtitle is not None:
            _ = self.subtitle.strip()
        return self'''

NO_VALIDATOR = '''def _unrelated(self) -> None:
        return None'''

CONSUMER_WRONG = '''
from shop.models import ArticleUpdate


def describe(patch: ArticleUpdate) -> str:
    if patch.subtitle is not None:
        return patch.subtitle.upper()
    return ''
'''

CONSUMER_RIGHT = '''
from sqlmodel_ext import Unset

from shop.models import ArticleUpdate


def describe(patch: ArticleUpdate) -> str:
    if patch.subtitle is not Unset and patch.subtitle is not None:
        return patch.subtitle.upper()
    return ''
'''


def _project(root: Path, *, validator: str, consumer: str, extra: dict[str, str] | None = None) -> Path:
    files = {
        'shop/__init__.py': '',
        'shop/models.py': MODELS.replace('{validator}', validator),
        'shop/handlers.py': consumer,
        **(extra or {}),
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(textwrap.dedent(content).lstrip('\n'), encoding='utf-8')
    return root


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'sqlmodel_ext.check_derived', 'shop', '--root', str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=root,
    )


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }


def test_consumer_is_not_none_check_is_caught(tmp_path: Path) -> None:
    """
    ``if dto.x is not None: dto.x.upper()`` on a partial DTO fails the check.

    Mutation checked: with the field declarations not rendered
    (``_render_block`` skipping ``expansion.decls``) this test fails -- the
    consumer error disappears and the exit status becomes 0.
    """
    root = _project(tmp_path, validator=NO_VALIDATOR, consumer=CONSUMER_WRONG)
    result = _run(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'consumer' in result.stdout
    assert 'shop/handlers.py' in result.stdout
    assert 'Cannot access attribute "upper" for class "MISSING"' in result.stdout


def test_tri_state_check_passes(tmp_path: Path) -> None:
    """The correct spelling (``is not Unset``) produces no blocking finding."""
    root = _project(tmp_path, validator=GUARD_TRI_STATE, consumer=CONSUMER_RIGHT)
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'consumer' not in result.stdout
    assert '_check_subtitle' not in result.stdout


def test_inherited_validator_is_expanded_along_the_mro(tmp_path: Path) -> None:
    """
    A validator inherited from the base, correct there, is caught once copied into the partial class.

    Mutation checked: with ``_inherited_members`` returning nothing this test
    fails -- the validator is no longer expanded and the exit status becomes 0.
    """
    root = _project(tmp_path, validator=GUARD_TOO_WEAK, consumer=CONSUMER_RIGHT)
    result = _run(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'ArticleUpdate <- _check_subtitle' in result.stdout
    assert 'Cannot access attribute "strip" for class "MISSING"' in result.stdout


def test_inherited_plain_method_is_latent_not_blocking(tmp_path: Path) -> None:
    """An inherited ordinary method that misuses a tri-state field is reported, but does not fail the run."""
    root = _project(tmp_path, validator=NO_VALIDATOR, consumer=CONSUMER_RIGHT)
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ArticleUpdate <- headline' in result.stdout
    assert 'blocking 0 / latent 1' in result.stdout


def test_working_tree_is_not_modified(tmp_path: Path) -> None:
    """
    Running the check leaves every project file byte-identical and creates no file (not even ``__pycache__``).

    Mutation checked: without ``sys.dont_write_bytecode = True`` in
    ``import_targets`` this test fails (``shop/__pycache__/*.pyc`` appears).
    """
    root = _project(tmp_path, validator=GUARD_TOO_WEAK, consumer=CONSUMER_WRONG)
    before = _tree_digest(root)
    assert before, 'empty digest -- the comparison below would be vacuous'
    result = _run(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _tree_digest(root) == before


def test_preexisting_errors_are_not_reported(tmp_path: Path) -> None:
    """
    Errors that exist without the expansion do not fail the check; only introduced ones count.

    Mutation checked: with the baseline subtraction disabled (``introduced =
    found``) this test fails -- the unrelated error is reported as a consumer
    finding and the exit status becomes 1.
    """
    root = _project(
        tmp_path,
        validator=GUARD_TRI_STATE,
        consumer=CONSUMER_RIGHT,
        extra={'shop/legacy.py': 'def broken() -> int:\n    return "not an int"\n'},
    )
    result = _run(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'legacy.py' not in result.stdout


def test_target_outside_root_aborts(tmp_path: Path) -> None:
    """A target that resolves outside ``--root`` (here: the library itself) aborts instead of checking other code."""
    _ = _project(tmp_path, validator=NO_VALIDATOR, consumer=CONSUMER_RIGHT)
    result = subprocess.run(
        [sys.executable, '-m', 'sqlmodel_ext.check_derived', 'sqlmodel_ext', '--root', str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert '[ABORT]' in result.stderr
    assert 'outside the project root' in result.stderr


def test_documented_demo_output() -> None:
    """``examples/check_derived_demo`` produces the findings quoted in ``examples/README.md`` and the docs."""
    demo = Path(__file__).resolve().parent.parent / 'examples' / 'check_derived_demo'
    result = _run(demo)
    assert result.returncode == 1, result.stdout + result.stderr
    for expected in (
        'ArticleUpdate <- headline',
        'ArticleUpdate <- _normalize_subtitle',
        'Cannot access attribute "strip" for class "MISSING"',
        '"__getitem__" method not defined on type "MISSING"',
        'blocking 2 / latent 1',
    ):
        assert expected in result.stdout, result.stdout


@pytest.mark.parametrize('config_name', ['pyrightconfig.json', 'pyproject.toml'])
def test_empty_scan_is_not_success(tmp_path: Path, config_name: str) -> None:
    """A configuration that includes no file must not pass as "no errors"."""
    root = _project(tmp_path, validator=NO_VALIDATOR, consumer=CONSUMER_WRONG)
    config = (
        '{\n  // JSONC comment\n  "include": ["nothing_here"],\n}\n'
        if config_name == 'pyrightconfig.json'
        else '[tool.basedpyright]\ninclude = ["nothing_here"]\n'
    )
    _ = (root / config_name).write_text(config, encoding='utf-8')
    result = _run(root)
    assert result.returncode != 0
    assert 'analyzed 0 files' in result.stderr
