"""Final-review regressions for ``check_derived`` safety wording and baseline subtraction."""

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

from sqlmodel_ext.check_derived import Diagnostic, introduced_by


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }


def test_import_side_effect_is_the_only_project_write_and_docs_disclose_it(tmp_path: Path) -> None:
    """The tool creates no project artifact itself, while docs must disclose imported-module effects."""
    module = tmp_path / 'models.py'
    module.write_text(
        textwrap.dedent(
            '''
            from pathlib import Path

            from sqlmodel_ext import SQLModelBase

            Path(__file__).with_name('IMPORT_SIDE_EFFECT.txt').write_text('imported', encoding='utf-8')


            class Base(SQLModelBase):
                value: int


            class Patch(Base, partial=True):
                pass
            '''
        ).lstrip(),
        encoding='utf-8',
    )
    (tmp_path / 'pyrightconfig.json').write_text(
        '{"include": ["models.py"], "pythonVersion": "3.12"}\n',
        encoding='utf-8',
    )
    before = _tree_digest(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'sqlmodel_ext.check_derived',
            'models',
            '--root',
            str(tmp_path),
            '--import-path',
            '.',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    after = _tree_digest(tmp_path)
    assert set(after) - set(before) == {'IMPORT_SIDE_EFFECT.txt'}
    assert (tmp_path / 'IMPORT_SIDE_EFFECT.txt').read_text(encoding='utf-8') == 'imported'
    assert not any(path.name == '__pycache__' for path in tmp_path.rglob('*'))

    repository = Path(__file__).resolve().parent.parent
    english = (repository / 'docs/en/how-to/check-partial-dtos.md').read_text(encoding='utf-8')
    chinese_bytes = (repository / 'docs/how-to/check-partial-dtos.md').read_bytes()
    assert 'module-level side effects run' in english
    assert 'import-safe modules' in english
    assert '模块顶层的副作用'.encode() in chinese_bytes
    assert '可安全导入'.encode() in chinese_bytes


def test_baseline_does_not_cancel_same_message_on_different_source() -> None:
    """A disappearing old error cannot hide a new same-message error on different code."""
    baseline = [Diagnostic('app.py', 'reportOperatorIssue', 'same message', 10, 'old_call(value)')]
    expanded = [Diagnostic('app.py', 'reportOperatorIssue', 'same message', 80, 'patch.title.upper()')]

    assert introduced_by(expanded, baseline) == expanded
    assert introduced_by(
        [Diagnostic('app.py', 'reportOperatorIssue', 'same message', 80, 'old_call(value)')],
        baseline,
    ) == []
