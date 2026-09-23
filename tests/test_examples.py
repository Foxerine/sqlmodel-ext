"""
Run every runnable script in ``examples/`` so the examples cannot drift from the library.

Each example is executed in its own interpreter (``subprocess``): examples
define table models with overlapping names, and a fresh process gives each one
its own ``SQLModel.metadata``. An example passes when it exits with status 0
and prints its ``[OK]`` line.

``11_type_errors_caught_by_basedpyright.py`` is intentionally ill-typed and is
not executed; it is covered by its own test below, which only checks that it
refuses to run.
"""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / 'examples'

RUNNABLE_EXAMPLES = sorted(
    path for path in EXAMPLES_DIR.glob('[0-9][0-9]_*.py')
    if not path.name.startswith('11_')
)


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=EXAMPLES_DIR.parent,
    )


def test_examples_are_discovered() -> None:
    """Guard against the glob silently matching nothing (which would make every test vacuous)."""
    assert [p.name[:2] for p in RUNNABLE_EXAMPLES] == [f"{i:02d}" for i in range(1, 11)]


@pytest.mark.parametrize('example', RUNNABLE_EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(example: Path) -> None:
    result = _run(example)
    assert result.returncode == 0, f"{example.name} failed:\n{result.stdout}\n{result.stderr}"
    assert f"[OK] {example.stem}" in result.stdout, result.stdout


def test_type_error_example_refuses_to_run() -> None:
    result = _run(EXAMPLES_DIR / '11_type_errors_caught_by_basedpyright.py')
    assert result.returncode != 0
    assert "basedpyright" in result.stderr
