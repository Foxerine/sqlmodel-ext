"""
**Experimental.** Type-check ``partial=True`` DTOs as they really are at runtime.

Usage::

    python -m sqlmodel_ext.check_derived app            # import package ``app``, check the project
    python -m sqlmodel_ext.check_derived app --root .   # explicit project root (default: cwd)
    python -m sqlmodel_ext.check_derived app --keep     # keep the throwaway copy for inspection

As a pre-commit hook it must run **before** any other static check, so that a
misuse of a derived DTO is reported as such rather than drowned in (or masked
by) the output of later hooks::

    - repo: local
      hooks:
        - id: check-derived
          name: check partial DTOs (sqlmodel-ext, experimental)
          entry: python -m sqlmodel_ext.check_derived app
          language: system
          pass_filenames: false
          types: [python]

**What it guards against**: ``partial=True`` makes fields tri-state
(``Unset | T``) at **runtime**, and basedpyright only reads **source**. So an
inherited guard (``if x is None``) or a consumer (``if dto.x is not None:
use(dto.x)``) is no longer sufficient, and nothing reports it. This tool
materializes the runtime facts as source code in a throwaway copy of the
project, so the type checker can report them. The expansion itself lives in
:mod:`sqlmodel_ext.derived_decls` (single source of truth); this module only
orchestrates.

**Steps**: import the given modules (the metaclass registers every
``partial`` class on creation) -> collect expansions -> copy the project's
``.py`` / ``.pyi`` files into a temporary directory -> write the expansions into
the copy -> run basedpyright on the copy -> if there are errors, run it again on
an unexpanded copy and report only the errors the expansion **introduced**.

**Exit status**: ``0`` no new blocking error; ``1`` new blocking errors;
``2`` the tool could not do its job (a class cannot be expanded, basedpyright
missing or returned no JSON, zero files analyzed).

**Severity**: an error inside an expanded ``model_validator`` /
``model_post_init`` (they run on every construction) or at a consumer (any
line outside the generated code) is **blocking**. An error inside an expanded
ordinary inherited method is **latent** -- it only fails if a PATCH DTO ever
calls that method -- and is printed without failing.

**Known blind spots**: see :mod:`sqlmodel_ext.derived_decls` -- most notably
``getattr(obj, 'field', None)``, which a type checker cannot follow.

**Invariants**:

1. **The project is never modified.** The copies live in the system temporary
   directory and are deleted afterwards (unless ``--keep``); bytecode writing
   is disabled while the project's modules are imported, so not even
   ``__pycache__`` appears. Enforced by :func:`main` (temporary directory must
   lie outside the project) and :func:`sqlmodel_ext.derived_decls.write_into`
   (refuses a destination inside the project).
2. **Only new errors are reported, not the total.** Pre-existing errors in the
   project must not make every run fail -- a gate that always fails gets
   bypassed. Enforced by the baseline run in :func:`main`.
3. **The baseline is compared by ``(file, rule, message)`` counts, without
   line numbers.** The expansion inserts lines, so every line number below it
   shifts; a key including the line would report every existing error as new.
4. **The checked code is the project's code.** Every imported target module
   must resolve to a file under ``--root``; otherwise the run aborts. An
   installed copy of the package shadowing the project would make every
   result meaningless while looking normal.

**Known limits**: the copy is taken from the **working tree** (for a git
repository: tracked files plus untracked, non-ignored ones), not from the
index.
"""
import argparse
import collections
import importlib
import json
import os
import pathlib
import pkgutil
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import tomllib
import typing

from sqlmodel_ext import derived_decls
from sqlmodel_ext.derived_decls import Span

SOURCE_SUFFIXES: typing.Final = frozenset({'.py', '.pyi'})
"""Only the files the type checker reads are copied."""

SKIPPED_DIRS: typing.Final = frozenset({
    '__pycache__', 'node_modules', 'site-packages', 'build', 'dist',
})
"""Directory names never copied when the project is not a git repository (hidden directories are skipped too)."""


def _is_under(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def import_targets(targets: list[str], import_paths: list[pathlib.Path], root: pathlib.Path) -> None:
    """
    Import every target module, and every submodule of target packages.

    The metaclass registers ``partial`` classes as a side effect of class
    creation, so every module that defines one must be imported.

    :raises SystemExit: a target resolves to a file outside ``root`` (invariant 4)
    """
    sys.dont_write_bytecode = True  # invariant 1: no __pycache__ in the project
    for path in reversed(import_paths):
        sys.path.insert(0, str(path))
    for target in targets:
        module = importlib.import_module(target)
        loaded = pathlib.Path(module.__file__ or '')
        if not module.__file__ or not _is_under(loaded, root):
            raise SystemExit(
                f"[ABORT] '{target}' was imported from {loaded or '<no file>'}, "
                f"which is outside the project root {root}. The results would describe "
                f"other code; aborting."
            )
        package_path: list[str] | None = getattr(module, '__path__', None)
        if package_path is None:
            continue
        for info in pkgutil.walk_packages(package_path, prefix=f'{target}.'):
            _ = importlib.import_module(info.name)


def list_sources(root: pathlib.Path) -> list[str]:
    """
    Project files to copy, relative to ``root`` (POSIX separators).

    In a git work tree: ``git ls-files`` plus untracked, non-ignored files -- a
    DTO written a minute ago and not yet added must be checked too. Otherwise:
    a directory walk skipping hidden directories, :data:`SKIPPED_DIRS` and
    virtual environments.
    """
    try:
        listed: list[str] = []
        for args in (['git', 'ls-files', '-z'], ['git', 'ls-files', '--others', '--exclude-standard', '-z']):
            out = subprocess.run(args, cwd=root, capture_output=True, check=True).stdout
            listed.extend(name for name in out.decode('utf-8').split('\0') if name)
    except (OSError, subprocess.CalledProcessError):
        listed = []
        for directory, dirnames, filenames in os.walk(root):
            here = pathlib.Path(directory)
            dirnames[:] = [
                name for name in dirnames
                if not name.startswith('.') and name not in SKIPPED_DIRS
                and not (here / name / 'pyvenv.cfg').is_file()
            ]
            listed.extend((here / name).relative_to(root).as_posix() for name in filenames)
    # ``ls-files`` lists the index: a file deleted but not yet ``git rm``-ed is still listed.
    return sorted(
        name for name in listed
        if pathlib.PurePosixPath(name).suffix in SOURCE_SUFFIXES and (root / name).is_file()
    )


def _strip_jsonc(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments (outside strings) and trailing commas -- basedpyright reads its config as JSONC."""
    out: list[str] = []
    # Index in ``out`` of the last comma outside a string not yet followed by a significant character.
    pending_comma: int | None = None
    index = 0
    in_string = False
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if char == '\\':
                out.append(text[index + 1:index + 2])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
        elif text.startswith('//', index):
            end = text.find('\n', index)
            index = len(text) if end == -1 else end
        elif text.startswith('/*', index):
            end = text.find('*/', index + 2)
            index = len(text) if end == -1 else end + 2
        else:
            if not char.isspace():
                if char in '}]' and pending_comma is not None:
                    out[pending_comma] = ''  # trailing comma
                pending_comma = len(out) if char == ',' else None
                in_string = char == '"'
            out.append(char)
            index += 1
    return ''.join(out)


def load_config(root: pathlib.Path) -> dict[str, typing.Any]:
    """
    The project's basedpyright configuration: ``pyrightconfig.json``, else ``[tool.basedpyright]`` / ``[tool.pyright]`` in ``pyproject.toml``, else empty.

    Same precedence as basedpyright itself.
    """
    json_config = root / 'pyrightconfig.json'
    if json_config.is_file():
        loaded: dict[str, typing.Any] = json.loads(_strip_jsonc(json_config.read_text(encoding='utf-8')))
        return loaded
    pyproject = root / 'pyproject.toml'
    if pyproject.is_file():
        tool: dict[str, typing.Any] = tomllib.loads(pyproject.read_text(encoding='utf-8')).get('tool', {})
        for key in ('basedpyright', 'pyright'):
            section: dict[str, typing.Any] | None = tool.get(key)
            if section is not None:
                return dict(section)
    return {}


def snapshot(
        root: pathlib.Path,
        dest: pathlib.Path,
        names: list[str],
        config: dict[str, typing.Any],
        extra_paths: list[str],
) -> None:
    """
    Copy ``names`` from ``root`` to ``dest`` and write the configuration for the copy.

    The configuration is the project's own, with two adjustments:

    - ``venvPath`` is made absolute (the copy has no virtual environment); when
      no ``venv`` is configured the caller passes ``--pythonpath`` instead;
    - the import paths used to load the project (e.g. ``src``) are added to
      ``extraPaths``. Without this, an editable install would make basedpyright
      resolve ``import app`` to the **original** files rather than the copy,
      and every consumer-side finding would silently disappear.
    """
    for name in names:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(root / name, target)

    adjusted = dict(config)
    venv_path = adjusted.get('venvPath')
    if isinstance(venv_path, str):
        adjusted['venvPath'] = str((root / venv_path).resolve())
    existing: list[str] = list(adjusted.get('extraPaths', []))
    adjusted['extraPaths'] = [*extra_paths, *(path for path in existing if path not in extra_paths)]
    _ = (dest / 'pyrightconfig.json').write_text(json.dumps(adjusted, indent=2), encoding='utf-8')


def find_basedpyright(explicit: str | None) -> pathlib.Path:
    """
    The basedpyright executable: ``--basedpyright``, else the current interpreter's scripts directory, else ``PATH``.

    :raises SystemExit: not found
    """
    if explicit is not None:
        path = pathlib.Path(explicit)
        if not path.is_file():
            raise SystemExit(f"[ABORT] --basedpyright {explicit}: no such file")
        return path
    scripts = pathlib.Path(sysconfig.get_path('scripts'))
    for name in ('basedpyright.exe', 'basedpyright'):
        if (scripts / name).is_file():
            return scripts / name
    found = shutil.which('basedpyright')
    if found is None:
        raise SystemExit(
            "[ABORT] basedpyright not found (install it: pip install 'basedpyright>=1.40.1', "
            "or pass --basedpyright PATH)"
        )
    return pathlib.Path(found)


class Diagnostic(typing.NamedTuple):
    """
    One error-level diagnostic.

    :attr:`line` is used for grading only and is **not** part of the baseline
    key (invariant 3).
    """

    path: str
    """Path relative to the copy root (POSIX)."""

    rule: str
    message: str
    """First line of the message."""

    line: int
    """0-based, as in pyright's JSON ``range.start.line``."""

    @property
    def identity(self) -> tuple[str, str, str]:
        """Baseline key -- deliberately without the line number."""
        return self.path, self.rule, self.message


def diagnose(copy: pathlib.Path, executable: pathlib.Path, python: str | None) -> list[Diagnostic]:
    """
    Run basedpyright on ``copy`` and return every error-level diagnostic.

    Only errors are considered: warnings are advisory and would bury the
    signal.

    :raises SystemExit: no JSON output, or zero files analyzed (an empty scan
        also reports "0 errors" and must not pass as success)
    """
    command = [str(executable), '--outputjson', '--project', str(copy / 'pyrightconfig.json')]
    if python is not None:
        command += ['--pythonpath', python]
    # Not ``check=True``: basedpyright exits non-zero whenever it reports errors, which is the normal case here.
    result = subprocess.run(command, cwd=copy, capture_output=True, text=True, encoding='utf-8', check=False)
    try:
        payload: dict[str, typing.Any] = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout[-4000:], file=sys.stderr)
        print(result.stderr[-4000:], file=sys.stderr)
        raise SystemExit("[ABORT] basedpyright did not return JSON (raw output above)") from None

    analyzed: int = payload.get('summary', {}).get('filesAnalyzed', 0)
    if analyzed == 0:
        raise SystemExit(
            "[ABORT] basedpyright analyzed 0 files -- check 'include' / 'exclude' in the configuration"
        )
    diagnostics: list[dict[str, typing.Any]] = payload['generalDiagnostics']
    return [
        Diagnostic(
            path=pathlib.Path(item['file']).resolve().relative_to(copy.resolve()).as_posix(),
            rule=item.get('rule', '-'),
            message=str(item['message']).splitlines()[0],
            line=item['range']['start']['line'],
        )
        for item in diagnostics
        if item.get('severity') == 'error'
    ]


def introduced_by(found: list[Diagnostic], baseline: list[Diagnostic]) -> list[Diagnostic]:
    """
    Diagnostics in ``found`` that the baseline does not have, line numbers kept for grading.

    Counts are subtracted rather than sets: the same message may legitimately
    occur several times in one file (one mixin expanded into several classes),
    and a set difference would erase the extra occurrences.
    """
    quota = collections.Counter(item.identity for item in found)
    quota.subtract(collections.Counter(item.identity for item in baseline))
    out: list[Diagnostic] = []
    for item in found:
        if quota[item.identity] > 0:
            quota[item.identity] -= 1
            out.append(item)
    return out


BLOCKING_BANNER: typing.Final = "Code that runs on construction, or a consumer, is wrong for tri-state fields"
BLOCKING_ADVICE: typing.Final = (
    "  model_validator / model_post_init run on every construction -- including a PATCH\n"
    "  body that sets a single field -- so these fail at runtime, not in theory.\n"
    "  Fix: test for omission with `x is Unset` (or `x is Unset or x is None`).\n"
    "  Do not use `if x:` -- bool(Unset) is True, and a truthiness test also\n"
    "  swallows legitimate 0 / '' / []."
)

LATENT_BANNER: typing.Final = "Inherited methods are wrong for tri-state fields (fail only when called)"
LATENT_ADVICE: typing.Final = (
    "  A PATCH DTO usually never calls these (they are behavior of the base class\n"
    "  that the DTO inherited), so they do not fail the check. They are real:\n"
    "  calling one on a PATCH DTO will fail."
)


def classify(
        introduced: list[Diagnostic],
        spans: list[Span],
) -> tuple[list[tuple[Diagnostic, str]], list[tuple[Diagnostic, str]]]:
    """
    Split new diagnostics into ``(blocking, latent)`` by the generated range they fall in.

    A diagnostic outside every generated range is a consumer (another place
    reading the field) and is always blocking: it is a real call site. Inside a
    generated range, the rules in
    :data:`~sqlmodel_ext.derived_decls.GENERATOR_ARTIFACT_RULES` are dropped:
    they describe the expansion, not the checked code.
    """
    blocking: list[tuple[Diagnostic, str]] = []
    latent: list[tuple[Diagnostic, str]] = []
    for item in introduced:
        owner = next(
            (span for span in spans if span.path == item.path and span.start <= item.line <= span.end),
            None,
        )
        if owner is not None and item.rule in derived_decls.GENERATOR_ARTIFACT_RULES:
            continue
        if owner is None:
            blocking.append((item, 'consumer'))
        elif owner.blocking:
            blocking.append((item, owner.label))
        else:
            latent.append((item, owner.label))
    return blocking, latent


def report(items: list[tuple[Diagnostic, str]], banner: str, advice: str) -> None:
    """Print findings folded by (origin, file, rule, message); repeats collapse into one line."""
    if not items:
        return
    folded: collections.Counter[tuple[str, str, str, str]] = collections.Counter(
        (label, item.path, item.rule, item.message) for item, label in items
    )
    print(f"\n{'=' * 78}\n{banner} ({len(items)})\n{'=' * 78}")
    for (label, path, rule, message), count in sorted(folded.items()):
        times = f" x{count}" if count > 1 else ''
        print(f"  {label}{times}\n    {path}  [{rule}]\n      {message}")
    print(advice)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m sqlmodel_ext.check_derived',
        description=(
            "Experimental: expand partial=True DTOs in a throwaway copy of the project "
            "and report the basedpyright errors that the expansion introduces."
        ),
    )
    _ = parser.add_argument(
        'targets', nargs='+', metavar='MODULE',
        help="module or package to import (dotted name); packages are imported recursively",
    )
    _ = parser.add_argument(
        '--root', type=pathlib.Path, default=None,
        help="project root to copy and check (default: current directory)",
    )
    _ = parser.add_argument(
        '--import-path', type=pathlib.Path, action='append', default=None, metavar='DIR',
        help=(
            "directory (relative to --root) to put on sys.path before importing; repeatable "
            "(default: the root, plus 'src' if it exists)"
        ),
    )
    _ = parser.add_argument('--basedpyright', default=None, metavar='PATH', help="basedpyright executable")
    _ = parser.add_argument(
        '--python', default=None, metavar='PATH',
        help=(
            "interpreter basedpyright resolves third-party packages from "
            "(default: the current one, unless the configuration sets 'venv')"
        ),
    )
    _ = parser.add_argument('--keep', action='store_true', help="keep the temporary copies and print their location")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; returns the exit status."""
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    root: pathlib.Path = (args.root or pathlib.Path.cwd()).resolve()
    raw_paths: list[pathlib.Path] | None = args.import_path
    if raw_paths is None:
        raw_paths = [pathlib.Path('.')] + ([pathlib.Path('src')] if (root / 'src').is_dir() else [])
    import_paths = [(root / path).resolve() for path in raw_paths]
    extra_paths = [path.relative_to(root).as_posix() for path in import_paths if _is_under(path, root)]

    executable = find_basedpyright(args.basedpyright)
    config = load_config(root)
    python: str | None = args.python
    if python is None and 'venv' not in config:
        python = sys.executable
    names = list_sources(root)

    targets: list[str] = args.targets
    import_targets(targets, import_paths, root)
    expansions, errors = derived_decls.collect_all(root)
    for message in errors:
        print(f"[ERROR] {message}", file=sys.stderr)
    if errors:
        print(f"\n[FAIL] {len(errors)} partial class(es) could not be expanded", file=sys.stderr)
        return 2

    holder = pathlib.Path(tempfile.mkdtemp(prefix='check-derived-'))
    try:
        if _is_under(holder, root):
            raise SystemExit(f"[ABORT] the temporary directory {holder} is inside the project {root}")
        expanded = holder / 'expanded'
        snapshot(root, expanded, names, config, extra_paths)
        spans = derived_decls.write_into(expanded, root, expansions)
        print(
            f"[..] expanded {len(expansions)} class(es) / "
            f"{sum(len(expansion.methods) for expansion in expansions)} inherited member(s) "
            f"({time.monotonic() - started:.1f}s)"
        )

        found = diagnose(expanded, executable, python)
        if not found:
            print(f"[OK] no errors after expansion ({time.monotonic() - started:.1f}s)")
            return 0

        # The second scan is only paid for when there is something to compare.
        print(f"[..] {len(found)} error(s) after expansion; running the unexpanded baseline")
        pristine = holder / 'pristine'
        snapshot(root, pristine, names, config, extra_paths)
        introduced = introduced_by(found, diagnose(pristine, executable, python))
        if not introduced:
            print(
                f"[OK] all {len(found)} error(s) already exist without expansion; none introduced "
                f"({time.monotonic() - started:.1f}s)"
            )
            return 0

        blocking, latent = classify(introduced, spans)
        report(latent, LATENT_BANNER, LATENT_ADVICE)
        report(blocking, BLOCKING_BANNER, BLOCKING_ADVICE)
        print(
            f"\n[{'FAIL' if blocking else 'OK'}] blocking {len(blocking)} / latent {len(latent)} "
            f"({time.monotonic() - started:.1f}s)"
        )
        return 1 if blocking else 0
    finally:
        if args.keep:
            print(f"[keep] copies kept in {holder}")
        else:
            shutil.rmtree(holder, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
