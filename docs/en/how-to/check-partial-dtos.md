# Check partial DTOs for misuse (check_derived)

::: warning Experimental
`sqlmodel_ext.check_derived` and `sqlmodel_ext.derived_decls` are **experimental**: the command-line options and the module API may change in any release. There is a defined exit plan — see [When it will be removed](#when-it-will-be-removed) at the end.
:::

**Goal**: make basedpyright speak about PATCH DTOs derived with `partial=True` — so that misuse such as `if dto.x is not None: dto.x.upper()` is flagged before you commit.

**Prerequisites**:

- sqlmodel-ext and `basedpyright>=1.40.1` installed in the project (see [Type-check with basedpyright](./type-check-with-basedpyright))
- familiarity with [Unset](/en/explanation/unset-three-state)

## 1. The problem

`partial=True` makes the metaclass rewrite fields into `Unset | T = Unset` **at runtime**, while basedpyright only reads **source code**. So to the checker a derived class still has the base types:

<!-- skip-run -->
```python
class ArticleBase(SQLModelBase):
    title: Str64
    subtitle: Str256 | None = None


class ArticleUpdate(ArticleBase, partial=True):
    pass                     # runtime: subtitle: Unset | Str256 | None; statically still Str256 | None


def preview(patch: ArticleUpdate) -> str:
    if patch.subtitle is not None:      # when omitted it is Unset, and Unset is not None is true
        return patch.subtitle[:20]      # TypeError at runtime, yet basedpyright reports 0 errors
    return ''
```

The same blind spot exists in **inherited methods**: a `@model_validator` that is correct in the base class (`if self.subtitle is not None`) is no longer sufficient once `ArticleUpdate` inherits it — and a validator runs on **every construction**, so a PATCH body that sets a single field is exactly its normal path.

This is a language boundary, not a configuration problem: Python has neither compile-time derive macros (Rust `#[derive]`) nor type-level mapped types (TypeScript `Partial<T>`).

Without this tool there is one remedy: **declare explicitly**, in the partial class body, the fields that need static narrowing as `Unset | T = Unset` (the class body wins over `partial`). `check_derived` adds a second one: leave the source alone and let the checker see the runtime tri-state directly.

## 2. Run it

```bash
python -m sqlmodel_ext.check_derived app            # import package app (all submodules, recursively) and check the project in the current directory
python -m sqlmodel_ext.check_derived app --root .   # explicit project root
python -m sqlmodel_ext.check_derived app --keep     # keep the temporary copy to look at the expansion
```

| Option | Default | Meaning |
|---|---|---|
| `MODULE ...` (positional, at least one) | — | Module or package to import (dotted name). Packages are imported recursively — the metaclass registers partial classes when they are created, so the modules that define them must be imported |
| `--root ROOT` | current directory | Project root to copy and check |
| `--import-path DIR` | the root, plus `src/` if it exists | Directory (relative to `--root`) to put on `sys.path` before importing; repeatable |
| `--basedpyright PATH` | the current interpreter's scripts directory, then `PATH` | basedpyright executable |
| `--python PATH` | the current interpreter (not passed when the configuration sets `venv`) | Interpreter basedpyright resolves third-party packages from |
| `--keep` | off | Keep the temporary copies and print their location |

Exit status: `0` no new blocking error; `1` new blocking errors; `2` (or a message starting with `[ABORT]`) the tool could not do its job — a class cannot be expanded, basedpyright is missing, basedpyright returned no JSON, or **zero files were analyzed** (an empty scan also says "0 errors" and must not pass as success).

## 3. Add it to pre-commit (before every other static check)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-derived          # must be the first static check
        name: check partial DTOs (sqlmodel-ext, experimental)
        entry: python -m sqlmodel_ext.check_derived app
        language: system
        pass_filenames: false
        types: [python]
  # ... basedpyright, ruff etc. come after it
```

**This hook must run before every other static check**: it reports errors the type checker cannot otherwise see. Running it first makes a misuse show up as what it is ("a field of a derived DTO is not checked for `Unset`") instead of being buried in the output of later hooks, or never running at all when a later hook fails. `language: system` makes the hook use the project's own virtual environment (it needs to import your modules).

In CI, run the same command directly.

## 4. What it does

1. **Imports** the given modules; the metaclass registers every `partial=True` class in `optional_dto_registry`. The checked set is these classes **and all their subclasses** (the tri-state annotations are inherited, and subclasses do not say `partial=True` themselves), limited to classes whose source file lies under `--root`.
2. **Copies** the project's `.py` / `.pyi` files into the system temporary directory (git repository: tracked files plus untracked, non-ignored ones; otherwise a directory walk that skips hidden directories and virtual environments).
3. **Expands in the copy** — two things, both required:
   - **field declarations**: for every field the metaclass really made tri-state (decided by the runtime `model_fields`, not by re-deriving the metaclass rules), it writes `name: Unset | <annotation copied verbatim from the base source>` into an `if TYPE_CHECKING:` block of the derived class — keeping aliases such as `Str64` instead of expanding them to `Annotated[...]`;
   - **methods expanded along the MRO**: members the derived class does not define, an ancestor does, and that read a tri-state field (`self.<field>`) — methods, properties, `model_validator`, `model_post_init` — are copied into the derived class. Their `self` then has the derived type, and an insufficient guard is reported by basedpyright natively.
4. **Runs basedpyright** on the copy (with the project's own configuration: `pyrightconfig.json`, else `[tool.basedpyright]` / `[tool.pyright]` in `pyproject.toml`).
5. If there are errors, runs it again on an **unexpanded** copy as the baseline and reports only the errors the expansion **introduced**. The comparison key is the count of `(file, rule, message, text of the flagged line)`, **without line numbers** — the expansion inserts lines, and a key with line numbers would count every existing error as new. The line's text moves with the code, so an existing error stays matched, while a new error with the same rule and message on a different line is still reported (without the text, an old error disappearing and a new one appearing in the same file would cancel out). Residual limit: two identical errors on textually identical lines of one file are interchangeable.
6. Deletes the temporary directory (kept with `--keep`).

**The tool itself never writes into your project**: every write goes to the temporary copy; bytecode writing is disabled while your modules are imported, so not even `__pycache__` appears; the writer refuses a destination inside the project. Nothing generated ever lands in the repository.

::: warning Your modules are imported
To read runtime facts, the target modules are **imported from your project**, exactly as a test run would import them, so their module-level side effects run: a module that writes a file, opens a connection or starts a server at import time will do so. Point `check_derived` only at import-safe modules (model definitions usually are).
:::

**Severity**:

| Where the new error is | Severity | Why |
|---|---|---|
| An expanded `model_validator` / `model_post_init` | blocking | runs on every construction, so it fails at runtime |
| Outside the generated code (a **consumer** reading the field elsewhere) | blocking | a real call site |
| An expanded ordinary method / property | latent (printed, does not fail) | only fails when called; PATCH DTOs usually never call the base class's domain methods |

## 5. Demo

[`examples/check_derived_demo`](https://github.com/Foxerine/sqlmodel-ext/tree/master/examples/check_derived_demo) in the repository is a minimal project: `shop/models.py` has a base-class validator and a derived `ArticleUpdate`, `shop/handlers.py` has one `is not None` misuse.

Plain basedpyright sees nothing wrong (run in the project directory with the virtual environment activated):

```text
$ basedpyright
0 errors, 0 warnings, 0 notes
```

Running `check_derived` (from the repository root, basedpyright 1.40.1, verbatim output):

```text
$ python -m sqlmodel_ext.check_derived shop --root examples/check_derived_demo
[..] expanded 1 class(es) / 2 inherited member(s) (0.0s)
[..] 6 error(s) after expansion; running the unexpanded baseline

==============================================================================
Inherited methods are wrong for tri-state fields (fail only when called) (1)
==============================================================================
  ArticleUpdate <- headline
    shop/models.py  [reportAttributeAccessIssue]
      Cannot access attribute "upper" for class "MISSING"
  A PATCH DTO usually never calls these (they are behavior of the base class
  that the DTO inherited), so they do not fail the check. They are real:
  calling one on a PATCH DTO will fail.

==============================================================================
Code that runs on construction, or a consumer, is wrong for tri-state fields (2)
==============================================================================
  ArticleUpdate <- _normalize_subtitle
    shop/models.py  [reportAttributeAccessIssue]
      Cannot access attribute "strip" for class "MISSING"
  consumer
    shop/handlers.py  [reportIndexIssue]
      "__getitem__" method not defined on type "MISSING"
  model_validator / model_post_init run on every construction -- including a PATCH
  body that sets a single field -- so these fail at runtime, not in theory.
  Fix: test for omission with `x is Unset` (or `x is Unset or x is None`).
  Do not use `if x:` -- bool(Unset) is True, and a truthiness test also
  swallows legitimate 0 / '' / [].

[FAIL] blocking 2 / latent 1 (2.6s)
```

(`MISSING` is `Unset` — the two are the same object. The other 3 of the "6 error(s) after expansion" are override diagnostics that the expansion produces by construction; the tool recognizes and drops them.)

The blocking validator is not theoretical: `ArticleUpdate(title='x')` really raises `AttributeError: 'sentinel' object has no attribute 'strip'` at runtime. The fix is a tri-state guard:

<!-- skip-run -->
```python
if self.subtitle is not Unset and self.subtitle is not None:
    self.subtitle = self.subtitle.strip()
```

::: tip Do not use `if x:` instead
`bool(Unset)` is true, so a truthiness test does not stop it; it also treats legitimate `0` / `''` / `[]` as "not provided".
:::

## 6. Known blind spots

- **String-based** attribute access such as `getattr(self, 'x', None)` — a type checker cannot infer the attribute from a string literal.
- Anything the type checker cannot see anyway: code outside the `include` set, `Any`-typed values, `# pyright: ignore` comments.
- Classes whose source file is not under `--root`, and members inherited from sqlmodel-ext / SQLModel / Pydantic themselves, are not expanded.
- Whether a member is expanded depends on its source containing `self.<tri-state field>`; access through another name (`other = self; other.x`) is not detected.
- The copy is taken from the **working tree**, not from the git index.

## When it will be removed

[PEP 827](https://peps.python.org/pep-0827/) (type manipulation, draft) proposes the piece Python is missing: a PATCH DTO written as a type-level transformation of its base, so the checker sees the tri-state directly. Once it (or an equivalent) is accepted and supported by basedpyright, `check_derived` and `derived_decls` will be deprecated and then removed. Because they never leave generated files in the repository, removal leaves nothing to clean up — delete the pre-commit entry and you are done.
