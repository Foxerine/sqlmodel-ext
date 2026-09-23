# Use with AI coding assistants

**Goal**: make Claude Code, Codex, Copilot and other AI coding assistants write models and queries **the way sqlmodel-ext intends** instead of code that merely looks right, and let basedpyright flag the remaining mistakes before anything runs.

**Prerequisites**:

- Your project already uses sqlmodel-ext
- You can run `basedpyright` in the project (`pip install basedpyright`; 1.40.1 or later is recommended -- the first release that narrows `x is Unset` correctly)

## Why

sqlmodel-ext is built around a **single source of truth**: every fact is declared in one place and everything else is derived from it. A field constraint is written once, in its type (`Str64`), and produces Pydantic validation, the database column type and the OpenAPI schema; update DTOs are derived from the model with `partial=True`; "not sent / sent null / sent a value" are the three states `Unset` / `None` / value.

The most common AI mistake is exactly **a second declaration of a fact that already exists**:

| What AI tends to write | Problem | Correct |
|---|---|---|
| A hand-written `class XxxUpdate: title: str \| None = None` | Cannot tell "not sent" from "clear"; constraints drift from the table model | `class XxxUpdate(XxxBase, partial=True)` |
| `if patch.summary is not None:` | Treats "not sent" and "sent null" alike | `if patch.summary is not Unset:` |
| `model_dump(exclude_unset=True)` | Redundant -- `Unset` fields never appear in a dump | just `await obj.update(session, patch)` |
| `title[:64]` | A second `64` that will not follow the type | `title[:max_length_of(Str64)]` |
| `count = x or 0` | Valid `0` / `''` get replaced too; the bug is hidden | an explicit `if x is None:`, or make the field required |
| `obj.save(session)` without using the result | The object you hold is expired; the next attribute read is I/O | `obj = await obj.save(session)` |
| Reading an unloaded relationship | Relationships default to `lazy='raise_on_sql'` and raise | `load=rel(Model.rel)` / `@requires_relations` |

The rule pack shipped with the repository turns these conventions into instructions an assistant can follow directly.

## 1. Install the rule pack

The rules live in the repository's [`ai-rules/`](https://github.com/Foxerine/sqlmodel-ext/tree/master/ai-rules) directory, as a single text:

| File | Read by | Content |
|---|---|---|
| `AGENTS.md` | Codex, Copilot, Cursor, Claude Code, ... | **The rules** (the only copy) |
| `CLAUDE.md` | Claude Code | Entry point that pulls in the rules with Claude Code's `@AGENTS.md` import |

**Your project has no `AGENTS.md` / `CLAUDE.md` yet** -- put both files in the repository root:

```bash
base=https://raw.githubusercontent.com/Foxerine/sqlmodel-ext/master/ai-rules
curl -fsSLO "$base/AGENTS.md" && curl -fsSLO "$base/CLAUDE.md"
```

**Your project already has them** -- keep the rules in their own file and reference it once:

```bash
curl -fsSL -o sqlmodel-ext.rules.md \
  https://raw.githubusercontent.com/Foxerine/sqlmodel-ext/master/ai-rules/AGENTS.md
echo '@sqlmodel-ext.rules.md' >> CLAUDE.md
printf '\nWhen writing SQLModel models or queries, follow sqlmodel-ext.rules.md.\n' >> AGENTS.md
```

Claude Code expands the `@` import at session start; Codex and similar tools have no import syntax and follow the sentence in `AGENTS.md`. Replace `master` with the branch or tag that matches your sqlmodel-ext version.

## 2. Rule overview

`AGENTS.md` has 11 sections; each rule states the rule, the reason, and a do / don't example:

1. **Model layout**: shared fields on `XxxBase`, the table and every DTO inherit it; import `Field` / `Relationship` from `sqlmodel`; describe fields with docstrings
2. **Constraints live in type aliases**: `Str64`, `NonNegativeInt`, `NonNegativeWriteDecimal38_18`, ...; use `max_length_of()` when code needs the number
3. **PATCH = `partial=True` + `Unset`**: decide "submitted?" with `is Unset`, never `is not None`; no `exclude_unset`
4. **No silent fallbacks**: no `or 0` / `or ''` / `.get(k, 0)`; raise instead of returning `None`
5. **Writing**: always use the return value of `save()` / `update()`; no `session.refresh()`
6. **Reading**: parameterized `get(condition)`, no `find_by_xxx` wrappers; `col()` / `cond()` / `rel()`
7. **Always preload relationships**: `load=`, `@requires_relations`, `ensure_relations_loaded_bulk`; nullable relationships as `'Target | None'`; no `from __future__ import annotations` in model modules
8. **Concurrency and transactions**: `with_for_update` + `@requires_for_update`; `OptimisticLockMixin` first in the MRO; side effects via `add_post_commit_callback`
9. **Cache**: `invalidate_on_commit()` after raw DML
10. **Deletes and integrity errors**: `ResourceReferencedError` -> 409, messages registered next to the parent model
11. **Boundary types**: `AwareDatetime` for client-supplied datetimes; Decimal aliases for money, never `float`

## 3. Pair it with basedpyright

The rules tell the assistant how to write the code; basedpyright checks that it did. **Running it to 0 errors after every change** is a mandatory step in the rule pack. The sqlmodel-ext API is typed so that most misuse is a type error:

| Misuse | basedpyright reports |
|---|---|
| An explicit `Unset \| T` field used without `is not Unset` | `"MISSING" is not assignable to "str"` |
| Only `is not None` checked on `Unset \| T \| None` | same -- a `None` check does not exclude `Unset` |
| `get()`'s default `T \| None` used as `T` | `"name" is not a known attribute of "None"` |
| `await Model.delete(session)` with nothing to delete | `No overloads for "delete"` |
| `load=Model.relation` without `rel()` | `"Target" is not assignable to "QueryableAttribute[Any]"` |
| `Model.field.in_(...)` without `col()` | `Cannot access attribute "in_"` |
| `select()` with more than 9 columns | `No overloads for "select"` |
| `save()`'s return value discarded | warning `reportUnusedCallResult` |

The complete real output is in the repository: [`examples/11_type_errors_caught_by_basedpyright.py`](https://github.com/Foxerine/sqlmodel-ext/blob/master/examples/11_type_errors_caught_by_basedpyright.py) and [`examples/README.md`](https://github.com/Foxerine/sqlmodel-ext/blob/master/examples/README.md).

::: warning The static blind spot of partial=True
The tri-state annotations of a `partial=True` class are generated by the metaclass at class-creation time, so basedpyright still sees the base annotations on the derived class (`title: str`) and **does not force** the `is Unset` check. Runtime semantics are unaffected. Where static enforcement matters there are two remedies: declare the field explicitly as `field: Unset | T = Unset` in the partial class body, or use the experimental `python -m sqlmodel_ext.check_derived` (see [Check partial DTOs for misuse](./check-partial-dtos)).
:::

Run basedpyright in CI and make "basedpyright: 0 errors" the definition of done in the instructions you give the assistant:

```bash
basedpyright           # from the project root; done means 0 errors
```

## Common pitfalls

- **Installing `CLAUDE.md` without `AGENTS.md`**: `CLAUDE.md` is only an import entry point; the rules are in `AGENTS.md`.
- **Copying the rule text into several files**: copies do not update together and drift apart -- the very problem this library removes. Keep one copy and reference it.
- **Letting the assistant silence errors with `# type: ignore`**: the rules forbid it; `# pyright: ignore[rule]  # reason` is only for defects in third-party stubs.
