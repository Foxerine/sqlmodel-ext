# AI coding rules for projects that use sqlmodel-ext

AI assistants write code that *looks* right. With an ORM toolkit the typical
result is a second declaration of something that already exists -- a hand-made
PATCH DTO, a repeated `max_length`, an `if x is not None` where "not sent" and
"clear it" must be told apart -- which then drifts from the original. The rules
in this directory teach Claude Code, Codex, Copilot and similar tools the
single-source-of-truth way of using sqlmodel-ext, and tell them to run
basedpyright until it reports 0 errors.

| File | Read by | Content |
|---|---|---|
| [`AGENTS.md`](AGENTS.md) | Codex, Copilot, Cursor, Claude Code, ... | **The rules** (single source) |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code | A two-line entry point that imports `AGENTS.md` with Claude Code's `@path` import |

There is only one copy of the rules. `CLAUDE.md` pulls `AGENTS.md` in through
an import, so editing `AGENTS.md` updates what every tool sees.

## Install

Pick the branch or tag that matches the sqlmodel-ext version you use
(`master` below).

**Your project has no `AGENTS.md` / `CLAUDE.md` yet** -- copy both to the repository root:

```bash
base=https://raw.githubusercontent.com/Foxerine/sqlmodel-ext/master/ai-rules
curl -fsSLO "$base/AGENTS.md" && curl -fsSLO "$base/CLAUDE.md"
```

**Your project already has them** -- keep the rules in their own file and
reference it once:

```bash
curl -fsSL -o sqlmodel-ext.rules.md \
  https://raw.githubusercontent.com/Foxerine/sqlmodel-ext/master/ai-rules/AGENTS.md
echo '@sqlmodel-ext.rules.md' >> CLAUDE.md
printf '\nWhen writing SQLModel models or queries, follow sqlmodel-ext.rules.md.\n' >> AGENTS.md
```

Claude Code expands the `@` import at session start. Codex and most other
tools have no import syntax; the sentence in `AGENTS.md` points them to the
file. If your models live in one package, you can instead copy the rules to
`<package>/AGENTS.md`: Codex applies nested `AGENTS.md` files to their
subtree, and Claude Code reads a subdirectory's `AGENTS.md` when it opens a
file there (as long as that directory has no `CLAUDE.md` of its own).

## Pair it with basedpyright

The rules only work if the checker runs. Install it (`pip install basedpyright`)
and make "0 errors" part of your definition of done -- in CI and in the
instructions you give the assistant. `examples/11_type_errors_caught_by_basedpyright.py`
in this repository shows the mistakes it reports.

## Updating

Re-download `AGENTS.md` when you upgrade sqlmodel-ext; do not edit your copy
by hand unless you also want to diverge from the library's guidance.
