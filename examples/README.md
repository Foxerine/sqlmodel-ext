# sqlmodel-ext examples

Every script is standalone: it uses in-memory SQLite (`sqlite+aiosqlite`) and,
where Redis is needed, `fakeredis` -- no external services. Each one asserts
its own claims and prints a line starting with `[OK]` when they all hold.
`tests/test_examples.py` runs 01-10 in CI, so these files cannot drift from the
library.

## Run

```bash
pip install "sqlmodel-ext[cache]" aiosqlite fakeredis   # from a checkout: uv pip install -e ".[dev]"
python examples/01_unset_patch.py
python -m pytest tests/test_examples.py   # all of them
```

`07_transactional_cache.py` needs `fakeredis` (part of the `dev` extra); the
others need only `sqlmodel-ext` and `aiosqlite`.

## Index

| Script | Shows |
|---|---|
| [`01_unset_patch.py`](01_unset_patch.py) | **Start here.** `partial=True` PATCH DTOs, the three states (omitted `Unset` / `null` / value), `update()` touching only submitted columns, inherited constraints, derived JSON Schema, opt-in `omitted_sentinel` wire value |
| [`02_single_source_types.py`](02_single_source_types.py) | One annotation (`Str64`, `NonNegativeWriteDecimal38_18`) drives validation, the column type and the OpenAPI schema; `max_length_of()` instead of a second literal; floats rejected for Decimal |
| [`03_keyset_pagination.py`](03_keyset_pagination.py) | `TableViewRequest` offset + keyset (`after_id`) pagination, `get_with_count()`, `KeysetCursorInvalidError`, `after_id` + `offset` rejected at construction |
| [`04_locking_and_isolation.py`](04_locking_and_isolation.py) | `with_for_update` + `@requires_for_update` / `@requires_locked_param`, lock scope ends at commit, `add_post_commit_callback()`, `commit_count`; PostgreSQL-only helpers described in the docstring |
| [`05_aggregates.py`](05_aggregates.py) | `count(distinct_column=)`, `distinct_column()`, `group_sum()` with `GroupSumRow`; write/sum Decimal aliases |
| [`06_referenced_delete.py`](06_referenced_delete.py) | `ResourceReferencedError` (409) and `register_fk_delete_restrict_message()`; SQLite enforces the FK but has no SQLSTATE, so the raw `IntegrityError` is shown and the PostgreSQL mapping is written out |
| [`07_transactional_cache.py`](07_transactional_cache.py) | `CachedTableBaseMixin` on fakeredis: ID-layer hit with zero SQL, invalidation after commit, rollback safety, raw DML + `invalidate_on_commit()`, `authoritative=True` |
| [`08_optimistic_lock.py`](08_optimistic_lock.py) | `OptimisticLockMixin` / `oplock_version`: default retry re-applies only your delta, `optimistic_retry_count=0` raises `OptimisticLockError` |
| [`09_relation_preload_bulk.py`](09_relation_preload_bulk.py) | Default `lazy='raise_on_sql'`, `load=rel(...)`, `@requires_relations`, `ensure_relations_loaded_bulk()` (1 query instead of N) |
| [`10_typed_select.py`](10_typed_select.py) | `select()` typed up to 9 columns, `col()`, `cond()`, `fetch_mode` return types |
| [`11_type_errors_caught_by_basedpyright.py`](11_type_errors_caught_by_basedpyright.py) | **Not runnable on purpose** -- nine common mistakes, each reported by basedpyright (output below) |
| [`check_derived_demo/`](check_derived_demo/) | **Experimental** `python -m sqlmodel_ext.check_derived`: finds `partial=True` misuse plain basedpyright cannot see (output below) |
| [`consumer_integration/run.py`](consumer_integration/run.py) | CI smoke test run against the built wheel in a clean virtualenv |

## What basedpyright catches

sqlmodel-ext is developed against basedpyright and its API is typed so that
most misuse becomes a type error. Output of
`basedpyright examples/11_type_errors_caught_by_basedpyright.py`
(basedpyright 1.40.1, run from the repository root; only the absolute path
prefix was trimmed):

```text
examples/11_type_errors_caught_by_basedpyright.py
  examples/11_type_errors_caught_by_basedpyright.py:62:15 - error: Argument of type "Str64 | MISSING" cannot be assigned to parameter "text" of type "str" in function "shout"
    Type "Str64 | MISSING" is not assignable to type "str"
      "MISSING" is not assignable to "str" (reportArgumentType)
  examples/11_type_errors_caught_by_basedpyright.py:64:19 - error: Argument of type "str | MISSING" cannot be assigned to parameter "text" of type "str" in function "shout"
    Type "str | MISSING" is not assignable to type "str"
      "MISSING" is not assignable to "str" (reportArgumentType)
  examples/11_type_errors_caught_by_basedpyright.py:66:21 - error: "name" is not a known attribute of "None" (reportOptionalMemberAccess)
  examples/11_type_errors_caught_by_basedpyright.py:67:22 - error: Type "Member | None" is not assignable to declared type "Member"
    Type "Member | None" is not assignable to type "Member"
      "None" is not assignable to "Member" (reportAssignmentType)
  examples/11_type_errors_caught_by_basedpyright.py:68:15 - error: No overloads for "delete" match the provided arguments
    Argument types: (AsyncSession) (reportCallIssue)
  examples/11_type_errors_caught_by_basedpyright.py:69:15 - error: No overloads for "get" match the provided arguments (reportCallIssue)
  examples/11_type_errors_caught_by_basedpyright.py:69:58 - error: Argument of type "Team" cannot be assigned to parameter "load" of type "QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None" in function "get"
    Type "Team" is not assignable to type "QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None"
      "Team" is not assignable to "QueryableAttribute[Any]"
      "Team" is not assignable to "list[QueryableAttribute[Any]]"
      "Team" is not assignable to "None" (reportArgumentType)
  examples/11_type_errors_caught_by_basedpyright.py:70:47 - error: Cannot access attribute "in_" for class "Str64"
    Attribute "in_" is unknown (reportAttributeAccessIssue)
  examples/11_type_errors_caught_by_basedpyright.py:71:9 - error: No overloads for "select" match the provided arguments
    Argument types: (Mapped[UUID], Mapped[Str64], Mapped[str | None], Mapped[UUID], Mapped[datetime], Mapped[datetime], Mapped[UUID], Mapped[Str64], Mapped[str | None], Mapped[UUID]) (reportCallIssue)
  examples/11_type_errors_caught_by_basedpyright.py:76:5 - warning: Result of call expression is of type "Member" and is not used; assign to variable "_" if this is intentional (reportUnusedCallResult)
9 errors, 1 warning, 0 notes
```

| Line | Mistake | Fix |
|---|---|---|
| 62 | Used an omissible field without checking it | `if patch.name is not Unset: ...` |
| 64 | Checked `is not None` on a tri-state field | `is not Unset` (then `None` means "clear") |
| 66, 67 | Treated `get()`'s default `T \| None` as `T` | `fetch_mode='one'`, or handle `None` |
| 68 | `delete()` with nothing to delete | `delete(session, instance)` or `delete(session, condition=...)` |
| 69 | Passed a relationship attribute to `load=` | `load=rel(Member.team)` |
| 70 | Column method on a bare attribute | `col(Member.name).in_([...])` |
| 71 | 10-column `select()` | at most 9 columns, or split the query |
| 76 | Discarded `save()`'s return value | `member = await member.save(session)` |

Limitation: the tri-state annotations of a `partial=True` DTO are created at
class-creation time, so the checker sees the *base* annotations there (lines
62/64 would not be flagged on a `partial=True` class). Declare
`field: Unset | T = Unset` explicitly where static enforcement matters, or run
the experimental `check_derived` (next section), and keep the `is Unset`
discipline everywhere.

## Closing the `partial=True` gap: `check_derived` (experimental)

[`check_derived_demo/`](check_derived_demo/) is a tiny stand-alone project
(its own `pyrightconfig.json`): `shop/models.py` defines `ArticleBase` with a
`model_validator` that is correct for the base, and derives
`ArticleUpdate(ArticleBase, partial=True)`; `shop/handlers.py` checks a PATCH
field with `is not None`. Plain basedpyright finds nothing (run inside
`examples/check_derived_demo` with the virtualenv activated):

```text
$ basedpyright
0 errors, 0 warnings, 0 notes
```

`check_derived` expands `ArticleUpdate` (its tri-state fields and the inherited
members that read them) in a throwaway copy, runs basedpyright there and reports
only what the expansion introduced. Verbatim output, from the repository root
(basedpyright 1.40.1):

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

The exit status is 1. Nothing is written into `check_derived_demo/` (not even
`__pycache__`). `tests/test_check_derived.py` runs this demo, so the findings
above cannot drift. As a pre-commit hook `check_derived` must run before every
other static check; see
[Check partial DTOs for misuse](../docs/en/how-to/check-partial-dtos.md).

## Rules for AI coding assistants

[`ai-rules/`](../ai-rules/) contains a rule file you can drop into your own
project so Claude Code, Codex, Copilot and similar tools use these APIs the way
the examples do.
