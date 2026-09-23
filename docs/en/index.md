---
layout: home

hero:
  name: sqlmodel-ext
  text: SQLModel Enhancement Library
  tagline: Declare every fact once — validation, columns, OpenAPI and PATCH DTOs are derived from it
  actions:
    - theme: brand
      text: Tutorials
      link: /en/tutorials/
    - theme: alt
      text: How-to
      link: /en/how-to/
    - theme: alt
      text: Reference
      link: /en/reference/
    - theme: alt
      text: Explanation
      link: /en/explanation/
    - theme: alt
      text: GitHub
      link: https://github.com/Foxerine/sqlmodel-ext

features:
  - title: Single source of truth & types
    details: "Unset three-state fields and partial=True PATCH DTOs; constrained aliases (Str64, Port, Decimal, ...) drive validation, column type and OpenAPI at once, and max_length_of() reflects bounds instead of repeating them; typed select() up to 9 columns"
  - title: CRUD & queries
    details: "One get(condition) with typed fetch_mode overloads, count(distinct_column=) / distinct_column() / group_sum(), keyset pagination with after_id, ResourceReferencedError on FK-restricted deletes, UUIDv7 primary keys, JTI / STI polymorphism"
  - title: Concurrency & transactions
    details: "with_for_update that always refreshes, skip_locked work queues, fail-closed @requires_for_update / isolation-level decorators, run_in_repeatable_read with 40001 retries, post-commit callbacks, optimistic locking on oplock_version with 3 retries by default"
  - title: Caching & relations
    details: "Transaction-transparent two-tier Redis cache that never publishes uncommitted state, @requires_relations and bulk relation preloading, RelationLoadChecker static analysis (RLC001–RLC014) against MissingGreenlet"
  - title: Built for basedpyright
    details: "Constraints and the three field states live in types, so basedpyright (≥ 1.40.1) flags almost every misuse before the code runs"
  - title: Rules for AI coding assistants
    details: "The repository ships CLAUDE.md / AGENTS.md / .claude/rules so Claude, Codex and friends use the library the intended way"
---

::: warning Work in progress
sqlmodel-ext is under active development. APIs may change without notice between releases, and there are no stability or backward-compatibility guarantees. Use it at your own risk. Upgrading from 0.4.x? Read the [0.5.0 migration guide](/en/how-to/migrate-to-0-5).
:::

## Design philosophy

> **Every fact is declared in exactly one place; everything else is derived from it.**
>
> A field's constraints are written once, in its type — that single declaration produces the Pydantic validation, the database column type and the OpenAPI schema. Update DTOs are derived from the table model. "Not sent", "sent as null" and "sent a value" are three different states, told apart by the type system instead of by convention.
>
> This matters even more in the age of AI-assisted coding. AI is best at producing code that *looks* right, and its most common mistake is restating a fact in a second place — after which the two copies slowly drift apart. sqlmodel-ext makes the restatement unnecessary, and turns as many of the remaining mistakes as possible into type errors: with basedpyright, most misuse is flagged before the code ever runs. The repository ships a rule set for AI coding assistants — drop it into your project and Claude, Codex and friends will use the library the intended way.
>
> Fail loudly: illegal states are rejected at construction time instead of being silently patched over in production.

```python
class ArticleBase(SQLModelBase):
    title: Str64                   # one declaration: validation + VARCHAR(64) + OpenAPI maxLength
    summary: Str64 | None = None   # null is a real value here

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    pass                           # UUIDv7 id, timestamps, async CRUD

class ArticleUpdate(ArticleBase, partial=True):
    pass                           # every field: Unset | T = Unset

# PATCH {"summary": null} -> writes only `summary`; {"title": null} -> 422
article = await article.update(session, ArticleUpdate.model_validate(payload))
```

## Design orientation

sqlmodel-ext is not a new ORM. It is a set of **opt-in Mixins** layered on top of SQLModel / Pydantic v2 / SQLAlchemy 2.0. The library only introduces two kinds of machinery:

- **Metaclass enhancements** — automatic SQLAlchemy column setup driven by `Annotated` type hints, `mapper_args` merging, `partial=True` PATCH DTOs with `Unset` fields, attribute-docstring inheritance, and a Python 3.14 (PEP 649) compatibility patch.
- **Composable Mixins** — `TableBaseMixin` (async CRUD), `PolymorphicBaseMixin` (JTI/STI), `OptimisticLockMixin` (version-column concurrency), `RelationPreloadMixin` (eager loading), `CachedTableBaseMixin` (Redis two-tier cache). Each Mixin stands alone and has no hard dependency on the others.

The underlying `select()`, query construction, and migration tooling remain native SQLAlchemy — there is no custom DSL, and the library does not take over the `engine` lifecycle. The only session-layer enhancement is the `sqlmodel_ext.AsyncSession` subclass (cache-aware commit, post-commit callbacks, lock and isolation tracking; plug it in via `async_sessionmaker(class_=AsyncSession)`). In practice that means you can drop a single Mixin into an existing SQLModel/SQLAlchemy project without rewriting your data-access layer.
