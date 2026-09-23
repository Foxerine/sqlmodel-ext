# How-to guides

> **Task-oriented.** How-to guides help you accomplish **a specific goal**. They assume you already know the basics (you've finished the [tutorials](/en/tutorials/) or you're already familiar with async SQLModel) and now face a concrete problem that needs a direct procedure.
> Every guide is a recipe for "how to do X": prerequisites, steps, common pitfalls.

::: tip Upgrading from 0.4.x?
Start with [Migrate from 0.4.x to 0.5.0](./migrate-to-0-5) — every breaking change and how to fix it.
:::

## By topic

### Single source of truth & type checking

- [Write PATCH endpoints](./write-patch-endpoints) — `partial=True` derived update DTOs + the `Unset` three states
- [Type-check with basedpyright](./type-check-with-basedpyright) — recommended config; misuse is flagged before runtime
- [Check partial DTOs for misuse](./check-partial-dtos) — experimental `python -m sqlmodel_ext.check_derived`
- [Use with AI coding assistants](./use-with-ai-assistants) — drop `ai-rules/` into your project

### API endpoints & queries

- [Paginate a list endpoint](./paginate-a-list-endpoint) — `TableViewRequest` + `ListResponse[T]`
- [Integrate with FastAPI](./integrate-with-fastapi) — Standard patterns for the 5 endpoint types (GET / POST / PATCH / DELETE / LIST)
- [Keyset cursor pagination](./keyset-pagination) — `after_id` / `PageWindowRequest`
- [Aggregate queries](./aggregate-queries) — `count(distinct_column=)` / `distinct_column()` / `group_sum()`

### Data models

- [Define JTI (joined table inheritance) models](./define-jti-models) — Use when subclasses have many distinct fields
- [Define STI (single table inheritance) models](./define-sti-models) — Use when subclasses add only 1–2 extra fields
- [Configure cascade delete](./configure-cascade-delete) — How to combine `cascade_delete` / `passive_deletes` / `ondelete`
- [Handle deletes of still-referenced rows](./handle-referenced-deletes) — `ResourceReferencedError` + message registration
- [More mixins](./extra-mixins) — quotas, fuzzy search, cross-table scans, migration cache invalidation

### Concurrency & consistency

- [Handle concurrent updates](./handle-concurrent-updates) — Use `OptimisticLockMixin` to prevent lost updates
- [Enforce row locks and isolation levels](./enforce-locking-and-isolation) — `@requires_for_update` / `run_in_repeatable_read` and friends
- [Prevent MissingGreenlet errors](./prevent-missing-greenlet) — `@requires_relations` + `lazy='raise_on_sql'` + static analysis: three layers of defense
- [Release the DB connection during long I/O](./release-connection-during-long-io) — Use `session.reset()` to keep external I/O from exhausting the pool

### Performance

- [Cache queries with Redis](./cache-queries) — `CachedTableBaseMixin` + `configure_redis()`

## What how-to guides are not

- **Not tutorials.** Guides assume you know the basic library idioms. If `await User.save(session)` doesn't ring a bell, start with the [tutorials](/en/tutorials/).
- **Not reference.** Guides only list the **parameters needed for the task at hand**, not every option. For full signatures see [Reference](/en/reference/).
- **They don't explain the "why".** If you want to know "why does sqlmodel-ext implement it this way", go to [Explanation](/en/explanation/).

## Can't find your guide?

If your task isn't listed, it might be:

1. **Tutorial-level** ("how do I create my first model") → see [Tutorials](/en/tutorials/)
2. **Reference-level** ("the full parameter list of `save()`") → see [Reference](/en/reference/)
3. **A new scenario** → please open an issue on [GitHub](https://github.com/Foxerine/sqlmodel-ext/issues) to propose a new guide
