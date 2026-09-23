# Explanation

> **Understanding-oriented.** Explanation tells you **why** — why sqlmodel-ext is designed the way it is, why a particular mechanism exists, what deeper problem it solves. It doesn't teach you how to do things (that's the job of [tutorials](/en/tutorials/) and [how-to guides](/en/how-to/)) and it doesn't list APIs (that's [reference](/en/reference/)).
> Read this section once you can already use the library and you're curious "why is it like this?"

::: tip Read this one first
[Design philosophy: a single source of truth](./single-source-of-truth) — **every fact is declared in exactly one place; everything else is derived from it.** Every other design decision in this library follows from that sentence.
:::

## Reading order

Explanation pieces have **no mandatory order** — pick whatever piques your curiosity. If you're entirely new to async SQLAlchemy internals, start with [Prerequisites](./prerequisites).

| Chapter | Difficulty | Question it answers |
|---------|-----------|---------------------|
| [Design philosophy: a single source of truth](./single-source-of-truth) | Beginner | Why does this library exist? Why does "declare each fact once" matter even more with AI-assisted coding? |
| [Unset: three states](./unset-three-state) | Beginner | Why must "not provided / null / a value" be three distinct states? |
| [Prerequisites](./prerequisites) | Beginner | What are ORM, Session, lazy loading, metaclass, `Annotated` types? |
| [Metaclass & SQLModelBase](./metaclass) | Intermediate | Why does sqlmodel-ext need a custom metaclass? What does it do at class creation time? |
| [CRUD pipeline](./crud-pipeline) | Core | How do `save()` / `get()` work internally? Why must you use the return value? |
| [Polymorphic inheritance internals](./polymorphic-internals) | Advanced | How are JTI and STI implemented at the SQLAlchemy level? What does the two-phase column registration solve? |
| [Optimistic lock mechanism](./optimistic-lock) | Intermediate | How does automatic retry turn "lost update" into a recoverable conflict? |
| [Relation preloading mechanism](./relation-preload) | Intermediate | How does `@requires_relations` declare dependencies without changing call sites? |
| [Cascade delete semantics](./cascade-delete-semantics) | Intermediate | What actually happens in the 18 combinations of `cascade_delete` × `passive_deletes` × `ondelete`? Why doesn't `raise_on_sql` (usually) fire during cascade? |
| [Redis cache mechanism](./cached-table) | Advanced | How does the dual-layer cache (ID + query) coordinate with automatic invalidation? Why `_cached_ancestors`? |
| [Cache transparency inside transactions](./transactional-cache-transparency) | Advanced | After writing a cached model inside a transaction, what do reads, rollback and commit each see? |
| [Static analyzer internals](./relation-load-checker) | Advanced | How does the AST find potential MissingGreenlet problems at startup? |

## Core design philosophy

Every design decision in sqlmodel-ext follows from one sentence: **every fact is declared in exactly one place; everything else is derived from it.** (The full argument is in [Design philosophy: a single source of truth](./single-source-of-truth).) Concretely, the user just declares a model definition; the framework handles all the SQLAlchemy plumbing behind the scenes.

Reaching that goal relies on several key techniques:

| Technique | Problem it solves | Read more |
|-----------|------------------|-----------|
| Custom metaclass `__DeclarativeMeta` | Auto `table=True`, JTI/STI detection, `sa_type` extraction, `__mapper_args__` merging | [Metaclass & SQLModelBase](./metaclass) |
| Mixin composition | CRUD, optimistic lock, caching, preloading — each capability independent and opt-in | every chapter |
| `__init_subclass__` hooks | Import-time validation of relation name typos, automatic polymorphic identity generation | [Polymorphic internals](./polymorphic-internals) |
| `__get_pydantic_core_schema__` | Custom types satisfy both Pydantic validation and SQLAlchemy column mapping | [Metaclass & SQLModelBase](./metaclass) |
| AST static analysis | Catch potential MissingGreenlet bugs before any request arrives | [Static analyzer](./relation-load-checker) |
| Dual-layer cache + version invalidation | Row-level precise caching + O(1) model-level invalidation | [Redis cache](./cached-table) |

## What explanation is not

- **Not a tutorial.** Explanation will not walk you through writing code from scratch.
- **Not API reference.** Explanation cites source snippets but does not list every parameter.
- **Not for everyone.** You can use sqlmodel-ext perfectly well without ever reading explanation. This section is for people who like to "open the hood and look at the engine."
