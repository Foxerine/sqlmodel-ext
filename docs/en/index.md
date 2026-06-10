---
layout: home

hero:
  name: sqlmodel-ext
  text: SQLModel Enhancement Library
  tagline: Define models, inherit Mixins, get a complete async CRUD API
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
  - title: One-line Async CRUD
    details: save / get / update / delete / count / get_with_count, with built-in pagination, time filtering, and relation preloading
  - title: Rich Field Types
    details: Str64, Port, HttpUrl, SafeHttpUrl, IPAddress, Array[T], high-precision Decimal (NUMERIC columns + fixed-point JSON-string serialization), etc. — Pydantic validation + SQLAlchemy column types in one step
  - title: Polymorphic Inheritance
    details: Zero-config support for Joined Table Inheritance (JTI) and Single Table Inheritance (STI), with automatic discriminator columns, subclass registration, and DeferredIndex for base-class indexes over subclass columns
  - title: Redis Query Caching
    details: CachedTableBaseMixin provides a dual-layer cache (ID + query); the enhanced AsyncSession synchronously invalidates on commit, with polymorphic inheritance support
  - title: Safe & Reliable
    details: Optimistic locking for concurrency control, @requires_relations to prevent MissingGreenlet, and AST static analysis at startup
---

## Design orientation

sqlmodel-ext is not a new ORM. It is a set of **opt-in Mixins** layered on top of SQLModel / Pydantic v2 / SQLAlchemy 2.0. The library only introduces two kinds of machinery:

- **Metaclass enhancements** — automatic SQLAlchemy column setup driven by `Annotated` type hints, `mapper_args` merging, on-demand `*UpdateRequest` DTO generation, attribute-docstring inheritance, and a Python 3.14 (PEP 649) compatibility patch.
- **Composable Mixins** — `TableBaseMixin` (async CRUD), `PolymorphicBaseMixin` (JTI/STI), `OptimisticLockMixin` (version-column concurrency), `RelationPreloadMixin` (eager loading), `CachedTableBaseMixin` (Redis two-tier cache). Each Mixin stands alone and has no hard dependency on the others.

The underlying `select()`, query construction, and migration tooling remain native SQLAlchemy — there is no custom DSL, and the library does not take over the `engine` lifecycle. The only session-layer enhancement is the optional `sqlmodel_ext.AsyncSession` (a thin subclass of sqlmodel's `AsyncSession`; caching setups plug it in via `async_sessionmaker(class_=AsyncSession)`). In practice that means you can drop a single Mixin into an existing SQLModel/SQLAlchemy project without rewriting your data-access layer.
