---
layout: home

hero:
  name: sqlmodel-ext
  text: SQLModel Enhancement Library
  tagline: Define models, inherit Mixins, get a complete async CRUD API
  actions:
    - theme: brand
      text: Usage Guide
      link: /en/guide/
    - theme: alt
      text: Implementation Internals
      link: /en/internals/
    - theme: alt
      text: GitHub
      link: https://github.com/Foxerine/sqlmodel-ext

features:
  - title: One-line Async CRUD
    details: save / get / update / delete / count / get_with_count, with built-in pagination, time filtering, and relation preloading
  - title: Rich Field Types
    details: Str64, Port, HttpUrl, SafeHttpUrl, IPAddress, Array[T], etc. — Pydantic validation + SQLAlchemy column types in one step
  - title: Polymorphic Inheritance
    details: Zero-config support for Joined Table Inheritance (JTI) and Single Table Inheritance (STI), with automatic discriminator columns and subclass registration
  - title: Redis Query Caching
    details: CachedTableBaseMixin provides a dual-layer cache (ID + query), with automatic invalidation on CRUD and polymorphic inheritance support
  - title: Safe & Reliable
    details: Optimistic locking for concurrency control, @requires_relations to prevent MissingGreenlet, and AST static analysis at startup
---
