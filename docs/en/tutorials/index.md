# Tutorials

> **Learning-oriented.** Tutorials are hand-held lessons that get you to your **first success**. You may be completely unfamiliar with the library, and you don't need to fully understand every line — just follow along and you'll see the expected result at every step.
> By the end of all three tutorials you'll have a working project and solid muscle memory for sqlmodel-ext's core capabilities.

## Learning path

Tutorials **have a fixed order**. Each one builds on the previous.

| # | Tutorial | Time | What you'll get |
|---|----------|------|-----------------|
| 1 | [Getting started](./01-getting-started) | 15 min | Library installed, first CRUD round-trip, the "model + Mixin = table" pattern |
| 2 | [Building a blog API](./02-building-a-blog-api) | 60 min | A full FastAPI backend: users, posts, comments, pagination, JOINs, relation preloading |
| 3 | [Adding Redis caching](./03-adding-redis-cache) | 30 min | Plug `CachedTableBaseMixin` into the previous project; verify cache hits and invalidation |

## What tutorials are not

Tutorials don't **exhaustively cover the API**. If a feature you need is missing from the tutorials, that's normal —

- Looking for a method's full parameter list? Go to [Reference](/en/reference/).
- Trying to accomplish a specific task ("how do I handle concurrent updates")? Go to [How-to guides](/en/how-to/).
- Want to understand why something is designed a certain way? Go to [Explanation](/en/explanation/).

## Prerequisites

All tutorials assume you know:

- Python 3.12+ (3.12 / 3.13 / 3.14 are all exercised in CI)
- Basic `async` / `await` syntax
- What SELECT / INSERT / UPDATE / DELETE do in SQL

You don't need prior experience with SQLAlchemy, SQLModel, or any ORM — tutorials introduce these concepts on demand.
