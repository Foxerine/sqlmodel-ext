import type { DefaultTheme } from 'vitepress'

export const en = {
  lang: 'en',
  description: 'SQLModel Enhancement Library — Tutorials, How-to, Reference, Explanation',

  themeConfig: {
    nav: [
      { text: 'Home', link: '/en/' },
      { text: 'Tutorials', link: '/en/tutorials/' },
      { text: 'How-to', link: '/en/how-to/' },
      { text: 'Reference', link: '/en/reference/' },
      { text: 'Explanation', link: '/en/explanation/' },
    ],

    sidebar: {
      '/en/tutorials/': [
        {
          text: 'Tutorials',
          items: [
            { text: 'Overview', link: '/en/tutorials/' },
            { text: '01 · Getting started', link: '/en/tutorials/01-getting-started' },
            { text: '02 · Building a blog API', link: '/en/tutorials/02-building-a-blog-api' },
            { text: '03 · Adding Redis caching', link: '/en/tutorials/03-adding-redis-cache' },
          ],
        },
      ],
      '/en/how-to/': [
        {
          text: 'How-to guides',
          items: [
            { text: 'Overview', link: '/en/how-to/' },
            { text: 'Migrate from 0.4.x to 0.5.0', link: '/en/how-to/migrate-to-0-5' },
          ],
        },
        {
          text: 'Single source of truth & type checking',
          items: [
            { text: 'Write PATCH endpoints', link: '/en/how-to/write-patch-endpoints' },
            { text: 'Type-check with basedpyright', link: '/en/how-to/type-check-with-basedpyright' },
            { text: 'Check partial DTOs for misuse', link: '/en/how-to/check-partial-dtos' },
            { text: 'Use with AI coding assistants', link: '/en/how-to/use-with-ai-assistants' },
          ],
        },
        {
          text: 'API endpoints & queries',
          items: [
            { text: 'Integrate with FastAPI', link: '/en/how-to/integrate-with-fastapi' },
            { text: 'Paginate a list endpoint', link: '/en/how-to/paginate-a-list-endpoint' },
            { text: 'Keyset cursor pagination', link: '/en/how-to/keyset-pagination' },
            { text: 'Aggregate queries', link: '/en/how-to/aggregate-queries' },
          ],
        },
        {
          text: 'Data models',
          items: [
            { text: 'Define JTI models', link: '/en/how-to/define-jti-models' },
            { text: 'Define STI models', link: '/en/how-to/define-sti-models' },
            { text: 'Configure cascade delete', link: '/en/how-to/configure-cascade-delete' },
            { text: 'Handle deletes of still-referenced rows', link: '/en/how-to/handle-referenced-deletes' },
            { text: 'More mixins', link: '/en/how-to/extra-mixins' },
          ],
        },
        {
          text: 'Concurrency & consistency',
          items: [
            { text: 'Handle concurrent updates', link: '/en/how-to/handle-concurrent-updates' },
            { text: 'Enforce row locks and isolation levels', link: '/en/how-to/enforce-locking-and-isolation' },
            { text: 'Prevent MissingGreenlet errors', link: '/en/how-to/prevent-missing-greenlet' },
            { text: 'Release the DB connection during long I/O', link: '/en/how-to/release-connection-during-long-io' },
          ],
        },
        {
          text: 'Performance',
          items: [
            { text: 'Cache queries with Redis', link: '/en/how-to/cache-queries' },
          ],
        },
      ],
      '/en/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Overview', link: '/en/reference/' },
            { text: 'Base classes', link: '/en/reference/base-classes' },
            { text: 'CRUD methods', link: '/en/reference/crud-methods' },
            { text: 'Field types', link: '/en/reference/field-types' },
            { text: 'Mixins', link: '/en/reference/mixins' },
            { text: 'Decorators & helpers', link: '/en/reference/decorators' },
            { text: 'Pagination types', link: '/en/reference/pagination-types' },
          ],
        },
      ],
      '/en/explanation/': [
        {
          text: 'Explanation',
          items: [
            { text: 'Design philosophy: a single source of truth', link: '/en/explanation/single-source-of-truth' },
            { text: 'Overview', link: '/en/explanation/' },
            { text: 'Unset: three states', link: '/en/explanation/unset-three-state' },
            { text: 'Prerequisites', link: '/en/explanation/prerequisites' },
            { text: 'Metaclass & SQLModelBase', link: '/en/explanation/metaclass' },
            { text: 'CRUD pipeline', link: '/en/explanation/crud-pipeline' },
            { text: 'Polymorphic internals', link: '/en/explanation/polymorphic-internals' },
            { text: 'Optimistic locking', link: '/en/explanation/optimistic-lock' },
            { text: 'Relation preloading', link: '/en/explanation/relation-preload' },
            { text: 'Cascade delete semantics', link: '/en/explanation/cascade-delete-semantics' },
            { text: 'Redis caching', link: '/en/explanation/cached-table' },
            { text: 'Cache transparency inside transactions', link: '/en/explanation/transactional-cache-transparency' },
            { text: 'Static analyzer', link: '/en/explanation/relation-load-checker' },
          ],
        },
      ],
    },

    outline: {
      level: [2, 3],
      label: 'On this page',
    },

    docFooter: {
      prev: 'Previous',
      next: 'Next',
    },
  } satisfies DefaultTheme.Config,
}
