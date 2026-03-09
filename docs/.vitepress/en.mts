import type { DefaultTheme } from 'vitepress'

export const en = {
  lang: 'en',
  description: 'SQLModel Enhancement Library: Usage Guide & Implementation Internals',

  themeConfig: {
    nav: [
      { text: 'Home', link: '/en/' },
      { text: 'Guide', link: '/en/guide/' },
      { text: 'Internals', link: '/en/internals/' },
    ],

    sidebar: {
      '/en/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Quick Start', link: '/en/guide/' },
            { text: 'Field Types', link: '/en/guide/field-types' },
            { text: 'CRUD Operations', link: '/en/guide/crud' },
            { text: 'Pagination & Lists', link: '/en/guide/pagination' },
            { text: 'Polymorphic Inheritance', link: '/en/guide/polymorphic' },
            { text: 'Optimistic Locking', link: '/en/guide/optimistic-lock' },
            { text: 'Relation Preloading', link: '/en/guide/relation-preload' },
            { text: 'Redis Caching', link: '/en/guide/cached-table' },
            { text: 'Static Analyzer', link: '/en/guide/relation-load-checker' },
          ],
        },
      ],
      '/en/internals/': [
        {
          text: 'Implementation Internals',
          items: [
            { text: 'Architecture Overview', link: '/en/internals/' },
            { text: 'Prerequisites', link: '/en/internals/prerequisites' },
            { text: 'Metaclass & SQLModelBase', link: '/en/internals/metaclass' },
            { text: 'CRUD Implementation', link: '/en/internals/crud' },
            { text: 'Polymorphic Inheritance', link: '/en/internals/polymorphic' },
            { text: 'Optimistic Locking', link: '/en/internals/optimistic-lock' },
            { text: 'Relation Preloading', link: '/en/internals/relation-preload' },
            { text: 'Redis Caching', link: '/en/internals/cached-table' },
            { text: 'Static Analyzer', link: '/en/internals/relation-load-checker' },
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
