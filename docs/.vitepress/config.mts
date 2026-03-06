import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'sqlmodel-ext',
  description: 'SQLModel 增强库：使用指南与实现原理',
  lang: 'zh-CN',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '指南', link: '/guide/' },
      { text: '原理', link: '/internals/' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: '指南',
          items: [
            { text: '快速开始', link: '/guide/' },
            { text: '字段类型', link: '/guide/field-types' },
            { text: 'CRUD 操作', link: '/guide/crud' },
            { text: '分页与列表', link: '/guide/pagination' },
            { text: '多态继承', link: '/guide/polymorphic' },
            { text: '乐观锁', link: '/guide/optimistic-lock' },
            { text: '关系预加载', link: '/guide/relation-preload' },
            { text: 'Redis 缓存', link: '/guide/cached-table' },
            { text: '静态分析器', link: '/guide/relation-load-checker' },
          ],
        },
      ],
      '/internals/': [
        {
          text: '实现原理',
          items: [
            { text: '架构总览', link: '/internals/' },
            { text: '前置知识', link: '/internals/prerequisites' },
            { text: '元类与 SQLModelBase', link: '/internals/metaclass' },
            { text: 'CRUD 实现', link: '/internals/crud' },
            { text: '多态继承机制', link: '/internals/polymorphic' },
            { text: '乐观锁机制', link: '/internals/optimistic-lock' },
            { text: '关系预加载机制', link: '/internals/relation-preload' },
            { text: 'Redis 缓存机制', link: '/internals/cached-table' },
            { text: '静态分析器原理', link: '/internals/relation-load-checker' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Foxerine/sqlmodel-ext' },
    ],

    outline: {
      level: [2, 3],
      label: '目录',
    },

    docFooter: {
      prev: '上一页',
      next: '下一页',
    },
  },

  markdown: {
    container: {
      tipLabel: '提示',
      warningLabel: '注意',
      dangerLabel: '危险',
      infoLabel: '信息',
      detailsLabel: '详细信息',
    },
  },
})
