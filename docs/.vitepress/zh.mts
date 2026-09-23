import type { DefaultTheme } from 'vitepress'

export const zh = {
  lang: 'zh-CN',
  description: 'SQLModel 增强库 — 教程、操作指南、参考、讲解',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '教程', link: '/tutorials/' },
      { text: '操作指南', link: '/how-to/' },
      { text: '参考', link: '/reference/' },
      { text: '讲解', link: '/explanation/' },
    ],

    sidebar: {
      '/tutorials/': [
        {
          text: '教程',
          items: [
            { text: '总览', link: '/tutorials/' },
            { text: '01 · 快速上手', link: '/tutorials/01-getting-started' },
            { text: '02 · 构建博客 API', link: '/tutorials/02-building-a-blog-api' },
            { text: '03 · 给博客加 Redis 缓存', link: '/tutorials/03-adding-redis-cache' },
          ],
        },
      ],
      '/how-to/': [
        {
          text: '操作指南',
          items: [
            { text: '总览', link: '/how-to/' },
            { text: '从 0.4.x 迁移到 0.5.0', link: '/how-to/migrate-to-0-5' },
          ],
        },
        {
          text: '单点真相与类型检查',
          items: [
            { text: '编写 PATCH 端点', link: '/how-to/write-patch-endpoints' },
            { text: '用 basedpyright 做类型检查', link: '/how-to/type-check-with-basedpyright' },
            { text: '检查 partial DTO 的误用', link: '/how-to/check-partial-dtos' },
            { text: '配合 AI 编码助手使用', link: '/how-to/use-with-ai-assistants' },
          ],
        },
        {
          text: 'API 端点与查询',
          items: [
            { text: '集成 FastAPI', link: '/how-to/integrate-with-fastapi' },
            { text: '给列表端点加分页', link: '/how-to/paginate-a-list-endpoint' },
            { text: 'Keyset 游标分页', link: '/how-to/keyset-pagination' },
            { text: '聚合查询', link: '/how-to/aggregate-queries' },
          ],
        },
        {
          text: '数据模型',
          items: [
            { text: '定义 JTI 联表继承模型', link: '/how-to/define-jti-models' },
            { text: '定义 STI 单表继承模型', link: '/how-to/define-sti-models' },
            { text: '配置级联删除', link: '/how-to/configure-cascade-delete' },
            { text: '处理"仍被引用"的删除', link: '/how-to/handle-referenced-deletes' },
            { text: '更多 Mixin', link: '/how-to/extra-mixins' },
          ],
        },
        {
          text: '并发与一致性',
          items: [
            { text: '处理并发更新', link: '/how-to/handle-concurrent-updates' },
            { text: '强制行锁与隔离级别', link: '/how-to/enforce-locking-and-isolation' },
            { text: '防止 MissingGreenlet 错误', link: '/how-to/prevent-missing-greenlet' },
            { text: '长 I/O 期间释放数据库连接', link: '/how-to/release-connection-during-long-io' },
          ],
        },
        {
          text: '性能优化',
          items: [
            { text: '给查询加 Redis 缓存', link: '/how-to/cache-queries' },
          ],
        },
      ],
      '/reference/': [
        {
          text: '参考',
          items: [
            { text: '总览', link: '/reference/' },
            { text: '基础类', link: '/reference/base-classes' },
            { text: 'CRUD 方法', link: '/reference/crud-methods' },
            { text: '字段类型', link: '/reference/field-types' },
            { text: 'Mixin 类', link: '/reference/mixins' },
            { text: '装饰器与辅助函数', link: '/reference/decorators' },
            { text: '分页类型', link: '/reference/pagination-types' },
          ],
        },
      ],
      '/explanation/': [
        {
          text: '讲解',
          items: [
            { text: '设计哲学：单点真相', link: '/explanation/single-source-of-truth' },
            { text: '总览', link: '/explanation/' },
            { text: 'Unset 三态', link: '/explanation/unset-three-state' },
            { text: '前置知识', link: '/explanation/prerequisites' },
            { text: '元类与 SQLModelBase', link: '/explanation/metaclass' },
            { text: 'CRUD 实现', link: '/explanation/crud-pipeline' },
            { text: '多态继承机制', link: '/explanation/polymorphic-internals' },
            { text: '乐观锁机制', link: '/explanation/optimistic-lock' },
            { text: '关系预加载机制', link: '/explanation/relation-preload' },
            { text: '级联删除语义', link: '/explanation/cascade-delete-semantics' },
            { text: 'Redis 缓存机制', link: '/explanation/cached-table' },
            { text: '事务内的缓存透明性', link: '/explanation/transactional-cache-transparency' },
            { text: '静态分析器原理', link: '/explanation/relation-load-checker' },
          ],
        },
      ],
    },

    outline: {
      level: [2, 3],
      label: '目录',
    },

    docFooter: {
      prev: '上一页',
      next: '下一页',
    },
  } satisfies DefaultTheme.Config,
}
