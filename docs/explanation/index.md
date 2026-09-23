# 讲解

> **理解导向。** 讲解告诉你**为什么**——为什么 sqlmodel-ext 要这么设计、为什么某个机制存在、它解决了什么深层问题。它不教你做事（那是 [教程](/tutorials/) 和 [操作指南](/how-to/) 的工作），也不列 API（那是 [参考](/reference/) 的工作）。
> 这部分适合你**已经会用**，但好奇"为什么是这样"的时候阅读。

::: tip 先读这一篇
[设计哲学：单点真相](./single-source-of-truth)——**每个事实只在一个地方声明，其余一切由它派生。** 本库的所有其他设计都从这一句推出。
:::

## 阅读建议

讲解类文档**没有强制顺序**，按兴趣挑选即可。但如果你完全不熟悉异步 SQLAlchemy 的内部机制，建议先看 [前置知识](./prerequisites)。

| 章节 | 难度 | 解答的问题 |
|------|------|----------|
| [设计哲学：单点真相](./single-source-of-truth) | 入门 | 这个库为什么存在？为什么"每个事实只声明一次"在 AI 辅助编码时代尤其重要？ |
| [Unset 三态](./unset-three-state) | 入门 | "没传 / 传了 null / 传了值"为什么必须是三个不同的状态？ |
| [前置知识](./prerequisites) | 入门 | ORM、Session、懒加载、元类、Annotated 类型——这些底层概念是什么？ |
| [元类与 SQLModelBase](./metaclass) | 中等 | 为什么 sqlmodel-ext 需要自定义元类？它在类创建瞬间做了什么？ |
| [CRUD 实现](./crud-pipeline) | 核心 | `save()` / `get()` 内部如何工作？为什么必须用返回值？ |
| [多态继承机制](./polymorphic-internals) | 高级 | JTI 和 STI 在 SQLAlchemy 层面如何实现？两阶段列注册解决了什么问题？ |
| [乐观锁机制](./optimistic-lock) | 中等 | 自动重试如何把"丢失更新"问题转成可重试的冲突？ |
| [关系预加载机制](./relation-preload) | 中等 | `@requires_relations` 如何在不改调用方代码的前提下声明依赖？ |
| [级联删除语义](./cascade-delete-semantics) | 中等 | `cascade_delete` × `passive_deletes` × `ondelete` 的 18 种组合究竟会发生什么？`raise_on_sql` 为什么（通常）不在级联期间触发？ |
| [Redis 缓存机制](./cached-table) | 高级 | 双层缓存（ID + 查询）如何配合自动失效？为什么要 `_cached_ancestors`？ |
| [事务内的缓存透明性](./transactional-cache-transparency) | 高级 | 事务里写了缓存模型，同一事务内的读、回滚、提交各自看到什么？ |
| [静态分析器原理](./relation-load-checker) | 高级 | AST 如何在启动时找出潜在的 MissingGreenlet 问题？ |

## 核心设计哲学

sqlmodel-ext 的所有设计决策都围绕一句话：**每个事实只在一个地方声明，其余一切由它派生。**（完整论述见 [设计哲学：单点真相](./single-source-of-truth)。）具体到实现，就是让用户只声明式地写模型定义，框架在幕后处理 SQLAlchemy 的所有配置细节。

实现这个目标依赖几项关键技术：

| 技术 | 解决的问题 | 详见 |
|------|----------|------|
| 自定义元类 `__DeclarativeMeta` | 自动 `table=True`、JTI/STI 检测、`sa_type` 提取、`__mapper_args__` 合并 | [元类与 SQLModelBase](./metaclass) |
| Mixin 组合模式 | CRUD、乐观锁、缓存、预加载——每个能力独立，按需混入 | 各章节 |
| `__init_subclass__` 钩子 | 导入时验证关系名拼写、自动生成多态 identity | [多态继承](./polymorphic-internals) |
| `__get_pydantic_core_schema__` | 让自定义类型同时满足 Pydantic 验证和 SQLAlchemy 列映射 | [元类与 SQLModelBase](./metaclass) |
| AST 静态分析 | 在请求到达之前发现 MissingGreenlet 隐患 | [静态分析器](./relation-load-checker) |
| 双层缓存 + 版本号失效 | 行级精确缓存 + 模型级 O(1) 失效 | [Redis 缓存](./cached-table) |

## 讲解不是什么

- **不是教程**。讲解不会带你从零写代码。
- **不是 API 参考**。讲解会引用源码片段，但不会列每个参数的完整定义。
- **不一定面向所有用户**。你可以完全不读讲解就把 sqlmodel-ext 用得很好。讲解面向想"打开发动机盖看一看"的人。
