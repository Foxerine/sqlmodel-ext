---
layout: home

hero:
  name: sqlmodel-ext
  text: SQLModel 增强库
  tagline: 每个事实只声明一次——校验、列类型、OpenAPI 与 PATCH DTO 都由它派生
  actions:
    - theme: brand
      text: 教程
      link: /tutorials/
    - theme: alt
      text: 操作指南
      link: /how-to/
    - theme: alt
      text: 参考
      link: /reference/
    - theme: alt
      text: 讲解
      link: /explanation/
    - theme: alt
      text: GitHub
      link: https://github.com/Foxerine/sqlmodel-ext

features:
  - title: 单点真相与类型
    details: "Unset 三态字段与 partial=True PATCH DTO；约束类型别名（Str64、Port、Decimal 等）一次同时驱动校验、列类型和 OpenAPI，max_length_of() 反推上限而不是重复数字；select() 类型重载到 9 列"
  - title: CRUD 与查询
    details: "统一的 get(condition) 与带类型的 fetch_mode 重载，count(distinct_column=) / distinct_column() / group_sum()，after_id keyset 分页，外键限制删除时抛 ResourceReferencedError，UUIDv7 主键，JTI / STI 多态"
  - title: 并发与事务
    details: "总是刷新的 with_for_update、skip_locked 工作队列、fail-closed 的 @requires_for_update 与隔离级别装饰器、带 40001 重试的 run_in_repeatable_read、post-commit 回调，基于 oplock_version 的乐观锁默认重试 3 次"
  - title: 缓存与关系
    details: "事务透明、从不发布未提交状态的两级 Redis 缓存，@requires_relations 与批量关系预加载，防 MissingGreenlet 的 RelationLoadChecker 静态分析（RLC001–RLC014）"
  - title: 为 basedpyright 而生
    details: "约束与字段三态都在类型里，basedpyright（≥ 1.40.1）能在运行前标出几乎所有误用"
  - title: 给 AI 编码助手的规则
    details: "仓库附带 AGENTS.md（CLAUDE.md 导入它，规则只有一份），让 Claude、Codex 等按正确方式使用本库"
---

::: warning 开发中
sqlmodel-ext 正在积极开发中。API 可能在版本之间发生不兼容变更，且不提供任何稳定性或向后兼容性保证。请自行承担使用风险。从 0.4.x 升级？请阅读 [0.5.0 迁移指南](/how-to/migrate-to-0-5)。
:::

## 设计哲学

> **每个事实只在一个地方声明，其余一切由它派生。**
>
> 字段的约束写在类型里一次——同一份声明同时产出 Pydantic 校验、数据库列类型和 OpenAPI schema；更新 DTO 从表模型派生；"没传 / 传了 null / 传了值"是三个不同的状态，由类型系统区分而不是靠约定。
>
> 这在 AI 辅助编码时代格外重要：AI 最擅长生成"看起来对"的代码，而最常见的错误正是在第二个地方重复声明同一个事实，然后两处慢慢漂移。sqlmodel-ext 让重复声明变得不必要，并让剩下的错误尽量成为类型错误——配合 basedpyright，绝大多数误用在运行前就被标红。仓库附带一套给 AI 编码助手的规则，放进你的项目，让 Claude / Codex 等按正确方式使用本库。
>
> 失败要响亮：非法状态在构造时就被拒绝，而不是在生产环境里被静默兜底。

```python
class ArticleBase(SQLModelBase):
    title: Str64                   # 一次声明：校验 + VARCHAR(64) + OpenAPI maxLength
    summary: Str64 | None = None   # 这里 null 是真实值

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    pass                           # UUIDv7 id、时间戳、异步 CRUD

class ArticleUpdate(ArticleBase, partial=True):
    pass                           # 每个字段：Unset | T = Unset

# PATCH {"summary": null} -> 只写 `summary`；{"title": null} -> 422
article = await article.update(session, ArticleUpdate.model_validate(payload))
```

## 设计取向

sqlmodel-ext 不是新的 ORM，而是构建在 SQLModel / Pydantic v2 / SQLAlchemy 2.0 之上的一组**可单独取用的 Mixin**。库本身只引入两类东西：

- **元类增强**——根据 `Annotated` 类型注解自动设置 SQLAlchemy 列、合并 `mapper_args`、生成字段为 `Unset` 的 `partial=True` PATCH DTO、保留属性 docstring，并打上 Python 3.14 (PEP 649) 兼容补丁。
- **可插拔 Mixin**——`TableBaseMixin`（异步 CRUD）、`PolymorphicBaseMixin`（JTI/STI）、`OptimisticLockMixin`（版本号并发）、`RelationPreloadMixin`（关系预加载）、`CachedTableBaseMixin`（Redis 双层缓存）。每个 Mixin 都可以独立使用，不互相依赖。

底层 `select()`、查询构造、迁移工具仍是原生 SQLAlchemy，没有自创 DSL，也不接管 `engine` 的生命周期。唯一的 session 层增强是 `sqlmodel_ext.AsyncSession` 子类（感知缓存的 commit、post-commit 回调、锁与隔离级别跟踪；用 `async_sessionmaker(class_=AsyncSession)` 接入即可）。换句话说：在已有 SQLModel/SQLAlchemy 项目里，可以单独引入需要的 Mixin，而不必重写既有数据访问层。
