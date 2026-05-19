---
layout: home

hero:
  name: sqlmodel-ext
  text: SQLModel 增强库
  tagline: 定义模型，继承 Mixin，获得完整的异步 CRUD API
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
  - title: 异步 CRUD 一行搞定
    details: save / get / update / delete / count / get_with_count，内置分页、时间过滤、关系预加载
  - title: 丰富的字段类型
    details: Str64、Port、HttpUrl、SafeHttpUrl、IPAddress、Array[T] 等，Pydantic 验证 + SQLAlchemy 列类型一步到位
  - title: 多态继承
    details: 联表继承 (JTI) 和单表继承 (STI) 的零配置支持，自动鉴别列与子类注册
  - title: Redis 查询缓存
    details: CachedTableBaseMixin 提供 ID 缓存 + 查询缓存双层架构，CRUD 时自动失效，支持多态继承
  - title: 安全与可靠
    details: 乐观锁并发控制、@requires_relations 防止 MissingGreenlet、启动时 AST 静态分析
---

## 设计取向

sqlmodel-ext 不是新的 ORM，而是构建在 SQLModel / Pydantic v2 / SQLAlchemy 2.0 之上的一组**可单独取用的 Mixin**。库本身只引入两类东西：

- **元类增强**——根据 `Annotated` 类型注解自动设置 SQLAlchemy 列、合并 `mapper_args`、按需生成 `*UpdateRequest` DTO、保留属性 docstring，并打上 Python 3.14 (PEP 649) 兼容补丁。
- **可插拔 Mixin**——`TableBaseMixin`（异步 CRUD）、`PolymorphicBaseMixin`（JTI/STI）、`OptimisticLockMixin`（版本号并发）、`RelationPreloadMixin`（关系预加载）、`CachedTableBaseMixin`（Redis 双层缓存）。每个 Mixin 都可以独立使用，不互相依赖。

底层 `Session`、`select()`、查询构造、迁移工具仍是原生 SQLAlchemy，没有自创 DSL，也不接管 `engine`/`session` 的生命周期。换句话说：在已有 SQLModel/SQLAlchemy 项目里，可以单独引入需要的 Mixin，而不必重写既有数据访问层。
