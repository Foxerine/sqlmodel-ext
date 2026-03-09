---
layout: home

hero:
  name: sqlmodel-ext
  text: SQLModel 增强库
  tagline: 定义模型，继承 Mixin，获得完整的异步 CRUD API
  actions:
    - theme: brand
      text: 使用指南
      link: /guide/
    - theme: alt
      text: 实现原理
      link: /internals/
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
