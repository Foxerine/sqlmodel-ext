# 操作指南

> **任务导向。** 操作指南帮你完成**一个具体的目标**。它假设你已经会用基础 API（看过 [教程](/tutorials/) 或本身就熟悉异步 SQLModel），现在面对一个具体问题需要直接的步骤。
> 每篇指南都是一个"如何完成 X"的菜谱：列出前置条件、给出步骤、提示常见陷阱。

## 按主题浏览

### API 端点

- [给列表端点加分页](./paginate-a-list-endpoint) — `TableViewRequest` + `ListResponse[T]`
- [集成 FastAPI](./integrate-with-fastapi) — 5 种端点的标准写法（GET/POST/PATCH/DELETE/LIST）

### 数据模型

- [定义 JTI 联表继承模型](./define-jti-models) — 子类字段差异大时用
- [定义 STI 单表继承模型](./define-sti-models) — 子类只多 1~2 个字段时用
- [配置级联删除](./configure-cascade-delete) — `cascade_delete` / `passive_deletes` / `ondelete` 三件套怎么搭

### 并发与一致性

- [处理并发更新](./handle-concurrent-updates) — 用 `OptimisticLockMixin` 防止丢失更新
- [防止 MissingGreenlet 错误](./prevent-missing-greenlet) — `@requires_relations` + `lazy='raise_on_sql'` + 静态分析三道防线
- [长 I/O 期间释放数据库连接](./release-connection-during-long-io) — 用 `session.reset()` 防连接池被外部 I/O 拖死

### 性能优化

- [给查询加 Redis 缓存](./cache-queries) — `CachedTableBaseMixin` + `configure_redis()`

## 操作指南不是什么

- **不是教程**。指南假设你已经知道这个库的基本范式。如果你看到 `await User.save(session)` 不知道是什么意思，先去看 [教程](/tutorials/)。
- **不是参考**。指南只列**完成任务必需的参数**，不会展开所有可选参数。完整签名去 [参考](/reference/) 查。
- **不解释为什么**。如果你想知道"为什么 sqlmodel-ext 用这种方式实现"，去 [讲解](/explanation/)。

## 找不到你要的指南？

如果你的任务不在列表中，可能：

1. **它是教程级别的**（"如何创建第一个模型"）→ 去 [教程](/tutorials/)
2. **它是 reference 级别的**（"`save()` 的所有参数"）→ 去 [参考](/reference/)
3. **它是新场景** → 欢迎在 [GitHub Issues](https://github.com/Foxerine/sqlmodel-ext/issues) 提议新增指南
