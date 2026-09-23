# 设计哲学：单点真相

> **每个事实只在一个地方声明，其余一切由它派生。**
>
> 字段的约束写在类型里一次——同一份声明同时产出 Pydantic 校验、数据库列类型和 OpenAPI schema；更新 DTO 从表模型派生；"没传 / 传了 null / 传了值"是三个不同的状态，由类型系统区分而不是靠约定。
>
> 这在 AI 辅助编码时代格外重要：AI 最擅长生成"看起来对"的代码，而最常见的错误正是在第二个地方重复声明同一个事实，然后两处慢慢漂移。sqlmodel-ext 让重复声明变得不必要，并让剩下的错误尽量成为类型错误——配合 basedpyright，绝大多数误用在运行前就被标红。仓库附带一套给 AI 编码助手的规则，放进你的项目，让 Claude / Codex 等按正确方式使用本库。
>
> 失败要响亮：非法状态在构造时就被拒绝，而不是在生产环境里被静默兜底。

这一页解释这段话背后的推理，以及本库的每个机制分别消除了哪一处重复声明。

## 漂移是怎么发生的

一个再普通不过的字段——"用户名最长 64 个字符"——在典型的 FastAPI + SQLAlchemy 项目里会被写多少次？

| 位置 | 写法 |
|---|---|
| 数据库列 | `Column(String(64))` |
| 请求 DTO | `name: str = Field(max_length=64)` |
| 更新 DTO | `name: str \| None = Field(default=None, max_length=64)` |
| 业务代码 | `if len(name) > 64: ...`、`name[:64]` |
| API 文档 | "最长 64 个字符" |

五处声明，没有任何机制保证它们一致。某天有人把列改成 128，另外四处原样留着：要么数据库能存、接口却拒收；要么截断了仍然合法的值。**每一处单独看都是对的**，这正是漂移难以被 code review 发现的原因。

## AI 让这个问题变得更严重

AI 编码助手最擅长的，是在局部生成**看起来正确**的代码。让它"加一个修改用户名的接口"，它会照着附近的样子再写一份 `max_length=64`、再写一个 `if body.name is not None`——每一行都合理，合起来就是第六处声明、第二种"没传"的约定。

人会在第三次复制时停下来想"是不是该抽出来"；AI 不会，它只会忠实地延续它看到的模式。所以对策不能是"请更小心"，而是：

1. **让重复声明变得不必要**——事实只有一个自然的声明位置，其余地方从它派生，AI 想重复也无从下笔；
2. **让剩下的错误成为类型错误**——派生不到的地方，用类型把约定固定下来，让检查器而不是审阅者去发现偏差。

## 本库的每个机制消除了哪一处重复

| 事实 | 在 sqlmodel-ext 里声明在哪里（只写一次） | 由它派生的东西 |
|---|---|---|
| 字段长度 / 范围 | 类型别名：`Str64`、`NonNegativeInt`、`Text10K` … | Pydantic 校验 + 数据库列类型（`VARCHAR(64)`、`BIGINT` …）+ OpenAPI `maxLength` / `minimum`；需要数字时 `max_length_of(Str64)` 反射出 `64` |
| 金额精度 | `NonNegativeDecimal38_18` 等 Decimal 别名 | 整数位 / 小数位校验、拒绝 float、`NUMERIC(38, 18)` 列、JSON 定点字符串输出、OpenAPI `string` schema；可汇总的列用 `*WriteDecimal38_18` 写、`SignedSumDecimal38_18` 读，差出的 3 位就是 `SUM()` 的余量 |
| 更新 DTO 的字段集合 | 基类（`ArticleBase`） | `class ArticleUpdate(ArticleBase, partial=True)`——不重写任何字段，约束、描述、别名全部继承 |
| "没传 / null / 值" | 注解：`Unset \| T \| None` | 校验（不可空字段拒绝 null）、序列化（没传的键不输出）、JSON Schema、类型检查器收窄 |
| 字段描述 | 字段下方的 docstring | Pydantic `description` → OpenAPI；派生 DTO 自动继承 |
| 是不是表 | 继承 `TableBaseMixin` / `UUIDTableBaseMixin` | 元类自动加 `table=True` |
| 自定义类型的列类型 | 类型自己的 `__get_pydantic_core_schema__` 元数据（`Array[T]`、`JSON100K`） | 元类提取 `sa_type` 并注入列定义 |
| JSON 列的体积 / 深度上限 | 字段注解 `JSON100K` / `JSONList100K` | 类创建时自动发现这些字段，构造时检查——连跳过 Pydantic 校验的 `table=True` 模型也覆盖 |
| 乐观锁 | 混入 `OptimisticLockMixin` | `oplock_version` 列、`version_id_col` 接线、默认重试 3 次（策略写在声明能力的地方，而不是每个调用点） |
| 缓存 TTL | 类关键字 `cache_ttl=` | `__cache_ttl__` |
| 方法需要行锁 | `@requires_for_update` | 运行时检查调用方确实用 `with_for_update=True` 取得了该实例；静态分析器也读取这个标记 |
| 管理员专属字段 | 一个只含这些字段的模型 | `body.submitted_fields_among(AdminOnlyFields)`——不维护字段名字符串列表 |
| 分页上限 | `MAX_PAGE_SIZE` 等常量 | OpenAPI `maximum`、`offset` 上界（`MAX_TABLE_VIEW_OFFSET` 由 `JS_MAX_SAFE_INTEGER` 推导） |

一个例子把前几行串起来：

```python
from sqlmodel_ext import SQLModelBase, Str64, UUIDTableBaseMixin, max_length_of


class UserBase(SQLModelBase):
    name: Str64
    """用户名。"""


class User(UserBase, UUIDTableBaseMixin, table=True):
    pass


class UserUpdate(UserBase, partial=True):
    pass


assert str(User.__table__.c.name.type) == 'VARCHAR(64)'                       # 数据库列
assert UserBase.model_json_schema()['properties']['name']['maxLength'] == 64   # OpenAPI
assert UserUpdate.model_fields['name'].description == "用户名。"              # 派生 DTO 继承描述
assert max_length_of(Str64) == 64                                              # 业务代码需要数字时
```

把 `Str64` 换成 `Str128`，上面四处一起变。

## 失败要响亮

派生消除了"两处不一致"；剩下的一类错误是"非法状态被悄悄接受"。本库的原则是在**最早能发现的地方**拒绝它，而不是在下游兜底：

| 非法状态 | 在哪里被拒绝 |
|---|---|
| 使用已删除的 `all_fields_optional` | 类创建时 `TypeError`，错误信息里附带迁移方法 |
| `partial=True` 与 `table=True` 同时使用 | 类创建时 `TypeError`（`Unset` 无法入库） |
| 在类体里声明 `oplock_version` | 类创建时 `TypeError`（该名字为乐观锁保留） |
| 不可空字段收到 `null`（PATCH） | 校验时 422，而不是在数据库 NOT NULL 处 500 |
| 给 Decimal 字段传 float | 校验时拒绝（float 已经丢了精度） |
| Decimal 整数位超出列宽 | 校验时拒绝，而不是留给数据库 |
| `PositiveFloat` 收到 `inf` / `nan` | 校验时拒绝 |
| 不带时区的 `datetime` 进入 `TimeFilterRequest` | 校验时拒绝（否则会被静默按数据库时区解释） |
| `after_id` 与非零 `offset` 同时出现 | 校验时拒绝（否则会静默跳过记录） |
| 嵌套过深 / 过大的 JSON 进入 `JSON100K` | 构造时拒绝，而不是在响应序列化时才炸 |
| `@requires_for_update` 找不到 `session` 参数 | 调用时 `RuntimeError`（0.4.x 会静默跳过检查） |
| `max_length_of()` 用在没有长度上限的类型上 | `TypeError`，而不是编一个数字 |

```python
from pydantic import ValidationError
from sqlmodel_ext import SQLModelBase

try:
    class Legacy(SQLModelBase, all_fields_optional=True):
        pass
except TypeError as e:
    assert "partial=True" in str(e)

try:
    UserUpdate.model_validate({'name': None})
except ValidationError:
    pass
else:
    raise AssertionError("null for a non-nullable field must be rejected")
```

## 与 basedpyright 的配合

派生和响亮失败解决了运行时；类型检查器解决"运行前"。sqlmodel-ext 的公开 API 尽量让误用成为类型错误：

- 可省略字段是 `Unset | T`，不收窄就用会报错，用 `is not None` 判断会被提示"条件永远为真"；
- `get()` 按 `fetch_mode` 重载返回类型（`T | None` / `T` / `list[T]`）；
- `@requires_for_update` 等装饰器用 `ParamSpec` 保留签名；
- `select()` 的多列重载覆盖到 9 列，行类型精确；
- `max_length_of()`、`distinct_column()`、`group_sum()` 都有精确的返回类型。

实际报错与推荐配置见 [用 basedpyright 做类型检查](/how-to/type-check-with-basedpyright)。那一页也如实列出了它**抓不到**的地方（例如模型构造参数）——那些地方由运行时校验兜底。

## 给 AI 编码助手的规则

仓库附带一套给 AI 编码助手的规则（见 [配合 AI 编码助手使用](/how-to/use-with-ai-assistants)），把本页的原则写成了助手能直接执行的指令：用 `partial=True` 而不是手写更新 DTO、用 `is Unset` 而不是 `is None` 判断"有没有传"、用类型别名而不是散落的 `max_length=`、用 `max_length_of()` 而不是重复数字……放进你的项目，让 Claude / Codex 等在生成代码时就走派生的路径，而不是再造一处声明。

## 延伸阅读

- [Unset 三态](./unset-three-state) —— 为什么 `None` 不够，以及 `Unset` 的完整语义
- [编写 PATCH 端点](/how-to/write-patch-endpoints) —— 派生 + 三态的端到端示例
- [元类与 SQLModelBase](./metaclass) —— 派生在类创建时具体是怎么发生的
