# 基础类

::: tip
本页是参考文档。要看怎么用这些类构建模型，去 [教程](/tutorials/01-getting-started) 或 [操作指南](/how-to/)。
:::

## `SQLModelBase`

```python
from sqlmodel_ext import SQLModelBase
```

所有 sqlmodel-ext 模型的根类。继承自 `SQLModel`，使用自定义元类 `__DeclarativeMeta`。

**`model_config`**（类型为 `SQLModelExtConfig`）：

| 键 | 值 | 说明 |
|---|---|---|
| `use_attribute_docstrings` | `True` | 字段下方的 `"""..."""` 自动作为字段描述 |
| `validate_by_name` | `True` | 允许通过字段名验证（即使有 alias） |
| `extra` | `'forbid'` | 传入未定义的字段会抛 `ValidationError` |
| `omitted_sentinel` | 未设置（= 关闭） | 见下方 `SQLModelExtConfig` |

**方法**：

```python
@staticmethod
def annotation_is_omissible(annotation: Any) -> bool
```

注解是否接受 `Unset`（即字段能否不传）。递归穿过 `Annotated` 与嵌套联合。判断依据是注解，不是默认值。

```python
@classmethod
def field_is_omissible(cls, field_name: str) -> bool
```

`annotation_is_omissible` 的按字段名版本。未知字段名返回 `False`（不抛错）。

```python
def submitted_fields_among(self, *models: type[SQLModelBase]) -> set[str]
```

返回 `self.model_fields_set` 与给定模型字段名并集的交集——"这次请求显式提交了其中哪些字段"。纯集合运算，命中意味着什么由调用方决定（典型用法：管理员专属字段的权限检查）。

```python
@classmethod
def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]
```

与 Pydantic 相同；当模型开启 `omitted_sentinel` 时，额外给每个可省略字段（及通过字段可达的嵌套模型）加上 `{"const": "__omitted__", "type": "string"}` 分支和 `"default": "__omitted__"`。

```python
@classmethod
def get_computed_field_names(cls) -> set[str]
```

返回所有 `@computed_field` 字段的名称集合。

```python
@classmethod
def validate_list(cls, items: Sequence[Any]) -> list[Self]
```

批量验证：把 ORM 实例 / dict 序列逐项 `model_validate(..., from_attributes=True)` 为当前模型类型，返回列表。常用于"查询结果列表 → 响应 DTO 列表"的转换。

**构造期检查**：`JSON100K` / `JSONList100K` 字段在类创建时被自动发现（`__orjson_checked_fields__`），并在 `model_post_init` 中检查可编码性与 100K 上限——`table=True` 模型跳过 Pydantic 校验，这道检查依然生效。**覆写 `model_post_init` 的子类必须调用 `super().model_post_init(context)`**，否则检查被静默跳过。

**其他修正**：字段类型是枚举 / 嵌套模型时，Pydantic 有时会生成不带 `description` 的 `$ref` 属性，`SQLModelBase` 会补回描述；子类覆盖字段但没写 docstring 时，描述从父类继承。

**常用类定义关键字参数**（由元类处理，详见[元类机制](/explanation/metaclass)）：

| 关键字 | 说明 |
|--------|------|
| `table_name` | 自定义表名（等价 `__tablename__`） |
| `table_args` | 表级约束/索引元组（等价 `__table_args__`；其中的 `CustomTableArg` 子类实例会被拦截延迟处理，如 `DeferredIndex`） |
| `mapper_args` | SQLAlchemy mapper 参数 dict（等价 `__mapper_args__`） |
| `polymorphic_on` / `polymorphic_identity` / `polymorphic_abstract` / `version_id_col` / `concrete` | mapper 参数的顶级快捷形式 |
| `cache_ttl` | Redis 缓存 TTL 秒数（正整数；仅 `CachedTableBaseMixin` 子类有效） |
| `partial` | `True` 时把继承字段变成可省略：`T` → `Unset \| T = Unset`，`T \| None` → `Unset \| T \| None = Unset`。约束、描述、字段级属性保留；本类自己声明的字段与 `Literal` 字段跳过。不能与 `table=True` 同用。详见 [Unset 三态](/explanation/unset-three-state) |
| `abstract` | 标记抽象类（等价 `__abstract__`） |

::: warning 已删除：`all_fields_optional`
0.5.0 起使用 `all_fields_optional` 会在类创建时抛 `TypeError`。改用 `partial=True`，见 [迁移到 0.5](/how-to/migrate-to-0-5)。
:::

**保留字段名**：`oplock_version` 为 `OptimisticLockMixin` 保留，任何 `SQLModelBase` 子类在自己的类体里声明它都会在类创建时抛 `TypeError`。

**继承场景**：

- `class XxxBase(SQLModelBase)` — 纯数据模型（不建表），用于 API 输入/输出
- `class Xxx(XxxBase, TableBaseMixin, table=True)` — 建表模型
- `class XxxUpdate(XxxBase, partial=True)` — PATCH 请求体

`sqlmodel_ext.base.optional_dto_registry: list[type]` 按创建顺序记录所有 `partial=True` 类，供契约测试等工具枚举。

## `SQLModelExtConfig`

```python
from sqlmodel_ext import SQLModelExtConfig
```

`SQLModelConfig`（Pydantic `ConfigDict`）加上 sqlmodel-ext 自己的配置键。`total=False` 的 TypedDict：缺省的键表示"使用默认值"，上游所有键都可用。

| 键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `omitted_sentinel` | `bool` | 关闭 | 为可省略字段提供可填写的线上哨兵值 `'__omitted__'`：入站 dict 中任意深度的该值在校验前被替换为 `Unset`；`model_json_schema()` 给可省略字段加上哨兵分支。供"无法省略键"的调用方使用（如 LLM 严格模式 function calling）。开启后该模型的字符串字段不能持有字面量 `'__omitted__'` |

```python
from sqlmodel_ext import SQLModelBase, SQLModelExtConfig


class ToolArguments(SQLModelBase):
    model_config = SQLModelExtConfig(omitted_sentinel=True)
```

## `Unset` 与 `OMITTED_SENTINEL`

```python
from sqlmodel_ext import Unset, OMITTED_SENTINEL
```

- `Unset`：Pydantic 官方的 `pydantic.experimental.missing_sentinel.MISSING`，表示"这个字段没有提供"。用 `x is Unset` 判断。值为 `Unset` 的字段不出现在 `model_dump()` / `model_dump_json()` 输出中。`copy.deepcopy(Unset) is Unset`。
- `OMITTED_SENTINEL`：`'__omitted__'`，线上哨兵值，只存在于开启 `omitted_sentinel` 的模型的 JSON Schema 与入站负载中，进门即被归一为 `Unset`。业务代码不应与它比较。

完整语义见 [Unset 三态](/explanation/unset-three-state)。

## `ExtraIgnoreModelBase`

```python
from sqlmodel_ext import ExtraIgnoreModelBase
```

继承自 `SQLModelBase`，但 `extra='ignore'`：未知字段被静默忽略，同时记录 WARNING 日志。

**`model_config`**：

| 键 | 值 | 说明 |
|---|---|---|
| `use_attribute_docstrings` | `True` | 同 `SQLModelBase` |
| `validate_by_name` | `True` | 同 `SQLModelBase` |
| `extra` | `'ignore'` | 未知字段被忽略（不报错） |

**校验器**：

```python
@model_validator(mode='before')
@classmethod
def _warn_unknown_fields(cls, data: Any) -> Any
```

如果输入是 dict 且包含未声明的字段，记录 WARNING 日志（包含模型名、未知字段数与最多 5 个样例字段名）。字段名、`alias`、`validation_alias`（字符串，或 `AliasChoices` 中的每个字符串选项）都算已知字段；`AliasPath` 是嵌套路径而不是顶层键，不计入。

**适用场景**：第三方 API 响应、外部 WebSocket 消息、JSON schema 可能变化的输入。

## `TableBaseMixin`

```python
from sqlmodel_ext import TableBaseMixin
```

为模型添加自增整数主键和 CRUD 方法。

**继承自**：`AsyncAttrs`（提供 `await obj.awaitable_attrs.xxx` 语法）。

**类属性**：

<!-- skip-run -->
```python
_has_table_mixin: ClassVar[bool] = True
```

让元类识别"这是 table 类"，自动添加 `table=True`。

<!-- skip-run -->
```python
__optimistic_retry_default__: ClassVar[int] = 0
```

`save()` / `update()` 在未传 `optimistic_retry_count`（`None`）时使用的重试次数。基类为 0；`OptimisticLockMixin` 覆盖为 3。

**字段**：

| 字段 | 类型 | 数据库行为 |
|------|------|----------|
| `id` | `int \| None` | 主键，自动生成（`SERIAL` / `INTEGER PRIMARY KEY`） |
| `created_at` | `datetime` | `TIMESTAMP WITH TIME ZONE`，创建时由 `default_factory=now` 设置 |
| `updated_at` | `datetime` | `TIMESTAMP WITH TIME ZONE`，`onupdate=now`；`save()` / `update()` 在有改动时显式赋值（联表继承只改子表列时也会推进） |

**方法**：CRUD 方法签名见 [CRUD 方法](./crud-methods)。

## `UUIDTableBaseMixin`

```python
from sqlmodel_ext import UUIDTableBaseMixin
```

`TableBaseMixin` 的 UUID 主键变体。

**字段**：

| 字段 | 类型 | 数据库行为 |
|------|------|----------|
| `id` | `uuid.UUID` | 主键，`default_factory=uuid7`（UUIDv7，时间有序） |
| `created_at` | `datetime` | 同 `TableBaseMixin` |
| `updated_at` | `datetime` | 同 `TableBaseMixin` |

UUIDv7 的前 48 位是毫秒级 Unix 时间戳：`ORDER BY id` 近似创建顺序，B-tree 插入集中在右侧。生成函数是 `sqlmodel_ext.mixins.uuid7`（Python 3.14+ 上即标准库 `uuid.uuid7`，更早版本使用符合 RFC 9562 的内置实现）。

已知限制：ID 以毫秒精度暴露创建时间（ID 是标识符不是凭证）；ID 顺序不是权威时间顺序；旧行保留原有 UUID 版本，v4/v7 混合仍是良定义的全序；需要确定性派生的主键请显式赋值。

**类型精确的 override**：

`UUIDTableBaseMixin` 重载了 `get_one()` / `get_exist_one()`，参数 `id` 类型为 `uuid.UUID`（而非 `int`）。

## `RecordNotFoundError`

```python
from sqlmodel_ext import RecordNotFoundError
```

未安装 FastAPI 时，`get_exist_one()` 找不到记录抛出此异常（`status_code = 404`，`detail` 为传入的 `detail` 参数，默认 `"Not found"`）。安装 FastAPI 时改为抛 `HTTPException(404)`。

**判定逻辑**：`sqlmodel_ext.mixins.table` 模块导入时尝试 `from fastapi import HTTPException`，成功则使用它。
