# Unset 三态：「没传」「传了 null」「传了值」

::: tip 源码位置
`src/sqlmodel_ext/unset.py` — `Unset`、`OMITTED_SENTINEL` 以及线上哨兵协议的形状常量

`src/sqlmodel_ext/base.py` — `partial=True`（`_apply_partial`）、`SQLModelExtConfig`、`SQLModelBase.annotation_is_omissible` / `field_is_omissible` / `submitted_fields_among` / `model_json_schema`
:::

`Unset` 是 sqlmodel-ext 0.5 的核心特性之一。它解决的是一个几乎每个 PATCH 端点都会踩、却很少被正面承认的问题：**`None` 同时被用来表达两件相反的事**。

## 为什么 `None` 不够

```python
from sqlmodel_ext import SQLModelBase


class ArticleUpdate(SQLModelBase):
    subtitle: str | None = None
```

读到 `body.subtitle is None` 时，它可能是：

| 调用方的意图 | 请求体 | 服务端应该做什么 |
|---|---|---|
| "这个字段我没打算改" | `{}` | **什么都不做** |
| "把副标题清空" | `{"subtitle": null}` | **写入 NULL** |

同一个拼写，两个相反的处理方式。传统解法都是**约定**：

- 调用方约定"不改就别传"，服务端用 `model_dump(exclude_unset=True)` 过滤——但只要有一处代码直接读 `body.subtitle`，约定就失效了；
- 或者约定"`None` 一律表示不改"——那么"清空"就再也无法表达。

约定是靠人记住的。忘记的那一次，没有任何报错。

## 解法：把「没传」搬到一个不是 `None` 的载体上

```python
from sqlmodel_ext import Unset
```

`Unset` 就是 Pydantic 官方的 `pydantic.experimental.missing_sentinel.MISSING`（PEP 661 风格的哨兵），sqlmodel-ext 只是给它换了个名字导出。「没传」由 `Unset` 承载之后，`None` 重新变回一个**普通的值**。

| 注解 | 含义 |
|---|---|
| `Unset \| T = Unset` | 可以不传；`null` 被拒绝 |
| `Unset \| T \| None = Unset` | 可以不传；`null` 是一个真实的值（例如"清空这一列"） |
| `T \| None` | 必须传；可以是 `null` |
| `T = <默认值>` | 可以不传，有自然的默认值；`null` 被拒绝 |

判断注解里要不要带 `None`，只看一件事：**`null` 对这个字段是不是一个有意义的值**（写入时"清空这一列"、查询时"筛选这一列 IS NULL"）——而**不是**看它能不能省略。能不能省略，由 `Unset` 表达。

::: info 为什么叫 `Unset` 而不是 `MISSING`
"missing" 暗示"本应存在却缺失了"，而这里是调用方**主动选择**不提供。它还会出现在类型位置（`Unset | T`），全大写的常量名放在那里读起来别扭。请从 `sqlmodel_ext` 导入 `Unset`，而不是从 Pydantic 导入 `MISSING`，让一个代码库对同一个概念只有一个名字。
:::

## 三个状态的运行时行为

```python
from pydantic import ValidationError
from sqlmodel_ext import NonNegativeInt, SQLModelBase, Str64, Unset


class ArticleBase(SQLModelBase):
    title: Str64
    """文章标题。"""
    subtitle: Str64 | None = None
    """副标题；null 表示清空。"""
    views: NonNegativeInt = 0
    """浏览量。"""


class ArticleUpdate(ArticleBase, partial=True):
    pass


empty = ArticleUpdate()
assert empty.title is Unset
assert empty.model_dump() == {}                    # 没传的字段不出现
assert empty.model_dump_json() == '{}'

cleared = ArticleUpdate(subtitle=None)
assert cleared.model_dump() == {'subtitle': None}  # 显式 null 保留

try:
    ArticleUpdate.model_validate_json('{"title": null}')   # title 在基类里不可空
except ValidationError:
    pass
else:
    raise AssertionError("title=null must be rejected")
```

- **没传** → 字段值是 `Unset`，并且在 `model_dump()` / `model_dump_json()` 的输出里**整个键都不出现**。不再需要 `exclude_unset=True`——这是 Pydantic `MISSING` 自身的保证。
- **传了 `null`** → 只有基类字段本身允许 `None` 时才接受，值就是 `None`，并且会被序列化出来。
- **传了值** → 照常校验（长度、范围等约束全部保留）。

::: warning 错误列表的顺序不要依赖
`Unset | T` 是一个联合类型，校验失败时 `ValidationError.errors()` 里会有每个分支各自的错误（例如上面 `title=null` 会同时出现 `missing_sentinel_error` 与 `string_type`，`loc` 里带有分支标签如 `'missing-sentinel'` / `'constrained-str'`）。成员的顺序不影响**接受/拒绝哪些输入**，但会影响错误条目的**顺序**；下游不要依赖这个顺序。
:::

## `partial=True`：从基类派生 PATCH DTO

手写 `Unset | T = Unset` 当然可以，但更新 DTO 通常就是"基类的每个字段都变成可省略"。`partial=True` 让你**不再重复声明任何字段**：

```python
class ArticleUpdate(ArticleBase, partial=True):
    pass
```

元类对每个继承来的字段做如下变换：

- 基类 `T` → `Unset | T`（`null` 被拒绝）
- 基类 `T | None` → `Unset | T | None`（`null` 是真实的值）
- 默认值统一变成 `Unset`；`default_factory` 不会被调用（没传的列表字段是 `Unset`，不是 `[]`）
- **约束原样保留**：`Unset` 由 pydantic-core 专门的 missing-sentinel 分支校验，永远不会进入约束校验器，所以 `Unset | Annotated[int, Field(ge=0)]` 不需要任何特殊嵌套
- 字段级属性（`exclude` / `alias` / `validation_alias` / `serialization_alias` / `repr` / `frozen`）会被提升到联合类型外层——否则 Pydantic 会静默丢弃它们。这些属性从基类已解析的字段读取，所以 `Annotated[T, Field(...)]`、`x: T = Field(...)` 以及两者混用都有效。`alias_generator` 生成的别名不提升：与普通继承一样，派生类会用自己的生成器重新生成
- `discriminator` 留在联合成员内部，嵌套的判别联合照常工作
- 字段描述（docstring）会从基类继承

跳过的字段：

- 本类自己在类体里声明的字段——作者的声明优先；
- `Literal` 字段（例如判别字段）——`Unset | Literal[...]` 会破坏 discriminated union。

两条硬限制，违反时在**类创建时**直接 `TypeError`：

- `partial=True` 不能和 `table=True` 同时使用——`Unset` 无法存进数据库；
- 旧关键字 `all_fields_optional` 已删除，使用它会抛出带迁移说明的 `TypeError`（见 [迁移到 0.5](/how-to/migrate-to-0-5)）。

所有用 `partial=True` 创建的类按创建顺序登记在 `sqlmodel_ext.base.optional_dto_registry` 里，方便写契约测试（例如断言每个 PATCH DTO 都能从 `{}` 构造并 dump 成 `{}`）。

## 在代码里区分三个状态

```python
from sqlmodel_ext import NonNegativeInt, SQLModelBase, Unset


class ArticleFilter(SQLModelBase):
    owner_id: Unset | int | None = Unset
    """不传：不过滤；null：没有作者的文章；整数：该作者的文章。"""
    limit: Unset | NonNegativeInt = Unset


def describe(f: ArticleFilter) -> str:
    if f.owner_id is Unset:
        return "no filter"
    if f.owner_id is None:
        return "owner IS NULL"
    return f"owner = {f.owner_id}"


assert describe(ArticleFilter()) == "no filter"
assert describe(ArticleFilter.model_validate({'owner_id': None})) == "owner IS NULL"
assert describe(ArticleFilter(owner_id=7)) == "owner = 7"
```

**用 `is Unset` / `is not Unset` 判断「有没有传」**，永远不要用 `is None`——那正是本特性要消灭的歧义。

多数时候你甚至不需要逐字段判断：把整个 partial DTO 交给 `instance.update(session, body)` 即可——没传的字段根本不会出现在 `model_dump()` 里，显式 `null` 会写成 `NULL`。完整端到端示例见 [编写 PATCH 端点](/how-to/write-patch-endpoints)。

## 通用判断：`annotation_is_omissible` / `field_is_omissible`

"这个字段能不能不传"与"这个字段能不能为 `null`"在三态语义下是**两个正交的问题**。sqlmodel-ext 在模型上提供了前者的判断：

```python
assert ArticleFilter.field_is_omissible('owner_id') is True
assert ArticleBase.field_is_omissible('title') is False
assert ArticleUpdate.field_is_omissible('title') is True
assert ArticleFilter.field_is_omissible('no_such_field') is False   # 未知字段返回 False，不抛错

annotation = ArticleFilter.model_fields['owner_id'].annotation
assert SQLModelBase.annotation_is_omissible(annotation) is True
```

- `annotation_is_omissible(annotation)` 是 `staticmethod`：判断依据是**注解**里是否含有 `Unset`（会递归穿过 `Annotated` 和嵌套联合），而不是默认值碰巧是不是 `Unset`——"能不能省略"由类型回答，"省略时得到什么"由默认值回答。
- `field_is_omissible(field_name)` 是它的按字段名版本。

## 按字段集合做权限判断：`submitted_fields_among`

同一个更新体里，有些字段只有管理员能改。与其维护一份字段名字符串列表，不如把这些字段声明成一个模型，再求交集：

```python
class ArticleAdminOnlyFields(SQLModelBase):
    is_featured: bool = False


class ArticleAdminUpdate(ArticleAdminOnlyFields, ArticleBase, partial=True):
    pass


body = ArticleAdminUpdate.model_validate({'title': 'x', 'is_featured': True})
assert body.submitted_fields_among(ArticleAdminOnlyFields) == {'is_featured'}
assert ArticleAdminUpdate.model_validate({'title': 'x'}).submitted_fields_among(ArticleAdminOnlyFields) == set()
```

它是纯集合运算：`self.model_fields_set` ∩ 给定模型的字段名。命中之后意味着什么（403？忽略？）由调用方决定。

## 序列化与 JSON Schema

- `model_dump()` / `model_dump_json()`：值为 `Unset` 的字段**不出现**。
- JSON Schema：可省略的字段不在 `required` 里，并且**默认不出现任何哨兵分支**——对 REST 客户端来说，它就是一个普通的可选字段；不可空的字段 schema 里也不会出现 `null`。

## 线上哨兵：给「无法省略键」的调用方（按模型开启）

`MISSING` 假设调用方**可以省略键**，因此 JSON Schema 里没有它的分支。但有些 schema 的消费方要求**每个键都必须出现**（例如 LLM 严格模式的 function calling）。在官方行为下，这样的调用方没有办法说"别动这个字段"：schema 只给出 `T`，而 `null` 意味着"清空"——省略与清空坍缩成了一件事。

"能不能省略键"是**调用方**的性质，而调用方是按**模型**区分的（REST 请求体 vs 工具调用参数模型），不是按字段。所以开关是一个 `model_config` 键，**默认关闭**：

```python
from sqlmodel_ext import OMITTED_SENTINEL, SQLModelExtConfig, Unset


class ArticleUpdateTool(ArticleUpdate):
    model_config = SQLModelExtConfig(omitted_sentinel=True)


assert OMITTED_SENTINEL == '__omitted__'
tool_args = ArticleUpdateTool.model_validate({'title': '__omitted__', 'subtitle': None, 'views': '__omitted__'})
assert tool_args.title is Unset
assert tool_args.model_dump() == {'subtitle': None}

title_schema = ArticleUpdateTool.model_json_schema()['properties']['title']
assert {'const': '__omitted__', 'type': 'string'} in title_schema['anyOf']
assert title_schema['default'] == '__omitted__'
assert 'anyOf' not in ArticleUpdate.model_json_schema()['properties']['title']   # REST 模型不受影响
```

开启后：

- 入站的 dict 负载里，任意嵌套深度出现的 `'__omitted__'` 都会在校验前被替换成 `Unset`；
- `model_json_schema()` 给每个可省略字段（包括通过字段可达的嵌套模型在 `$defs` 里的条目、以及自引用模型本身）加上 `{"const": "__omitted__", "type": "string"}` 分支和 `"default": "__omitted__"`；`title` / `description` 等注释性关键字保留在外层。

于是同一个领域 DTO，被 REST 模型继承时得到干净的 schema，被严格模式的工具模型继承时得到带哨兵的 schema。三条不变式：

1. 值为 `Unset` 的字段永远不出现在 `model_dump()` / `model_dump_json()` 的输出里。
2. 线上哨兵只存在于 JSON Schema 与入站负载，进门即被归一为 `Unset`；业务代码永远不应该和 `'__omitted__'` 比较。
3. `Unset` 和 `None` 是两样东西，永远不互相替代。

代价：开启后，该模型的字符串字段不能再持有字面量 `'__omitted__'` 本身。

## 深拷贝与 FastAPI 默认值

`typing_extensions.Sentinel` 原生不支持 `copy.deepcopy`（它的 `__getstate__` 会抛错）。而框架会深拷贝默认值——FastAPI 对缺失的查询参数就会这么做——于是一个裸的 `Unset` 默认值会在请求时失败。sqlmodel-ext 注册了一个 `copyreg` reducer，让 `copy.deepcopy(Unset)` 返回 `Unset` 本身（只作用于 `Unset` 这一个对象，其他哨兵保持上游行为；真正的 `pickle` 往返仍然失败——哨兵不应该跨进程复活）。

```python
import copy
from typing import Annotated

from fastapi import FastAPI, Query

assert copy.deepcopy(Unset) is Unset

app = FastAPI()


@app.get("/articles")
async def list_articles(limit: Annotated[Unset | int, Query()] = Unset) -> dict[str, str]:
    return {'limit': 'omitted' if limit is Unset else str(limit)}
```

`Unset` 默认值同样可以用在请求体模型和 `Annotated[Model, Depends()]` 查询参数模型里。

## 类型检查器的收窄

`Unset` 用 `typing.Final` 注解，因此它既能出现在类型位置（`Unset | T`），也能出现在值位置（`= Unset`）。静态收窄 `Unset | T` 需要支持 PEP 661 的类型检查器：**pyright ≥ 1.1.414 / basedpyright ≥ 1.40.1**。

```python
def next_limit(f: ArticleFilter) -> int:
    if f.limit is Unset:
        return 50
    return f.limit + 1       # 这里 f.limit 已被收窄为 int
```

对**手写**的 `Unset | T` 字段，basedpyright 会强制你先收窄：忘了判断直接 `f.limit + 1` 会报 `reportOperatorIssue`；用 `is not None` 判断则会被提示"条件永远为真"，并且错误依然存在（真实输出见 [用 basedpyright 做类型检查](/how-to/type-check-with-basedpyright)）。

::: warning `partial=True` 派生的字段在静态层面是基类类型
`partial=True` 的注解是在**运行时**由元类生成的，静态类型检查器看到的仍然是基类的注解：`ArticleUpdate.title` 在 basedpyright 眼里是 `str`，而不是 `Unset | str`。所以单独运行 basedpyright 时，检查器**不会**替你强制收窄；从基类继承来的 validator 在三态下守卫不足，同样不会被报出。规则因此是：

- 优先把整个 DTO 交给 `update()` / `model_dump()`，不逐字段读；
- 必须逐字段读时，一律用 `is Unset` / `is not Unset`（运行时完全正确），不要用 `is None`；
- 需要检查器强制收窄的字段（例如查询过滤条件），手写 `Unset | T = Unset`；
- 或者用实验性的 [`check_derived`](/how-to/check-partial-dtos)：它在临时副本里把派生类的三态字段与继承来的方法展开，再跑 basedpyright，只报告新增错误，不改动工作树。
:::

## `EXCLUDE_IF_NONE` 与 `Unset` 的区别

两者都会让键从输出里消失，但回答的是不同的问题：

```python
from typing import Annotated

from sqlmodel_ext import EXCLUDE_IF_NONE


class Event(SQLModelBase):
    marker: Annotated[bool | None, EXCLUDE_IF_NONE] = None
    note: Unset | str | None = Unset


assert Event().model_dump() == {}
assert Event(marker=None, note=None).model_dump() == {'note': None}
assert Event(marker=True).model_dump() == {'marker': True}
```

| | `EXCLUDE_IF_NONE` | `Unset` |
|---|---|---|
| 表达的意思 | "值是 `None` 时不要输出这个键" | "没有提供这个字段" |
| `None` 能否被输出 | 不能——`None` 与"缺席"在线上合并 | 能——显式 `null` 照常输出 |
| 典型场景 | 给老版本消费方（`extra='forbid'` 严格反序列化）的结构新增可空字段：滚动部署期间新生产者发出 `"new_field": null` 会让旧消费者拒收整条消息，加上标记后键干脆不出现 | PATCH 请求体、查询过滤条件——需要区分"没传"和"传了 null" |
| 写法要求 | `Annotated[T \| None, EXCLUDE_IF_NONE] = None` 三件套缺一不可（没有 `= None` 默认值，dump 出去的 JSON 读不回来） | `Unset \| T [\| None] = Unset` |

一句话：**`EXCLUDE_IF_NONE` 是输出格式的约定，`Unset` 是输入语义的第三个状态。**

## 已知限制

- Pydantic 仍把该特性标为 experimental（模块路径 `pydantic.experimental.missing_sentinel`，自 Pydantic 2.12 起提供），因此 sqlmodel-ext 0.5 要求 `pydantic>=2.12`。
- 静态收窄需要 PEP 661 支持（见上文版本要求）；partial 派生字段在单独运行 basedpyright 时不参与静态收窄，需显式声明或用 [`check_derived`](/how-to/check-partial-dtos)（见上文）。
- 开启 `omitted_sentinel` 的模型，字符串字段不能持有字面量 `'__omitted__'`。
