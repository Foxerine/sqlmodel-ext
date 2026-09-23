# 用 basedpyright 做类型检查

**目标**：让误用在运行前就被标红。sqlmodel-ext 的设计目标之一，是把"约定"尽量变成"类型"——`Unset` 三态、`get()` 按 `fetch_mode` 重载的返回类型、保留签名的装饰器、`select()` 的多列重载、`max_length_of()` 的返回类型……这些只有在类型检查器跑起来时才发挥全部作用。**配合 basedpyright 使用效果最佳**：库本身以 basedpyright 零 error 为门禁，本页列出的每一条报错都是实际运行得到的。

**前置条件**：Python ≥ 3.12，项目已安装 sqlmodel-ext。

## 1. 安装

```bash
pip install "basedpyright>=1.40.1"
# 或
uv add --dev "basedpyright>=1.40.1"
```

版本下限来自 `Unset`：静态收窄 `Unset | T` 需要支持 PEP 661 的检查器（basedpyright ≥ 1.40.1，对应 pyright ≥ 1.1.414）。更老的版本会把 `Unset` 当成普通变量，收窄失效。

## 2. 推荐配置

在项目根目录放一个 `pyrightconfig.json`（basedpyright 按 JSONC 读取，可以写注释）：

```jsonc
{
  "pythonVersion": "3.12",
  "include": ["app"],

  // SQLAlchemy / Pydantic 的 stub 里 Any 与部分未知类型无处不在（Row 值、model_dump 输出、
  // inspect() 等），逐条报告会淹没真正的诊断。
  "reportAny": false,
  "reportExplicitAny": false,
  "reportUnknownVariableType": false,
  "reportUnknownMemberType": false,
  "reportUnknownArgumentType": false,
  "reportUnknownParameterType": false,

  // redis / alembic / pgvector / numpy 等可选依赖没有完整 stub。
  "reportMissingTypeStubs": false
}
```

其余规则保持 basedpyright 默认（`recommended` 级别）。这组关闭项与 sqlmodel-ext 仓库自身的 `pyrightconfig.json` 一致（仓库那份额外关闭了 `reportImportCycles`、`reportIncompatibleMethodOverride`、`reportIncompatibleVariableOverride`、`reportPrivateLocalImportUsage`，那是库内部 mixin 重新声明上游成员所需；用户的模型代码在实测中不需要）。

在 CI 里以 **0 error** 为门禁：

```bash
basedpyright --level error
```

注意：不带 `--level error` 时，basedpyright 1.40.1 在**只有 warning** 的情况下退出码也是 1（实测）；想让 warning 也挡住合并就去掉这个参数。

::: warning 空扫描也是"0 errors"
`include` 写错（或被 `exclude` 排除）时，basedpyright 输出 `No source files found.` 和 `0 errors, 0 warnings, 0 notes`（1.40.1 实测退出码为 3）。只看摘要行的 CI 脚本会把它当成通过。加 `--verbose` 并断言输出里有 `Found N source files` 且 N > 0。
:::

## 3. 它能抓到的典型误用

以下示例都基于这组模型（`app/models.py`）：

<!-- skip-run -->
```python
from sqlmodel_ext import (
    AsyncSession, NonNegativeInt, SQLModelBase, Str64, UUIDTableBaseMixin, Unset, requires_for_update,
)


class UserBase(SQLModelBase):
    name: Str64
    nickname: Str64 | None = None
    age: NonNegativeInt = 0


class User(UserBase, UUIDTableBaseMixin, table=True):
    @requires_for_update
    async def rename(self, session: AsyncSession, *, name: str) -> None:
        _ = session
        self.name = name


class UserFilter(SQLModelBase):
    min_age: Unset | NonNegativeInt = Unset
    """不传：没有下限。"""
```

每条下面的输出是用上面的推荐配置在 basedpyright 1.40.1 上实际运行得到的（只去掉了文件路径前缀）。

### 3.1 可省略字段没收窄就使用

<!-- skip-run -->
```python
def next_age(f: UserFilter) -> int:
    return f.min_age + 1
```

```text
error: Operator "+" not supported for types "NonNegativeInt | MISSING" and "Literal[1]"
    Operator "+" not supported for types "MISSING" and "Literal[1]" when expected type is "int" (reportOperatorIssue)
```

（`MISSING` 就是 `Unset`——两者是同一个对象。）

### 3.2 用 `is not None` 判断"有没有传"

<!-- skip-run -->
```python
def next_age(f: UserFilter) -> int:
    if f.min_age is not None:
        return f.min_age + 1
    return 0
```

```text
warning: Condition will always evaluate to True since the types "int | MISSING" and "None" have no overlap (reportUnnecessaryComparison)
error: Operator "+" not supported for types "NonNegativeInt | MISSING" and "Literal[1]"
    Operator "+" not supported for types "MISSING" and "Literal[1]" when expected type is "int" (reportOperatorIssue)
warning: Code is unreachable (reportUnreachable)
```

正确写法是 `if f.min_age is Unset: return 0`，之后 `f.min_age` 被收窄为 `int`。

### 3.3 把 `Unset` 赋给不可省略的字段

<!-- skip-run -->
```python
def clear_name(user: User) -> None:
    user.name = Unset
```

```text
error: Cannot assign to attribute "name" for class "User"
    "MISSING" is not assignable to "str" (reportAttributeAccessIssue)
```

### 3.4 `get()` 默认返回 `T | None`

<!-- skip-run -->
```python
async def user_name(session: AsyncSession, user_id: UUID) -> str:
    user = await User.get(session, col(User.id) == user_id)
    return user.name
```

```text
error: "name" is not a known attribute of "None" (reportOptionalMemberAccess)
```

`get()` 按 `fetch_mode` 重载返回类型：`'first'`（默认）→ `User | None`，`'one'` → `User`（查不到时抛错），`'all'` → `list[User]`。

### 3.5 `fetch_mode='all'` 返回的是列表

<!-- skip-run -->
```python
async def all_names(session: AsyncSession) -> list[str]:
    users = await User.get(session, fetch_mode='all')
    return users.name
```

```text
error: Cannot access attribute "name" for class "list[User]"
    Attribute "name" is unknown (reportAttributeAccessIssue)
```

### 3.6 `@requires_for_update` 保留了被装饰方法的签名

<!-- skip-run -->
```python
async def rename(session: AsyncSession, user: User) -> None:
    await user.rename(session, new_name="x")
```

```text
error: Argument missing for parameter "name" (reportCallIssue)
error: No parameter named "new_name" (reportCallIssue)
```

`requires_for_update` 用 `ParamSpec` 标注，装饰后参数仍被检查（`requires_locked_param` / `requires_read_committed` / `requires_repeatable_read` 同理）。

### 3.7 `max_length_of()` 返回 `int`

<!-- skip-run -->
```python
def name_limit() -> str:
    return max_length_of(Str64)
```

```text
error: Type "int" is not assignable to return type "str"
    "int" is not assignable to "str" (reportReturnType)
```

### 3.8 列方法要用 `col()` 包起来

<!-- skip-run -->
```python
async def named(session: AsyncSession) -> list[User]:
    return await User.get(session, User.name.in_(["alice", "bob"]), fetch_mode='all')
```

```text
error: Cannot access attribute "in_" for class "Str64"
    Attribute "in_" is unknown (reportAttributeAccessIssue)
```

在类型层面，模型类属性是它的 Python 值类型（`User.name: Str64`），没有 `.in_()` / `.is_()` / `.asc()` 这些列方法。写成 `col(User.name).in_([...])` 即可。

`select()` 的投影则不需要 `col()`：1–4 列的重载与上游 `sqlmodel.select` 完全一致，`select(User.id, User.name)` 直接推断为 `Select[tuple[UUID, str]]`；5–9 列的裸属性同样可用。唯一的例外是 5 列以上且**混用** SQL 函数表达式（如 `func.count()`）时——此时该表达式的元素类型会被放宽成联合类型，把所有列都包进 `col()` 就能拿到精确类型（见下一节）。

## 4. 正确写法得到的类型

<!-- skip-run -->
```python
def next_age(f: UserFilter) -> int:
    if f.min_age is Unset:
        return 0
    return f.min_age + 1                      # 收窄为 int，无诊断


async def must_get_name(session: AsyncSession, user_id: UUID) -> str:
    user = await User.get(session, col(User.id) == user_id, fetch_mode='one')
    return user.name                          # User，无诊断


async def ages(session: AsyncSession) -> list[int]:
    return await User.distinct_column(session, col(User.age))   # list[int]


def projection() -> None:
    stmt = select(col(User.id), col(User.name), col(User.nickname), col(User.age), col(User.created_at))
    reveal_type(stmt)


async def totals(session: AsyncSession) -> None:
    rows = await User.group_sum(session, [col(User.age)], group_by=col(User.nickname))
    reveal_type(rows)
```

```text
information: Type of "stmt" is "Select[tuple[UUID, str, str | None, int, datetime]]"
information: Type of "rows" is "list[GroupSumRow[str | None]]"
```

`select()` 的重载覆盖到 9 列（上游 SQLModel 止于 4 列）；超过 9 列会报错，而不是静默退化成 `Any`。

## 5. 它**抓不到**的东西

类型检查不是万能的。以下几处在 0.5.0 里需要靠运行时校验或约定兜底，请心里有数：

- **模型构造参数**：SQLModel 的 `__init__` 签名是 `(**data: Any)`，所以 `UserBase(nme="x")`、`UserBase(name=3)` 在 basedpyright 里都**没有**诊断。DTO 在运行时由 Pydantic 校验（`extra='forbid'` 会拒绝拼错的字段名）；而 `table=True` 模型构造时跳过 Pydantic 校验——外部输入应先进 DTO，再构造表模型。
- **`partial=True` 派生的字段**：派生注解在运行时生成，单独运行 basedpyright 时静态层面仍是基类类型（`ArticleUpdate.title: str`），检查器不会强制你收窄；继承来的 validator 在三态下守卫不足也不会被报出。两个办法：把需要收窄的字段在 partial 类体里显式声明为 `Unset | T = Unset`，或者用实验性的 [`check_derived`](./check-partial-dtos) 在临时副本里展开派生类再跑 basedpyright（不改动工作树）。见 [Unset 三态](/explanation/unset-three-state#类型检查器的收窄)。

`sqlmodel_ext.AsyncSession` 的 `execute()` / `exec()` / `scalar()` / `stream()` / `stream_scalars()` 保留上游 SQLModel 的类型签名：`await session.exec(select(User))` 推断为 `ScalarResult[User]`，多列 `select()` 的行类型同样保留。

## 6. 配合 `check_derived`（实验性）

`partial=True` 的盲区可以用 `python -m sqlmodel_ext.check_derived <你的包>` 补上：它把派生 DTO 的运行时三态与继承来的方法展开进一个临时副本，在副本上跑 basedpyright，只报告展开**新增**的错误。在 pre-commit 里它必须**排在所有其他静态检查之前**。完整用法、真实输出与已知盲区见 [检查 partial DTO 的误用](./check-partial-dtos)。

## 常见噪音

- `reportUnannotatedClassAttribute`：在模型方法里给字段赋值（如 `self.name = name`）时，basedpyright 可能要求在非 `@final` 类上重新标注该属性。它不影响正确性，按团队偏好决定是否在配置里关闭。
- `reportUnusedParameter`：`@requires_for_update` 方法需要 `session` 参数来定位锁记录，即使方法体没用到它；写 `_ = session` 或关闭该规则。
