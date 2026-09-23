# Type-check with basedpyright

**Goal**: have misuse flagged before the code runs. One of sqlmodel-ext's design goals is to turn "conventions" into "types" wherever possible — `Unset` tri-state, `get()` return types overloaded on `fetch_mode`, signature-preserving decorators, multi-column `select()` overloads, the return type of `max_length_of()`… These only pay off fully when a type checker runs. **It works best together with basedpyright**: the library itself is gated on zero basedpyright errors, and every diagnostic on this page was obtained by actually running it.

**Prerequisites**: Python ≥ 3.12, sqlmodel-ext installed in the project.

## 1. Install

```bash
pip install "basedpyright>=1.40.1"
# or
uv add --dev "basedpyright>=1.40.1"
```

The version floor comes from `Unset`: narrowing `Unset | T` statically requires a checker with PEP 661 support (basedpyright ≥ 1.40.1, i.e. pyright ≥ 1.1.414). Older versions treat `Unset` as a plain variable and narrowing does not work.

## 2. Recommended configuration

Put a `pyrightconfig.json` in the project root (basedpyright reads it as JSONC, so comments are allowed):

```jsonc
{
  "pythonVersion": "3.12",
  "include": ["app"],

  // The SQLAlchemy / Pydantic stubs expose Any and partially unknown types everywhere
  // (Row values, model_dump output, inspect(), ...); reporting each one drowns real diagnostics.
  "reportAny": false,
  "reportExplicitAny": false,
  "reportUnknownVariableType": false,
  "reportUnknownMemberType": false,
  "reportUnknownArgumentType": false,
  "reportUnknownParameterType": false,

  // Optional dependencies (redis / alembic / pgvector / numpy) ship without complete stubs.
  "reportMissingTypeStubs": false
}
```

Every other rule keeps basedpyright's default (`recommended` level). These switches match sqlmodel-ext's own repository `pyrightconfig.json` (that file additionally disables `reportImportCycles`, `reportIncompatibleMethodOverride`, `reportIncompatibleVariableOverride` and `reportPrivateLocalImportUsage`, which the library's mixins need to re-declare upstream members; user model code did not need them in our tests).

Gate CI on **0 errors**:

```bash
basedpyright --level error
```

Note: without `--level error`, basedpyright 1.40.1 also exits with status 1 when there are **only warnings** (verified); drop the flag if warnings should block merges too.

::: warning An empty scan is also "0 errors"
If `include` is wrong (or excluded by `exclude`), basedpyright prints `No source files found.` and `0 errors, 0 warnings, 0 notes` (exit status 3 on 1.40.1). A CI script that only looks at the summary line will treat it as a pass. Add `--verbose` and assert that the output contains `Found N source files` with N > 0.
:::

## 3. Typical misuse it catches

All examples use these models (`app/models.py`):

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
    """Omitted: no lower bound."""
```

The output under each item was obtained by running basedpyright 1.40.1 with the recommended configuration above (only the file path prefix was removed).

### 3.1 Using an omissible field without narrowing

<!-- skip-run -->
```python
def next_age(f: UserFilter) -> int:
    return f.min_age + 1
```

```text
error: Operator "+" not supported for types "NonNegativeInt | MISSING" and "Literal[1]"
    Operator "+" not supported for types "MISSING" and "Literal[1]" when expected type is "int" (reportOperatorIssue)
```

(`MISSING` is `Unset` — they are the same object.)

### 3.2 Checking "was it provided" with `is not None`

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

The correct form is `if f.min_age is Unset: return 0`; afterwards `f.min_age` is narrowed to `int`.

### 3.3 Assigning `Unset` to a non-omissible field

<!-- skip-run -->
```python
def clear_name(user: User) -> None:
    user.name = Unset
```

```text
error: Cannot assign to attribute "name" for class "User"
    "MISSING" is not assignable to "str" (reportAttributeAccessIssue)
```

### 3.4 `get()` returns `T | None` by default

<!-- skip-run -->
```python
async def user_name(session: AsyncSession, user_id: UUID) -> str:
    user = await User.get(session, col(User.id) == user_id)
    return user.name
```

```text
error: "name" is not a known attribute of "None" (reportOptionalMemberAccess)
```

`get()` overloads its return type on `fetch_mode`: `'first'` (default) → `User | None`, `'one'` → `User` (raises when nothing is found), `'all'` → `list[User]`.

### 3.5 `fetch_mode='all'` returns a list

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

### 3.6 `@requires_for_update` keeps the decorated method's signature

<!-- skip-run -->
```python
async def rename(session: AsyncSession, user: User) -> None:
    await user.rename(session, new_name="x")
```

```text
error: Argument missing for parameter "name" (reportCallIssue)
error: No parameter named "new_name" (reportCallIssue)
```

`requires_for_update` is typed with `ParamSpec`, so the parameters are still checked after decoration (the same holds for `requires_locked_param` / `requires_read_committed` / `requires_repeatable_read`).

### 3.7 `max_length_of()` returns `int`

<!-- skip-run -->
```python
def name_limit() -> str:
    return max_length_of(Str64)
```

```text
error: Type "int" is not assignable to return type "str"
    "int" is not assignable to "str" (reportReturnType)
```

### 3.8 Wrap columns in `col()` to call column methods

<!-- skip-run -->
```python
async def named(session: AsyncSession) -> list[User]:
    return await User.get(session, User.name.in_(["alice", "bob"]), fetch_mode='all')
```

```text
error: Cannot access attribute "in_" for class "Str64"
    Attribute "in_" is unknown (reportAttributeAccessIssue)
```

At the type level a model class attribute is its Python value type (`User.name: Str64`), which has no column methods such as `.in_()` / `.is_()` / `.asc()`. Write `col(User.name).in_([...])`.

`select()` projections do not need `col()`: for 1–4 columns the overloads are exactly upstream `sqlmodel.select`'s, so `select(User.id, User.name)` is inferred as `Select[tuple[UUID, str]]`; bare attributes work for 5–9 columns as well. The one exception is a projection of 5+ columns that **mixes** in a SQL function expression (such as `func.count()`) — that expression's element type is widened to a union; wrap every column in `col()` to get the precise type (next section).

## 4. Types you get from the correct forms

<!-- skip-run -->
```python
def next_age(f: UserFilter) -> int:
    if f.min_age is Unset:
        return 0
    return f.min_age + 1                      # narrowed to int, no diagnostic


async def must_get_name(session: AsyncSession, user_id: UUID) -> str:
    user = await User.get(session, col(User.id) == user_id, fetch_mode='one')
    return user.name                          # User, no diagnostic


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

The `select()` overloads cover up to 9 columns (upstream SQLModel stops at 4); beyond 9 the checker reports an error instead of silently degrading to `Any`.

## 5. What it does **not** catch

Type checking is not magic. In 0.5.0 the following rely on runtime validation or conventions; be aware of them:

- **Model constructor arguments**: SQLModel's `__init__` signature is `(**data: Any)`, so `UserBase(nme="x")` and `UserBase(name=3)` produce **no** basedpyright diagnostic. DTOs are validated by Pydantic at runtime (`extra='forbid'` rejects misspelled field names); `table=True` models skip Pydantic validation on construction — put external input through a DTO first, then build the table model.
- **Fields derived by `partial=True`**: the derived annotations are generated at runtime, so with basedpyright alone they keep the base type statically (`ArticleUpdate.title: str`) and the checker does not force narrowing; an inherited validator whose guard is insufficient under the tri-state is not reported either. Two remedies: declare the fields that need narrowing explicitly as `Unset | T = Unset` in the partial class body, or use the experimental [`check_derived`](./check-partial-dtos), which expands the derived classes in a temporary copy and runs basedpyright there (the working tree is not touched). See [Unset](/en/explanation/unset-three-state#narrowing-in-the-type-checker).

The `execute()` / `exec()` / `scalar()` / `stream()` / `stream_scalars()` methods of `sqlmodel_ext.AsyncSession` keep upstream SQLModel's type signatures: `await session.exec(select(User))` is inferred as `ScalarResult[User]`, and multi-column `select()` row types are kept as well.

## 6. Together with `check_derived` (experimental)

The `partial=True` blind spot can be closed with `python -m sqlmodel_ext.check_derived <your package>`: it expands the runtime tri-state of derived DTOs and their inherited methods into a temporary copy, runs basedpyright on the copy, and reports only the errors the expansion **introduced**. In pre-commit it must run **before every other static check**. Full usage, real output and known blind spots: [Check partial DTOs for misuse](./check-partial-dtos).

## Common noise

- `reportUnannotatedClassAttribute`: assigning a field inside a model method (e.g. `self.name = name`) may make basedpyright ask for the attribute to be re-annotated on a non-`@final` class. It does not affect correctness; disable it in the configuration if your team prefers.
- `reportUnusedParameter`: a `@requires_for_update` method needs its `session` parameter to locate the lock records even if the body does not use it; write `_ = session` or disable the rule.
