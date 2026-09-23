# Design philosophy: a single source of truth

> **Every fact is declared in exactly one place; everything else is derived from it.**
>
> A field's constraints are written once, in its type — that single declaration produces the Pydantic validation, the database column type and the OpenAPI schema. Update DTOs are derived from the table model. "Not sent", "sent as null" and "sent a value" are three different states, told apart by the type system instead of by convention.
>
> This matters even more in the age of AI-assisted coding. AI is best at producing code that *looks* right, and its most common mistake is restating a fact in a second place — after which the two copies slowly drift apart. sqlmodel-ext makes the restatement unnecessary, and turns as many of the remaining mistakes as possible into type errors: with basedpyright, most misuse is flagged before the code ever runs. The repository ships a rule set for AI coding assistants — drop it into your project and Claude, Codex and friends will use the library the intended way.
>
> Fail loudly: illegal states are rejected at construction time instead of being silently patched over in production.

This page explains the reasoning behind that statement, and which duplicated declaration each mechanism of the library removes.

## How drift happens

Take a completely ordinary field — "a user name is at most 64 characters". How many times is it written in a typical FastAPI + SQLAlchemy project?

| Place | Spelling |
|---|---|
| Database column | `Column(String(64))` |
| Request DTO | `name: str = Field(max_length=64)` |
| Update DTO | `name: str \| None = Field(default=None, max_length=64)` |
| Business code | `if len(name) > 64: ...`, `name[:64]` |
| API docs | "at most 64 characters" |

Five declarations and nothing keeping them consistent. One day someone changes the column to 128 and leaves the other four alone: either the database can store a value the API rejects, or valid values get truncated. **Each place is correct on its own**, which is exactly why drift slips through code review.

## AI makes it worse

What an AI coding assistant does best is generate locally **plausible** code. Ask it to "add an endpoint to rename a user" and it will copy the nearby pattern: another `max_length=64`, another `if body.name is not None` — every line reasonable, together a sixth declaration and a second convention for "not provided".

A person stops at the third copy and asks "shouldn't this be factored out?"; an AI does not — it faithfully continues the pattern it sees. So the remedy cannot be "please be more careful". It has to be:

1. **Make the second declaration unnecessary** — each fact has one natural place to be declared and everything else derives from it, so there is nothing for the AI to copy;
2. **Turn what is left into type errors** — where derivation cannot reach, pin the convention down in a type so that a checker, not a reviewer, finds the deviation.

## Which duplication each mechanism removes

| Fact | Where sqlmodel-ext declares it (once) | What is derived from it |
|---|---|---|
| Field length / range | Type aliases: `Str64`, `NonNegativeInt`, `Text10K`, … | Pydantic validation + database column type (`VARCHAR(64)`, `BIGINT`, …) + OpenAPI `maxLength` / `minimum`; when code needs the number, `max_length_of(Str64)` reflects `64` |
| Monetary precision | Decimal aliases such as `NonNegativeDecimal38_18` | integer/fractional digit validation, float rejection, `NUMERIC(38, 18)` column, fixed-point JSON string output, OpenAPI `string` schema; summable columns are written through `*WriteDecimal38_18` and read through `SignedSumDecimal38_18`, the 3-digit difference being the headroom for `SUM()` |
| The field set of an update DTO | The base class (`ArticleBase`) | `class ArticleUpdate(ArticleBase, partial=True)` — no field re-declared; constraints, descriptions and aliases are inherited |
| "Not provided / null / a value" | The annotation: `Unset \| T \| None` | validation (null rejected for non-nullable fields), serialization (omitted keys are not output), JSON Schema, type-checker narrowing |
| Field description | The docstring below the field | Pydantic `description` → OpenAPI; inherited by derived DTOs |
| Whether it is a table | Inheriting `TableBaseMixin` / `UUIDTableBaseMixin` | the metaclass adds `table=True` |
| Column type of a custom type | The type's own `__get_pydantic_core_schema__` metadata (`Array[T]`, `JSON100K`) | the metaclass extracts `sa_type` and injects it into the column definition |
| Size / depth limit of a JSON column | The field annotation `JSON100K` / `JSONList100K` | such fields are discovered at class creation and checked at construction — including `table=True` models that skip Pydantic validation |
| Optimistic locking | Mixing in `OptimisticLockMixin` | the `oplock_version` column, the `version_id_col` wiring, 3 retries by default (the policy lives where the capability is declared, not at every call site) |
| Cache TTL | The class keyword `cache_ttl=` | `__cache_ttl__` |
| A method needs a row lock | `@requires_for_update` | a runtime check that the caller really obtained the instance with `with_for_update=True`; the static analyzer reads the same marker |
| Admin-only fields | A model containing only those fields | `body.submitted_fields_among(AdminOnlyFields)` — no list of field-name strings |
| Pagination limits | Constants such as `MAX_PAGE_SIZE` | OpenAPI `maximum`, the `offset` bound (`MAX_TABLE_VIEW_OFFSET` derived from `JS_MAX_SAFE_INTEGER`) |

One example tying the first rows together:

```python
from sqlmodel_ext import SQLModelBase, Str64, UUIDTableBaseMixin, max_length_of


class UserBase(SQLModelBase):
    name: Str64
    """User name."""


class User(UserBase, UUIDTableBaseMixin, table=True):
    pass


class UserUpdate(UserBase, partial=True):
    pass


assert str(User.__table__.c.name.type) == 'VARCHAR(64)'                       # database column
assert UserBase.model_json_schema()['properties']['name']['maxLength'] == 64   # OpenAPI
assert UserUpdate.model_fields['name'].description == "User name."            # derived DTO inherits the description
assert max_length_of(Str64) == 64                                              # when business code needs the number
```

Change `Str64` to `Str128` and all four change together.

## Fail loudly

Derivation removes "two places disagree"; the remaining class of bugs is "an invalid state is quietly accepted". The library's rule is to reject it at the **earliest point where it can be detected**, not to patch it over downstream:

| Invalid state | Where it is rejected |
|---|---|
| Using the removed `all_fields_optional` | `TypeError` at class creation, with migration instructions in the message |
| `partial=True` together with `table=True` | `TypeError` at class creation (`Unset` cannot be stored) |
| Declaring `oplock_version` in a class body | `TypeError` at class creation (the name is reserved for optimistic locking) |
| `null` for a non-nullable field (PATCH) | 422 at validation, instead of a 500 at the database NOT NULL constraint |
| A float for a Decimal field | rejected at validation (a float has already lost precision) |
| Decimal integer digits wider than the column | rejected at validation, not left to the database |
| `inf` / `nan` for `PositiveFloat` | rejected at validation |
| A naive `datetime` in `TimeFilterRequest` | rejected at validation (it would otherwise be silently interpreted in the database's timezone) |
| `after_id` together with a non-zero `offset` | rejected at validation (it would otherwise silently skip records) |
| Too deeply nested / too large JSON in `JSON100K` | rejected at construction, not when the response is serialized |
| `@requires_for_update` cannot find the `session` argument | `RuntimeError` at call time (0.4.x silently skipped the check) |
| `max_length_of()` on a type without a length bound | `TypeError` instead of inventing a number |

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

## Working with basedpyright

Derivation and loud failures cover runtime; a type checker covers "before runtime". sqlmodel-ext's public API tries to make misuse a type error:

- omissible fields are `Unset | T`; using one without narrowing is an error, and checking it with `is not None` is flagged as "condition will always evaluate to True";
- `get()` overloads its return type on `fetch_mode` (`T | None` / `T` / `list[T]`);
- decorators such as `@requires_for_update` keep the signature through `ParamSpec`;
- the multi-column `select()` overloads cover up to 9 columns with precise row types;
- `max_length_of()`, `distinct_column()` and `group_sum()` all have precise return types.

Real diagnostics and a recommended configuration are in [Type-check with basedpyright](/en/how-to/type-check-with-basedpyright). That page also lists honestly what it does **not** catch (e.g. model constructor arguments) — those places are covered by runtime validation.

## Rules for AI coding assistants

The repository ships a set of rules for AI coding assistants (see [Use with AI coding assistants](/en/how-to/use-with-ai-assistants)) that turns the principles on this page into instructions an assistant can follow: use `partial=True` instead of hand-written update DTOs, `is Unset` instead of `is None` for "was it provided", type aliases instead of scattered `max_length=`, `max_length_of()` instead of repeating the number… Put them in your project so Claude, Codex and friends take the derived path when generating code instead of creating another declaration.

## Further reading

- [Unset](./unset-three-state) — why `None` is not enough, and the full semantics of `Unset`
- [Write PATCH endpoints](/en/how-to/write-patch-endpoints) — derivation plus tri-state, end to end
- [Metaclass & SQLModelBase](./metaclass) — how derivation actually happens at class creation
