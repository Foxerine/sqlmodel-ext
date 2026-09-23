# Unset: "not provided", "null" and "a value"

::: tip Source location
`src/sqlmodel_ext/unset.py` — `Unset`, `OMITTED_SENTINEL` and the shape constants of the wire-sentinel protocol

`src/sqlmodel_ext/base.py` — `partial=True` (`_apply_partial`), `SQLModelExtConfig`, `SQLModelBase.annotation_is_omissible` / `field_is_omissible` / `submitted_fields_among` / `model_json_schema`
:::

`Unset` is one of the core features of sqlmodel-ext 0.5. It solves a problem that almost every PATCH endpoint runs into but rarely admits: **`None` is used to say two opposite things**.

## Why `None` is not enough

```python
from sqlmodel_ext import SQLModelBase


class ArticleUpdate(SQLModelBase):
    subtitle: str | None = None
```

When you read `body.subtitle is None`, it may mean:

| What the caller meant | Request body | What the server should do |
|---|---|---|
| "I don't intend to change this field" | `{}` | **nothing** |
| "Clear the subtitle" | `{"subtitle": null}` | **write NULL** |

Same spelling, opposite handling. The traditional fixes are all **conventions**:

- the caller agrees to "omit what you don't change" and the server filters with `model_dump(exclude_unset=True)` — but the convention breaks as soon as one piece of code reads `body.subtitle` directly;
- or everyone agrees "`None` always means don't change" — and then "clear" can no longer be expressed.

Conventions live in people's heads. The one time someone forgets, nothing raises.

## The fix: move "not provided" onto a carrier that is not `None`

```python
from sqlmodel_ext import Unset
```

`Unset` **is** Pydantic's official `pydantic.experimental.missing_sentinel.MISSING` (a PEP 661-style sentinel); sqlmodel-ext only exports it under another name. Once "not provided" is carried by `Unset`, `None` goes back to being an **ordinary value**.

| Annotation | Meaning |
|---|---|
| `Unset \| T = Unset` | may be omitted; `null` is rejected |
| `Unset \| T \| None = Unset` | may be omitted; `null` is a real value (e.g. "clear this column") |
| `T \| None` | must be provided; may be `null` |
| `T = <default>` | may be omitted, has a natural default; `null` is rejected |

Whether the annotation includes `None` depends on one thing only: **is `null` a meaningful value for this field** ("clear this column" on write, "filter rows where the column IS NULL" on read) — **not** on whether the field may be left out. Omission is carried by `Unset`.

::: info Why `Unset` and not `MISSING`
"missing" suggests something that *should* be present but is absent, whereas here the caller *chose* not to provide the field. The name also appears in type position (`Unset | T`), where an ALL_CAPS constant reads oddly. Import `Unset` from `sqlmodel_ext` rather than `MISSING` from Pydantic so a codebase has a single name for the concept.
:::

## Runtime behavior of the three states

```python
from pydantic import ValidationError
from sqlmodel_ext import NonNegativeInt, SQLModelBase, Str64, Unset


class ArticleBase(SQLModelBase):
    title: Str64
    """Article title."""
    subtitle: Str64 | None = None
    """Subtitle; null clears it."""
    views: NonNegativeInt = 0
    """View count."""


class ArticleUpdate(ArticleBase, partial=True):
    pass


empty = ArticleUpdate()
assert empty.title is Unset
assert empty.model_dump() == {}                    # omitted fields do not appear
assert empty.model_dump_json() == '{}'

cleared = ArticleUpdate(subtitle=None)
assert cleared.model_dump() == {'subtitle': None}  # explicit null is kept

try:
    ArticleUpdate.model_validate_json('{"title": null}')   # title is not nullable in the base
except ValidationError:
    pass
else:
    raise AssertionError("title=null must be rejected")
```

- **Not provided** → the value is `Unset`, and the **whole key is absent** from `model_dump()` / `model_dump_json()`. No `exclude_unset=True` needed — that is guaranteed by Pydantic's `MISSING` itself.
- **`null`** → accepted only where the base field allows `None`; the value is `None` and it is serialized.
- **A value** → validated as usual (length, range and every other constraint is kept).

::: warning Do not rely on the order of errors
`Unset | T` is a union, so a failed validation lists one error per branch in `ValidationError.errors()` (for `title=null` above you get both `missing_sentinel_error` and `string_type`, with branch tags such as `'missing-sentinel'` / `'constrained-str'` in `loc`). The order of union members does not change **which** inputs are accepted or rejected, but it does change the **order** of error entries; do not depend on that order downstream.
:::

## `partial=True`: derive a PATCH DTO from the base

You can write `Unset | T = Unset` by hand, but an update DTO is usually "every base field becomes omissible". `partial=True` means you **re-declare no field at all**:

```python
class ArticleUpdate(ArticleBase, partial=True):
    pass
```

The metaclass transforms every inherited field:

- base `T` → `Unset | T` (`null` rejected)
- base `T | None` → `Unset | T | None` (`null` is a real value)
- the default becomes `Unset`; a `default_factory` is not called (an omitted list field is `Unset`, not `[]`)
- **constraints are preserved**: `Unset` is validated by pydantic-core's dedicated missing-sentinel branch and never reaches constraint validators, so `Unset | Annotated[int, Field(ge=0)]` needs no special nesting
- field-level attributes (`exclude` / `alias` / `validation_alias` / `serialization_alias` / `repr` / `frozen`) are hoisted outside the union — otherwise Pydantic would silently drop them. They are read from the base's resolved field, so `Annotated[T, Field(...)]`, `x: T = Field(...)` and mixes of the two all work. Aliases produced by an `alias_generator` are not hoisted: as in plain inheritance, a derived class regenerates them with its own generator
- `discriminator` stays on the union member, where a nested tagged union works as usual
- field descriptions (docstrings) are inherited from the base

Skipped fields:

- fields the class declares in its own body — the author's declaration wins;
- `Literal` fields (e.g. discriminators) — `Unset | Literal[...]` would break discriminated unions.

Two hard limits, each raising `TypeError` **at class creation**:

- `partial=True` cannot be combined with `table=True` — `Unset` cannot be stored;
- the old keyword `all_fields_optional` was removed; using it raises a `TypeError` with migration instructions (see [Migrate to 0.5](/en/how-to/migrate-to-0-5)).

Every class created with `partial=True` is recorded, in creation order, in `sqlmodel_ext.base.optional_dto_registry`, which makes contract tests easy (e.g. assert that every PATCH DTO builds from `{}` and dumps to `{}`).

## Telling the three states apart in code

```python
from sqlmodel_ext import NonNegativeInt, SQLModelBase, Unset


class ArticleFilter(SQLModelBase):
    owner_id: Unset | int | None = Unset
    """Omitted: no filter; null: articles without an owner; int: that owner's articles."""
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

**Check "was it provided" with `is Unset` / `is not Unset`**, never with `is None` — that is exactly the ambiguity this feature removes.

Most of the time you do not even need per-field checks: hand the whole partial DTO to `instance.update(session, body)` — omitted fields never appear in `model_dump()`, and an explicit `null` is written as `NULL`. For a full end-to-end example see [Write PATCH endpoints](/en/how-to/write-patch-endpoints).

## Generic checks: `annotation_is_omissible` / `field_is_omissible`

Under tri-state semantics, "may this field be left out" and "may this field be `null`" are **two orthogonal questions**. sqlmodel-ext answers the first one on the model:

```python
assert ArticleFilter.field_is_omissible('owner_id') is True
assert ArticleBase.field_is_omissible('title') is False
assert ArticleUpdate.field_is_omissible('title') is True
assert ArticleFilter.field_is_omissible('no_such_field') is False   # unknown names return False, no error

annotation = ArticleFilter.model_fields['owner_id'].annotation
assert SQLModelBase.annotation_is_omissible(annotation) is True
```

- `annotation_is_omissible(annotation)` is a `staticmethod`: the criterion is whether the **annotation** contains `Unset` (searched recursively through `Annotated` and nested unions), not whether the default happens to be `Unset` — "can it be omitted" is answered by the type, "what do I get when it is omitted" by the default.
- `field_is_omissible(field_name)` is its by-field-name variant.

## Permission checks on a set of fields: `submitted_fields_among`

In one update body, some fields may only be changed by admins. Instead of maintaining a list of field-name strings, declare those fields as a model and intersect:

```python
class ArticleAdminOnlyFields(SQLModelBase):
    is_featured: bool = False


class ArticleAdminUpdate(ArticleAdminOnlyFields, ArticleBase, partial=True):
    pass


body = ArticleAdminUpdate.model_validate({'title': 'x', 'is_featured': True})
assert body.submitted_fields_among(ArticleAdminOnlyFields) == {'is_featured'}
assert ArticleAdminUpdate.model_validate({'title': 'x'}).submitted_fields_among(ArticleAdminOnlyFields) == set()
```

It is pure set arithmetic: `self.model_fields_set` ∩ the field names of the given models. What a hit means (403? ignore?) is up to the caller.

## Serialization and JSON Schema

- `model_dump()` / `model_dump_json()`: fields whose value is `Unset` **do not appear**.
- JSON Schema: omissible fields are not in `required`, and **by default no sentinel branch appears** — to a REST client it is just an optional field; a non-nullable field's schema does not contain `null` either.

## The wire sentinel: for callers that cannot omit keys (per model)

`MISSING` assumes the caller **can omit keys**, so its branch never appears in the JSON Schema. Some schema consumers, however, require **every key to be present** (for example LLM function calling in strict mode). Under the official behavior such a caller cannot say "leave this field alone": the schema only offers `T`, and `null` means "clear". Omission and clearing collapse into one.

Whether a caller can omit keys is a property of the **caller**, and callers are distinguished by **model** (a REST body vs. a tool-call argument model), not by field. Hence the switch is a `model_config` key, **off by default**:

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
assert 'anyOf' not in ArticleUpdate.model_json_schema()['properties']['title']   # the REST model is unaffected
```

When it is on:

- every occurrence of `'__omitted__'` in an inbound dict payload, at any nesting depth, is replaced by `Unset` before validation;
- `model_json_schema()` adds a `{"const": "__omitted__", "type": "string"}` branch and `"default": "__omitted__"` to every omissible field (including the `$defs` entries of nested models reachable from the fields, and a self-referencing model itself); annotation keywords such as `title` / `description` stay on the outer object.

The same domain DTO therefore yields a clean schema when inherited by a REST model and a sentinel-aware schema when inherited by a strict-mode tool model. Three invariants:

1. Fields whose value is `Unset` never appear in `model_dump()` / `model_dump_json()` output.
2. The wire sentinel only exists in JSON Schema and inbound payloads and is normalized to `Unset` on entry; application code should never compare against `'__omitted__'`.
3. `Unset` and `None` are different things and never substitute for each other.

Cost: with the switch on, string fields of that model can no longer hold the literal `'__omitted__'` itself.

## Deep copies and FastAPI defaults

`typing_extensions.Sentinel` does not support `copy.deepcopy` natively (its `__getstate__` raises). Frameworks deep-copy defaults — FastAPI does so for missing query parameters — so a bare `Unset` default would fail at request time. sqlmodel-ext registers a `copyreg` reducer that makes `copy.deepcopy(Unset)` return `Unset` itself (it only acts on the `Unset` object; other sentinels keep the upstream behavior; a real `pickle` round-trip still fails — sentinels should not be revived across processes).

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

An `Unset` default works the same way in request-body models and in `Annotated[Model, Depends()]` query-parameter models.

## Narrowing in the type checker

`Unset` is annotated `typing.Final`, so it works both in type position (`Unset | T`) and in value position (`= Unset`). Narrowing `Unset | T` statically requires a type checker with PEP 661 support: **pyright ≥ 1.1.414 / basedpyright ≥ 1.40.1**.

```python
def next_limit(f: ArticleFilter) -> int:
    if f.limit is Unset:
        return 50
    return f.limit + 1       # f.limit is narrowed to int here
```

For **hand-written** `Unset | T` fields basedpyright forces you to narrow first: `f.limit + 1` without a check reports `reportOperatorIssue`; checking with `is not None` is flagged as "condition will always evaluate to True" and the error remains (real output in [Type-check with basedpyright](/en/how-to/type-check-with-basedpyright)).

::: warning Fields derived by `partial=True` have the base type statically
The `partial=True` annotations are generated by the metaclass **at runtime**; static type checkers still see the base class annotations: to basedpyright, `ArticleUpdate.title` is `str`, not `Unset | str`. So with basedpyright alone the checker does **not** force narrowing on partial DTOs, and a validator inherited from the base whose guard is insufficient under the tri-state is not reported either. The rules are therefore:

- prefer handing the whole DTO to `update()` / `model_dump()` instead of reading fields one by one;
- when you must read a field, always use `is Unset` / `is not Unset` (fully correct at runtime), never `is None`;
- for fields where you want the checker to force narrowing (e.g. query filters), write `Unset | T = Unset` by hand;
- or use the experimental [`check_derived`](/en/how-to/check-partial-dtos): it expands the derived classes' tri-state fields and inherited methods in a temporary copy, runs basedpyright there, and reports only new errors, without touching the working tree.
:::

## `EXCLUDE_IF_NONE` vs. `Unset`

Both make a key disappear from the output, but they answer different questions:

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
| Meaning | "when the value is `None`, do not output the key" | "this field was not provided" |
| Can `None` be output | No — `None` and "absent" merge on the wire | Yes — an explicit `null` is output as usual |
| Typical use | Adding a nullable field to a structure that older consumers deserialize strictly (`extra='forbid'`): during a rolling deployment a new producer emitting `"new_field": null` would make old consumers reject the whole message; with the marker the key is simply absent | PATCH bodies, query filters — where "not provided" and "null" must be told apart |
| Required shape | `Annotated[T \| None, EXCLUDE_IF_NONE] = None`, all three parts (without the `= None` default the dumped JSON cannot be read back) | `Unset \| T [\| None] = Unset` |

In one sentence: **`EXCLUDE_IF_NONE` is an output-format convention; `Unset` is a third state of input semantics.**

## Known limitations

- Pydantic still marks the feature experimental (module `pydantic.experimental.missing_sentinel`, available since Pydantic 2.12), hence sqlmodel-ext 0.5 requires `pydantic>=2.12`.
- Static narrowing requires PEP 661 support (versions above); with basedpyright alone, partial-derived fields do not take part in static narrowing — declare them explicitly or use [`check_derived`](/en/how-to/check-partial-dtos) (see above).
- In models with `omitted_sentinel` enabled, string fields cannot hold the literal `'__omitted__'`.
