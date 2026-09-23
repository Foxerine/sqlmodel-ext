# sqlmodel-ext: rules for AI coding assistants

These rules apply to every file that defines SQLModel models / DTOs or calls
their query, save, lock or cache methods. They are written for AI coding
assistants (Claude Code, Codex, Copilot, Cursor, ...) and for the humans who
review their output. Every API named here exists in `sqlmodel_ext`; if a rule
seems to require an API you cannot find, stop and ask instead of inventing one.

## The one idea behind all rules

**Every fact is declared in exactly one place; everything else is derived from it.**
A field's constraint is written once, in its type, and that one declaration
produces Pydantic validation, the database column type and the OpenAPI schema.
Update DTOs are derived from the model. "Not sent", "sent null" and "sent a
value" are three different states, told apart by the type system rather than by
convention. Failures are loud: an illegal state is rejected when it is
constructed, not silently patched over in production.

The most common AI mistake is writing a correct-looking second declaration of a
fact that already exists -- a re-declared field, a repeated `64`, a hand-written
PATCH DTO -- which then drifts from the first. Before adding anything, look for
the existing declaration and derive from it.

## Workflow (mandatory)

1. Read the model and its base class before touching a field; confirm every
   attribute you use exists.
2. After every change run `basedpyright` on the changed files and fix it to
   **0 errors**. Most misuse of this library is a type error (see the table at
   the end); a red checker means the code is wrong, not the checker.
3. If the project defines `partial=True` DTOs, also run
   `python -m sqlmodel_ext.check_derived <package>` (experimental) after every
   change and fix what it reports as blocking. It expands the derived DTOs in a
   throwaway copy and runs basedpyright there, catching `is not None` checks
   and inherited validators that plain basedpyright cannot see. The tool itself
   never writes into the working tree; do not try to "apply" its expansion to
   the source. It does import the target modules, so run it only on
   import-safe modules (no side effects at import time).
4. Never write `# type: ignore`. `# pyright: ignore[ruleName]  # <reason>` is
   allowed only for a genuine defect in a third-party stub, with the rule name
   and the reason on the same line.

---

## 1. Model layout: one Base, everything inherits

- Put the fields shared by the table and **all** of its DTOs on a `XxxBase(SQLModelBase)`.
  The table is `class Xxx(XxxBase, UUIDTableBaseMixin, table=True)`; create /
  response / update DTOs inherit `XxxBase` too. A field that only some DTOs
  expose (e.g. admin-only) goes on the table class or a mixin, never on the Base.
- Data carriers (DTOs, method results) inherit `SQLModelBase`, not
  `@dataclass`, `TypedDict` or `NamedTuple` -- one validation and serialization stack.
- Import `Field` / `Relationship` from `sqlmodel`, never from `pydantic`:
  `pydantic.Field(foreign_key=...)` silently stores `foreign_key` in
  `json_schema_extra` and no foreign key is created.
- Describe fields with a docstring under the field (becomes the schema
  `description`), not `Field(description=...)`.
- Only use `Field(...)` when there is more than a default; `x: int = 3` otherwise.
- Do not give a default to a field every creator must decide (prices, owners,
  quotas): a default turns "forgot to pass it" into a silent wrong value.

```python
# DO
class ProductBase(SQLModelBase):
    name: Str64
    """Display name."""
    price: NonNegativeWriteDecimal38_18
    """Unit price."""

class Product(ProductBase, UUIDTableBaseMixin, table=True):
    internal_note: Text1K | None = None
    """Admin-only; therefore not on ProductBase."""

class ProductCreate(ProductBase): ...
class ProductUpdate(ProductBase, partial=True): ...

# DON'T: a second, drifting declaration
class ProductCreate(SQLModelBase):
    name: str = Field(max_length=64)
    price: float
```

## 2. Constraints live in type aliases

- Use the library aliases (`Str64`, `Text1K`, `NonEmptyStrippedStr64`,
  `NonNegativeInt`, `Port`, `NonNegativeWriteDecimal38_18`, `HttpUrl`, ...)
  instead of bare `str` / `int` or `Field(max_length=...)`. One alias yields
  validation, column type and schema.
- For a one-off constraint write `Annotated[int, Field(ge=1, le=10)]` once, on the Base.
- When code needs the number, ask the type: `max_length_of(Str64)` -- never a
  literal `64` or a second constant.
- A nullable constrained field is simply `Str64 | None`. But when the
  `Annotated` carries field-level attributes such as `alias`, put `| None`
  inside: `Annotated[str | None, Field(alias='x')]`. The form
  `Annotated[str, Field(alias='x')] | None` silently drops the alias.

```python
# DO
title = raw_title[:max_length_of(Str64)]
# DON'T
MAX_TITLE = 64
title = raw_title[:MAX_TITLE]
```

## 3. PATCH = `partial=True` + `Unset` (the three states)

- Derive every update DTO with `class XxxUpdate(XxxBase, partial=True)`. Each
  inherited field becomes `Unset | T = Unset`; nullability is kept exactly as
  declared on the Base (`T` rejects `null`, `T | None` accepts it as a real value).
- Decide "was this field submitted?" with `is Unset` / `is not Unset`. Never
  with `is None` / `is not None` -- `None` is a legitimate value ("clear it").
- Do not pass or add `exclude_unset=True`: `Unset` fields never appear in
  `model_dump()` / `model_dump_json()`. Apply the patch with
  `obj = await obj.update(session, patch)`.
- Put `| None` on a field only when `null` has a meaning for it (clear the
  column, filter `IS NULL`); omission is always `Unset`'s job.
- Do not hand-write `field: T | None = None` PATCH DTOs, and do not compare
  against the wire string `'__omitted__'` (it is normalized to `Unset` on entry).
- A caller that cannot omit keys (e.g. LLM strict-mode tool calls) gets a
  subclass with `model_config = SQLModelExtConfig(omitted_sentinel=True)`; the
  REST model stays unchanged.
- `partial=True` cannot be combined with `table=True`.
- Checker limitation: `partial=True` builds its annotations at runtime, so
  basedpyright sees the Base annotations on the derived class and will not force
  the `is Unset` check (nor an inherited validator whose guard is too weak for
  `Unset`). Apply it anyway. Where static enforcement matters, either declare
  `field: Unset | T = Unset` explicitly -- basedpyright then rejects every use
  that skips the check -- or run the experimental
  `python -m sqlmodel_ext.check_derived <package>` (see Workflow step 3).

```python
# DO
class ArticleUpdate(ArticleBase, partial=True): ...

if patch.summary is Unset:
    pass                      # not sent: leave the column alone
elif patch.summary is None:
    ...                       # explicit null: clear it
else:
    ...                       # new value
article = await article.update(session, patch)

# DON'T
class ArticleUpdate(SQLModelBase):
    title: str | None = None           # cannot say "clear" vs "not sent"
if patch.summary is not None: ...      # treats "not sent" and "clear" alike
data = patch.model_dump(exclude_unset=True)
```

## 4. No silent fallbacks -- fail loudly

- No `x or 0`, `x or ''`, `x or []`, `x if x else default`, `d.get(k, 0)`,
  `getattr(obj, 'attr', default)` to paper over a missing value: `0`, `''` and
  `False` are valid values and get replaced too. Write an explicit
  `if x is None:` branch, or better, make the field required.
- Do not return `None` to signal an error; raise. Do not clamp or cap values in
  money/quota computations to hide a bug; raise.
- Illegal combinations belong in the DTO (`model_validator`) so the invalid
  object cannot be constructed, not in scattered `if` checks.
- No `assert` for runtime guards in production code (stripped by `python -O`); raise.
- `SQLModelBase` forbids unknown fields (`extra='forbid'`). Parse third-party
  payloads with `ExtraIgnoreModelBase` instead of loosening your own models.

## 5. Writing: use the return value; no `session.refresh()`

- `save()` and `update()` commit and expire the session's objects; they return
  the fresh instance. Always rebind: `obj = await obj.save(session)`.
- Need a relationship on the result? `obj = await obj.save(session, load=rel(Obj.items))`.
- Read what you need from an instance before the next commit (e.g. store `.id`
  in a local); touching an expired attribute afterwards is I/O and fails.
- `update()` takes a model (the DTO); extra columns go in `extra_data={...}`,
  excluded ones in `exclude={...}`. Do not pass a raw dict.
- Do not call `session.refresh(obj)`; re-read with `Model.get(...)` (it honours
  the cache layer and polymorphic loading).
- Batch inserts: `save(session, commit=False)` for all but the last, or
  `Model.add(session, [...])`.
- Delete with `await Model.delete(session, obj)` or
  `await Model.delete(session, condition=...)`; the call without either is a type error.

## 6. Reading: parameterized `get()`, `col()`, `cond()`, `rel()`

- Query through `Model.get(session, condition, fetch_mode=..., load=..., order_by=..., table_view=...)`,
  `Model.count(...)`, `Model.get_with_count(...)`, `Model.group_sum(...)`,
  `Model.distinct_column(...)`. Do not add `find_by_xxx()` / `list_for_user()`
  methods whose body only builds a condition and calls `get()` -- the condition
  is the parameter.
- `fetch_mode` decides the type: `'first'` (default) -> `T | None`, `'one'`
  -> `T` (raises if absent), `'all'` -> `list[T]`. Use `get_exist_one(session, id)`
  for "404 if missing".
- `Model.field == value` is typed `bool` by checkers. Use `col(Model.field)` for
  column methods (`.in_()`, `.is_(None)`, `.asc()`); wrap comparisons in
  `cond(...)` to combine them with `&` / `|`. Pass one combined condition, not
  several positional ones.
- Pass relationships as `load=rel(Model.relation)` (a list for chains).
- `from sqlmodel_ext import select` types projections up to 9 columns. Bare
  attributes are fine (`select(User.id, User.name)` -> `Select[tuple[UUID, str]]`);
  only in a 5+ column projection that mixes in a SQL function expression
  (e.g. `func.count()`) wrap every column in `col()` to keep the precise type.
- List endpoints take a `TableViewRequest` (`offset` / `limit` / `order` /
  `desc` / `after_id` + time bounds). `after_id` cannot be combined with
  `offset`, custom `order_by` or `join`.

## 7. Relationships: always preload

- Relationships default to `lazy='raise_on_sql'`: reading an unloaded one
  raises. Never "fix" that by changing `lazy`; load it.
- Load at query time (`load=rel(...)`), or declare it on the method that reads
  it with `@requires_relations('rel_name', Other.nested)` on a
  `RelationPreloadMixin` model, or batch a list with
  `await Model.ensure_relations_loaded_bulk(session, items, {Model: ('rel_name',)})`
  (one query per target table instead of N).
- A response DTO field that reads a relationship requires that relationship to
  be loaded by the endpoint that returns it.
- Annotate nullable relationships as `'Target | None'` (one PEP 604 string),
  not `Optional['Target']`.
- Do not use `from __future__ import annotations` in model modules: it turns
  `list['Target']` relationship annotations into strings SQLAlchemy cannot resolve.

## 8. Concurrency and transactions

- Read-modify-write on a row that others may change: load it with
  `get(..., with_for_update=True)` and mark the method `@requires_for_update`
  (raises if the instance was not locked in this transaction). For batch
  parameters use `@requires_locked_param('param_name')`.
- Optimistic locking: `class Xxx(OptimisticLockMixin, ..., UUIDTableBaseMixin, table=True)`
  with the mixin **first**. `save()` / `update()` retry conflicts 3 times by
  default, re-applying only your changes; catch `OptimisticLockError` when
  retries are exhausted or `optimistic_retry_count=0`. `oplock_version` is
  reserved -- do not declare a field with that name.
- Use `sqlmodel_ext.AsyncSession` (e.g. `SessionFactory(engine, class_=AsyncSession)`)
  as the session class. Irreversible side effects (external deletes,
  notifications) go through `session.add_post_commit_callback(coro_fn)` so they
  run only after a real commit.
- Work that needs one consistent snapshot across statements (PostgreSQL):
  `await session_factory.run_in_repeatable_read(operation, description=...)`;
  `operation` must commit itself and capture only immutable inputs.
- Queue workers claiming rows: `with_for_update=True, skip_locked=True`.

## 9. Cache (`CachedTableBaseMixin`)

- Writes through `save` / `update` / `delete` and plain ORM mutations are
  invalidated automatically after commit by the enhanced `AsyncSession`.
- After raw DML (`update(...)`, `delete(...)`, `text(...)`) on a cached table,
  register `Model.invalidate_on_commit(session, *ids)` (or call
  `await Model.invalidate_all()`) -- otherwise the cache serves stale rows.
- Reads that authorize something use `get(..., authoritative=True)`.

## 10. Deletes, integrity errors, HTTP mapping

- Register user-facing messages next to the constraint's model:
  `TableBaseMixin.register_fk_delete_restrict_message(name, message)` (and the
  `register_unique_violation_message` / `register_check_violation_message` siblings).
- Map `ResourceReferencedError` to 409, `KeysetCursorError` to 422,
  `RecordNotFoundError` to 404 (each has `status_code`). Never return raw
  database error text to clients.

## 11. Types at the boundary

- Datetimes supplied by clients (query parameters, request bodies) are
  `AwareDatetime`, never bare `datetime`: naive values are silently read in the
  database's time zone.
- Money and other exact numbers use the Decimal aliases -- never `float`.
  Columns that will be summed are written through `*WriteDecimal38_18`; sums are
  read through `SignedSumDecimal38_18` (e.g. `Model.group_sum(...)` totals).
- Use `None` for "not configured / absent", never `''` or `0`.

---

## What basedpyright catches for you

| You wrote | basedpyright says |
|---|---|
| used an explicit `Unset \| T` field without `is not Unset` | `"MISSING" is not assignable to "str"` |
| `if patch.x is not None: use(patch.x)` on `Unset \| T \| None` | same -- `None` check does not exclude `Unset` |
| `(await Model.get(...)).name` | `"name" is not a known attribute of "None"` |
| `await Model.delete(session)` | `No overloads for "delete"` |
| `load=Model.relation` | `"Target" is not assignable to "QueryableAttribute[Any]"` |
| `Model.field.in_(...)` | `Cannot access attribute "in_"` -- use `col()` |
| `select()` with 10+ columns | `No overloads for "select"` |
| `await obj.save(session)` result dropped | warning `reportUnusedCallResult` |

If basedpyright reports one of these, fix the code; do not suppress the rule.
