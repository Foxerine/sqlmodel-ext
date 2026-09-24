# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **The project is still WIP / alpha.** Any release may contain breaking changes;
> there is no stability or backward-compatibility guarantee between versions, and
> you use it at your own risk. Pin the exact version you depend on.

## [0.5.2]

### Fixed

- A table class field declared with a type alias and a plain right-hand value
  (`x: NonNegativeDecimal38_18 = Decimal(0)`, `s: Str64 = 'a'`) had its default only in
  `model_fields`: the JSON schema listed the field as `required` with no `default`, and
  `model_validate({})` rejected the payload with "Field required". With an alias that
  carries its own default (`opt: OptionalNonNegativeDecimal38_18 = Decimal(1)`),
  `model_fields` even showed the alias's default (`None`) instead of the declared one.
  The metaclass set the value on the field after constructing it, so Pydantic did not
  count it as set when it built the core schema. A metaclass rebuild (an inherited field
  description, a relationship) hid the defect in some classes. The same applied to a
  field re-declared in the table class body without a right-hand side, which takes the
  base's default. The table class now matches a non-table class with the same
  declaration (schema, validation and `model_fields`). Non-table classes and fields the
  table class inherits without re-declaring were not affected. Also present in 0.5.0
  and 0.5.1.
- **Behavior change.** A table class field declared with **several** `Field(...)` in one
  `Annotated` and no right-hand side (`value: Legacy` with
  `Legacy = Annotated[str, Field(alias='legacy', title='T'), Field(unique=True)]`, or a
  nested alias such as `Annotated[Str64, Field(unique=True)]`) now gets its Pydantic
  attributes as Pydantic merges them -- the same field a non-table class, plain
  SQLModel and a table class inheriting the declaration already got. A later `Field`
  overrides an earlier one, `None` included, and sqlmodel's `Field()` passes
  `alias` / `validation_alias` / `serialization_alias` / `title` / `description` /
  `default_factory` / `discriminator` / `exclude` explicitly as `None` and `repr` as
  `True`. So when an earlier `Field` of the annotation sets any of these and a later
  one does not, the table class no longer keeps the earlier value: the alias / title /
  description are gone, and a `default_factory` is dropped, which makes the field
  **required**. A `default` is not affected (sqlmodel's `Field()` does not pass it).
  None of the bundled aliases combines several `Field`s or sets such an attribute. To keep the value, set it in the last `Field` or on a right-hand
  `= Field(...)`. The same holds for a non-table class whose field needs an injected
  `sa_type` (`Annotated[FilePathType, Field(alias='p'), Field(index=True)]`), which kept
  the earlier `alias` too. The same path also applied the `Field` of an alias nested in a
  union with no right-hand side, which Pydantic does not merge at field level:
  `x: OptionalNonNegativeDecimal38_18 | None` declared in a table class had
  `default=None`; it is now **required**, as it already was in a non-table class, when
  inherited, and with a right-hand `Field` that sets no default (0.5.1). Drop the
  redundant `| None` or write `= None`. The column attributes of all the declaration's `Field`s are
  still folded into the column, now with the precedence Pydantic uses for the Pydantic
  attributes: the right-hand `Field` first, then the annotation's `Field`s from the last
  to the first; `False` still never switches off `True`. So a table class **inheriting**
  such a declaration agrees with one declaring it directly:
  `Annotated[NonNegativeDecimal38_18, Field(sa_type=Numeric(20, 2))]` gives
  `NUMERIC(20, 2)` in both (inheriting it gave `NUMERIC(38, 18)`, the alias's
  `sa_type`). Plain SQLModel, which reads only the first carrier, still gives
  `NUMERIC(38, 18)`. Also present in 0.5.0 and 0.5.1. This only changes the metadata
  the models build: existing databases are not altered. Tables created from now on
  (`create_all`) get the new column type; to align an existing table, write a migration.

## [0.5.1]

### Added

- **`query_dependency(Model)`**: FastAPI dependency for query-parameter DTOs
  (`TableViewRequest`, `PaginationRequest`, `PageWindowRequest`, `TimeFilterRequest`,
  `TrgmSearchRequest`, your subclasses):
  `Annotated[TableViewRequest, Depends(query_dependency(TableViewRequest))]`. It declares
  one query parameter per field carrying the field's type, constraints, default, `title`,
  description, `examples`, `deprecated`, `json_schema_extra` and `discriminator` (each
  parameter's OpenAPI schema is the field's JSON schema; a callable `json_schema_extra`
  or a `Discriminator` object is a `TypeError`), builds the model itself and
  re-raises a failed construction as `RequestValidationError` with locations prefixed by
  `'query'`, so FastAPI answers 422. Requires the `fastapi` extra when called. See
  [Paginate a list endpoint](docs/en/how-to/paginate-a-list-endpoint.md).

### Fixed

- A bare mutation of a cached model that was flushed before the enhanced `commit()` -- inside
  a savepoint (`begin_nested()` + `flush()`), by a manual `session.flush()`, or by an
  autoflush -- was never invalidated: the cache kept serving the pre-commit row after the
  commit. Pending invalidations were collected by scanning `new` / `dirty` / `deleted`
  right before the commit, and a flushed object is no longer in those sets. Every flush
  now registers the cached instances it writes (`after_flush`); an outermost rollback
  drops them, a savepoint rollback keeps them (at most one extra invalidation). Also
  present in 0.5.0.
- A table class inheriting a field declared with a type alias and a right-hand
  `Field(...)` (`language: Str64 = Field(index=True)` in a non-table base) lost that
  `Field`'s attributes: the column had no index / unique constraint / primary key /
  foreign key / `nullable` / `sa_type` / `sa_column_kwargs` / `ondelete`, and the
  Pydantic field could lose attributes set only there (e.g. `title`). The metaclass
  rebuilt the inherited field from the alias's `Field` alone. The inherited field is now
  the base class's resolved field (`Base.model_fields[name]`, in which Pydantic has
  already merged the alias's `Field` and the right-hand one): its Pydantic attributes
  (`default`, `alias`, `validation_alias`, `title`, `description`, ...) are taken as-is,
  `None` included, so a base `Field(alias=None)` still clears the alias's `alias` in the
  table class. Only the SQLModel column attributes are completed: the alias's and the
  right-hand `Field`'s column settings are folded into one (the right-hand side wins on
  conflicts; `False` never switches off `True`), and constraints of an alias nested in
  a union (`Alias | None = Field(...)`), which Pydantic does not merge at field level,
  are added. Plain-typed inherited fields (`int = Field(index=True)`) were not affected.
  Also present in 0.5.0.
- A cross-field validation failure of `TableViewRequest` (and the other query DTOs) used
  as a FastAPI dependency (`after_id` with a non-zero `offset`, `after_id` with a mutable
  `order`, an inverted time range) was a 500. Declared through `query_dependency()` it is
  a 422 located at the offending query parameter; the documented `ValidationError`
  exception-handler workaround is no longer needed.
- Every commit of a cached model through the enhanced `AsyncSession` logged a spurious
  `WARNING` "fallback compensation triggered: ..." and invalidated the cache a second
  time. The `after_commit` event now hands the committed pendings over to the enhanced
  `commit()`, which invalidates each of them once. Commits that bypass the enhanced
  `commit()` (a plain SQLAlchemy / SQLModel session, `run_sync(lambda s: s.commit())`)
  keep the fire-and-forget compensation and its `WARNING`; it now also takes over when
  the enhanced `commit()` fails or is cancelled after the database committed.

### Changed

- A table class field declared with a type alias **and** a right-hand `Field(...)`
  (`name: Alias = Field(...)`) now gets its Pydantic attributes exactly as Pydantic
  resolves that declaration -- the same field plain SQLModel builds, and the same field
  a table class gets when it inherits the declaration from a base. The right-hand
  `Field` overrides the alias's Pydantic attributes, `None` included; sqlmodel's
  `Field()` passes every one of them explicitly, so with a right-hand `Field` the
  alias's `alias` / `validation_alias` / `serialization_alias` / `title` /
  `description` no longer apply (`Field(alias=None)` clears the alias's `alias`;
  previously the metaclass skipped `None` and kept the alias's values). The alias's
  `default` still applies unless the right-hand `Field` sets one -- except for an alias
  nested in a union (`Alias | None = Field(...)`), whose `Field` Pydantic does not
  merge: such a field is required unless the right-hand `Field` sets a default. Of the
  bundled aliases only the `Optional*Decimal*` ones carry a Pydantic attribute
  (`default=None`), so `OptionalNonNegativeDecimal38_18 | None = Field(index=True)` is
  now required; drop the redundant `| None` or add `default=None`. Column attributes are
  unchanged: the alias's and the right-hand `Field`'s column settings are still folded
  together.
- The cross-field rules of `PaginationRequest` and `TimeFilterRequest` are field validators:
  a violation is still a `ValidationError` with the same message, but located at the
  field the rule constrains (`after_id`, `created_before_datetime`,
  `updated_before_datetime`) instead of the model root, and independent violations are
  reported together. `TimeFilterRequest` no longer overrides `model_post_init`.
- `PolymorphicBaseMixin._is_joined_table_inheritance()` is renamed
  `is_joined_table_inheritance()` (it was documented and called across modules; the
  underscore was not a real boundary). Update direct callers.
- `get(table_view=...)` is annotated `TableViewArgument` (`TimeFilterRequest |
  PageWindowRequest`), matching what it already accepted at runtime (e.g. a bare
  `PageWindowRequest`). The `join=(Model, onclause)` ON clause is annotated with the
  new `OnClauseArgument` alias (SQLAlchemy's public `OnClauseRole`) instead of
  SQLAlchemy's private `_OnClauseArgument`.
- The type-check gate of the repository is now 0 errors **and 0 warnings**
  (`failOnWarnings: true`).

## [0.5.0]

Upgrade guide: [Migrate from 0.4.x to 0.5.0](docs/en/how-to/migrate-to-0-5.md)
([中文](docs/how-to/migrate-to-0-5.md)).

### Breaking

- **`all_fields_optional=True` removed; use `partial=True`.** The old keyword raises
  `TypeError` at class creation. `partial=True` turns inherited fields into
  `Unset | T = Unset` (`Unset | T | None` when the base field is nullable): omitted
  fields are `Unset` and never appear in `model_dump()` (no `exclude_unset=True`
  needed); explicit `null` is only accepted where the base field allows `None`.
  Replace `is None` / `is not None` "was it sent" checks with `is Unset` /
  `is not Unset`. `partial=True` cannot be combined with `table=True`.
- **Optimistic-lock column renamed `version` → `oplock_version`**, now `BIGINT`
  (bounded by `JS_MAX_SAFE_INTEGER`) with `server_default 0`, and excluded from
  `model_dump()`. Requires a database migration (see the upgrade guide). The
  rename is **not rolling-compatible**: 0.4.x and 0.5.0 instances cannot run
  against the same table at once, so switch all instances together. If you
  use `CachedTableBaseMixin`, also clear the Redis cache at the switch: cache
  keys are unchanged and a 0.4.x entry can validate under 0.5.0 with a
  different meaning.
- **`oplock_version` is a reserved name**: declaring it in the class body of any
  `SQLModelBase` subclass raises `TypeError`.
- **`OptimisticLockMixin` retries conflicts 3 times by default**
  (`__optimistic_retry_default__`); `optimistic_retry_count` now defaults to `None`
  ("use the model policy"). Pass `optimistic_retry_count=0` to fail immediately.
- **`delete()` normalizes errors**: an optimistic-lock conflict raises
  `OptimisticLockError` (previously a raw `StaleDataError`); deleting a row that is
  still referenced through a foreign key raises `ResourceReferencedError`
  (`status_code = 409`; previously a raw `IntegrityError`).
- **Default primary keys are UUIDv7** (`UUIDTableBaseMixin.id` and
  `create_subclass_id_mixin()`), previously UUIDv4. Existing rows need no migration.
- **`TimeFilterRequest` bounds are `AwareDatetime`**: naive datetimes are rejected.
  `PaginationRequest.offset` is now bounded by `MAX_TABLE_VIEW_OFFSET`.
- **`@requires_for_update` fails closed**: it raises `RuntimeError` when it cannot
  find an `AsyncSession` in the call arguments (previously the check was skipped).
- **Renamed / removed internals**: `RelationPreloadMixin._ensure_relations_loaded` →
  `ensure_relations_loaded`; `CachedTableBaseMixin._warn_raw_dml_on_cached` →
  `register_raw_dml_write`; `CachedTableBaseMixin._refresh_via_cache` and
  `_has_pending_invalidation` removed.
- **`AsyncSession.refresh()` always reads the database** (no longer routed through
  the Redis cache).
- **`get(with_for_update=True)` always implies `populate_existing`** (no opt-out).
- **`JSON100K` / `JSONList100K` serialize as objects / arrays** in every dump mode
  (previously a JSON string). Dict / list input is now subject to the 100K-character
  and nesting-depth limits as well; table models are checked at construction.
  Subclasses overriding `model_post_init` must call `super().model_post_init(context)`.
- **`PositiveFloat` / `NonNegativeFloat` reject `inf` and `nan`.**
- **Decimal aliases enforce integer digits** (previously only total digits and
  decimal places were validated, e.g. `SignedDecimal20_10` accepted a 15-digit integer).
- **Dependencies**: `pydantic>=2.12` (was `>=2.0`) and `typing-extensions>=4.14.1`.
- **RelationLoadChecker** (experimental) reports more findings (RLC014, session
  subclasses) and honors `# noqa: RLCxxx` on every public entry point; endpoint
  `noqa` comments must be on the first decorator line.

### Added

- **Tri-state fields**: `Unset` (Pydantic's `MISSING` sentinel), `partial=True`,
  `SQLModelBase.annotation_is_omissible()` / `field_is_omissible()` /
  `submitted_fields_among()`, `sqlmodel_ext.base.optional_dto_registry`.
- **Wire sentinel for callers that cannot omit keys**: `SQLModelExtConfig(omitted_sentinel=True)`
  and `OMITTED_SENTINEL` (`'__omitted__'`).
- `EXCLUDE_IF_NONE` field marker (the key is dropped from every dump while the value is `None`).
- `sqlmodel_ext.select`: typed `select()` overloads for 5–9 column projections
  (same object as `sqlmodel.select` at runtime). For 1–4 columns the overload set is
  exactly upstream's, so bare attributes (`select(User.id, User.name)` →
  `Select[tuple[UUID, str]]`) type-check; bare attributes also work for 5–9 columns,
  and only a 5+ projection that mixes in a SQL function expression (`func.count()`)
  should wrap every column in `col()` to keep the precise type.
- **Experimental `check_derived`**: `python -m sqlmodel_ext.check_derived <package>`
  (console script `sqlmodel-ext-check-derived`) expands `partial=True` DTOs and their
  inherited methods in a throwaway copy and runs basedpyright there, reporting only the
  errors the expansion introduced. See
  [Check partial DTOs for misuse](docs/en/how-to/check-partial-dtos.md).
- **AI rule pack** in [`ai-rules/`](ai-rules/) (`AGENTS.md` plus a `CLAUDE.md` entry
  point) so AI coding assistants use the library the intended way. See
  [Use with AI coding assistants](docs/en/how-to/use-with-ai-assistants.md).
- **`py.typed`**: the package is now PEP 561 typed, so type checkers read its inline
  annotations.
- `pyrightconfig.json` and a basedpyright CI gate (the job fails on errors and when no
  source files are scanned). See
  [Type-check with basedpyright](docs/en/how-to/type-check-with-basedpyright.md).
- Runnable [`examples/`](examples/) for the new features, executed by
  `tests/test_examples.py`.
- `AsyncSession`: `commit_count`, `add_post_commit_callback()`, `set_local_timeouts()`,
  `enter_repeatable_read()`, `commit(fail_soft_when_observed=...)`,
  `rollback(best_effort_budget_seconds=...)`, a `begin()` that routes through the
  enhanced commit / rollback, raw-DML registration on `execute` / `exec` / `scalar` /
  `stream` / `stream_scalars`.
- `sqlmodel_ext.session.SessionFactory.run_in_repeatable_read()` (whole-session retry on
  SQLSTATE 40001), `RepeatableReadSnapshotConflictError`, `SerializationRetryExhaustedError`.
- CRUD: `get(skip_locked=..., authoritative=...)`, `get_one(authoritative=...)`,
  `get_exist_one(with_for_update=...)`, `count(distinct_column=...)`, `distinct_column()`,
  `group_sum()` / `GroupSumRow`.
- Keyset pagination: `PaginationRequest.after_id`, `order="id"`, `PageWindowRequest`,
  `KeysetCursorError` / `KeysetCursorInvalidError` / `KeysetCursorUnsupportedError`.
- `ResourceReferencedError`, `FK_DELETE_RESTRICT_FALLBACK_MESSAGE`,
  `TableBaseMixin.register_fk_delete_restrict_message()`.
- `sqlmodel_ext.mixins.uuid7` (standard-library `uuid.uuid7` on Python 3.14+).
- Transaction contracts: `requires_locked_param`, `requires_read_committed`,
  `requires_repeatable_read`, `validate_locked_instances`, `SESSION_REPEATABLE_READ_KEY`.
- Relation preloading: `ensure_relations_loaded_bulk()`, `bulk_preload_unsupported_reason()`.
- Cache: `CachedTableBaseMixin.invalidate_on_commit()`, `invalidate_all(strict=...)`,
  `register_raw_dml_write()`; `CACHE_TTL_HOT` / `CACHE_TTL_WARM` / `CACHE_TTL_COLD`;
  `collect_migration_invalidation_tasks()` / `run_pending_migration_cache_invalidations()`
  (new optional extra `alembic`).
- New mixins: `ResourceQuotaMixin` (with `QuotaExceededError`, `QuotaOwnerNotFoundError`,
  `CallerDidNotCommitError`), `TrgmSearchableMixin` / `TrgmSearchRequest` /
  `BIGRAM_FUNCTION_SQL`, `MixinTableScanMixin`.
- Field types: `max_length_of()`, `Str1`, `Text3072`, `Text4K`, `NonEmptyStrippedStr32`,
  `SingleLineStr64`, `SearchQueryStr64`, `HttpHeaderName`, `SignedBigInt`, `INT32_MIN`,
  `ClientIPAddress`, `OptionalSignedDecimal38_18`, `NullableNonNegativeDecimal20_10`,
  write/sum Decimal pairs (`SignedWriteDecimal38_18`, `NonNegativeWriteDecimal38_18`,
  `PositiveWriteDecimal38_18`, `OptionalNonNegativeWriteDecimal38_18`,
  `OptionalSignedWriteDecimal38_18`, `SignedSumDecimal38_18`) and
  `DECIMAL_38_18_COLUMN_DIGITS` / `DECIMAL_38_18_WRITE_DIGITS` / `DECIMAL_38_18_PLACES`;
  `ensure_json_within_limits()` in `sqlmodel_ext.field_types.dialects.postgresql`.
- RelationLoadChecker: RLC014 and module-level `conditional_commit_methods` /
  `explicit_commit_methods` / `dependency_commit_methods`.

### Fixed

- A raw `insert(...)` into a cached table executed through the enhanced
  `AsyncSession` left cached query results stale after commit (until TTL). It
  now registers a query-level invalidation that the enhanced `commit()` runs.
  `text()` writes and writes nested in a `SELECT` (writable CTEs) are still not
  invalidated after commit -- register `invalidate_on_commit` or call
  `invalidate_all` yourself.
- An explicit `default=None` inside merged `Field` metadata was dropped, silently making
  the field required.
- On sqlmodel ≥ 0.0.32, `id: NonNegativeInt = Field(primary_key=True)` lost its primary key.
- Cached-model serialization dropped `exclude=True` columns.
- The cache could serve stale reads or publish data that was not yet committed.
- Read-modify-write after `with_for_update=True` could lose an update through a stale
  identity-map object.
- A joined-table-inheritance update that touched only child-table columns did not
  advance `updated_at`.
- `ExtraIgnoreModelBase` warned about keys accepted through `AliasChoices`.
- `model_json_schema()` did not inject the sentinel branch into self-referencing models.
- `AsyncSession.execute()` / `exec()` / `scalar()` / `stream()` / `stream_scalars()`
  no longer degrade their return type to `Any`: they keep upstream SQLModel's
  signatures (`await session.exec(select(User))` is `ScalarResult[User]`).
- Optimistic-lock models no longer emit Pydantic's "Field name ... shadows an
  attribute in parent" `UserWarning` for the version column.

### Changed

- Documentation rewritten around the single-source-of-truth design, with new pages for
  `Unset`, PATCH endpoints, basedpyright and this migration.

### Known limitations

- A Redis failure during post-commit cache invalidation is logged, not raised (the
  database has already committed): the affected cache entries keep serving
  pre-commit data until their TTL expires. Use `no_cache=True` for reads that must
  not rely on the cache.
- On Python 3.12, `JSON100K | None` as a `table=True` field raises
  `has no matching SQLAlchemy type`; declare the column explicitly with
  `Field(default=None, sa_type=JSONB)` (pre-existing).

## 0.4.2 and earlier

Not recorded in this file; see the git history (tags `v0.4.0` and earlier).
