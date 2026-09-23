"""
sqlmodel_ext -- Extended SQLModel infrastructure.

Smart metaclass, async CRUD mixins, polymorphic inheritance, optimistic locking,
relation preloading, and reusable field types for SQLModel.

Quick start::

    from sqlmodel_ext import SQLModelBase, TableBaseMixin, UUIDTableBaseMixin

    class UserBase(SQLModelBase):
        name: str
        email: str

    class User(UserBase, UUIDTableBaseMixin, table=True):
        pass

    # CRUD
    user = User(name="Alice", email="alice@example.com")
    user = await user.save(session)
    users = await User.get(session, fetch_mode="all")
"""
__version__ = "0.5.1"

# Every re-export below uses the redundant ``X as X`` form: the package ships
# ``py.typed``, and type checkers treat a plain ``from m import X`` in a typed
# package as private (consumers would get ``reportPrivateImportUsage``).

# Base
from sqlmodel_ext.base import (
    SQLModelBase as SQLModelBase,
    ExtraIgnoreModelBase as ExtraIgnoreModelBase,
    CustomTableArg as CustomTableArg,
    SQLModelExtConfig as SQLModelExtConfig,
)

# Tri-state semantics ("not provided" vs null) and the optional wire value
from sqlmodel_ext.unset import Unset as Unset, OMITTED_SENTINEL as OMITTED_SENTINEL

# Field metadata markers
from sqlmodel_ext.constants import EXCLUDE_IF_NONE as EXCLUDE_IF_NONE

# ``sqlmodel.select`` with overloads for 5-9 column projections (same object at runtime)
from sqlmodel_ext.select import select as select

# Enhanced session (cache-aware commit/reset/refresh/execute, REPEATABLE READ retries)
from sqlmodel_ext.session import (
    AsyncSession as AsyncSession,
    SessionFactory as SessionFactory,
    RepeatableReadSnapshotConflictError as RepeatableReadSnapshotConflictError,
    SerializationRetryExhaustedError as SerializationRetryExhaustedError,
)

# Exceptions
from sqlmodel_ext._exceptions import RecordNotFoundError as RecordNotFoundError

# Pagination
from sqlmodel_ext.pagination import (
    DEFAULT_PAGE_SIZE as DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE as MAX_PAGE_SIZE,
    MAX_TABLE_VIEW_OFFSET as MAX_TABLE_VIEW_OFFSET,
    ListResponse as ListResponse,
    TimeFilterRequest as TimeFilterRequest,
    PageWindowRequest as PageWindowRequest,
    PaginationRequest as PaginationRequest,
    TableViewRequest as TableViewRequest,
)

# FastAPI dependency for query-parameter DTOs (cross-field errors -> 422; requires the `fastapi` extra when called)
from sqlmodel_ext.dependencies import query_dependency as query_dependency

# Mixins
from sqlmodel_ext.mixins import (
    # Table
    SESSION_FOR_UPDATE_KEY as SESSION_FOR_UPDATE_KEY,
    SESSION_REPEATABLE_READ_KEY as SESSION_REPEATABLE_READ_KEY,
    GroupSumRow as GroupSumRow,
    TableBaseMixin as TableBaseMixin,
    UUIDTableBaseMixin as UUIDTableBaseMixin,
    rel as rel,
    cond as cond,
    uuid7 as uuid7,
    # Delete / keyset exceptions
    FK_DELETE_RESTRICT_FALLBACK_MESSAGE as FK_DELETE_RESTRICT_FALLBACK_MESSAGE,
    ResourceReferencedError as ResourceReferencedError,
    KeysetCursorError as KeysetCursorError,
    KeysetCursorInvalidError as KeysetCursorInvalidError,
    KeysetCursorUnsupportedError as KeysetCursorUnsupportedError,
    # Polymorphic
    PolymorphicBaseMixin as PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin as AutoPolymorphicIdentityMixin,
    DeferredIndex as DeferredIndex,
    create_subclass_id_mixin as create_subclass_id_mixin,
    register_sti_columns_for_all_subclasses as register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses as register_sti_column_properties_for_all_subclasses,
    # Optimistic Lock
    OptimisticLockMixin as OptimisticLockMixin,
    OptimisticLockError as OptimisticLockError,
    # Relation Preload
    RelationPreloadMixin as RelationPreloadMixin,
    requires_relations as requires_relations,
    requires_for_update as requires_for_update,
    requires_locked_param as requires_locked_param,
    requires_read_committed as requires_read_committed,
    requires_repeatable_read as requires_repeatable_read,
    validate_locked_instances as validate_locked_instances,
    # Cached Table
    CachedTableBaseMixin as CachedTableBaseMixin,
    CACHE_TTL_HOT as CACHE_TTL_HOT,
    CACHE_TTL_WARM as CACHE_TTL_WARM,
    CACHE_TTL_COLD as CACHE_TTL_COLD,
    collect_migration_invalidation_tasks as collect_migration_invalidation_tasks,
    run_pending_migration_cache_invalidations as run_pending_migration_cache_invalidations,
    # Info Response DTOs
    IntIdInfoMixin as IntIdInfoMixin,
    UUIDIdInfoMixin as UUIDIdInfoMixin,
    DatetimeInfoMixin as DatetimeInfoMixin,
    IntIdDatetimeInfoMixin as IntIdDatetimeInfoMixin,
    UUIDIdDatetimeInfoMixin as UUIDIdDatetimeInfoMixin,
    # Resource quota
    ResourceQuotaMixin as ResourceQuotaMixin,
    QuotaExceededError as QuotaExceededError,
    QuotaOwnerNotFoundError as QuotaOwnerNotFoundError,
    CallerDidNotCommitError as CallerDidNotCommitError,
    # Mixin table scan
    MixinTableScanMixin as MixinTableScanMixin,
    # Trigram search (PostgreSQL pg_trgm)
    BIGRAM_FUNCTION_SQL as BIGRAM_FUNCTION_SQL,
    TrgmSearchableMixin as TrgmSearchableMixin,
    TrgmSearchRequest as TrgmSearchRequest,
)

# Field Types
from sqlmodel_ext.field_types import (
    # Path types
    DirectoryPathType as DirectoryPathType,
    FilePathType as FilePathType,
    # String constraints
    max_length_of as max_length_of,
    Str1 as Str1,
    Str16 as Str16,
    Str24 as Str24,
    Str32 as Str32,
    Str36 as Str36,
    Str48 as Str48,
    Str64 as Str64,
    Str100 as Str100,
    Str128 as Str128,
    Str255 as Str255,
    Str256 as Str256,
    Str500 as Str500,
    Str512 as Str512,
    Str2048 as Str2048,
    Text1K as Text1K,
    Text1024 as Text1024,
    Text2K as Text2K,
    Text2500 as Text2500,
    Text3K as Text3K,
    Text3072 as Text3072,
    Text4K as Text4K,
    Text5K as Text5K,
    Text8K as Text8K,
    Text10K as Text10K,
    Text16K as Text16K,
    Text32K as Text32K,
    Text48K as Text48K,
    Text60K as Text60K,
    Text64K as Text64K,
    Text100K as Text100K,
    Text128K as Text128K,
    Text1M as Text1M,
    NonEmptyStr64 as NonEmptyStr64,
    NonEmptyStr128 as NonEmptyStr128,
    NonEmptyStr256 as NonEmptyStr256,
    NonEmptyStrippedStr32 as NonEmptyStrippedStr32,
    NonEmptyStrippedStr64 as NonEmptyStrippedStr64,
    NonEmptyStrippedStr128 as NonEmptyStrippedStr128,
    NonEmptyStrippedStr256 as NonEmptyStrippedStr256,
    Sha256Hex as Sha256Hex,
    BCP47LanguageCode as BCP47LanguageCode,
    SingleLineStr64 as SingleLineStr64,
    SearchQueryStr64 as SearchQueryStr64,
    HttpHeaderName as HttpHeaderName,
    # Numeric constraints
    INT32_MIN as INT32_MIN,
    INT32_MAX as INT32_MAX,
    INT64_MAX as INT64_MAX,
    JS_MAX_SAFE_INTEGER as JS_MAX_SAFE_INTEGER,
    Port as Port,
    Percentage as Percentage,
    PositiveInt as PositiveInt,
    NonNegativeInt as NonNegativeInt,
    PositiveBigInt as PositiveBigInt,
    NonNegativeBigInt as NonNegativeBigInt,
    SignedBigInt as SignedBigInt,
    PositiveFloat as PositiveFloat,
    NonNegativeFloat as NonNegativeFloat,
    # Decimal constraints (NUMERIC(p, s) + sign + JSON-string serialization)
    SignedDecimal38_18 as SignedDecimal38_18,
    NonNegativeDecimal38_18 as NonNegativeDecimal38_18,
    PositiveDecimal38_18 as PositiveDecimal38_18,
    OptionalNonNegativeDecimal38_18 as OptionalNonNegativeDecimal38_18,
    OptionalSignedDecimal38_18 as OptionalSignedDecimal38_18,
    SignedWriteDecimal38_18 as SignedWriteDecimal38_18,
    NonNegativeWriteDecimal38_18 as NonNegativeWriteDecimal38_18,
    PositiveWriteDecimal38_18 as PositiveWriteDecimal38_18,
    OptionalNonNegativeWriteDecimal38_18 as OptionalNonNegativeWriteDecimal38_18,
    OptionalSignedWriteDecimal38_18 as OptionalSignedWriteDecimal38_18,
    SignedSumDecimal38_18 as SignedSumDecimal38_18,
    DECIMAL_38_18_COLUMN_DIGITS as DECIMAL_38_18_COLUMN_DIGITS,
    DECIMAL_38_18_WRITE_DIGITS as DECIMAL_38_18_WRITE_DIGITS,
    DECIMAL_38_18_PLACES as DECIMAL_38_18_PLACES,
    SignedDecimal20_10 as SignedDecimal20_10,
    NonNegativeDecimal20_10 as NonNegativeDecimal20_10,
    OptionalNonNegativeDecimal20_10 as OptionalNonNegativeDecimal20_10,
    NullableNonNegativeDecimal20_10 as NullableNonNegativeDecimal20_10,
    # Bounded-length list aliases
    List as List,
    List1 as List1,
    List2 as List2,
    List3 as List3,
    List7 as List7,
    List10 as List10,
    List16 as List16,
    List20 as List20,
    List32 as List32,
    List40 as List40,
    List50 as List50,
    List64 as List64,
    List100 as List100,
    List128 as List128,
    List200 as List200,
    List256 as List256,
    List1024 as List1024,
    # Custom types
    IPAddress as IPAddress,
    ClientIPAddress as ClientIPAddress,
    Url as Url,
    HttpUrl as HttpUrl,
    WebSocketUrl as WebSocketUrl,
    SafeHttpUrl as SafeHttpUrl,
    UnsafeURLError as UnsafeURLError,
    validate_not_private_host as validate_not_private_host,
    ModuleNameMixin as ModuleNameMixin,
)

# Relation Load Checker (static analysis)
from sqlmodel_ext.relation_load_checker import (
    RelationLoadChecker as RelationLoadChecker,
    RelationLoadWarning as RelationLoadWarning,
    RelationLoadCheckMiddleware as RelationLoadCheckMiddleware,
    run_model_checks as run_model_checks,
    mark_app_check_completed as mark_app_check_completed,
)
