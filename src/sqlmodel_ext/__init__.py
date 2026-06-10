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
__version__ = "0.4.0"

# Base
from sqlmodel_ext.base import SQLModelBase, ExtraIgnoreModelBase, CustomTableArg

# Enhanced session (cache-aware commit/reset/refresh/execute)
from sqlmodel_ext.session import AsyncSession

# Exceptions
from sqlmodel_ext._exceptions import RecordNotFoundError

# Pagination
from sqlmodel_ext.pagination import (
    ListResponse,
    TimeFilterRequest,
    PaginationRequest,
    TableViewRequest,
)

# Mixins
from sqlmodel_ext.mixins import (
    # Table
    SESSION_FOR_UPDATE_KEY,
    TableBaseMixin,
    UUIDTableBaseMixin,
    rel,
    cond,
    # Polymorphic
    PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin,
    DeferredIndex,
    create_subclass_id_mixin,
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
    # Optimistic Lock
    OptimisticLockMixin,
    OptimisticLockError,
    # Relation Preload
    RelationPreloadMixin,
    requires_relations,
    requires_for_update,
    # Cached Table
    CachedTableBaseMixin,
    # Info Response DTOs
    IntIdInfoMixin,
    UUIDIdInfoMixin,
    DatetimeInfoMixin,
    IntIdDatetimeInfoMixin,
    UUIDIdDatetimeInfoMixin,
)

# Field Types
from sqlmodel_ext.field_types import (
    # Path types
    DirectoryPathType,
    FilePathType,
    # String constraints
    Str16,
    Str24,
    Str32,
    Str36,
    Str48,
    Str64,
    Str100,
    Str128,
    Str255,
    Str256,
    Str500,
    Str512,
    Str2048,
    Text1K,
    Text1024,
    Text2K,
    Text2500,
    Text3K,
    Text5K,
    Text8K,
    Text10K,
    Text16K,
    Text32K,
    Text48K,
    Text60K,
    Text64K,
    Text100K,
    Text128K,
    Text1M,
    NonEmptyStr64,
    NonEmptyStr128,
    NonEmptyStr256,
    NonEmptyStrippedStr64,
    NonEmptyStrippedStr128,
    NonEmptyStrippedStr256,
    Sha256Hex,
    BCP47LanguageCode,
    # Numeric constraints
    INT32_MAX,
    INT64_MAX,
    JS_MAX_SAFE_INTEGER,
    Port,
    Percentage,
    PositiveInt,
    NonNegativeInt,
    PositiveBigInt,
    NonNegativeBigInt,
    PositiveFloat,
    NonNegativeFloat,
    # Decimal constraints (NUMERIC(p, s) + sign + JSON-string serialization)
    SignedDecimal38_18,
    NonNegativeDecimal38_18,
    PositiveDecimal38_18,
    OptionalNonNegativeDecimal38_18,
    SignedDecimal20_10,
    NonNegativeDecimal20_10,
    OptionalNonNegativeDecimal20_10,
    # Bounded-length list aliases
    List,
    List1,
    List2,
    List3,
    List7,
    List10,
    List16,
    List20,
    List32,
    List40,
    List50,
    List64,
    List100,
    List128,
    List200,
    List256,
    List1024,
    # Custom types
    IPAddress,
    Url,
    HttpUrl,
    WebSocketUrl,
    SafeHttpUrl,
    UnsafeURLError,
    validate_not_private_host,
    ModuleNameMixin,
)

# Relation Load Checker (static analysis)
from sqlmodel_ext.relation_load_checker import (
    RelationLoadChecker,
    RelationLoadWarning,
    RelationLoadCheckMiddleware,
    run_model_checks,
    mark_app_check_completed,
)
