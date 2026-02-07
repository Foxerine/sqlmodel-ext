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
__version__ = "0.1.1"

# Base
from sqlmodel_ext.base import SQLModelBase

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
    TableBaseMixin,
    UUIDTableBaseMixin,
    # Polymorphic
    PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
    # Optimistic Lock
    OptimisticLockMixin,
    OptimisticLockError,
    # Relation Preload
    RelationPreloadMixin,
    requires_relations,
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
    Str24,
    Str32,
    Str36,
    Str48,
    Str64,
    Str100,
    Str128,
    Str255,
    Str256,
    Text1K,
    Text1024,
    Text2K,
    Text2500,
    Text3K,
    Text10K,
    Text60K,
    Text64K,
    Text100K,
    # Numeric constraints
    Port,
    Percentage,
    PositiveInt,
    NonNegativeInt,
    PositiveFloat,
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
