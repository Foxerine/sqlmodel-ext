"""
sqlmodel_ext.mixins -- Mixin classes for SQLModel table models.

Re-exports all mixins for convenient access.
"""
from .polymorphic import (
    PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)
from .optimistic_lock import (
    OptimisticLockMixin,
    OptimisticLockError,
)
from .table import (
    SESSION_FOR_UPDATE_KEY,
    TableBaseMixin,
    UUIDTableBaseMixin,
    rel,
    cond,
    safe_reset,
)
from .relation_preload import (
    RelationPreloadMixin,
    requires_relations,
    requires_for_update,
)
from .cached_table import (
    CachedTableBaseMixin,
)
from .info_response import (
    IntIdInfoMixin,
    UUIDIdInfoMixin,
    DatetimeInfoMixin,
    IntIdDatetimeInfoMixin,
    UUIDIdDatetimeInfoMixin,
)
