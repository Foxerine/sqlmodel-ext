"""
sqlmodel_ext.mixins -- Mixin classes for SQLModel table models.

Re-exports all mixins for convenient access.
"""
from ._uuid import uuid7
from .exceptions import (
    FK_DELETE_RESTRICT_FALLBACK_MESSAGE,
    KeysetCursorError,
    KeysetCursorInvalidError,
    KeysetCursorUnsupportedError,
    ResourceReferencedError,
)
from .polymorphic import (
    PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin,
    DeferredIndex,
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
    SESSION_REPEATABLE_READ_KEY,
    GroupSumRow,
    TableBaseMixin,
    UUIDTableBaseMixin,
    rel,
    cond,
)
from .relation_preload import (
    RelationPreloadMixin,
    requires_relations,
    requires_for_update,
    requires_locked_param,
    requires_read_committed,
    requires_repeatable_read,
    validate_locked_instances,
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
from .constants import (
    CACHE_TTL_HOT,
    CACHE_TTL_WARM,
    CACHE_TTL_COLD,
)
from .resource_quota import (
    ResourceQuotaMixin,
    QuotaExceededError,
    QuotaOwnerNotFoundError,
    CallerDidNotCommitError,
)
from .mixin_table_scan import (
    MixinTableScanMixin,
)
from .trgm_searchable import (
    BIGRAM_FUNCTION_SQL,
    TrgmSearchableMixin,
    TrgmSearchRequest,
)
from .migration_cache_invalidation import (
    collect_migration_invalidation_tasks,
    run_pending_migration_cache_invalidations,
)
