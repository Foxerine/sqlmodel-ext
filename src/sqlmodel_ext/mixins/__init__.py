"""
sqlmodel_ext.mixins -- Mixin classes for SQLModel table models.

Re-exports all mixins for convenient access. Every re-export uses the redundant
``X as X`` form: the package ships ``py.typed``, and type checkers treat a plain
``from .m import X`` in a typed package as private (consumers would get
``reportPrivateImportUsage``).
"""
from ._uuid import uuid7 as uuid7
from .exceptions import (
    FK_DELETE_RESTRICT_FALLBACK_MESSAGE as FK_DELETE_RESTRICT_FALLBACK_MESSAGE,
    KeysetCursorError as KeysetCursorError,
    KeysetCursorInvalidError as KeysetCursorInvalidError,
    KeysetCursorUnsupportedError as KeysetCursorUnsupportedError,
    ResourceReferencedError as ResourceReferencedError,
)
from .polymorphic import (
    PolymorphicBaseMixin as PolymorphicBaseMixin,
    AutoPolymorphicIdentityMixin as AutoPolymorphicIdentityMixin,
    DeferredIndex as DeferredIndex,
    create_subclass_id_mixin as create_subclass_id_mixin,
    register_sti_columns_for_all_subclasses as register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses as register_sti_column_properties_for_all_subclasses,
)
from .optimistic_lock import (
    OptimisticLockMixin as OptimisticLockMixin,
    OptimisticLockError as OptimisticLockError,
)
from .table import (
    SESSION_FOR_UPDATE_KEY as SESSION_FOR_UPDATE_KEY,
    SESSION_REPEATABLE_READ_KEY as SESSION_REPEATABLE_READ_KEY,
    GroupSumRow as GroupSumRow,
    TableBaseMixin as TableBaseMixin,
    UUIDTableBaseMixin as UUIDTableBaseMixin,
    rel as rel,
    cond as cond,
)
from .relation_preload import (
    RelationPreloadMixin as RelationPreloadMixin,
    requires_relations as requires_relations,
    requires_for_update as requires_for_update,
    requires_locked_param as requires_locked_param,
    requires_read_committed as requires_read_committed,
    requires_repeatable_read as requires_repeatable_read,
    validate_locked_instances as validate_locked_instances,
)
from .cached_table import (
    CachedTableBaseMixin as CachedTableBaseMixin,
)
from .info_response import (
    IntIdInfoMixin as IntIdInfoMixin,
    UUIDIdInfoMixin as UUIDIdInfoMixin,
    DatetimeInfoMixin as DatetimeInfoMixin,
    IntIdDatetimeInfoMixin as IntIdDatetimeInfoMixin,
    UUIDIdDatetimeInfoMixin as UUIDIdDatetimeInfoMixin,
)
from .constants import (
    CACHE_TTL_HOT as CACHE_TTL_HOT,
    CACHE_TTL_WARM as CACHE_TTL_WARM,
    CACHE_TTL_COLD as CACHE_TTL_COLD,
)
from .resource_quota import (
    ResourceQuotaMixin as ResourceQuotaMixin,
    QuotaExceededError as QuotaExceededError,
    QuotaOwnerNotFoundError as QuotaOwnerNotFoundError,
    CallerDidNotCommitError as CallerDidNotCommitError,
)
from .mixin_table_scan import (
    MixinTableScanMixin as MixinTableScanMixin,
)
from .trgm_searchable import (
    BIGRAM_FUNCTION_SQL as BIGRAM_FUNCTION_SQL,
    TrgmSearchableMixin as TrgmSearchableMixin,
    TrgmSearchRequest as TrgmSearchRequest,
)
from .migration_cache_invalidation import (
    collect_migration_invalidation_tasks as collect_migration_invalidation_tasks,
    run_pending_migration_cache_invalidations as run_pending_migration_cache_invalidations,
)
