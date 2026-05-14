"""
Smoke-test every public export in ``sqlmodel_ext.__init__``.

Catches accidental import-time breakage before it reaches PyPI consumers
(e.g. a broken relative import, a missing symbol, a module that fails to
initialise). This runs first in CI because it's the cheapest check.
"""
from __future__ import annotations

import pytest


def test_toplevel_import() -> None:
    """``import sqlmodel_ext`` alone must not raise."""
    import sqlmodel_ext  # noqa: F401


@pytest.mark.parametrize(
    "name",
    [
        # Base
        "SQLModelBase",
        "ExtraIgnoreModelBase",
        # Exceptions
        "RecordNotFoundError",
        # Pagination
        "ListResponse",
        "TimeFilterRequest",
        "PaginationRequest",
        "TableViewRequest",
        # Mixins - Table
        "SESSION_FOR_UPDATE_KEY",
        "TableBaseMixin",
        "UUIDTableBaseMixin",
        "rel",
        "cond",
        "safe_reset",
        # Mixins - Polymorphic
        "PolymorphicBaseMixin",
        "AutoPolymorphicIdentityMixin",
        "create_subclass_id_mixin",
        "register_sti_columns_for_all_subclasses",
        "register_sti_column_properties_for_all_subclasses",
        # Mixins - Optimistic Lock
        "OptimisticLockMixin",
        "OptimisticLockError",
        # Mixins - Relation Preload
        "RelationPreloadMixin",
        "requires_relations",
        "requires_for_update",
        # Mixins - Cached Table
        "CachedTableBaseMixin",
        # Mixins - Info DTOs
        "IntIdInfoMixin",
        "UUIDIdInfoMixin",
        "DatetimeInfoMixin",
        "IntIdDatetimeInfoMixin",
        "UUIDIdDatetimeInfoMixin",
        # Field types - path/string/numeric/custom
        "DirectoryPathType",
        "FilePathType",
        # Str ladder (whole range — guards against export drift)
        "Str16",
        "Str24",
        "Str32",
        "Str36",
        "Str48",
        "Str64",
        "Str100",
        "Str128",
        "Str255",
        "Str256",
        "Str500",
        "Str512",
        "Str2048",
        # Text ladder (whole range — guards against export drift)
        "Text1K",
        "Text1024",
        "Text2K",
        "Text2500",
        "Text3K",
        "Text5K",
        "Text10K",
        "Text16K",
        "Text32K",
        "Text60K",
        "Text64K",
        "Text100K",
        "Text128K",
        "Text1M",
        "Port",
        "Percentage",
        "PositiveInt",
        "NonNegativeInt",
        "PositiveBigInt",
        "PositiveFloat",
        "IPAddress",
        "Url",
        "HttpUrl",
        "WebSocketUrl",
        "SafeHttpUrl",
        "UnsafeURLError",
        "validate_not_private_host",
        "ModuleNameMixin",
        # RLC
        "RelationLoadChecker",
        "RelationLoadWarning",
        "RelationLoadCheckMiddleware",
        "run_model_checks",
        "mark_app_check_completed",
    ],
)
def test_public_symbol_is_exported(name: str) -> None:
    """Every documented public symbol must be importable from top level."""
    import sqlmodel_ext

    assert hasattr(sqlmodel_ext, name), f"sqlmodel_ext.{name} is not exported"
