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
        "SQLModelExtConfig",
        "Unset",
        "OMITTED_SENTINEL",
        "EXCLUDE_IF_NONE",
        "select",
        # Exceptions
        "RecordNotFoundError",
        # Pagination
        "ListResponse",
        "TimeFilterRequest",
        "PaginationRequest",
        "TableViewRequest",
        # Enhanced session
        "AsyncSession",
        # Mixins - Table
        "SESSION_FOR_UPDATE_KEY",
        "TableBaseMixin",
        "UUIDTableBaseMixin",
        "rel",
        "cond",
        # Mixins - Polymorphic
        "PolymorphicBaseMixin",
        "AutoPolymorphicIdentityMixin",
        "DeferredIndex",
        "CustomTableArg",
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
        "Str1",
        "max_length_of",
        # Text ladder (whole range — guards against export drift)
        "Text1K",
        "Text1024",
        "Text2K",
        "Text2500",
        "Text3K",
        "Text5K",
        "Text8K",
        "Text10K",
        "Text16K",
        "Text32K",
        "Text48K",
        "Text60K",
        "Text64K",
        "Text100K",
        "Text128K",
        "Text1M",
        "Text3072",
        "Text4K",
        # Non-empty string ladders
        "NonEmptyStr64",
        "NonEmptyStr128",
        "NonEmptyStr256",
        "NonEmptyStrippedStr64",
        "NonEmptyStrippedStr128",
        "NonEmptyStrippedStr256",
        "Sha256Hex",
        "BCP47LanguageCode",
        "NonEmptyStrippedStr32",
        "SingleLineStr64",
        "SearchQueryStr64",
        "HttpHeaderName",
        "INT32_MIN",
        "SignedBigInt",
        "Port",
        "Percentage",
        "PositiveInt",
        "NonNegativeInt",
        "PositiveBigInt",
        "PositiveFloat",
        # Decimal ladder
        "SignedDecimal38_18",
        "NonNegativeDecimal38_18",
        "PositiveDecimal38_18",
        "OptionalNonNegativeDecimal38_18",
        "SignedDecimal20_10",
        "NonNegativeDecimal20_10",
        "OptionalNonNegativeDecimal20_10",
        "OptionalSignedDecimal38_18",
        "NullableNonNegativeDecimal20_10",
        "SignedWriteDecimal38_18",
        "NonNegativeWriteDecimal38_18",
        "PositiveWriteDecimal38_18",
        "OptionalNonNegativeWriteDecimal38_18",
        "OptionalSignedWriteDecimal38_18",
        "SignedSumDecimal38_18",
        "DECIMAL_38_18_COLUMN_DIGITS",
        "DECIMAL_38_18_WRITE_DIGITS",
        "DECIMAL_38_18_PLACES",
        # Bounded-length list aliases
        "List",
        "List1",
        "List2",
        "List3",
        "List7",
        "List10",
        "List16",
        "List20",
        "List32",
        "List40",
        "List50",
        "List64",
        "List100",
        "List128",
        "List200",
        "List256",
        "List1024",
        "IPAddress",
        "ClientIPAddress",
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
        # Session (REPEATABLE READ / factory)
        "SessionFactory",
        "RepeatableReadSnapshotConflictError",
        "SerializationRetryExhaustedError",
        # Pagination (page window / keyset)
        "PageWindowRequest",
        "DEFAULT_PAGE_SIZE",
        "MAX_PAGE_SIZE",
        "MAX_TABLE_VIEW_OFFSET",
        # FastAPI dependency for query-parameter DTOs
        "query_dependency",
        # Table extras
        "SESSION_REPEATABLE_READ_KEY",
        "GroupSumRow",
        "uuid7",
        "FK_DELETE_RESTRICT_FALLBACK_MESSAGE",
        "ResourceReferencedError",
        "KeysetCursorError",
        "KeysetCursorInvalidError",
        "KeysetCursorUnsupportedError",
        # Lock / isolation decorators
        "requires_locked_param",
        "requires_read_committed",
        "requires_repeatable_read",
        "validate_locked_instances",
        # Cache
        "CACHE_TTL_HOT",
        "CACHE_TTL_WARM",
        "CACHE_TTL_COLD",
        "collect_migration_invalidation_tasks",
        "run_pending_migration_cache_invalidations",
        # Resource quota
        "ResourceQuotaMixin",
        "QuotaExceededError",
        "QuotaOwnerNotFoundError",
        "CallerDidNotCommitError",
        # Mixin table scan / trigram search
        "MixinTableScanMixin",
        "BIGRAM_FUNCTION_SQL",
        "TrgmSearchableMixin",
        "TrgmSearchRequest",
    ],
)
def test_public_symbol_is_exported(name: str) -> None:
    """Every documented public symbol must be importable from top level."""
    import sqlmodel_ext

    assert hasattr(sqlmodel_ext, name), f"sqlmodel_ext.{name} is not exported"


def test_package_ships_py_typed_marker() -> None:
    """PEP 561: without ``py.typed`` type checkers treat the package as untyped."""
    from importlib.resources import files

    assert files("sqlmodel_ext").joinpath("py.typed").is_file()


_REEXPORTING_PACKAGES = (
    "sqlmodel_ext",
    "sqlmodel_ext.mixins",
    "sqlmodel_ext.field_types.mixins",
    "sqlmodel_ext.field_types.dialects.postgresql",
)


@pytest.mark.parametrize("package", _REEXPORTING_PACKAGES)
def test_reexports_use_redundant_alias(package: str) -> None:
    """Every ``from ... import X`` in a re-exporting ``__init__`` must be ``X as X``.

    In a ``py.typed`` package, pyright/basedpyright treat a plain
    ``from m import X`` inside ``__init__.py`` as a private import, so a consumer
    writing ``from sqlmodel_ext import X`` gets ``reportPrivateImportUsage``.
    """
    import ast
    import importlib
    from pathlib import Path

    module = importlib.import_module(package)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    offenders = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname != alias.name
    ]
    assert offenders == [], f"{package}: re-exported without 'X as X': {offenders}"
