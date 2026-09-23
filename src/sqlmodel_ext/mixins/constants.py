"""
Semantic cache TTL constants for ``CachedTableBaseMixin``.

Named values for the ``__cache_ttl__`` ClassVar / ``cache_ttl=`` class keyword::

    class Character(CachedTableBaseMixin, CharacterBase, UUIDTableBaseMixin, table=True, cache_ttl=CACHE_TTL_WARM):
        ...
"""

CACHE_TTL_HOT: int = 600
"""TTL for frequently changing models (10 minutes)."""

CACHE_TTL_WARM: int = 1800
"""TTL for models that change occasionally (30 minutes)."""

CACHE_TTL_COLD: int = 3600
"""TTL for rarely changing models (1 hour; the ``CachedTableBaseMixin`` default)."""
