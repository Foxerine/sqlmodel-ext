"""
Relation Preloading Mixin

Provides method-level relationship declaration and on-demand incremental loading,
preventing MissingGreenlet errors while maintaining optimal SQL query count.

Design principles:
- On-demand: Only loads relationships needed by the called method
- Incremental: Already-loaded relationships are not re-loaded
- Optimal: Same relationship queried only once, different relationships loaded incrementally
- Zero-invasive: Callers need no changes
- Commit-safe: Uses SQLAlchemy inspect to detect real loading state

Usage::

    from sqlmodel_ext.mixins import RelationPreloadMixin
    from sqlmodel_ext.mixins.relation_preload import requires_relations

    class MyFunction(RelationPreloadMixin, Function, table=True):
        generator: Generator = Relationship(...)

        @requires_relations('generator', Generator.config)
        async def cost(self, params, context, session) -> int:
            return self.generator.config.price  # auto-loaded
"""
import inspect as python_inspect
import logging
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Any

from sqlalchemy import inspect as sa_inspect
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.main import RelationshipInfo

logger = logging.getLogger(__name__)

P = ParamSpec('P')
R = TypeVar('R')


def _extract_session(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AsyncSession | None:
    """
    Extract AsyncSession from method parameters.

    Search order:
    1. kwargs named 'session'
    2. Positional arg at 'session' parameter position
    3. kwargs with AsyncSession type
    """
    if 'session' in kwargs:
        return kwargs['session']

    try:
        sig = python_inspect.signature(func)
        param_names = list(sig.parameters.keys())

        if 'session' in param_names:
            idx = param_names.index('session') - 1  # subtract self
            if 0 <= idx < len(args):
                return args[idx]
    except (ValueError, TypeError):
        pass

    for value in kwargs.values():
        if isinstance(value, AsyncSession):
            return value

    return None


def _is_obj_relation_loaded(obj: Any, rel_name: str) -> bool:
    """
    Check if an object's relationship is loaded.

    :param obj: The object to check
    :param rel_name: Relationship attribute name
    :returns: True if loaded, False if unloaded or expired
    """
    try:
        state = sa_inspect(obj)
        return rel_name not in state.unloaded
    except Exception:
        return False


def _find_relation_to_class(from_class: type, to_class: type) -> str | None:
    """
    Find a relationship attribute name pointing from one class to another.

    :param from_class: Source class
    :param to_class: Target class
    :returns: Relationship attribute name, or None
    """
    for attr_name in dir(from_class):
        try:
            attr = getattr(from_class, attr_name, None)
            if attr is None:
                continue
            if hasattr(attr, 'property') and hasattr(attr.property, 'mapper'):
                target_class = attr.property.mapper.class_
                if target_class == to_class:
                    return attr_name
        except AttributeError:
            continue
    return None


def requires_relations(*relations: str | RelationshipInfo) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator declaring method's required relationships with auto-loading.

    Parameter formats:
    - String: attribute name on this class (e.g. ``'generator'``)
    - RelationshipInfo: external class attribute (e.g. ``Generator.config``)

    Behavior:
    - Checks if relationships are loaded before method execution
    - Unloaded relationships are incrementally loaded (single query)
    - Already-loaded relationships are skipped

    Supports both regular async methods and async generators.

    Example::

        @requires_relations('generator', Generator.config)
        async def cost(self, params, context, session) -> int:
            return self.generator.config.price
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        is_async_gen = python_inspect.isasyncgenfunction(func)

        if is_async_gen:
            @wraps(func)
            async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
                session = _extract_session(func, args, kwargs)
                if session is not None:
                    await self._ensure_relations_loaded(session, relations)
                async for item in func(self, *args, **kwargs):
                    yield item  # type: ignore
        else:
            @wraps(func)
            async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
                session = _extract_session(func, args, kwargs)
                if session is not None:
                    await self._ensure_relations_loaded(session, relations)
                return await func(self, *args, **kwargs)

        wrapper._required_relations = relations  # type: ignore
        return wrapper

    return decorator


class RelationPreloadMixin:
    """
    Relation Preloading Mixin.

    Provides on-demand incremental loading to ensure optimal SQL query count.

    Features:
    - On-demand: Only loads relationships needed by the called method
    - Incremental: Already-loaded relationships are not re-loaded
    - In-place update: Modifies self directly, no instance replacement
    - Import-time validation: String relationship names verified at class creation
    - Commit-safe: Uses SQLAlchemy inspect for real state detection
    """

    def __init_subclass__(cls, **kwargs) -> None:
        """Validate all @requires_relations declarations at class creation time."""
        super().__init_subclass__(**kwargs)

        all_annotations: set[str] = set()
        for klass in cls.__mro__:
            if hasattr(klass, '__annotations__'):
                all_annotations.update(klass.__annotations__.keys())

        sqlmodel_relationships: set[str] = set()
        for klass in cls.__mro__:
            if hasattr(klass, '__sqlmodel_relationships__'):
                sqlmodel_relationships.update(klass.__sqlmodel_relationships__.keys())

        all_available_names = all_annotations | sqlmodel_relationships

        for method_name in dir(cls):
            if method_name.startswith('__'):
                continue

            try:
                method = getattr(cls, method_name, None)
            except AttributeError:
                continue

            if method is None or not hasattr(method, '_required_relations'):
                continue

            for spec in method._required_relations:
                if isinstance(spec, str):
                    if spec not in all_available_names and not hasattr(cls, spec):
                        raise AttributeError(
                            f"{cls.__name__}.{method_name} declares relation '{spec}', "
                            f"but {cls.__name__} has no such attribute"
                        )

    def _is_relation_loaded(self, rel_name: str) -> bool:
        """
        Check if a relationship is truly loaded (via SQLAlchemy inspect).

        Handles commit-induced expiration automatically.

        :param rel_name: Relationship attribute name
        :returns: True if loaded, False if unloaded or expired
        """
        try:
            state = sa_inspect(self)
            return rel_name not in state.unloaded
        except Exception:
            return False

    async def _ensure_relations_loaded(
        self,
        session: AsyncSession,
        relations: tuple[str | RelationshipInfo, ...],
    ) -> None:
        """
        Ensure specified relationships are loaded, incrementally loading missing ones.

        :param session: Database session
        :param relations: Required relationship specs
        """
        to_load: list[str | RelationshipInfo] = []
        direct_keys: set[str] = set()
        nested_parent_keys: set[str] = set()

        for rel in relations:
            if isinstance(rel, str):
                if not self._is_relation_loaded(rel):
                    to_load.append(rel)
                    direct_keys.add(rel)
            else:
                parent_class = rel.parent.class_
                parent_attr = _find_relation_to_class(self.__class__, parent_class)

                if parent_attr is None:
                    logger.warning(
                        f"Cannot find relationship path from {self.__class__.__name__} "
                        f"to {parent_class.__name__}, cannot check if {rel.key} is loaded"
                    )
                    to_load.append(rel)
                    continue

                if not self._is_relation_loaded(parent_attr):
                    if parent_attr not in direct_keys and parent_attr not in nested_parent_keys:
                        to_load.append(parent_attr)
                        nested_parent_keys.add(parent_attr)
                    to_load.append(rel)
                else:
                    parent_obj = getattr(self, parent_attr)
                    if not _is_obj_relation_loaded(parent_obj, rel.key):
                        if parent_attr not in direct_keys and parent_attr not in nested_parent_keys:
                            to_load.append(parent_attr)
                            nested_parent_keys.add(parent_attr)
                        to_load.append(rel)

        if not to_load:
            return

        load_options = self._specs_to_load_options(to_load)
        if not load_options:
            return

        state = sa_inspect(self)
        pk_tuple = state.key[1] if state.key else None
        if pk_tuple is None:
            logger.warning(f"Cannot get primary key for {self.__class__.__name__}")
            return
        pk_value = pk_tuple[0]

        fresh = await self.__class__.get(
            session,
            self.__class__.id == pk_value,
            load=load_options,
        )

        if fresh is None:
            logger.warning(f"Cannot load relations: {self.__class__.__name__} id={self.id} not found")
            return

        all_direct_keys = direct_keys | nested_parent_keys
        for key in all_direct_keys:
            value = getattr(fresh, key, None)
            object.__setattr__(self, key, value)

    def _specs_to_load_options(
        self,
        specs: list[str | RelationshipInfo],
    ) -> list[RelationshipInfo]:
        """
        Convert relationship specs to load parameters.

        - String -> ``cls.{name}``
        - RelationshipInfo -> used directly
        """
        result: list[RelationshipInfo] = []

        for spec in specs:
            if isinstance(spec, str):
                rel = getattr(self.__class__, spec, None)
                if rel is not None:
                    result.append(rel)
                else:
                    logger.warning(f"Relation '{spec}' not found on {self.__class__.__name__}")
            else:
                result.append(spec)

        return result

    # ==================== Optional manual preload API ====================

    @classmethod
    def get_relations_for_method(cls, method_name: str) -> list[RelationshipInfo]:
        """
        Get relationships declared by a specific method.

        :param method_name: Method name
        :returns: List of RelationshipInfo
        """
        method = getattr(cls, method_name, None)
        if method is None or not hasattr(method, '_required_relations'):
            return []

        result: list[RelationshipInfo] = []
        for spec in method._required_relations:
            if isinstance(spec, str):
                rel = getattr(cls, spec, None)
                if rel:
                    result.append(rel)
            else:
                result.append(spec)

        return result

    @classmethod
    def get_relations_for_methods(cls, *method_names: str) -> list[RelationshipInfo]:
        """
        Get deduplicated relationships for multiple methods.

        :param method_names: Method names
        :returns: Deduplicated list of RelationshipInfo
        """
        seen: set[str] = set()
        result: list[RelationshipInfo] = []

        for method_name in method_names:
            for rel in cls.get_relations_for_method(method_name):
                key = rel.key
                if key not in seen:
                    seen.add(key)
                    result.append(rel)

        return result

    async def preload_for(self, session: AsyncSession, *method_names: str) -> 'RelationPreloadMixin':
        """
        Manually preload relationships for specified methods.

        Usually not needed -- the decorator handles this automatically.

        :param session: Database session
        :param method_names: Method names whose relationships to preload
        :returns: self (supports chaining)
        """
        all_relations: list[str | RelationshipInfo] = []

        for method_name in method_names:
            method = getattr(self.__class__, method_name, None)
            if method and hasattr(method, '_required_relations'):
                all_relations.extend(method._required_relations)

        if all_relations:
            await self._ensure_relations_loaded(session, tuple(all_relations))

        return self
