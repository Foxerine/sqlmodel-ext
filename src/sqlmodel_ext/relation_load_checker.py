"""
Relation Load Checker -- static analysis for async SQLAlchemy relationship access.

.. warning::

    **Experimental**. This module is off by default. The AST analyzer is
    tightly coupled to a specific project layout (FastAPI endpoints, STI
    inheritance conventions, internal naming patterns) and may produce false
    positives or crash on code it has not seen before. Opt in explicitly by
    setting ``check_on_startup = True`` AFTER evaluating whether your project
    matches the assumptions listed in the README. The module API is not
    covered by semver stability guarantees.

Startup-time AST analysis to detect unloaded relationship access in coroutines,
preventing MissingGreenlet errors before any request is served.

Dual-layer protection:
    1. AST static analysis (primary, this module)
    2. ``lazy='raise_on_sql'`` runtime safety net (user must inject via metaclass)

Analysis scope:
    - SQLModel model methods (auto, after configure_mappers)
    - FastAPI endpoints (auto, ASGI middleware on startup)
    - Project coroutines in all imported modules (auto, same as above)

Detection rules:
    - RLC001: response_model contains relationship fields not preloaded
    - RLC002: access to relationship after save()/update() without load=
    - RLC003: access to relationship without prior load= (only for locally obtained vars)
    - RLC005: dependency function does not preload relationships required by response_model
    - RLC007: column access on expired (post-commit) object triggers synchronous lazy load -> MissingGreenlet
    - RLC008: calling business methods on expired (post-commit) object (method internals may access expired columns -> MissingGreenlet)
    - RLC010: passing expired ORM objects as arguments to functions/methods (callee may access expired columns -> MissingGreenlet)
    - RLC011: implicit dunder method triggers relationship access (e.g. ``if not obj:`` triggers __len__() / ``for x in obj:`` triggers __iter__())
    - RLC012: response_model contains STI subclass-specific columns while the endpoint returns STI base-class query results (heterogeneous serialization accesses missing columns -> MissingGreenlet)

Opt-in auto-check::

    import sqlmodel_ext.relation_load_checker as rlc
    rlc.check_on_startup = True  # experimental; off by default

    # In your package __init__.py, after configure_mappers():
    from sqlmodel_ext.relation_load_checker import run_model_checks
    run_model_checks(SQLModelBase)

    # In your main.py:
    from sqlmodel_ext.relation_load_checker import RelationLoadCheckMiddleware
    app.add_middleware(RelationLoadCheckMiddleware)

Manual check (fallback)::

    from sqlmodel_ext.relation_load_checker import RelationLoadChecker
    checker = RelationLoadChecker(SQLModelBase)
    warnings = checker.check_model_methods()
    warnings += checker.check_app(app)
"""
import atexit
import ast
import inspect as python_inspect
import logging
import os
import re
import sys
import textwrap
import types
import typing
from dataclasses import dataclass, field
from typing import Annotated, Any, Self, TypeVar, Union, override

from sqlalchemy import inspect as sa_inspect
from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSession

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = '<unknown>'

# Conditional FastAPI import: the library must be usable without FastAPI installed.
try:
    from fastapi.params import Depends as _FastAPIDependsClass
    _HAS_FASTAPI = True
except ImportError:
    _FastAPIDependsClass = None  # type: ignore
    _HAS_FASTAPI = False


# ========================= Auto-check configuration =========================

check_on_startup: bool = False
"""Auto-check switch on startup.

Defaults to ``False``: the relation load checker is **experimental** and must
be opted into explicitly. Set to ``True`` AFTER evaluating whether your project
layout matches the analyzer's assumptions (FastAPI endpoints, STI inheritance
conventions, ``save``/``update``/``delete`` naming, etc.). When the flag is
``False``, ``run_model_checks``, ``RelationLoadCheckMiddleware``, and every
auto-check short-circuit immediately.
"""

_base_class: type | None = None
"""Cached base_class reference (set by run_model_checks)."""

_model_check_completed: bool = False
"""Whether model method checks have completed."""

_app_check_completed: bool = False
"""Whether app endpoint/coroutine checks have completed."""

_PROJECT_ROOT: str = os.getcwd()
"""Auto-detected project root directory (defaults to cwd for the standalone library)."""


@dataclass
class RelationLoadWarning:
    """Relation load static analysis warning."""
    code: str
    """Rule code (RLC001-RLC012)."""
    file: str
    """File path."""
    line: int
    """Line number."""
    message: str
    """Warning details."""

    @override
    def __str__(self) -> str:
        return f"[{self.code}] {self.file}:{self.line} - {self.message}"


# save/update return refreshed self (used for fine-grained tracking within commit methods)
_REFRESH_METHODS = frozenset({'save', 'update'})


@dataclass
class _TrackedVar:
    """Tracked variable state."""
    model_name: str
    """Model class name."""
    loaded_rels: set[str] = field(default_factory=set)
    """Set of loaded relationship names."""
    post_commit: bool = False
    """Whether the object has been through save/update/delete (may be expired)."""
    caller_provided: bool = False
    """Caller-provided param (e.g. self, function params); pre-commit access skips RLC003."""
    line: int = 0
    """Definition/last-update line number."""


def _collect_noreturn_names(func: Any) -> frozenset[str]:
    """
    Collect NoReturn function names from the module namespace of a function.

    Uses runtime type annotations to detect functions returning ``NoReturn``.
    Used by ``_branch_unconditionally_returns()`` to recognize calls that never return
    (e.g. ``raise_bad_request()``).
    """
    module = python_inspect.getmodule(func)
    if module is None:
        return frozenset()
    names: set[str] = set()
    for attr_name, obj in vars(module).items():
        if not callable(obj):
            continue
        try:
            hints = typing.get_type_hints(obj)
        except Exception:
            continue
        if hints.get('return') is typing.NoReturn:
            names.add(attr_name)
    return frozenset(names)


class RelationLoadChecker:
    """
    Startup-time relation load static analyzer.

    Uses AST analysis to detect unloaded relationship access in coroutines.
    Run after ``configure_mappers()`` and before serving requests.
    """

    def __init__(self, base_class: type) -> None:
        # model class name -> set of relationship attribute names
        self.model_relationships: dict[str, set[str]] = {}
        # model class name -> {relationship name -> target model class name}
        self.model_rel_targets: dict[str, dict[str, str]] = {}
        # model class name -> set of column attribute names (includes PK)
        self.model_columns: dict[str, set[str]] = {}
        # model class name -> actual class object
        self.model_classes: dict[str, type] = {}
        # Analyzed function ids for dedup
        self._analyzed_func_ids: set[int] = set()
        # Auto-discovered method behaviors (type system as single source of truth)
        self.commit_methods: frozenset[str] = frozenset()
        self.model_returning_methods: frozenset[str] = frozenset()
        self.sync_model_returning_methods: frozenset[str] = frozenset()
        # For model-returning commit methods, the set where the return comes
        # from save/update(commit!=False). These method return values are
        # refreshed inside save(), so callers may safely access column attrs.
        self.refreshing_commit_methods: frozenset[str] = frozenset()
        # Per-class commit methods (MRO-aware, model class name -> effective commit method set)
        self._model_commit_methods: dict[str, frozenset[str]] = {}
        # Intermediate state (for _discover_non_model_commit_methods to extend incrementally)
        self._method_asts: dict[str, list[tuple[str, str, ast.Module]]] = {}
        self._class_commit: dict[str, set[str]] = {}

        self._build_knowledge_base(base_class)
        (
            self.commit_methods, self.model_returning_methods,
            self.sync_model_returning_methods, self.refreshing_commit_methods,
        ) = self._discover_method_behaviors()
        # Extend commit method set: scan non-model project classes (e.g. Messages)
        # for transitive commit methods.
        self._discover_non_model_commit_methods()
        # model_name -> {dunder_name -> set of accessed relationship names}
        self.model_dunder_rels: dict[str, dict[str, set[str]]] = {}
        self._scan_dunder_relationship_access()

    def _build_knowledge_base(self, base_class: type) -> None:
        """Build model knowledge base from SQLAlchemy mappers."""
        for mapper in base_class._sa_registry.mappers:
            cls = mapper.class_
            cls_name = cls.__name__
            rel_names: set[str] = set()
            rel_targets: dict[str, str] = {}
            for rel in mapper.relationships:
                rel_names.add(rel.key)
                rel_targets[rel.key] = rel.mapper.class_.__name__
            self.model_relationships[cls_name] = rel_names
            self.model_rel_targets[cls_name] = rel_targets
            self.model_columns[cls_name] = {
                col.key for col in mapper.column_attrs
            }
            self.model_classes[cls_name] = cls

    def _discover_method_behaviors(self) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
        """
        Single-pass discovery of commit methods and model-returning methods.

        Uses the type system as single source of truth:

        **Commit methods** (anchored on ``AsyncSession.commit()``):

        1. Scan all model classes and their bases for async methods accepting ``AsyncSession``
        2. Check if the method body calls ``.commit()`` / ``.rollback()`` on that parameter
        3. Transitive closure: methods that call step-2 methods passing the session are also commit methods

        **Model-returning methods** (anchored on return type annotations):

        1. Inspect the method's return type annotation
        2. If it returns ``Self``, a concrete model class, ``Self | None``, ``T``, etc. -> model-returning
        3. Sync methods only used for variable tracking (not added to safe_methods)

        :returns: (commit_methods, model_returning_methods, sync_model_returning_methods, refreshing_commit_methods)
        """
        # method_name -> ALL versions of (owning_class, session_param_name, AST) across class hierarchy
        # All versions are kept (no dedup) so that commit-method discovery can
        # check base-class session.commit() sites. For example,
        # CachedTableBaseMixin.save() does not directly call session.commit(),
        # but TableBaseMixin.save() does -- both versions must participate in Phase 1.
        # owning_class records the class in which the method is defined,
        # used for per-class commit tracking in Phase 2.
        method_asts: dict[str, list[tuple[str, str, ast.Module]]] = {}
        # method_name -> owning class name (for resolving Self -> concrete class)
        method_owners: dict[str, str] = {}
        # method_name -> return type hint
        method_return_hints: dict[str, Any] = {}
        seen_func_ids: set[int] = set()

        for cls_name, cls in self.model_classes.items():
            for klass in cls.__mro__:
                if klass is object:
                    continue
                for attr_name in vars(klass):
                    if attr_name.startswith('__') and attr_name.endswith('__'):
                        continue
                    raw = vars(klass)[attr_name]
                    func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
                    if not callable(func):
                        continue
                    func_id = id(func)
                    if func_id in seen_func_ids:
                        continue
                    seen_func_ids.add(func_id)

                    # Only analyze async methods
                    if not (python_inspect.iscoroutinefunction(func)
                            or python_inspect.isasyncgenfunction(func)):
                        continue

                    # Get type hints (for session param discovery and return type analysis)
                    # get_type_hints may fail due to third-party ForwardRef
                    # (e.g. SQLAlchemy's _OnClauseArgument with ForwardRef('ColumnElement[_T]'))
                    # Fall back to __annotations__ in that case
                    try:
                        hints = typing.get_type_hints(func)
                    except Exception:
                        hints = None

                    # Find the AsyncSession-typed parameter name
                    session_param: str | None = None
                    if hints is not None:
                        for param_name, hint in hints.items():
                            if hint is _AsyncSession:
                                session_param = param_name
                                break
                    if session_param is None:
                        # Fallback: check __annotations__ for AsyncSession type
                        raw_annotations = getattr(func, '__annotations__', {})
                        for param_name, hint in raw_annotations.items():
                            if hint is _AsyncSession:
                                session_param = param_name
                                break

                    if session_param is None:
                        continue

                    try:
                        source = textwrap.dedent(python_inspect.getsource(func))
                        tree = ast.parse(source)
                    except (OSError, TypeError, SyntaxError):
                        continue

                    # All versions are saved for commit-method discovery
                    # (Phase 1/2 must check every version).
                    method_asts.setdefault(attr_name, []).append((klass.__name__, session_param, tree))

                    # Metadata only keeps the first version (MRO order, most specific subclass first)
                    if attr_name not in method_owners:
                        method_owners[attr_name] = cls_name

                    # Record return type (only the first version)
                    if attr_name not in method_return_hints:
                        return_hint = (
                            hints.get('return') if hints is not None
                            else getattr(func, '__annotations__', {}).get('return')
                        )
                        if return_hint is not None:
                            method_return_hints[attr_name] = return_hint

        # -------- Commit method discovery --------

        # Phase 1: methods that directly call session.commit() / session.rollback()
        # Check every version: if any version contains a direct session.commit(),
        # mark the method as a commit method.
        commit_methods: set[str] = set()
        # Per-class commit tracking: defining_class -> set of commit method names
        class_commit: dict[str, set[str]] = {}
        for method_name, versions in method_asts.items():
            for owning_cls, sp, tree in versions:
                if _ast_has_typed_commit(tree, sp):
                    commit_methods.add(method_name)
                    class_commit.setdefault(owning_cls, set()).add(method_name)

        # Phase 2: transitive closure -- methods that call commit methods passing the session
        # Per-class tracking: when the callee type can be resolved from the AST,
        # check that type's commit state to avoid false positives from same-named
        # methods on different classes (e.g. QQStats.get_instance commits but
        # S3APIClient.get_instance does not).
        model_class_names = frozenset(self.model_classes.keys())
        changed = True
        while changed:
            changed = False
            for method_name, versions in method_asts.items():
                for owning_cls, sp, tree in versions:
                    if method_name in class_commit.get(owning_cls, set()):
                        continue
                    if _ast_calls_commit_method_with_session(
                        tree, sp, frozenset(commit_methods),
                        owning_class=owning_cls,
                        class_commit=class_commit,
                        model_classes=self.model_classes,
                        model_class_names=model_class_names,
                    ):
                        class_commit.setdefault(owning_cls, set()).add(method_name)
                        commit_methods.add(method_name)
                        changed = True

        # Save intermediate state for _discover_non_model_commit_methods to extend incrementally
        self._method_asts = method_asts
        self._class_commit = class_commit

        # -------- Per-model-class commit methods (MRO-aware) --------
        self._rebuild_model_commit_methods(class_commit)

        # -------- Model-returning method discovery --------

        def _is_model_type(hint: Any) -> bool:
            """Check if type is a known model class."""
            return isinstance(hint, type) and hint.__name__ in self.model_relationships

        def _hint_returns_model(hint: Any) -> bool:
            """Recursively check if return type annotation contains a model type."""
            # Self -> returns model
            if hint is Self:
                return True

            # Direct model class
            if _is_model_type(hint):
                return True

            # TypeVar (T) -> only used in known contexts, treat as model (conservative but safe)
            if isinstance(hint, TypeVar):
                # Check if bound is a model class
                if hint.__bound__ is not None and _is_model_type(hint.__bound__):
                    return True
                # Unbound TypeVar (e.g. save's T) -- in model method context, treat as model
                return True

            origin = typing.get_origin(hint)

            # Union/Optional: Self | None, T | None
            # Handle both typing.Union (Optional[X], Union[X, Y]) and types.UnionType (X | Y)
            if origin is Union or origin is types.UnionType:  # pyright: ignore[reportDeprecated]
                return any(
                    _hint_returns_model(arg)
                    for arg in typing.get_args(hint)
                    if arg is not type(None)
                )

            # list[T], list[Self]
            if origin is list:
                args = typing.get_args(hint)
                if args:
                    return _hint_returns_model(args[0])

            # tuple[T, bool] (get_or_create pattern)
            if origin is tuple:
                args = typing.get_args(hint)
                if args:
                    return _hint_returns_model(args[0])

            # String forward reference (e.g. -> 'UserCharacterConfig')
            if isinstance(hint, str) and hint in self.model_relationships:
                return True

            return False

        model_returning: set[str] = set()
        for method_name, return_hint in method_return_hints.items():
            if _hint_returns_model(return_hint):
                model_returning.add(method_name)

        # -------- Sync model-returning method discovery --------
        # Sync methods can't commit, but sync methods returning model instances need tracking
        # (e.g. get_tool_by_name -> Tool), so that post-commit calls on expired objects
        # can be detected by RLC008.
        # Only checks return type annotations, no AST parsing (sync methods have no session ops).
        sync_model_returning: set[str] = set()
        sync_seen_func_ids: set[int] = set()
        for cls in self.model_classes.values():
            for klass in cls.__mro__:
                if klass is object:
                    continue
                for attr_name in vars(klass):
                    if attr_name.startswith('__') and attr_name.endswith('__'):
                        continue
                    raw = vars(klass)[attr_name]
                    func = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
                    if not callable(func):
                        continue
                    func_id = id(func)
                    if func_id in sync_seen_func_ids:
                        continue
                    sync_seen_func_ids.add(func_id)
                    # Only process sync methods (async already handled above)
                    if (python_inspect.iscoroutinefunction(func)
                            or python_inspect.isasyncgenfunction(func)):
                        continue
                    try:
                        hints = typing.get_type_hints(func)
                    except Exception:
                        hints = getattr(func, '__annotations__', {})
                    return_hint = hints.get('return') if hints else None
                    if return_hint is not None and _hint_returns_model(return_hint):
                        sync_model_returning.add(attr_name)

        # -------- Phase 3: internally-refreshing model-returning commit methods (transitive closure) --------
        # If a method's return comes from save/update(commit!=False) or from another refreshing method,
        # the return value has already been refreshed internally. Callers may safely access column attrs.
        # Transitive closure: fill_from_video_url -> fill_from_url -> fill_from_file_path -> save()
        refreshing_commit: set[str] = set(_REFRESH_METHODS)
        changed = True
        while changed:
            changed = False
            for method_name, versions in method_asts.items():
                if method_name in refreshing_commit:
                    continue
                if method_name not in commit_methods or method_name not in model_returning:
                    continue
                for _owning_cls, _sp, tree in versions:
                    if _method_returns_from_refreshing(tree, frozenset(refreshing_commit)):
                        refreshing_commit.add(method_name)
                        changed = True
                        break
        refreshing_commit -= _REFRESH_METHODS  # save/update already handled via _REFRESH_METHODS

        logger.debug(f"Auto-discovered commit methods: {sorted(commit_methods)}")
        logger.debug(f"Auto-discovered model-returning methods: {sorted(model_returning)}")
        if sync_model_returning:
            logger.debug(f"Auto-discovered sync model-returning methods: {sorted(sync_model_returning)}")
        if refreshing_commit:
            logger.debug(f"Auto-discovered refreshing commit methods: {sorted(refreshing_commit)}")
        return (
            frozenset(commit_methods), frozenset(model_returning),
            frozenset(sync_model_returning), frozenset(refreshing_commit),
        )

    def _rebuild_model_commit_methods(
            self,
            class_commit: dict[str, set[str]],
    ) -> None:
        """
        Build the per-model-class commit method set (MRO-aware).

        For each model class, walk the MRO looking for the most specific
        definition to determine which commit methods are effective for it.
        """
        model_commit_methods: dict[str, frozenset[str]] = {}
        for mcls_name, mcls in self.model_classes.items():
            effective: set[str] = set()
            seen_attrs: set[str] = set()
            for klass in mcls.__mro__:
                if klass is object:
                    continue
                klass_name = klass.__name__
                for attr_name in vars(klass):
                    if attr_name in seen_attrs:
                        continue
                    seen_attrs.add(attr_name)
                    if attr_name in class_commit.get(klass_name, set()):
                        effective.add(attr_name)
            model_commit_methods[mcls_name] = frozenset(effective)
        self._model_commit_methods = model_commit_methods

    # ========================= Dunder relationship access scanning =========================

    _DUNDERS_TO_SCAN: tuple[str, ...] = ('__len__', '__bool__', '__iter__', '__contains__', '__getitem__')

    def _scan_dunder_relationship_access(self) -> None:
        """
        Scan model classes' dunder methods for implicit relationship attribute access.

        When code uses ``if not obj:``, ``for x in obj:``, ``len(obj)`` etc.,
        Python implicitly calls __bool__/__len__/__iter__ etc. dunder methods.
        If these methods internally access unloaded relationship attributes,
        they will trigger ``lazy='raise_on_sql'`` errors.

        Scan strategy:
        - Iterate each model class's MRO (excluding object)
        - AST-analyze each dunder method to find ``self.attr`` accesses
        - Cross-reference attrs with model's relationship set
        - Record results to model_dunder_rels

        Typical case::

            class ToolSetBase(SQLModelBase):
                def __len__(self) -> int:
                    return len(self.tools)  # accesses 'tools' relationship

            class ToolSet(ToolSetBase, UUIDTableBaseMixin):
                tools: list[Tool] = Relationship(...)

            # Dangerous: if not tool_set: -> __len__() -> self.tools -> raise_on_sql
        """
        for model_name, cls in self.model_classes.items():
            rels = self.model_relationships.get(model_name, set())
            if not rels:
                continue

            dunder_rels: dict[str, set[str]] = {}

            for klass in cls.__mro__:
                if klass is object:
                    continue
                for dunder in self._DUNDERS_TO_SCAN:
                    if dunder in dunder_rels:
                        continue  # MRO order: more specific subclass takes priority
                    method = vars(klass).get(dunder)
                    if method is None:
                        continue
                    # AST-analyze the dunder method body for self.attr access
                    try:
                        source = textwrap.dedent(python_inspect.getsource(method))
                        tree = ast.parse(source)
                    except (OSError, TypeError, SyntaxError):
                        continue

                    accessed_rels: set[str] = set()
                    for node in ast.walk(tree):
                        if (
                            isinstance(node, ast.Attribute)
                            and isinstance(node.value, ast.Name)
                            and node.value.id == 'self'
                            and node.attr in rels
                        ):
                            accessed_rels.add(node.attr)

                    # Record even if accessed_rels is empty (indicates the dunder exists
                    # but doesn't access relations). Important for __bool__/__len__ fallback:
                    # if __bool__ exists (even without rel access), Python won't fall back to __len__.
                    dunder_rels[dunder] = accessed_rels

            if dunder_rels:
                self.model_dunder_rels[model_name] = dunder_rels

        if self.model_dunder_rels:
            # Only log models that actually access relationships, filter out empty-set noise
            interesting = {
                model: dunders
                for model, dunders in self.model_dunder_rels.items()
                if any(rels for rels in dunders.values())
            }
            if interesting:
                logger.debug(f"Found dunder relationship access: {interesting}")

    # ========================= Non-model class commit method discovery =========================

    def _discover_non_model_commit_methods(self) -> None:
        """
        Scan imported non-model project classes for transitive commit methods
        and propagate the findings back to model methods.

        Model-class commit methods are covered by ``_discover_method_behaviors()``.
        Non-model classes (e.g. ``Messages``) may internally call ``model.save(session)``
        which triggers a commit, forming a transitive commit chain.

        Flow:

        1. Collect async method ASTs from non-model classes accepting ``AsyncSession``
        2. Phase 1: detect direct ``session.commit()`` calls
        3. Phase 2: merge model + non-model AST sets and compute the transitive closure (bi-directional)
        4. Update ``commit_methods`` and ``_model_commit_methods``
        """
        project_root = _PROJECT_ROOT.replace('\\', '/')

        # -------- Collect non-model class method ASTs --------
        non_model_asts: dict[str, list[tuple[str, str, ast.Module]]] = {}
        seen_func_ids: set[int] = set()

        for _module_name, module in list(sys.modules.items()):
            module_file = getattr(module, '__file__', None)
            if module_file is None:
                continue
            module_file_normalized = module_file.replace('\\', '/')
            if not module_file_normalized.startswith(project_root):
                continue
            if '/site-packages/' in module_file_normalized:
                continue

            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue
                if not python_inspect.isclass(attr):
                    continue
                if attr.__module__ != _module_name:
                    continue
                if attr.__name__ in self.model_classes:
                    continue

                for method_name in vars(attr):
                    if method_name.startswith('__') and method_name.endswith('__'):
                        continue
                    raw = vars(attr)[method_name]
                    func = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
                    if not callable(func):
                        continue
                    func_id = id(func)
                    if func_id in seen_func_ids:
                        continue
                    seen_func_ids.add(func_id)

                    if not (python_inspect.iscoroutinefunction(func)
                            or python_inspect.isasyncgenfunction(func)):
                        continue

                    # Check AsyncSession parameter.
                    # Non-model classes commonly import AsyncSession under TYPE_CHECKING
                    # (as a string annotation at runtime), so get_type_hints() may fail
                    # because of the ForwardRef. The fallback matches both the string
                    # ``'AsyncSession'`` and the real class object.
                    try:
                        hints = typing.get_type_hints(func)
                    except Exception:
                        hints = None
                    session_param: str | None = None
                    if hints is not None:
                        for pname, hint in hints.items():
                            if hint is _AsyncSession:
                                session_param = pname
                                break
                    if session_param is None:
                        raw_annotations = getattr(func, '__annotations__', {})
                        for pname, hint in raw_annotations.items():
                            if hint is _AsyncSession or hint == 'AsyncSession':
                                session_param = pname
                                break
                    if session_param is None:
                        continue

                    try:
                        source = textwrap.dedent(python_inspect.getsource(func))
                        tree = ast.parse(source)
                    except (OSError, TypeError, SyntaxError):
                        continue

                    non_model_asts.setdefault(method_name, []).append(
                        (attr.__name__, session_param, tree)
                    )

        if not non_model_asts:
            return

        # -------- Merge AST sets --------
        combined_asts: dict[str, list[tuple[str, str, ast.Module]]] = {}
        for name, versions in self._method_asts.items():
            combined_asts.setdefault(name, []).extend(versions)
        for name, versions in non_model_asts.items():
            combined_asts.setdefault(name, []).extend(versions)

        # -------- Phase 1: non-model methods calling session.commit() directly --------
        class_commit = self._class_commit
        new_commits: set[str] = set()
        for method_name, versions in non_model_asts.items():
            for owning_cls, sp, tree in versions:
                if _ast_has_typed_commit(tree, sp):
                    new_commits.add(method_name)
                    class_commit.setdefault(owning_cls, set()).add(method_name)

        # -------- Phase 2: merge transitive closure (model + non-model bidirectional) --------
        all_commit = set(self.commit_methods) | new_commits
        model_class_names = frozenset(self.model_classes.keys())
        changed = True
        while changed:
            changed = False
            for method_name, versions in combined_asts.items():
                for owning_cls, sp, tree in versions:
                    if method_name in class_commit.get(owning_cls, set()):
                        continue
                    if _ast_calls_commit_method_with_session(
                        tree, sp, frozenset(all_commit),
                        owning_class=owning_cls,
                        class_commit=class_commit,
                        model_classes=self.model_classes,
                        model_class_names=model_class_names,
                    ):
                        class_commit.setdefault(owning_cls, set()).add(method_name)
                        all_commit.add(method_name)
                        changed = True

        new_methods = all_commit - set(self.commit_methods)
        if new_methods:
            logger.debug(f"Non-model class transitive commit methods: {sorted(new_methods)}")
            self.commit_methods = frozenset(all_commit)
            self._rebuild_model_commit_methods(class_commit)

    # ========================= Public API =========================

    def check_app(self, app: Any) -> list[RelationLoadWarning]:
        """
        Analyze all registered FastAPI route endpoints.

        :param app: FastAPI application instance
        :returns: all detected warnings
        """
        warnings: list[RelationLoadWarning] = []

        for route in app.routes:
            if not hasattr(route, 'endpoint'):
                continue
            endpoint = route.endpoint
            self._analyzed_func_ids.add(id(endpoint))
            response_model = getattr(route, 'response_model', None)
            path = getattr(route, 'path', '???')

            try:
                endpoint_warnings = self._check_endpoint(
                    endpoint, response_model, path,
                )
                warnings.extend(endpoint_warnings)
            except Exception as e:
                logger.debug(f"Error analyzing endpoint {path}: {e}")

        return warnings

    def check_model_methods(self) -> list[RelationLoadWarning]:
        """
        Analyze all mapped model classes' async methods (rich model methods).

        Iterates all mapper-registered model classes, analyzes their directly
        defined async methods. Traverses MRO to analyze inherited methods
        (consistent with _discover_method_behaviors).

        For ``self`` parameter:
        - Marked as caller_provided (caller responsible for preloading, skips RLC003)
        - Parses ``@requires_relations`` decorator for self's loaded relations
        - save()/update() still triggers RLC002 (post-commit expiration)
        """
        warnings: list[RelationLoadWarning] = []

        for cls_name, cls in self.model_classes.items():
            # Traverse MRO to analyze inherited methods (e.g. ToolSetBase.execute_tool via ToolSet)
            # Consistent with _discover_method_behaviors, ensuring all methods are analyzed
            for klass in cls.__mro__:
                if klass is object:
                    continue
                for attr_name in vars(klass):
                    if attr_name.startswith('__') and attr_name.endswith('__'):
                        continue

                    raw_attr = vars(klass)[attr_name]
                    # Unwrap staticmethod/classmethod
                    func = raw_attr.__func__ if isinstance(raw_attr, (staticmethod, classmethod)) else raw_attr

                    if not (python_inspect.iscoroutinefunction(func)
                            or python_inspect.isasyncgenfunction(func)):
                        continue

                    if id(func) in self._analyzed_func_ids:
                        continue
                    self._analyzed_func_ids.add(id(func))

                    label = f"{cls_name}.{attr_name}"

                    try:
                        method_warnings = self._check_model_method(
                            func=func,
                            cls_name=cls_name,
                            label=label,
                        )
                        warnings.extend(method_warnings)
                    except Exception as e:
                        logger.warning(f"Error analyzing model method {label} (possible mixed type annotation issue): {e}")
                        # Try to get source file info for better error reporting
                        try:
                            source_file = python_inspect.getfile(func)
                            line_num = python_inspect.getsourcelines(func)[1]
                        except (TypeError, OSError):
                            source_file = UNKNOWN_LABEL
                            line_num = 0
                        warnings.append(RelationLoadWarning(
                            code='RLC009',
                            file=source_file,
                            line=line_num,
                            message=(
                                f"Type annotation parse failure for {label}: {e}. "
                                f"Check for mixed resolved types and string forward references "
                                f"(e.g. `type[T] | 'tuple[...]'`); wrap the entire union in a string: "
                                f"`'type[T] | tuple[...]'`"
                            ),
                        ))

        return self._filter_noqa_suppressions(warnings)

    @staticmethod
    def _filter_noqa_suppressions(
            warnings: list[RelationLoadWarning],
    ) -> list[RelationLoadWarning]:
        """
        Filter warnings suppressed by ``# noqa: RLCxxx`` comments.

        Supported formats::

            return result  # noqa: RLC007
            return result  # noqa: RLC007, RLC010

        :param warnings: raw warning list
        :return: filtered warning list
        """
        if not warnings:
            return warnings

        source_cache: dict[str, list[str]] = {}
        filtered: list[RelationLoadWarning] = []
        _noqa_re = re.compile(r'#\s*noqa:\s*(.+)')
        _code_re = re.compile(r'RLC\d+')

        for w in warnings:
            if w.file not in source_cache:
                try:
                    with open(w.file, encoding='utf-8') as f:
                        source_cache[w.file] = f.readlines()
                except (OSError, UnicodeDecodeError):
                    source_cache[w.file] = []

            lines = source_cache[w.file]
            suppressed = False
            if 0 < w.line <= len(lines):
                m = _noqa_re.search(lines[w.line - 1])
                if m:
                    codes = set(_code_re.findall(m.group(1)))
                    if w.code in codes:
                        suppressed = True

            if not suppressed:
                filtered.append(w)

        return filtered

    def check_project_coroutines(
        self,
        project_root: str,
        skip_paths: list[str] | None = None,
        skip_third_party_attrs: bool = False,
    ) -> list[RelationLoadWarning]:
        """
        Scan all imported modules' async functions and async generators.

        Iterates sys.modules, analyzing coroutine functions and async generators
        from project source files. Includes module-level functions and methods
        of non-SQLModel classes (e.g. command handlers, service classes).
        Automatically skips functions already analyzed by check_app/check_model_methods.

        :param project_root: absolute path to project root directory
        :param skip_paths: list of path fragments to skip (e.g. ['/base/', '/mixin/'])
        :param skip_third_party_attrs: skip third-party library attributes in project modules.
            When imported third-party libs use lazy proxies (e.g. openai.AudioProxy),
            inspect operations may trigger client initialization and raise exceptions.
            When enabled, those exceptions are caught and the attribute is skipped.
        """
        warnings: list[RelationLoadWarning] = []
        # Normalize path separators
        project_root_normalized = project_root.replace('\\', '/')

        default_skip = skip_paths or []

        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
            module_file = getattr(module, '__file__', None)
            if module_file is None:
                continue
            module_file_normalized = module_file.replace('\\', '/')
            if not module_file_normalized.startswith(project_root_normalized):
                continue
            # Skip third-party libraries (venv site-packages paths may start with
            # the project root but are not project code)
            if '/site-packages/' in module_file_normalized:
                continue
            # Skip configured paths
            if any(skip in module_file_normalized for skip in default_skip):
                continue

            # Collect functions to analyze: module-level + class methods
            funcs_to_check: list[tuple[str, Any]] = []

            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue

                # Third-party lazy proxies (e.g. openai.AudioProxy) may trigger
                # client initialization and raise on attribute access
                try:
                    is_async = self._is_async_callable(attr)
                    is_class = not is_async and python_inspect.isclass(attr)
                except Exception:
                    if skip_third_party_attrs:
                        continue
                    raise

                if is_async:
                    # Module-level async function / async generator.
                    # Unwrap decorator wrappers (e.g. pytest FixtureFunctionDefinition)
                    # and analyze the original function -- the wrapper may be missing
                    # __annotations__/__globals__.
                    actual_func = getattr(attr, '__wrapped__', attr)
                    func_module = getattr(actual_func, '__module__', None)
                    if func_module == module_name:
                        funcs_to_check.append((f"{module_name}.{attr_name}", actual_func))
                elif is_class:
                    # Non-model class methods (model classes already covered by check_model_methods)
                    if attr.__module__ != module_name:
                        continue
                    if attr.__name__ in self.model_classes:
                        continue  # Model classes analyzed by check_model_methods
                    for method_name in vars(attr):
                        if method_name.startswith('__') and method_name.endswith('__'):
                            continue
                        raw = vars(attr)[method_name]
                        # Unwrap classmethod / staticmethod
                        func = raw
                        if isinstance(raw, (classmethod, staticmethod)):
                            func = raw.__func__
                        try:
                            is_func_async = self._is_async_callable(func)
                        except Exception:
                            if skip_third_party_attrs:
                                continue
                            raise
                        if is_func_async:
                            # Unwrap __wrapped__ (same logic as module-level functions)
                            actual_func = getattr(func, '__wrapped__', func)
                            funcs_to_check.append(
                                (f"{module_name}.{attr.__name__}.{method_name}", actual_func),
                            )

            for label, func in funcs_to_check:
                if id(func) in self._analyzed_func_ids:
                    continue
                self._analyzed_func_ids.add(id(func))

                try:
                    func_warnings = self._check_coroutine(
                        func=func, label=label,
                    )
                    warnings.extend(func_warnings)
                except Exception as e:
                    logger.debug(f"Error analyzing coroutine {label}: {e}")

        return warnings

    @staticmethod
    def _is_async_callable(obj: Any) -> bool:
        """Check whether obj is an async callable (coroutine function or async generator).

        Supports __wrapped__ unwrapping (e.g. decorator wrappers like pytest
        FixtureFunctionDefinition).
        """
        if python_inspect.iscoroutinefunction(obj) or python_inspect.isasyncgenfunction(obj):
            return True
        # Unwrap __wrapped__ (PEP 362 / functools.wraps protocol)
        wrapped = getattr(obj, '__wrapped__', None)
        if wrapped is not None:
            return (python_inspect.iscoroutinefunction(wrapped)
                    or python_inspect.isasyncgenfunction(wrapped))
        return False

    def check_function(self, func: Any) -> list[RelationLoadWarning]:
        """
        Analyze a single function (for testing or standalone checks).

        :param func: function to analyze
        :returns: detected warnings
        """
        return self._check_coroutine(func, label='<standalone>')

    # ========================= Internal analysis methods =========================

    def _check_endpoint(
        self,
        endpoint: Any,
        response_model: type | None,
        path: str,
    ) -> list[RelationLoadWarning]:
        """Check a FastAPI endpoint (with response_model and Depends analysis)."""
        # 1. Resolve parameter model types
        param_models = self._resolve_param_models(endpoint)

        # 2. Get response_model relationship fields
        required_rels = self._get_response_model_relationships(response_model)

        # 3. Analyze dependency function load= usage
        dep_loads = self._analyze_dependencies(endpoint)

        # 4. AST analysis
        # Endpoint params come from Depends, their load= tracked via dep_loads,
        # so not marked as caller_provided (RLC003 checks normally)
        warnings, analyzer = self._analyze_function_body(
            func=endpoint,
            param_models=param_models,
            required_rels=required_rels,
            dep_loads=dep_loads,
            label=path,
            caller_provided_params=set(),
        )

        # 5. RLC005: dependency not preloading response_model required rels
        if required_rels and analyzer:
            self._check_rlc005(
                warnings, required_rels, dep_loads,
                param_models, analyzer, endpoint, path,
            )

        # 6. RLC012: STI response_model column compatibility check
        self._check_rlc012(warnings, response_model, endpoint, path)

        return warnings

    def _check_model_method(
        self,
        func: Any,
        cls_name: str,
        label: str,
    ) -> list[RelationLoadWarning]:
        """
        Check a model method (with self tracking and @requires_relations parsing).

        self is marked as caller_provided; pre-commit access skips RLC003.
        But self.save() followed by relationship access still triggers RLC002.
        """
        # Parse AST to extract @requires_relations
        source_file, tree, line_offset = self._parse_function_source(func)
        if tree is None:
            return []

        func_node = self._find_function_node(tree, func.__name__)
        if func_node is None:
            return []

        # Extract declared loaded relations from @requires_relations
        decorator_loads = self._extract_requires_relations_loads(func_node)

        # Build param_models
        param_models = self._resolve_param_models(func)

        # Detect instance method or classmethod
        sig = python_inspect.signature(func)
        first_param = next(iter(sig.parameters), None)
        caller_provided_params: set[str] = set()

        # cls -> class alias (classmethod's cls parameter doesn't enter tracked_vars,
        # only used for resolving class-level calls)
        class_aliases: dict[str, str] = {}
        if first_param == 'self':
            param_models['self'] = cls_name
            caller_provided_params.add('self')
        elif first_param == 'cls':
            class_aliases['cls'] = cls_name

        # @requires_relations declared rels as self's dep_loads
        dep_loads: dict[str, set[str]] = {}
        if 'self' in param_models and decorator_loads:
            dep_loads['self'] = decorator_loads

        warnings, _ = self._analyze_function_body(
            func=func,
            param_models=param_models,
            required_rels={},
            dep_loads=dep_loads,
            label=label,
            caller_provided_params=caller_provided_params,
            pre_parsed=(source_file, tree, line_offset),
            class_aliases=class_aliases,
        )
        return warnings

    def _check_coroutine(
        self,
        func: Any,
        label: str,
    ) -> list[RelationLoadWarning]:
        """Check a regular coroutine function (background tasks, stream handlers, etc.)."""
        param_models = self._resolve_param_models(func)

        # All params are caller-provided, skip RLC003
        warnings, _ = self._analyze_function_body(
            func=func,
            param_models=param_models,
            required_rels={},
            dep_loads={},
            label=label,
            caller_provided_params=set(param_models.keys()),
        )
        return warnings

    def _analyze_function_body(
        self,
        func: Any,
        param_models: dict[str, str],
        required_rels: dict[str, str],
        dep_loads: dict[str, set[str]],
        label: str,
        caller_provided_params: set[str],
        pre_parsed: tuple[str, ast.Module, int] | None = None,
        class_aliases: dict[str, str] | None = None,
    ) -> tuple[list[RelationLoadWarning], '_FunctionAnalyzer | None']:
        """
        Core AST analysis: parse function body and run _FunctionAnalyzer.

        :param caller_provided_params: set of caller-provided parameter names, skip RLC003
        :param pre_parsed: pre-parsed (source_file, tree, line_offset) to avoid re-parsing
        :param class_aliases: class alias mapping (e.g. cls -> UserFile) for resolving classmethod calls
        """
        if pre_parsed is not None:
            source_file, tree, line_offset = pre_parsed
        else:
            source_file, tree, line_offset = self._parse_function_source(func)

        if tree is None:
            return [], None

        func_node = self._find_function_node(tree, func.__name__)
        if func_node is None:
            return [], None

        noreturn_names = _collect_noreturn_names(func)

        # Extract AsyncSession-typed parameter names (used to distinguish model commit
        # calls from same-named non-model calls)
        session_param_names: set[str] = set()
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = getattr(func, '__annotations__', {})
        for p_name, p_hint in hints.items():
            if p_hint is _AsyncSession:
                session_param_names.add(p_name)

        analyzer = _FunctionAnalyzer(
            model_relationships=self.model_relationships,
            model_columns=self.model_columns,
            param_models=param_models,
            dep_loads=dep_loads,
            required_rels=required_rels,
            source_file=source_file,
            line_offset=line_offset,
            path=label,
            caller_provided_params=caller_provided_params,
            commit_methods=self.commit_methods,
            model_returning_methods=self.model_returning_methods,
            sync_model_returning_methods=self.sync_model_returning_methods,
            class_aliases=class_aliases,
            model_dunder_rels=self.model_dunder_rels,
            noreturn_names=noreturn_names,
            session_param_names=frozenset(session_param_names),
            model_commit_methods=self._model_commit_methods,
            model_rel_targets=self.model_rel_targets,
            refreshing_commit_methods=self.refreshing_commit_methods,
        )
        analyzer.visit(func_node)

        return list(analyzer.warnings), analyzer

    # ========================= RLC005 check =========================

    def _check_rlc005(
        self,
        warnings: list[RelationLoadWarning],
        required_rels: dict[str, str],
        dep_loads: dict[str, set[str]],
        param_models: dict[str, str],
        analyzer: '_FunctionAnalyzer',
        endpoint: Any,
        path: str,
    ) -> None:
        """RLC005: dependency does not preload response_model required relationships."""
        try:
            source_file = python_inspect.getfile(endpoint)
        except (TypeError, OSError):
            source_file = UNKNOWN_LABEL
        try:
            line_offset = python_inspect.getsourcelines(endpoint)[1] - 1
        except (OSError, TypeError):
            line_offset = 0

        for rel_name, model_name in required_rels.items():
            loaded_anywhere = False
            # Check if loaded in dependencies
            for param_name, loaded_set in dep_loads.items():
                if param_name in param_models and param_models[param_name] == model_name:
                    if rel_name in loaded_set:
                        loaded_anywhere = True
                        break
            # Check if loaded in function body
            if not loaded_anywhere:
                for var in analyzer.tracked_vars.values():
                    if var.model_name == model_name and rel_name in var.loaded_rels:
                        loaded_anywhere = True
                        break
            if not loaded_anywhere:
                warnings.append(RelationLoadWarning(
                    code='RLC005',
                    file=source_file,
                    line=line_offset + 1,
                    message=(
                        f"Endpoint {path}: response_model requires {model_name}.{rel_name}, "
                        f"but no corresponding load= found in dependency or endpoint body"
                    ),
                ))

    # ========================= Type resolution =========================

    def _resolve_param_models(self, func: Any) -> dict[str, str]:
        """
        Resolve function parameter model types.

        Handles ``Annotated[Model, Depends(...)]`` type aliases.
        First tries ``get_type_hints()`` for batch resolution; if it fails
        (e.g. TYPE_CHECKING forward references), falls back to per-parameter
        ``__annotations__`` parsing to ensure resolvable params aren't missed.

        :returns: param_name -> model_class_name
        """
        param_models: dict[str, str] = {}

        try:
            hints = typing.get_type_hints(func, include_extras=True)
        except Exception:
            # get_type_hints may fail due to TYPE_CHECKING forward references.
            # Fall back to per-parameter __annotations__ (skip unresolvable string annotations)
            hints = {}
            annotations: dict[str, Any] = getattr(func, '__annotations__', {})
            for param_name, annotation in annotations.items():
                if isinstance(annotation, str):
                    continue  # Skip unresolvable string annotations
                hints[param_name] = annotation

        for param_name, hint in hints.items():
            model_name = self._extract_model_from_hint(hint)
            if model_name is not None:
                param_models[param_name] = model_name

        # Discover model attributes in non-model parameter types (e.g. CommandContext.user: User)
        # Generate chain tracking keys (e.g. "ctx.user" -> "User") so ctx.user.attr can be detected
        for param_name, hint in hints.items():
            if param_name in param_models or param_name == 'return':
                continue
            actual_type = self._unwrap_to_class(hint)
            if actual_type is None:
                continue
            # Skip known model types (already handled as direct params)
            if actual_type.__name__ in self.model_relationships:
                continue
            # Check class's __init__ annotations for model type attributes
            try:
                init_hints = typing.get_type_hints(actual_type.__init__)
            except Exception:
                continue
            for attr_name, attr_hint in init_hints.items():
                if attr_name in ('self', 'return'):
                    continue
                attr_model = self._extract_model_from_hint(attr_hint)
                if attr_model is not None:
                    param_models[f"{param_name}.{attr_name}"] = attr_model

        return param_models

    def _extract_model_from_hint(self, hint: Any) -> str | None:
        """
        Extract a model class name from a type annotation.

        Delegates to ``_unwrap_to_class()`` to strip ``Annotated``, ``X | None``
        etc. wrappers, then checks whether the unwrapped class is a known ORM
        model.
        """
        cls = self._unwrap_to_class(hint)
        if cls is not None and cls.__name__ in self.model_relationships:
            return cls.__name__
        return None

    def _unwrap_to_class(self, hint: Any) -> type | None:
        """
        Extract the actual class from a type annotation.

        Handles ``Annotated[X, ...]``, ``X | None``, ``Optional[X]`` etc. wrappers,
        returning the innermost actual type.

        :returns: actual type or None (when extraction is not possible)
        """
        origin = typing.get_origin(hint)

        # Annotated[X, ...] -> unwrap X
        if origin is Annotated:
            args = typing.get_args(hint)
            if args:
                return self._unwrap_to_class(args[0])

        # X | None (UnionType) or Optional[X] (Union[X, None])
        # typing.Union is deprecated in 3.10+ but still needed for Optional's legacy Union
        if origin is types.UnionType or origin is Union:  # pyright: ignore[reportDeprecated]
            args = typing.get_args(hint)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._unwrap_to_class(non_none[0])

        # Direct class
        if isinstance(hint, type):
            return hint

        return None

    def _get_response_model_relationships(
        self,
        response_model: type | None,
    ) -> dict[str, str]:
        """
        Get relationship fields from response_model.

        Traverses response_model MRO to find the corresponding table model's relationships.

        :returns: field_name -> model_class_name
        """
        if response_model is None:
            return {}

        # Handle generic types like ListResponse[T]
        origin = typing.get_origin(response_model)
        if origin is not None:
            args = typing.get_args(response_model)
            if args:
                return self._get_response_model_relationships(args[0])

        if not hasattr(response_model, 'model_fields'):
            return {}

        required: dict[str, str] = {}

        # Find the corresponding table model in response_model's MRO
        for base in response_model.__mro__:
            base_name = base.__name__
            if base_name not in self.model_relationships:
                continue
            rels = self.model_relationships[base_name]
            for field_name in response_model.model_fields:
                if field_name in rels:
                    required[field_name] = base_name
            break  # Use only the nearest table model

        return required

    # ========================= RLC012: STI column compatibility check =========================

    @staticmethod
    def _get_pydantic_generic_args(hint: Any) -> tuple[Any, ...]:
        """
        Extract the concrete generic arguments from a Pydantic parameterized model.

        Pydantic v2's ``ListResponse[T]`` creates a fully concretized
        ``ModelMetaclass`` instance where both ``typing.get_origin()`` and
        ``typing.get_args()`` return empty values. The real generic arguments
        are stored in ``__pydantic_generic_metadata__['args']``.

        :returns: tuple of generic arguments, or an empty tuple when missing
        """
        # Try the standard typing API first
        args = typing.get_args(hint)
        if args:
            return args
        # Fallback for concretized Pydantic generics
        pgm = getattr(hint, '__pydantic_generic_metadata__', None)
        if pgm is not None:
            return pgm.get('args', ())
        return ()

    def _unwrap_generic_to_orm_class(self, hint: Any) -> type | None:
        """
        Recursively extract an ORM model class from a type annotation.

        Handles generic containers (e.g. ``ListResponse[ImageGenerator]``),
        ``Annotated``, ``X | None`` wrappers, and Pydantic v2 concretized generics.
        Differs from ``_unwrap_to_class``: when direct unwrap fails, it recurses
        into the generic arguments until it finds a registered ORM class.

        :returns: ORM model class or None
        """
        # Try direct unwrap first (Annotated/Union/direct class)
        cls = self._unwrap_to_class(hint)
        if cls is not None and cls.__name__ in self.model_classes:
            return cls

        # Recurse into generic arguments (including Pydantic concretized generics)
        for arg in self._get_pydantic_generic_args(hint):
            if arg is type(None):
                continue
            result = self._unwrap_generic_to_orm_class(arg)
            if result is not None:
                return result

        return None

    def _unwrap_generic_to_dto_class(self, hint: Any) -> type | None:
        """
        Recursively extract a DTO class (one with ``model_fields``) from a type annotation.

        Skips container types (e.g. ``ListResponse``) and recurses into the
        generic arguments to fetch the inner DTO.

        :returns: inner DTO class or None
        """
        # Check if this is a Pydantic generic container
        # (has __pydantic_generic_metadata__ and the origin is non-null)
        pgm = getattr(hint, '__pydantic_generic_metadata__', None)
        is_pydantic_generic = pgm is not None and pgm.get('origin') is not None

        if not is_pydantic_generic:
            cls = self._unwrap_to_class(hint)
            if cls is not None and hasattr(cls, 'model_fields'):
                return cls

        # Recurse into generic arguments
        for arg in self._get_pydantic_generic_args(hint):
            if arg is type(None):
                continue
            result = self._unwrap_generic_to_dto_class(arg)
            if result is not None:
                return result

        return None

    def _check_rlc012(
        self,
        warnings: list[RelationLoadWarning],
        response_model: type | None,
        endpoint: Any,
        path: str,
    ) -> None:
        """
        RLC012: STI response_model column compatibility check.

        When an endpoint returns STI base-class / intermediate-abstract-class query
        results while its response_model uses a specific subclass DTO, check that
        each column field in the DTO is present on the mapper of every concrete
        STI subclass that could be returned.

        Typical problematic scenario::

            @router.get("", response_model=ListResponse[NanoBananaGeneratorInfoResponse])
            async def list_generators(...) -> ListResponse[ImageGenerator]:
                return await ImageGenerator.get_with_count(...)

        ``NanoBananaGeneratorInfoResponse`` exposes ``input_price``/``llm_id``
        while ``TencentGEM25ImageGenerator`` (a sibling STI subclass that may be
        returned) lacks those columns on its mapper. FastAPI serialization of the
        heterogeneous result triggers a deferred column load -> MissingGreenlet.
        """
        if response_model is None:
            return

        # 1. Extract the ORM model from the endpoint's return type annotation
        try:
            hints = typing.get_type_hints(endpoint)
        except Exception:
            return
        return_hint = hints.get('return')
        if return_hint is None:
            return

        return_model_cls = self._unwrap_generic_to_orm_class(return_hint)
        if return_model_cls is None:
            return

        return_model_name = return_model_cls.__name__

        # 2. Check whether this is an STI polymorphic class
        try:
            mapper = sa_inspect(return_model_cls)
        except Exception:
            return

        if mapper.polymorphic_on is None:
            return  # Not a polymorphic class

        # 3. Collect model_fields of all concrete subclasses (Pydantic layer).
        # Note: we cannot use mapper.column_attrs (STI shared tables mean every
        # subclass has the same column set). We must use model_fields to reflect
        # the fields declared by the Python class. Pydantic serialization calls
        # getattr() on the mapper descriptor; if the field is absent from
        # model_fields the value is not loaded -> deferred IO -> MissingGreenlet.
        concrete_descendants: list[tuple[str, set[str]]] = []
        for sub_mapper in mapper.self_and_descendants:
            if sub_mapper is mapper:
                continue
            if sub_mapper.polymorphic_identity is None:
                continue  # Abstract intermediate class
            sub_cls = sub_mapper.class_
            sub_name = sub_cls.__name__
            sub_fields = set(sub_cls.model_fields.keys()) if hasattr(sub_cls, 'model_fields') else set()
            concrete_descendants.append((sub_name, sub_fields))

        if len(concrete_descendants) < 2:
            return  # Only one concrete subclass, no heterogeneous risk

        # 4. Extract DTO fields from response_model
        dto_cls = self._unwrap_generic_to_dto_class(response_model)
        if dto_cls is None:
            return

        dto_fields = set(dto_cls.model_fields.keys())

        # 5. Build the union of model_fields across all concrete subclasses
        all_sub_fields: set[str] = set()
        for _, fields in concrete_descendants:
            all_sub_fields |= fields

        # 6. For each DTO field, check whether it exists on every subclass's model_fields
        source_file = path
        line = 0
        try:
            source_file = python_inspect.getfile(endpoint)
            _, start_line = python_inspect.getsourcelines(endpoint)
            line = start_line
        except (OSError, TypeError):
            pass

        for field_name in dto_fields:
            if field_name not in all_sub_fields:
                continue  # Not a subclass field at all (computed_field / DTO-only)

            missing_in: list[str] = []
            for sub_name, sub_fields in concrete_descendants:
                if field_name not in sub_fields:
                    missing_in.append(sub_name)

            if missing_in:
                warnings.append(RelationLoadWarning(
                    code='RLC012',
                    file=source_file,
                    line=line,
                    message=(
                        f"response_model field '{field_name}' is missing from the "
                        f"model_fields of the following STI subclasses: "
                        f"{', '.join(missing_in)}. The endpoint returns "
                        f"{return_model_name} (STI base class); the query may yield "
                        f"subclasses lacking this field, so serialization will invoke "
                        f"getattr() and trigger a deferred column load -> MissingGreenlet. "
                        f"Suggestion: build the response_model from fields shared by all "
                        f"subclasses, or filter the query by polymorphic_identity"
                    ),
                ))

    # ========================= Dependency chain analysis =========================

    def _analyze_dependencies(self, endpoint: Any) -> dict[str, set[str]]:
        """
        Analyze load= usage in endpoint dependency functions.

        :returns: param_name -> set of loaded relationship names
        """
        dep_loads: dict[str, set[str]] = {}

        if not _HAS_FASTAPI:
            return dep_loads

        try:
            hints = typing.get_type_hints(endpoint, include_extras=True)
        except Exception:
            return dep_loads

        for param_name, hint in hints.items():
            origin = typing.get_origin(hint)
            if origin is not Annotated:
                continue

            args = typing.get_args(hint)
            for metadata in args[1:]:
                if _FastAPIDependsClass is not None and isinstance(metadata, _FastAPIDependsClass):
                    dep_func = metadata.dependency
                    if dep_func is not None:
                        loaded = self._extract_loads_from_function(dep_func)
                        dep_loads[param_name] = loaded
                    break

        return dep_loads

    def _extract_loads_from_function(self, func: Any) -> set[str]:
        """
        Extract relationship names from load= parameters in a function's AST.

        Handles factory functions (returning closures).
        """
        loaded: set[str] = set()

        # Handle factory functions: e.g. require_character_access("read") returns checker
        actual_func = func
        if hasattr(func, '__wrapped__'):
            actual_func = func.__wrapped__
        # functools.partial
        if hasattr(func, 'func'):
            actual_func = func.func

        try:
            source = python_inspect.getsource(actual_func)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return loaded

        # Extract all load= keyword arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'load':
                loaded.update(_extract_load_value(node.value))

        return loaded

    # ========================= AST utilities =========================

    @staticmethod
    def _parse_function_source(func: Any) -> tuple[str, ast.Module | None, int]:
        """
        Parse function source code.

        Handles ``@wraps`` decorator: uses ``inspect.unwrap()`` to unwrap the
        ``__wrapped__`` chain and get the original function's source code
        (instead of the wrapper's).

        :returns: (source_file, ast_tree_or_None, line_offset)
        """
        # Unwrap @wraps and similar decorator chains to get the original function
        unwrapped = python_inspect.unwrap(func, stop=lambda f: not hasattr(f, '__wrapped__'))

        try:
            source_file = python_inspect.getfile(unwrapped)
        except (TypeError, OSError):
            source_file = UNKNOWN_LABEL

        try:
            source = python_inspect.getsource(unwrapped)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return source_file, None, 0

        try:
            line_offset = python_inspect.getsourcelines(unwrapped)[1] - 1
        except (OSError, TypeError):
            line_offset = 0

        return source_file, tree, line_offset

    @staticmethod
    def _find_function_node(
        tree: ast.Module,
        func_name: str,
    ) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
        """Find a function node by name in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name == func_name:
                    return node
        return None

    @staticmethod
    def _extract_requires_relations_loads(
        func_node: ast.AsyncFunctionDef | ast.FunctionDef,
    ) -> set[str]:
        """
        Extract loaded relationship names from ``@requires_relations`` decorator.

        Supports::

            @requires_relations('rel_name')          -> {'rel_name'}
            @requires_relations('r1', Model.nested)  -> {'r1', 'nested'}
        """
        loaded: set[str] = set()
        for decorator in func_node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            is_requires = (
                (isinstance(func, ast.Name) and func.id == 'requires_relations')
                or (isinstance(func, ast.Attribute) and func.attr == 'requires_relations')
            )
            if not is_requires:
                continue
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    loaded.add(arg.value)
                elif isinstance(arg, ast.Attribute):
                    loaded.add(arg.attr)
        return loaded


# ========================= load= value extraction =========================


def _extract_load_single(node: ast.expr) -> str | None:
    """Extract a relationship name from a single AST node, supporting Model.rel and rel(Model.rel) forms."""
    if isinstance(node, ast.Attribute):
        # load=Model.rel_name
        return node.attr
    elif isinstance(node, ast.Call) and node.args:
        # load=rel(Model.rel_name) -- rel() is a type conversion wrapper
        return _extract_load_single(node.args[0])
    return None


def _extract_load_value(node: ast.expr) -> set[str]:
    """
    Extract relationship names from a load= AST value node.

    Supports:
    - ``load=Model.rel`` -> ``{'rel'}``
    - ``load=rel(Model.rel)`` -> ``{'rel'}``
    - ``load=[Model.r1, rel(Model.r2)]`` -> ``{'r1', 'r2'}``
    """
    result: set[str] = set()

    if isinstance(node, ast.List):
        # load=[Model.r1, rel(Model.r2), ...]
        for elt in node.elts:
            name = _extract_load_single(elt)
            if name is not None:
                result.add(name)
    else:
        # load=Model.rel or load=rel(Model.rel)
        name = _extract_load_single(node)
        if name is not None:
            result.add(name)

    return result


# ========================= Commit method auto-discovery =========================


def _ast_has_typed_commit(tree: ast.Module, session_param: str) -> bool:
    """
    Check if the AST calls ``.commit()`` or ``.rollback()`` on the specified session parameter.

    Only matches ``<session_param>.commit()`` / ``<session_param>.rollback()``,
    anchored by parameter name to avoid false positives.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {'commit', 'rollback'}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == session_param
        ):
            return True
    return False


def _ast_has_keyword_false_static(call: ast.Call, keyword: str) -> bool:
    """Statically check whether an AST Call node has a ``keyword=False`` argument."""
    for kw in call.keywords:
        if (kw.arg == keyword
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False):
            return True
    return False


def _resolve_callee_type(
    value_node: ast.expr,
    owning_class: str | None,
    model_class_names: frozenset[str],
) -> str | None:
    """
    Resolve the model type of a call target from its AST node.

    Supported patterns:

    - ``self.method()`` -> owning_class
    - ``cls.method()`` -> owning_class
    - ``super().method()`` -> owning_class (conservative: same class hierarchy)
    - ``ClassName.method()`` -> ClassName (if it is a known model class)
    """
    if isinstance(value_node, ast.Name):
        if value_node.id in ('self', 'cls') and owning_class is not None:
            return owning_class
        if value_node.id in model_class_names:
            return value_node.id
    elif isinstance(value_node, ast.Call):
        # super().method() pattern
        if (isinstance(value_node.func, ast.Name)
                and value_node.func.id == 'super'
                and owning_class is not None):
            return owning_class
    return None


def _method_returns_from_refreshing(tree: ast.AST, refreshing_methods: frozenset[str]) -> bool:
    """Check whether a method's return value comes from a known refreshing method call.

    Used by the auto-discovery of "internally refreshing" model-returning commit
    methods (transitive closure): if the method's ``return`` statement returns a
    variable/expression whose value came from one of ``refreshing_methods``
    (without an explicit ``commit=False``), the return value has already been
    refreshed internally.

    Patterns checked:
    1. ``return await obj.method(...)`` -- direct return of a refreshing call
    2. ``var = await obj.method(...)`` + ``return var`` -- indirect return
    """

    def _is_refreshing_call(call: ast.Call) -> bool:
        if isinstance(call.func, ast.Attribute) and call.func.attr in refreshing_methods:
            for kw in call.keywords:
                if kw.arg == 'commit' and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    return False
            return True
        return False

    # Collect all variable names assigned by refreshing method calls
    refreshed_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if (isinstance(target, ast.Name)
                    and isinstance(value, ast.Await)
                    and isinstance(value.value, ast.Call)
                    and _is_refreshing_call(value.value)):
                refreshed_vars.add(target.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        # return await obj.refreshing_method(...)
        if (isinstance(node.value, ast.Await)
                and isinstance(node.value.value, ast.Call)
                and _is_refreshing_call(node.value.value)):
            return True
        # return var
        if isinstance(node.value, ast.Name) and node.value.id in refreshed_vars:
            return True

    return False


def _is_commit_for_resolved_type(
    callee_type: str,
    method_name: str,
    class_commit: dict[str, set[str]],
    model_classes: dict[str, type],
) -> bool:
    """
    Check whether a method is a commit method for the resolved type (MRO-aware).

    Walks the MRO to find the most specific definition of the method and checks
    whether that defining class marks it as a commit method. This correctly
    handles subclass overrides: if a subclass overrides a base-class commit
    method without committing, the subclass version wins.

    For non-model classes (not in ``model_classes``), ``class_commit`` is checked
    directly (no MRO resolution).
    """
    cls = model_classes.get(callee_type)
    if cls is None:
        # Non-model class (e.g. Messages): check class_commit directly
        return method_name in class_commit.get(callee_type, set())
    for klass in cls.__mro__:
        if klass is object:
            continue
        klass_name = klass.__name__
        if method_name in vars(klass):
            # Most specific definition found -> check that class's commit state
            return method_name in class_commit.get(klass_name, set())
    return False


def _ast_calls_commit_method_with_session(
    tree: ast.Module,
    session_param: str,
    commit_methods: frozenset[str],
    *,
    owning_class: str | None = None,
    class_commit: dict[str, set[str]] | None = None,
    model_classes: dict[str, type] | None = None,
    model_class_names: frozenset[str] | None = None,
) -> bool:
    """
    Check if the AST calls a known commit method and passes the session parameter.

    Matches patterns:

    - ``await obj.save(session, ...)``
    - ``await cls.from_remote_url(session=session, ...)``

    Excluded patterns (not treated as commit):

    - ``await obj.save(session, commit=False)`` -- commit explicitly disabled

    Enhanced mode (when per-class arguments are provided):

    When the call target's type is resolvable from the AST (self/cls/explicit
    class name), per-class commit state is used to avoid false matches between
    same-named methods on different classes. When the type cannot be resolved,
    falls back to the global name-based match (conservative strategy).
    """
    use_per_class = (
        class_commit is not None
        and model_classes is not None
        and model_class_names is not None
    )

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in commit_methods
        ):
            continue

        method_name = node.func.attr

        # commit=False -> not treated as a commit call
        if _ast_has_keyword_false_static(node, 'commit'):
            continue

        # Check if session is passed as an argument
        found_session = False
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id == session_param:
                found_session = True
                break
        if not found_session:
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id == session_param:
                    found_session = True
                    break
        if not found_session:
            continue

        # Per-class type resolution (enhanced mode)
        if use_per_class:
            assert class_commit is not None
            assert model_classes is not None
            assert model_class_names is not None
            callee_type = _resolve_callee_type(
                node.func.value, owning_class, model_class_names,
            )
            if callee_type is not None:
                # Type resolved -> check that class's commit methods (MRO-aware)
                if _is_commit_for_resolved_type(
                    callee_type, method_name, class_commit, model_classes,
                ):
                    return True
                continue  # This type's method does not commit -> skip this call

        # Type unresolved or per-class mode disabled -> conservative strategy (global name match)
        return True

    return False


# ========================= AST function analyzer =========================


class _FunctionAnalyzer(ast.NodeVisitor):
    """
    AST function analyzer.

    Traverses function body, tracking variable state (model type, loaded rels,
    post-commit status) and detecting unloaded relationship access patterns.

    caller_provided semantics:
    - self / function params provided by caller
    - Pre-commit access skips RLC003 (caller responsible for preloading)
    - Post-commit still triggers RLC002 (save/update expires object)
    """

    def __init__(
        self,
        model_relationships: dict[str, set[str]],
        model_columns: dict[str, set[str]],
        param_models: dict[str, str],
        dep_loads: dict[str, set[str]],
        required_rels: dict[str, str],
        source_file: str,
        line_offset: int,
        path: str,
        caller_provided_params: set[str],
        commit_methods: frozenset[str] | None = None,
        model_returning_methods: frozenset[str] | None = None,
        sync_model_returning_methods: frozenset[str] | None = None,
        class_aliases: dict[str, str] | None = None,
        model_dunder_rels: dict[str, dict[str, set[str]]] | None = None,
        noreturn_names: frozenset[str] | None = None,
        session_param_names: frozenset[str] | None = None,
        model_commit_methods: dict[str, frozenset[str]] | None = None,
        model_rel_targets: dict[str, dict[str, str]] | None = None,
        refreshing_commit_methods: frozenset[str] | None = None,
    ) -> None:
        self.model_relationships: dict[str, set[str]] = model_relationships
        self.model_columns: dict[str, set[str]] = model_columns
        self.model_rel_targets: dict[str, dict[str, str]] = model_rel_targets or {}
        self.required_rels: dict[str, str] = required_rels
        self.source_file: str = source_file
        self.line_offset: int = line_offset
        self.path: str = path
        self.warnings: list[RelationLoadWarning] = []
        self._parent_map: dict[int, ast.AST] = {}
        self.commit_methods: frozenset[str] = commit_methods or frozenset()
        self.model_commit_methods: dict[str, frozenset[str]] = model_commit_methods or {}
        self.model_returning_methods: frozenset[str] = model_returning_methods or frozenset()
        self.noreturn_names: frozenset[str] = noreturn_names or frozenset()
        self.session_param_names: frozenset[str] = session_param_names or frozenset()
        # Complete model-returning set for variable tracking (includes sync methods).
        # Sync methods are NOT added to safe_methods because calling sync methods
        # on expired objects is equally dangerous.
        _sync = sync_model_returning_methods or frozenset()
        self._all_model_returning: frozenset[str] = self.model_returning_methods | _sync
        self.refreshing_commit_methods: frozenset[str] = refreshing_commit_methods or frozenset()
        self.class_aliases: dict[str, str] = class_aliases or {}
        self.model_dunder_rels: dict[str, dict[str, set[str]]] = model_dunder_rels or {}

        # Initialize tracked vars from parameter type annotations
        self.tracked_vars: dict[str, _TrackedVar] = {}
        for param_name, model_name in param_models.items():
            loaded = dep_loads.get(param_name, set())
            self.tracked_vars[param_name] = _TrackedVar(
                model_name=model_name,
                loaded_rels=loaded.copy(),
                post_commit=False,
                caller_provided=param_name in caller_provided_params,
                line=0,
            )

    def _abs_line(self, node: ast.AST) -> int:
        """Get absolute line number."""
        return self.line_offset + getattr(node, 'lineno', 0)

    def _resolve_class_name(self, name: str | None) -> str | None:
        """
        Resolve a name to a known model class name.

        Handles classmethod ``cls`` parameter aliases (e.g. ``cls`` -> ``UserFile``).
        """
        if name is None:
            return None
        resolved = self.class_aliases.get(name, name)
        if resolved in self.model_relationships:
            return resolved
        return None

    def _is_commit_for_call(self, call: ast.Call, method_name: str) -> bool:
        """
        Check whether a method call is a commit call (per-class aware).

        When the call target's model type is known, use the per-class commit
        method set to avoid false matches between same-named methods on
        different classes. Falls back to the global ``commit_methods`` when the
        type is unknown.
        """
        if not self.model_commit_methods:
            return method_name in self.commit_methods

        # Try to resolve the call object's model type
        obj_name = self._get_call_object_name(call)
        resolved_type: str | None = None

        if obj_name is not None:
            if obj_name in self.tracked_vars:
                resolved_type = self.tracked_vars[obj_name].model_name
            else:
                resolved_type = self._resolve_class_name(obj_name)

        if resolved_type is not None:
            # Type resolved -> use per-class commit set
            class_commits = self.model_commit_methods.get(resolved_type, frozenset())
            return method_name in class_commits

        # Type unresolved -> conservative strategy (global match)
        return method_name in self.commit_methods

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        """Check assignment statement."""
        self._handle_attribute_writes(node.targets, node.value)
        # Visit RHS expression in pre-commit state: Python evaluates arguments
        # BEFORE executing the call, so attribute accesses in args/kwargs
        # must be checked before _check_assign potentially expires all tracked vars
        self.visit(node.value)
        self._check_assign(node.targets, node.value, node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check annotated assignment statement."""
        if node.target and node.value:
            self._handle_attribute_writes([node.target], node.value)
            self.visit(node.value)
            self._check_assign([node.target], node.value, node)

    def _handle_attribute_writes(self, targets: list[ast.expr], value: ast.expr) -> None:
        """
        Track attribute writes on tracked variables.

        Handles two patterns:
        1. ``self.attr = fresh_var.attr`` -- same model type from fresh var -> identity map refresh.
           ``Model.get(session, ...)`` returns an object sharing the identity map with ``self``;
           assigning query result attributes to self indicates self was refreshed via identity map,
           clearing the ``post_commit`` flag (all columns and loaded rels are up-to-date).

        2. ``self.rel = some_value`` -- simple attribute write -> only marks that relationship as loaded.
        """
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            # Resolve tracking key: direct variable (user.attr) or chain (ctx.user.attr)
            var_name: str | None = None
            if isinstance(target.value, ast.Name):
                var_name = target.value.id
            else:
                var_name = self._build_chain_key(target.value)
            if var_name is None:
                continue
            attr_name = target.attr

            if var_name not in self.tracked_vars:
                continue
            var_info = self.tracked_vars[var_name]
            if not var_info.post_commit:
                continue

            # Pattern 1: self.attr = fresh_var.attr (same model type -> identity map refresh)
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                src_var_name = value.value.id
                if src_var_name in self.tracked_vars:
                    src_var = self.tracked_vars[src_var_name]
                    if (src_var.model_name == var_info.model_name
                            and not src_var.post_commit):
                        # Identity map: self and fresh_var are the same database row.
                        # The query refreshed all column attributes; rels follow fresh_var's state.
                        var_info.post_commit = False
                        var_info.loaded_rels = src_var.loaded_rels.copy()
                        return

            # Pattern 2: simple relationship attribute write (don't clear post_commit,
            # only mark that specific relationship as loaded)
            rels = self.model_relationships.get(var_info.model_name, set())
            if attr_name in rels:
                var_info.loaded_rels.add(attr_name)

    @override
    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Check expression statement (await without assignment).

        Tracks actual SQLAlchemy commit behavior (auto-discovered, no hardcoded method names):
        - session.commit() / session.rollback() expires all objects
        - Auto-discovered commit methods (save/update/delete/create_duplicate/...) same behavior
        - commit=False only flushes, no expiration
        - Model.get(session, ...) restores column attrs (un-expires, hits Redis cache)
        """
        if isinstance(node.value, ast.Await):
            call = node.value.value
            if isinstance(call, ast.Call):
                # RLC010: check args for expired ORM objects (before commit side effects)
                self._check_expired_call_args(call, node)

                # Visit call children in pre-commit state: Python evaluates arguments
                # BEFORE executing the call, so attribute accesses in args/kwargs
                # must be checked before commit processing potentially expires all tracked vars
                self.visit(call)

                method_name = self._get_method_name(call)

                # session.commit() / session.rollback()
                if method_name in {'commit', 'rollback'}:
                    self._expire_all_tracked_vars()

                # session.refresh(obj)
                elif method_name == 'refresh':
                    self._handle_session_refresh(call)

                # Auto-discovered commit methods (no assignment)
                elif self._is_commit_for_call(call, method_name):
                    obj_name = self._get_call_object_name(call)
                    is_model_call = (
                        obj_name is not None
                        and (obj_name in self.tracked_vars
                             or self._resolve_class_name(obj_name) is not None)
                    )
                    if is_model_call:
                        # RLC008: calling commit method on already-expired object.
                        # Commit methods internally access self.id etc. to build SQL;
                        # on expired objects this triggers synchronous lazy load -> MissingGreenlet.
                        # save/update use ORM primitives (session.add/merge), don't directly
                        # access column attrs, so they are exempt.
                        # self excluded: self's column access already covered by RLC007.
                        if (obj_name != 'self'
                                and obj_name in self.tracked_vars
                                and self.tracked_vars[obj_name].post_commit
                                and method_name not in _REFRESH_METHODS):
                            self.warnings.append(RelationLoadWarning(
                                code='RLC008',
                                file=self.source_file,
                                line=self._abs_line(node),
                                message=(
                                    f"Calling commit method '{method_name}()' on expired "
                                    f"post-commit object '{obj_name}'. The method may internally "
                                    f"access expired column attributes (e.g. self.id) to build SQL, "
                                    f"causing MissingGreenlet. "
                                    f"Suggestion: refresh first with "
                                    f"{obj_name} = await Type.get(session, Type.id == {obj_name}.id)"
                                ),
                            ))
                        commit_disabled = self._has_keyword_false(call, 'commit')
                        if not commit_disabled:
                            self._expire_all_tracked_vars()
                            # save/update default refresh=True, refreshes the call object in-place
                            if (obj_name in self.tracked_vars
                                    and method_name in _REFRESH_METHODS
                                    and not self._has_keyword_false(call, 'refresh')):
                                self.tracked_vars[obj_name].post_commit = False
                            # Model-returning commit method called on self -> identity map refreshes self.
                            # Example: super().fill_from_file_path() internally calls self.save() -> self is refreshed.
                            elif (obj_name == 'self'
                                    and obj_name in self.tracked_vars
                                    and method_name in self._all_model_returning):
                                self.tracked_vars[obj_name].post_commit = False
                    else:
                        # Untracked variable calling a commit method (e.g. loop var user_file.fill_from_image_url()).
                        # Only expire when a session parameter is passed (to distinguish
                        # from non-model calls like redis.delete(key)).
                        if (not self._has_keyword_false(call, 'commit')
                                and self._call_passes_session(call)):
                            self._expire_all_tracked_vars()

                # call children already visited above in pre-commit state
                return

        self.generic_visit(node)

    # ========================= Assignment analysis =========================

    def _check_assign(
        self,
        targets: list[ast.expr],
        value: ast.expr | None,
        node: ast.AST,
    ) -> None:
        """Analyze assignment statement."""
        if value is None:
            return

        # Extract target variable name
        var_name: str | None = None
        for target in targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                break

        if var_name is None:
            return

        # RLC010: check call args for expired ORM objects (before commit side effects)
        # Supports IfExp unwrapping (consistent with the _await_call extraction below)
        if isinstance(value, ast.Await) and isinstance(value.value, ast.Call):
            self._check_expired_call_args(value.value, node)
        elif (isinstance(value, ast.IfExp)
                and isinstance(value.body, ast.Await)
                and isinstance(value.body.value, ast.Call)):
            self._check_expired_call_args(value.body.value, node)
        elif isinstance(value, ast.Call):
            self._check_expired_call_args(value, node)

        # Check await expression.
        # Supports two forms:
        # 1. var = await Model.get(...)                          -> Await(Call)
        # 2. var = (await Model.get(...) if cond else None)      -> IfExp(body=Await(Call), orelse=None)
        # In Python the ``await`` precedence is lower than the ternary, so
        # ``(await X if c else None)`` parses as IfExp(Await(Call), None).
        _await_call: ast.Call | None = None
        if isinstance(value, ast.Await) and isinstance(value.value, ast.Call):
            _await_call = value.value
        elif (isinstance(value, ast.IfExp)
                and isinstance(value.body, ast.Await)
                and isinstance(value.body.value, ast.Call)
                and isinstance(value.orelse, ast.Constant)
                and value.orelse.value is None):
            _await_call = value.body.value

        if _await_call is not None:
            call = _await_call
            method_name = self._get_method_name(call)
            loaded_rels = self._extract_load_from_call(call)

            # Auto-discovered commit methods (with assignment)
            # Includes save/update/delete/get_or_create/create_duplicate etc.
            # Tracks actual SQLAlchemy commit behavior:
            #   commit=True  -> session.commit() -> all objects expire
            #   refresh=True (default, save/update only) -> cls.get() -> column attrs restored
            #   load= -> cls.get(load=) -> specified rels loaded
            #   commit=False -> session.flush() -> no expiration
            if self._is_commit_for_call(call, method_name):
                obj_name = self._get_call_object_name(call)
                class_name = self._get_call_class_name(call)

                # Instance method call (obj.save/update/create_duplicate/...)
                if obj_name and obj_name in self.tracked_vars:
                    old_var = self.tracked_vars[obj_name]

                    # RLC008: calling commit method on already-expired object
                    # (same check as in visit_Expr)
                    if (obj_name != 'self'
                            and old_var.post_commit
                            and method_name not in _REFRESH_METHODS):
                        self.warnings.append(RelationLoadWarning(
                            code='RLC008',
                            file=self.source_file,
                            line=self._abs_line(node),
                            message=(
                                f"Calling commit method '{method_name}()' on expired "
                                f"post-commit object '{obj_name}'. The method may internally "
                                f"access expired column attributes (e.g. self.id) to build SQL, "
                                f"causing MissingGreenlet. "
                                f"Suggestion: refresh first with "
                                f"{obj_name} = await Type.get(session, Type.id == {obj_name}.id)"
                            ),
                        ))

                    commit_disabled = self._has_keyword_false(call, 'commit')

                    if not commit_disabled:
                        self._expire_all_tracked_vars()

                    if method_name in _REFRESH_METHODS:
                        # save/update: returns self, supports refresh= and load= parameters
                        refresh_disabled = self._has_keyword_false(call, 'refresh')
                        if var_name == obj_name:
                            old_var.caller_provided = False
                            if commit_disabled:
                                old_var.post_commit = False
                                old_var.loaded_rels |= loaded_rels
                            elif refresh_disabled:
                                pass  # _expire_all already marked
                            else:
                                old_var.post_commit = False
                                old_var.loaded_rels = loaded_rels
                            old_var.line = self._abs_line(node)
                        else:
                            # save/update returns self; in-place refresh means
                            # both variables point to the refreshed object
                            if commit_disabled:
                                new_rels = old_var.loaded_rels | loaded_rels
                                new_post_commit = False
                            elif refresh_disabled:
                                new_rels = set()
                                new_post_commit = True
                            else:
                                new_rels = loaded_rels
                                new_post_commit = False
                            self.tracked_vars[var_name] = _TrackedVar(
                                model_name=old_var.model_name,
                                loaded_rels=new_rels,
                                post_commit=new_post_commit,
                                caller_provided=False,
                                line=self._abs_line(node),
                            )
                            if not commit_disabled and not refresh_disabled:
                                old_var.post_commit = False
                                old_var.loaded_rels = loaded_rels
                    else:
                        # Other commit methods (non-save/update).
                        # Only model-returning methods' return values are model instances
                        # that need tracking. Non-model-returning methods (e.g. calculate_cost -> int)
                        # don't create tracking variables, avoiding false RLC010 positives
                        # from scalar return values being treated as ORM objects.
                        #
                        # refreshing_commit_methods: methods whose return comes from
                        # save/update(commit!=False); the return value has been refreshed
                        # inside save() -> post_commit=False.
                        # Other commit methods: conservatively post_commit=True (return value may be expired).
                        if method_name in self._all_model_returning:
                            is_refreshing = method_name in self.refreshing_commit_methods
                            self.tracked_vars[var_name] = _TrackedVar(
                                model_name=old_var.model_name,
                                loaded_rels=loaded_rels,
                                post_commit=not commit_disabled and not is_refreshing,
                                caller_provided=False,
                                line=self._abs_line(node),
                            )
                            # Model-returning commit method called on self -> identity map refreshes self
                            if obj_name == 'self':
                                old_var.post_commit = False
                                old_var.loaded_rels = loaded_rels

                # Class method call (Model.get_or_create / cls.get_or_create /...)
                else:
                    resolved = self._resolve_class_name(class_name)
                    if resolved is not None:
                        if not self._has_keyword_false(call, 'commit'):
                            self._expire_all_tracked_vars()
                        self.tracked_vars[var_name] = _TrackedVar(
                            model_name=resolved,
                            loaded_rels=loaded_rels,
                            post_commit=False,
                            caller_provided=False,
                            line=self._abs_line(node),
                        )
                    else:
                        # Untracked variable calling a commit method
                        # (e.g. loop variable user_file.fill_from_image_url()).
                        # Only expire when session is passed (distinguishes from
                        # non-model calls like redis.delete(key)).
                        if (not self._has_keyword_false(call, 'commit')
                                and self._call_passes_session(call)):
                            self._expire_all_tracked_vars()

            # Auto-discovered model-returning methods (non-commit, pure query)
            # Includes get/find_by_content_hash/get_exist_one etc. (including sync methods,
            # e.g. get_tool_by_name)
            # Discovered via return type annotations, no hardcoded method names
            elif method_name in self._all_model_returning:
                obj_name = self._get_call_object_name(call)
                class_name = self._get_call_class_name(call)
                resolved = self._resolve_class_name(class_name)

                # Class method call (Model.get / cls.find_by_content_hash /...)
                if resolved is not None:
                    if self._has_keyword(call, 'options'):
                        effective_rels = self.model_relationships[resolved].copy()
                    else:
                        effective_rels = loaded_rels
                    self.tracked_vars[var_name] = _TrackedVar(
                        model_name=resolved,
                        loaded_rels=effective_rels,
                        post_commit=False,
                        caller_provided=False,
                        line=self._abs_line(node),
                    )

                # Instance method call (tracked_var.some_query_method/...)
                elif obj_name and obj_name in self.tracked_vars:
                    old_var = self.tracked_vars[obj_name]
                    self.tracked_vars[var_name] = _TrackedVar(
                        model_name=old_var.model_name,
                        loaded_rels=loaded_rels,
                        post_commit=False,
                        caller_provided=False,
                        line=self._abs_line(node),
                    )

        # Model constructor call (sync): var = Model(...)
        # Track model instances created by constructors so that subsequent
        # var.save() is recognized as a commit operation
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            class_name = value.func.id
            resolved = self._resolve_class_name(class_name)
            if resolved is not None:
                self.tracked_vars[var_name] = _TrackedVar(
                    model_name=resolved,
                    loaded_rels=set(),
                    post_commit=False,
                    caller_provided=False,
                    line=self._abs_line(node),
                )
            # type(tracked_var) pattern: tool_class = type(tool)
            # When tool is a tracked variable, tool_class becomes an alias for
            # that model class, so subsequent tool = await tool_class.get(...)
            # is correctly resolved as a model class method call
            elif (class_name == 'type'
                    and len(value.args) == 1
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id in self.tracked_vars):
                self.class_aliases[var_name] = self.tracked_vars[value.args[0].id].model_name

        # Relationship attribute extraction: var = tracked_var.relationship_attr
        # Example: llm = self.text_llm -- extracts the model object from a tracked
        # self's relationship and starts tracking it, so subsequent commits can
        # correctly mark var as post_commit and detect MissingGreenlet from
        # column attribute access.
        # caller_provided=True: the extracted object comes from a caller-preloaded
        # relationship, so its own relationship loading state is caller's
        # responsibility; pre-commit RLC003 is not triggered. Post-commit
        # RLC002/RLC007/RLC008 still fire normally.
        if isinstance(value, ast.Attribute):
            src_var_key: str | None = None
            if isinstance(value.value, ast.Name):
                src_var_key = value.value.id
            else:
                src_var_key = self._build_chain_key(value.value)
            if src_var_key is not None and src_var_key in self.tracked_vars:
                src_var = self.tracked_vars[src_var_key]
                rel_attr = value.attr
                target_model = self.model_rel_targets.get(
                    src_var.model_name, {},
                ).get(rel_attr)
                if target_model is not None and target_model in self.model_relationships:
                    self.tracked_vars[var_name] = _TrackedVar(
                        model_name=target_model,
                        loaded_rels=set(),
                        post_commit=src_var.post_commit,
                        caller_provided=True,
                        line=self._abs_line(node),
                    )

    # ========================= Attribute access detection =========================

    def _build_chain_key(self, node: ast.expr) -> str | None:
        """
        Build a chain tracking key from an Attribute node.

        Converts ``ctx.user`` form AST node to ``"ctx.user"`` string,
        used for looking up chained attributes in tracked_vars
        (model attributes inside non-model containers).

        Only handles ``Name.attr`` single-level chains; deeper chains not supported.

        :param node: AST expression node
        :returns: chain key (e.g. ``"ctx.user"``) or None
        """
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return None

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Detect attribute access.

        Handles two forms:
        1. Direct access: ``user.name`` -- ``node.value`` is ``ast.Name``
        2. Chain access: ``ctx.user.name`` -- ``node.value`` is ``ast.Attribute``,
           resolved via ``_build_chain_key()`` to a chain key in tracked_vars
        """
        # Resolve tracking key: direct variable (user) or chain (ctx.user)
        var_key: str | None = None
        if isinstance(node.value, ast.Name):
            var_key = node.value.id
        else:
            var_key = self._build_chain_key(node.value)

        if var_key is None or var_key not in self.tracked_vars:
            self.generic_visit(node)
            return

        var_info = self.tracked_vars[var_key]
        attr_name = node.attr

        # Skip method calls (e.g. obj.save())
        parent = self._get_parent(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            # RLC008: calling business methods on post-commit object.
            # commit expires all session objects; business methods internally access
            # column attributes triggering synchronous lazy load
            # -> MissingGreenlet in async context.
            # Typical: obj_a.save() then obj_b.some_method() (obj_b not refreshed)
            #
            # Only check non-self variables: self's methods often operate on runtime
            # attributes (_queue, _cache etc.), not necessarily database columns.
            # self's column access already covered by RLC007.
            # External objects (e.g. s3_client) almost certainly access database columns.
            if var_key != 'self' and var_info.post_commit:
                method_name = attr_name
                # Known safe methods don't need warnings (handled by other rules
                # or don't trigger database access).
                # commit_methods are in this list: visit_Expr/visit_Assign already checked
                # them with correct pre-commit state; checking again here after generic_visit
                # runs post-expire would produce false positives.
                safe_methods = (
                    self.commit_methods
                    | self.model_returning_methods
                    | frozenset({'refresh', 'commit', 'rollback'})
                )
                if method_name not in safe_methods:
                    self.warnings.append(RelationLoadWarning(
                        code='RLC008',
                        file=self.source_file,
                        line=self._abs_line(node),
                        message=(
                            f"Calling method '{method_name}()' on expired post-commit "
                            f"object '{var_key}'. The method may internally access expired "
                            f"column attributes, causing MissingGreenlet. "
                            f"Suggestion: call before commit, or refresh first with "
                            f"await session.refresh({var_key})"
                        ),
                    ))
            self.generic_visit(node)
            return

        # Skip assignment targets (self.attr = value is a write, does not trigger lazy load)
        if isinstance(parent, ast.Assign) and node in parent.targets:
            self.generic_visit(node)
            return

        # Check relationship and column attribute access
        rels = self.model_relationships.get(var_info.model_name, set())
        cols = self.model_columns.get(var_info.model_name, set())

        if attr_name in rels and attr_name not in var_info.loaded_rels:
            if var_info.post_commit:
                # RLC002: accessing unloaded relationship after save/update.
                # Triggers regardless of caller_provided (post-commit expiration).
                self.warnings.append(RelationLoadWarning(
                    code='RLC002',
                    file=self.source_file,
                    line=self._abs_line(node),
                    message=(
                        f"Accessing '{var_key}.{attr_name}' relationship after "
                        f"save()/update() without load= parameter. "
                        f"Suggestion: {var_info.model_name}.{attr_name}"
                    ),
                ))
            elif not var_info.caller_provided:
                # RLC003: accessing unloaded relationship.
                # Only triggers for locally obtained vars; caller_provided is skipped.
                self.warnings.append(RelationLoadWarning(
                    code='RLC003',
                    file=self.source_file,
                    line=self._abs_line(node),
                    message=(
                        f"Accessing '{var_key}.{attr_name}' relationship "
                        f"without load= parameter. "
                        f"Suggestion: load={var_info.model_name}.{attr_name}"
                    ),
                ))
        elif var_info.post_commit and attr_name in cols:
            # RLC007: column access on expired (post-commit) object (including PK).
            # After commit state.dict is cleared (including PK); accessing any column
            # triggers _load_expired -> synchronous SELECT -> MissingGreenlet in
            # async context. The identity map retains the PK for object lookup,
            # but the attribute descriptor still takes the expired path.
            # Safe alternative: sa_inspect(obj).identity[0] (no DB query),
            # or extract values into local variables before commit.
            # Typical scenario: obj_a.save() then obj_b.column (obj_b not refreshed)
            self.warnings.append(RelationLoadWarning(
                code='RLC007',
                file=self.source_file,
                line=self._abs_line(node),
                message=(
                    f"Accessing column '{var_key}.{attr_name}' on expired object "
                    f"after commit. The object was not refreshed and access will "
                    f"trigger synchronous lazy load -> MissingGreenlet. "
                    f"Suggestion: extract the needed values into local variables "
                    f"before commit, or refresh with Type.get() after commit"
                ),
            ))

        self.generic_visit(node)

    @override
    def visit_Return(self, node: ast.Return) -> None:
        """Check return statement."""
        if node.value is None:
            return

        # return var
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            self._check_return_var(var_name, node)

        # return [var1, var2, ...] / return (var1, var2, ...)
        # FastAPI serializes each element in list/tuple, so column attributes are accessed.
        elif isinstance(node.value, (ast.List, ast.Tuple)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    self._check_return_var(elt.id, node)

        # return await obj.save(...)
        if isinstance(node.value, ast.Await):
            call = node.value.value
            if isinstance(call, ast.Call):
                method_name = self._get_method_name(call)
                loaded_rels = self._extract_load_from_call(call)

                # return await obj.save/update/create_duplicate/... (commit methods)
                if self._is_commit_for_call(call, method_name):
                    obj_name = self._get_call_object_name(call)
                    if obj_name and obj_name in self.tracked_vars:
                        model_name = self.tracked_vars[obj_name].model_name
                        self._check_return_loaded(model_name, loaded_rels, node)
                    else:
                        # Class method call (Model.get_or_create/...)
                        class_name = self._get_call_class_name(call)
                        if class_name:
                            self._check_return_loaded(class_name, loaded_rels, node)

                # return await Model.get/find_by_content_hash/... (model-returning methods)
                elif method_name in self.model_returning_methods:
                    obj_name = self._get_call_object_name(call)
                    if obj_name and obj_name in self.tracked_vars:
                        model_name = self.tracked_vars[obj_name].model_name
                        self._check_return_loaded(model_name, loaded_rels, node)
                    else:
                        class_name = self._get_call_class_name(call)
                        if class_name:
                            self._check_return_loaded(class_name, loaded_rels, node)

        self.generic_visit(node)

    def _check_expired_call_args(self, call: ast.Call, context_node: ast.AST) -> None:
        """
        RLC010: check function/method call args for post-commit expired ORM variables.

        Passing expired ORM objects to functions/methods is dangerous because the
        callee may internally access column attributes, triggering synchronous
        lazy load and causing MissingGreenlet in async context.

        Skips session.refresh(obj) and Model.get() -- these are the ways to restore expired objects.
        """
        method_name = self._get_method_name(call)

        # session.refresh(obj) / Model.get() are ways to restore expired objects, skip
        if method_name in {'refresh', 'get'}:
            return

        # Check positional arguments
        for arg in call.args:
            # Resolve tracking key: direct variable (user) or chain (ctx.user)
            var_key: str | None = None
            if isinstance(arg, ast.Name):
                var_key = arg.id
            else:
                var_key = self._build_chain_key(arg)
            if var_key is not None and var_key in self.tracked_vars:
                var_info = self.tracked_vars[var_key]
                if var_info.post_commit:
                    self.warnings.append(RelationLoadWarning(
                        code='RLC010',
                        file=self.source_file,
                        line=self._abs_line(context_node),
                        message=(
                            f"Passing expired post-commit ORM object '{var_key}' "
                            f"({var_info.model_name}) as argument to "
                            f"'{method_name or '<function>'}()' call. "
                            f"Callee may access column attributes triggering synchronous "
                            f"lazy load -> MissingGreenlet. "
                            f"Suggestion: refresh first with "
                            f"{var_key} = await Type.get(session, Type.id == {var_key}.id)"
                        ),
                    ))

        # Check keyword arguments
        for kw in call.keywords:
            var_key_kw: str | None = None
            if isinstance(kw.value, ast.Name):
                var_key_kw = kw.value.id
            else:
                var_key_kw = self._build_chain_key(kw.value)
            if var_key_kw is not None and var_key_kw in self.tracked_vars:
                var_info = self.tracked_vars[var_key_kw]
                if var_info.post_commit:
                    self.warnings.append(RelationLoadWarning(
                        code='RLC010',
                        file=self.source_file,
                        line=self._abs_line(context_node),
                        message=(
                            f"Passing expired post-commit ORM object '{var_key_kw}' "
                            f"({var_info.model_name}) as keyword argument to "
                            f"'{method_name or '<function>'}()' call. "
                            f"Callee may access column attributes triggering synchronous "
                            f"lazy load -> MissingGreenlet. "
                            f"Suggestion: refresh first with "
                            f"{var_key_kw} = await Type.get(session, Type.id == {var_key_kw}.id)"
                        ),
                    ))

    def _check_return_var(self, var_name: str, node: ast.AST) -> None:
        """Check if returned variable satisfies response_model requirements and is not expired."""
        if var_name not in self.tracked_vars:
            return
        var_info = self.tracked_vars[var_name]
        # RLC007: returning a post-commit expired object -> FastAPI serialization accesses
        # column attributes -> MissingGreenlet
        if var_info.post_commit:
            self.warnings.append(RelationLoadWarning(
                code='RLC007',
                file=self.source_file,
                line=self._abs_line(node),
                message=(
                    f"Returning expired post-commit object '{var_name}' "
                    f"({var_info.model_name}). FastAPI will access column attributes "
                    f"when serializing response_model, triggering synchronous lazy load "
                    f"-> MissingGreenlet. "
                    f"Suggestion: refresh before return with {var_info.model_name}.get()"
                ),
            ))
        self._check_return_loaded(var_info.model_name, var_info.loaded_rels, node)

    def _check_return_loaded(
        self,
        model_name: str,
        loaded_rels: set[str],
        node: ast.AST,
    ) -> None:
        """Check if returned model has loaded all response_model required relationships."""
        for rel_name, req_model in self.required_rels.items():
            if req_model == model_name and rel_name not in loaded_rels:
                self.warnings.append(RelationLoadWarning(
                    code='RLC001',
                    file=self.source_file,
                    line=self._abs_line(node),
                    message=(
                        f"Returning {model_name} instance, response_model requires "
                        f"'{rel_name}' relationship but it was not preloaded via load=. "
                        f"Suggestion: load={model_name}.{rel_name}"
                    ),
                ))

    # ========================= RLC011: implicit dunder relationship access =========================

    @staticmethod
    def _extract_boolean_context_vars(node: ast.expr) -> list[str]:
        """
        Recursively extract variable names from a boolean context AST expression.

        Handles patterns:
        - ``if obj:`` / ``if not obj:`` -> ['obj']
        - ``if obj1 and obj2:`` -> ['obj1', 'obj2']
        - ``if not (obj1 or obj2):`` -> ['obj1', 'obj2']
        """
        result: list[str] = []
        if isinstance(node, ast.Name):
            result.append(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            result.extend(_FunctionAnalyzer._extract_boolean_context_vars(node.operand))
        elif isinstance(node, ast.BoolOp):
            for val in node.values:
                result.extend(_FunctionAnalyzer._extract_boolean_context_vars(val))
        return result

    def _check_truthiness_test(self, test: ast.expr) -> None:
        """
        Detect whether a boolean test implicitly triggers dunder methods
        that access unloaded relationships.

        Python truthiness protocol: ``if obj:`` calls ``__bool__()``;
        if no ``__bool__``, falls back to ``__len__()``.
        If these dunder methods internally access unloaded relationships,
        ``lazy='raise_on_sql'`` runtime errors occur.

        Typical scenario::

            # ToolSetBase.__len__ accesses self.tools (a Relationship)
            tool_set = await ToolSet.get(session, ...)
            if not tool_set:  # triggers __len__() -> self.tools -> raise_on_sql
                ...
        """
        var_names = self._extract_boolean_context_vars(test)
        for var_name in var_names:
            if var_name not in self.tracked_vars:
                continue
            var_info = self.tracked_vars[var_name]
            dunder_rels = self.model_dunder_rels.get(var_info.model_name, {})
            if not dunder_rels:
                continue
            # Python truthiness protocol: __bool__ first, then falls back to __len__.
            # If __bool__ is recorded in dunder_rels (even with an empty rel set),
            # it means that dunder is defined and Python won't fall back to __len__.
            for dunder in ('__bool__', '__len__'):
                if dunder not in dunder_rels:
                    continue  # dunder not defined, try next (fallback)
                accessed_rels = dunder_rels[dunder]
                unloaded = accessed_rels - var_info.loaded_rels
                if unloaded:
                    self.warnings.append(RelationLoadWarning(
                        code='RLC011',
                        file=self.source_file,
                        line=self._abs_line(test),
                        message=(
                            f"Boolean test on '{var_name}' ({var_info.model_name}) will "
                            f"implicitly call {dunder}() accessing unloaded relations {unloaded}. "
                            f"Suggestion: use 'if {var_name} is None:' instead of 'if not {var_name}:'"
                        ),
                    ))
                break  # __bool__ defined (whether or not it has unloaded rels), no fallback to __len__

    def _check_iteration_context(self, iter_node: ast.expr) -> None:
        """
        Detect whether iteration context implicitly triggers __iter__
        accessing unloaded relationships.

        Typical scenario::

            tool_set = await ToolSet.get(session, ...)
            for tool in tool_set:  # triggers __iter__() -> self.tools -> raise_on_sql
                ...
        """
        if not isinstance(iter_node, ast.Name):
            return
        var_name = iter_node.id
        if var_name not in self.tracked_vars:
            return
        var_info = self.tracked_vars[var_name]
        dunder_rels = self.model_dunder_rels.get(var_info.model_name, {})
        iter_rels = dunder_rels.get('__iter__', set())
        unloaded = iter_rels - var_info.loaded_rels
        if unloaded:
            self.warnings.append(RelationLoadWarning(
                code='RLC011',
                file=self.source_file,
                line=self._abs_line(iter_node),
                message=(
                    f"Iterating over '{var_name}' ({var_info.model_name}) will "
                    f"implicitly call __iter__() accessing unloaded relations {unloaded}. "
                    f"Suggestion: preload relations before iterating, or access the relation attribute directly"
                ),
            ))

    def _check_dunder_call(self, var_name: str, dunder: str, node: ast.AST) -> None:
        """
        Generic dunder relationship access check.

        Checks if the specified dunder method on a tracked variable
        accesses unloaded relationships.
        """
        if var_name not in self.tracked_vars:
            return
        var_info = self.tracked_vars[var_name]
        dunder_rels = self.model_dunder_rels.get(var_info.model_name, {})
        accessed_rels = dunder_rels.get(dunder, set())
        unloaded = accessed_rels - var_info.loaded_rels
        if unloaded:
            self.warnings.append(RelationLoadWarning(
                code='RLC011',
                file=self.source_file,
                line=self._abs_line(node),
                message=(
                    f"Operation on '{var_name}' ({var_info.model_name}) will implicitly "
                    f"call {dunder}() accessing unloaded relations {unloaded}. "
                    f"Suggestion: preload relations first"
                ),
            ))

    # Builtin function to dunder mapping
    _BUILTIN_TO_DUNDER: dict[str, str] = {
        'len': '__len__',
        'bool': '__bool__',
        'iter': '__iter__',
        'list': '__iter__',
        'tuple': '__iter__',
        'set': '__iter__',
        'frozenset': '__iter__',
        'sorted': '__iter__',
        'sum': '__iter__',
        'any': '__iter__',
        'all': '__iter__',
        'min': '__iter__',
        'max': '__iter__',
        'enumerate': '__iter__',
    }

    @override
    def visit_Call(self, node: ast.Call) -> None:
        """
        Detect len(obj) / bool(obj) / list(obj) etc. builtin function calls
        that implicitly trigger dunder methods on tracked variables.
        """
        if (isinstance(node.func, ast.Name)
                and node.func.id in self._BUILTIN_TO_DUNDER
                and node.args
                and isinstance(node.args[0], ast.Name)):
            var_name = node.args[0].id
            dunder = self._BUILTIN_TO_DUNDER[node.func.id]
            self._check_dunder_call(var_name, dunder, node)
        self.generic_visit(node)

    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Detect obj[i] implicit __getitem__ call on tracked variables."""
        if isinstance(node.value, ast.Name):
            self._check_dunder_call(node.value.id, '__getitem__', node)
        self.generic_visit(node)

    @override
    def visit_Compare(self, node: ast.Compare) -> None:
        """
        Detect ``x in obj`` implicit __contains__ / __iter__ call on tracked variables.

        Python's ``in`` operator first looks for ``__contains__``;
        if not found, falls back to ``__iter__``.
        """
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comparator, ast.Name):
                var_name = comparator.id
                if var_name in self.tracked_vars:
                    var_info = self.tracked_vars[var_name]
                    model_dunders = self.model_dunder_rels.get(var_info.model_name, {})
                    # __contains__ takes priority; falls back to __iter__ if not found
                    if '__contains__' in model_dunders:
                        self._check_dunder_call(var_name, '__contains__', node)
                    else:
                        self._check_dunder_call(var_name, '__iter__', node)
        self.generic_visit(node)

    # ========================= Branch-aware traversal =========================

    @override
    def visit_If(self, node: ast.If) -> None:
        """
        Branch-aware if/elif/else traversal.

        if body and orelse are mutually exclusive branches -- only one runs at runtime.
        Therefore orelse's starting state should be pre-if state (not post-body state),
        and the final state is a pessimistic merge of both branches
        (post_commit takes OR, loaded_rels takes intersection).

        Branches that unconditionally return are dead ends; their state doesn't
        participate in the merge.

        Avoids common false positive pattern::

            if action == 'search':
                results = await self._search(session, s3_client)  # commit
                return results  # unconditional return
            elif action == 'download':
                file = await self._download(session, s3_client)  # s3_client shouldn't be marked expired
        """
        # RLC011: detect implicit dunder relationship access in boolean tests
        self._check_truthiness_test(node.test)

        # Visit the test expression (may contain attribute access checks)
        self.visit(node.test)

        # Save pre-if state (common starting point for both branches)
        pre_if = self._snapshot_tracked_vars()

        # ---- if body ----
        for child in node.body:
            self.visit(child)
        body_returns = self._branch_unconditionally_returns(node.body)
        post_body = self._snapshot_tracked_vars()

        if node.orelse:
            # ---- orelse (elif/else) starts from pre-if state ----
            self._restore_tracked_vars(pre_if)
            for child in node.orelse:
                self.visit(child)
            else_returns = self._branch_unconditionally_returns(node.orelse)
            post_else = self._snapshot_tracked_vars()

            # ---- Merge both branches' state ----
            if body_returns and else_returns:
                # Both branches unconditionally return: subsequent code unreachable, restore pre-if
                self._restore_tracked_vars(pre_if)
            elif body_returns:
                # Only body returns: subsequent code only reached via orelse
                self._restore_tracked_vars(post_else)
            elif else_returns:
                # Only orelse returns: subsequent code only reached via body
                self._restore_tracked_vars(post_body)
            else:
                # Neither branch returns: pessimistic merge
                self._merge_tracked_vars(post_body, post_else)
        else:
            # No orelse: if body may or may not execute
            if body_returns:
                # body returns: subsequent code only reached when body doesn't execute
                self._restore_tracked_vars(pre_if)
            else:
                # body may or may not execute: pessimistic merge
                self._merge_tracked_vars(pre_if, post_body)

    @override
    def visit_Try(self, node: ast.Try) -> None:
        """
        Branch-aware try/except/else/finally traversal.

        - try body unconditionally returns: subsequent code only reached via handlers,
          restore pre-try state (exception may be raised before commit)
        - except handlers unconditionally return: state changes don't leak to subsequent code
        - else/finally: visited normally
        """
        # Save state before entering try
        pre_try = self._snapshot_tracked_vars()

        # Visit try body
        for child in node.body:
            self.visit(child)

        # If try body unconditionally returns, subsequent code only reached via handlers
        # Restore pre-try state (exception may be raised before commit)
        if self._branch_unconditionally_returns(node.body):
            self._restore_tracked_vars(pre_try)

        # Visit except handlers
        for handler in node.handlers:
            pre_handler = self._snapshot_tracked_vars()
            for child in handler.body:
                self.visit(child)
            if self._branch_unconditionally_returns(handler.body):
                self._restore_tracked_vars(pre_handler)

        # Visit else (runs when try succeeds with no exception)
        for child in node.orelse:
            self.visit(child)

        # Visit finally
        for child in node.finalbody:
            self.visit(child)

    @override
    def visit_While(self, node: ast.While) -> None:
        """Detect implicit dunder relationship access in while conditions."""
        self._check_truthiness_test(node.test)
        self.generic_visit(node)

    @override
    def visit_For(self, node: ast.For) -> None:
        """Detect implicit __iter__ relationship access in for loop iterables."""
        self._check_iteration_context(node.iter)
        self.generic_visit(node)

    # ========================= Comprehension scoping =========================

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        elements: list[ast.expr],
    ) -> None:
        """
        Visit comprehension with proper scoping for iteration variables.

        Python 3 comprehensions (ListComp/SetComp/DictComp/GeneratorExp) have
        independent scopes -- iteration variables don't shadow outer variables.
        Temporarily remove same-named tracked vars to prevent false positives
        from attribute access on comprehension-local iteration variables being
        reported as access on outer expired variables.
        """
        shadowed: dict[str, _TrackedVar] = {}
        for gen in generators:
            for name in self._extract_target_names(gen.target):
                if name in self.tracked_vars:
                    shadowed[name] = self.tracked_vars.pop(name)

        for gen in generators:
            self.visit(gen.iter)
            for if_clause in gen.ifs:
                self.visit(if_clause)
        for elt in elements:
            self.visit(elt)

        self.tracked_vars.update(shadowed)

    @staticmethod
    def _extract_target_names(target: ast.expr) -> list[str]:
        """Extract all variable names from an assignment target (supports tuple unpacking)."""
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            names: list[str] = []
            for elt in target.elts:
                names.extend(_FunctionAnalyzer._extract_target_names(elt))
            return names
        return []

    @override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    @override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    @override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    @override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

    # ========================= AST utility methods =========================

    @staticmethod
    def _get_method_name(call: ast.Call) -> str:
        """Extract method name from a Call node."""
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ''

    @staticmethod
    def _get_call_class_name(call: ast.Call) -> str | None:
        """
        Extract class name from Model.get() call.

        Matches: Model.get(...), Model.get_with_count(...)
        """
        if isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name):
                return call.func.value.id
        return None

    @staticmethod
    def _get_call_object_name(call: ast.Call) -> str | None:
        """
        Extract object name from obj.save() call.

        Matches: variable.save(...), variable.update(...)
        Special: super().method(...) -> 'self' (super() is called on self)
        """
        if isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name):
                return call.func.value.id
            # super().method() -> treated as self.method()
            if (isinstance(call.func.value, ast.Call)
                    and isinstance(call.func.value.func, ast.Name)
                    and call.func.value.func.id == 'super'):
                return 'self'
        return None

    def _call_passes_session(self, call: ast.Call) -> bool:
        """
        Check whether a function call passes an AsyncSession parameter.

        Used to distinguish model commit methods
        (e.g. ``user_file.fill_from_image_url(session, ...)``) from same-named
        non-model methods (e.g. ``redis.delete(key)``).
        """
        for arg in call.args:
            if isinstance(arg, ast.Name) and arg.id in self.session_param_names:
                return True
        for kw in call.keywords:
            if isinstance(kw.value, ast.Name) and kw.value.id in self.session_param_names:
                return True
        return False

    def _expire_all_tracked_vars(self) -> None:
        """
        Mark all tracked variables as post-commit.

        Simulates session.commit() behavior: commit expires ALL objects in the session,
        not just the one being saved.
        """
        for var in self.tracked_vars.values():
            var.post_commit = True
            var.loaded_rels.clear()

    # ========================= Branch-aware state management =========================

    def _snapshot_tracked_vars(self) -> dict[str, _TrackedVar]:
        """
        Deep-copy tracked_vars state for branch analysis.

        Save state before entering if/try branches; unconditionally returning
        branches can be restored, preventing commit state from leaking.
        """
        return {
            name: _TrackedVar(
                model_name=var.model_name,
                loaded_rels=var.loaded_rels.copy(),
                post_commit=var.post_commit,
                caller_provided=var.caller_provided,
                line=var.line,
            )
            for name, var in self.tracked_vars.items()
        }

    def _restore_tracked_vars(self, snapshot: dict[str, _TrackedVar]) -> None:
        """
        Restore tracked_vars state from a snapshot.

        Removes variables added in a branch and restores existing variables' state.
        Used after an unconditionally returning branch ends, to undo that branch's
        commit impact.
        """
        self.tracked_vars.clear()
        self.tracked_vars.update(snapshot)

    def _merge_tracked_vars(
        self,
        state_a: dict[str, _TrackedVar],
        state_b: dict[str, _TrackedVar],
    ) -> None:
        """
        Pessimistic merge of two branches' tracked_vars state into self.tracked_vars.

        Used when if/else branches both don't return, merging visible state for
        subsequent code.
        Rules (either branch may execute, take worst case):

        - post_commit: OR (if either branch committed, object is expired)
        - loaded_rels: intersection (only rels loaded in both branches are certain)
        - caller_provided: AND (only if both branches mark it)

        Only variables existing in both states are preserved (branch-specific new
        variables don't exist in the other branch).
        """
        merged: dict[str, _TrackedVar] = {}
        for name in state_a.keys() & state_b.keys():
            a = state_a[name]
            b = state_b[name]
            merged[name] = _TrackedVar(
                model_name=a.model_name,
                loaded_rels=a.loaded_rels & b.loaded_rels,
                post_commit=a.post_commit or b.post_commit,
                caller_provided=a.caller_provided and b.caller_provided,
                line=max(a.line, b.line),
            )
        self.tracked_vars.clear()
        self.tracked_vars.update(merged)

    def _branch_unconditionally_returns(self, stmts: list[ast.stmt]) -> bool:
        """
        Check if a statement list unconditionally exits (return/raise/continue/break/NoReturn call).

        Recursively checks nested if/elif/else and try/except:
        - if/elif/else: all branches must exit to be considered unconditional (must have else)
        - try/except: try body and all handlers must exit to be considered unconditional
        """
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            return True
        # NoReturn function calls (e.g. raise_bad_request(), raise_internal_error())
        if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call):
            call_func = last.value.func
            if isinstance(call_func, ast.Name) and call_func.id in self.noreturn_names:
                return True
        # if/elif/else: all branches must exit
        if isinstance(last, ast.If):
            if not last.orelse:
                return False  # no else -> may not exit
            return (
                self._branch_unconditionally_returns(last.body)
                and self._branch_unconditionally_returns(last.orelse)
            )
        # try/except: try body and all handlers must exit
        if isinstance(last, ast.Try):
            body_returns = self._branch_unconditionally_returns(last.body)
            handlers_return = all(
                self._branch_unconditionally_returns(h.body)
                for h in last.handlers
            ) if last.handlers else False
            return body_returns and handlers_return
        return False

    def _handle_session_refresh(self, call: ast.Call) -> None:
        """
        Handle ``await session.refresh(obj)`` / ``session.refresh(obj, attribute_names=[...])`` call.

        - ``refresh(obj)`` restores column attributes (un-expires the object), but doesn't load relationships.
        - ``refresh(obj, attribute_names=['rel1', 'rel2'])`` also loads specified relationship attributes.
        """
        if not call.args:
            return
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Name):
            obj_name = first_arg.id
            if obj_name in self.tracked_vars:
                var = self.tracked_vars[obj_name]
                # refresh restores column attrs, un-expire the object
                var.post_commit = False
                # Check attribute_names parameter for specified relationship attributes
                for kw in call.keywords:
                    if kw.arg == 'attribute_names':
                        if isinstance(kw.value, ast.List):
                            for elt in kw.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    # attribute_names strings may be rels or columns;
                                    # if it's a relationship, mark as loaded
                                    model_rels = self.model_relationships.get(var.model_name, set())
                                    if elt.value in model_rels:
                                        var.loaded_rels.add(elt.value)
                        break

    @staticmethod
    def _extract_load_from_call(call: ast.Call) -> set[str]:
        """Extract relationship names from load= keyword argument in a Call node."""
        for kw in call.keywords:
            if kw.arg == 'load':
                return _extract_load_value(kw.value)
        return set()

    @staticmethod
    def _has_keyword_false(call: ast.Call, keyword: str) -> bool:
        """Check if call has keyword=False argument (delegates to module-level helper, DRY)."""
        return _ast_has_keyword_false_static(call, keyword)

    @staticmethod
    def _has_keyword(call: ast.Call, keyword: str) -> bool:
        """Check if call has the specified keyword argument."""
        return any(kw.arg == keyword for kw in call.keywords)

    def _get_parent(self, node: ast.AST) -> ast.AST | None:
        """Get the parent node."""
        return self._parent_map.get(id(node))

    @override
    def visit(self, node: ast.AST) -> None:
        """Override visit to build parent map."""
        for child in ast.iter_child_nodes(node):
            self._parent_map[id(child)] = node
        super().visit(node)


# ========================= Auto-check entry points =========================


def run_model_checks(base_class: type) -> None:
    """
    Run model method relation load static analysis.

    Called automatically in your package's ``__init__.py`` after ``configure_mappers()``.
    Checks all model classes' async methods for relationship loading issues.

    :param base_class: SQLModelBase class
    :raises RuntimeError: if issues are found (blocks startup)
    """
    global _model_check_completed, _base_class
    if not check_on_startup:
        return
    if _model_check_completed:
        return

    _base_class = base_class
    checker = RelationLoadChecker(base_class)
    warnings = checker.check_model_methods()
    _model_check_completed = True

    if warnings:
        for w in warnings:
            logger.error(str(w))
        # In the test environment warn without blocking: WIP code may trigger
        # checks unrelated to the current test run.
        if 'pytest' in sys.modules or '_pytest' in sys.modules:
            logger.warning(
                f"Test environment: relation load static analysis found {len(warnings)} "
                f"issues (non-blocking). See error log above for details."
            )
        else:
            raise RuntimeError(
                f"Relation load static analysis found {len(warnings)} model method issues. "
                f"Fix them before restarting. See error log above for details."
            )
    else:
        logger.info("Model method relation load analysis passed")


def mark_app_check_completed() -> None:
    """Mark app endpoint/coroutine checks as completed."""
    global _app_check_completed
    _app_check_completed = True


class RelationLoadCheckMiddleware:
    """
    ASGI middleware: auto-check FastAPI endpoints and project coroutines on startup.

    Runs checks once after lifespan startup completes.
    Passes if clean, raises RuntimeError to block startup if issues found.

    Usage::

        from sqlmodel_ext.relation_load_checker import RelationLoadCheckMiddleware
        app.add_middleware(RelationLoadCheckMiddleware)

    Custom project root::

        app.add_middleware(RelationLoadCheckMiddleware, project_root="/path/to/project")

    Skip certain paths::

        app.add_middleware(
            RelationLoadCheckMiddleware,
            skip_paths=['/base/', '/mixin/'],
        )

    Skip third-party library lazy proxy attributes (e.g. openai.AudioProxy
    triggering client initialization during inspect)::

        app.add_middleware(RelationLoadCheckMiddleware, skip_third_party_attrs=True)
    """

    def __init__(
        self,
        app: Any,
        *,
        project_root: str | None = None,
        skip_paths: list[str] | None = None,
        skip_third_party_attrs: bool = False,
    ) -> None:
        self.app: Any = app
        self.project_root: str = project_root or _PROJECT_ROOT
        self.skip_paths: list[str] | None = skip_paths
        self.skip_third_party_attrs: bool = skip_third_party_attrs
        self._checked: bool = False

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope['type'] == 'lifespan':
            async def send_wrapper(message: dict[str, Any]) -> None:
                if (
                    message['type'] == 'lifespan.startup.complete'
                    and not self._checked
                ):
                    self._checked = True
                    self._run_checks()
                await send(message)
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

    def _run_checks(self) -> None:
        """Run endpoint and coroutine checks."""
        if not check_on_startup:
            mark_app_check_completed()
            return

        if _base_class is None:
            logger.warning(
                "RelationLoadCheckMiddleware: base_class not set. "
                "Ensure your models package is properly imported "
                "and run_model_checks() was called."
            )
            return

        # Walk middleware chain to find the app with routes
        routes_app = self._find_app_with_routes()
        if routes_app is None:
            logger.warning(
                "RelationLoadCheckMiddleware: "
                "no app with routes found, skipping endpoint checks"
            )
            return

        checker = RelationLoadChecker(_base_class)
        warnings = checker.check_app(routes_app)
        warnings.extend(checker.check_project_coroutines(
            self.project_root,
            skip_paths=self.skip_paths,
            skip_third_party_attrs=self.skip_third_party_attrs,
        ))

        mark_app_check_completed()

        if warnings:
            for w in warnings:
                logger.error(str(w))
            raise RuntimeError(
                f"Relation load static analysis found {len(warnings)} issues. "
                f"Fix them before restarting. See error log above for details."
            )
        logger.info("Endpoint and coroutine relation load analysis passed")

    def _find_app_with_routes(self) -> Any:
        """Walk middleware chain to find the app with .routes attribute."""
        current: Any = self.app
        while current is not None:
            if hasattr(current, 'routes'):
                return current
            current = getattr(current, 'app', None)
        return None


def _check_completion_warning() -> None:
    """Warn at process exit if the app check was missed.

    Uses sys.stderr rather than the logger: atexit callbacks run during process
    shutdown, when the logging handlers may already be closed and emitting via
    the logger can raise ``ValueError: I/O operation on closed file``.
    """
    if check_on_startup and _model_check_completed and not _app_check_completed:
        msg = (
            "WARNING: Model method checks completed, but endpoint/coroutine "
            "checks were not run.\n"
            "Add the middleware:\n"
            "  from sqlmodel_ext.relation_load_checker import RelationLoadCheckMiddleware\n"
            "  app.add_middleware(RelationLoadCheckMiddleware)\n"
            "Or call manually:\n"
            "  checker = RelationLoadChecker(base_class)\n"
            "  checker.check_app(app)\n"
            "To disable:\n"
            "  import sqlmodel_ext.relation_load_checker as rlc\n"
            "  rlc.check_on_startup = False\n"
        )
        try:
            sys.stderr.write(msg)
        except (ValueError, OSError):
            pass  # stderr already closed, silently ignore


atexit.register(_check_completion_warning)
