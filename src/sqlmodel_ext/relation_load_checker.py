"""
Relation Load Checker -- static analysis for async SQLAlchemy relationship access.

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

Auto-check (recommended)::

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

Configuration::

    import sqlmodel_ext.relation_load_checker as rlc
    rlc.check_on_startup = False  # disable all auto-checks
"""
import atexit
import ast
import inspect as python_inspect
import logging
import os
import pathlib
import sys
import textwrap
import types
import typing
from dataclasses import dataclass, field
from typing import Annotated, Any, Self, TypeVar, Union, override

from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSession

logger = logging.getLogger(__name__)

# Conditional FastAPI import
try:
    from fastapi.params import Depends as _FastAPIDependsClass
    _HAS_FASTAPI = True
except ImportError:
    _FastAPIDependsClass = None  # type: ignore
    _HAS_FASTAPI = False


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


# ========================= Auto-check configuration =========================

check_on_startup: bool = True
"""Auto-check switch on startup (default on). Set False to disable all auto-checks."""

_base_class: type | None = None
"""Cached base_class reference (set by run_model_checks)."""

_model_check_completed: bool = False
"""Whether model method checks have completed."""

_app_check_completed: bool = False
"""Whether app endpoint/coroutine checks have completed."""

_PROJECT_ROOT: str = os.getcwd()
"""Auto-detected project root directory (defaults to cwd)."""


@dataclass
class RelationLoadWarning:
    """Relation load static analysis warning."""
    code: str
    """Rule code (RLC001-RLC011)."""
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


class RelationLoadChecker:
    """
    Startup-time relation load static analyzer.

    Uses AST analysis to detect unloaded relationship access in coroutines.
    Run after ``configure_mappers()`` and before serving requests.
    """

    def __init__(self, base_class: type) -> None:
        # model class name -> set of relationship attribute names
        self.model_relationships: dict[str, set[str]] = {}
        # model class name -> set of column attribute names
        self.model_columns: dict[str, set[str]] = {}
        # model class name -> set of primary key column attribute names (PKs don't expire)
        self.model_pk_columns: dict[str, set[str]] = {}
        # model class name -> actual class object
        self.model_classes: dict[str, type] = {}
        # analyzed function ids for dedup
        self._analyzed_func_ids: set[int] = set()
        # auto-discovered method behaviors (type system as single source of truth)
        self.commit_methods: frozenset[str] = frozenset()
        self.model_returning_methods: frozenset[str] = frozenset()
        self.sync_model_returning_methods: frozenset[str] = frozenset()

        self._build_knowledge_base(base_class)
        self.commit_methods, self.model_returning_methods, self.sync_model_returning_methods = (
            self._discover_method_behaviors()
        )
        # model_name -> {dunder_name -> set of accessed relationship names}
        self.model_dunder_rels: dict[str, dict[str, set[str]]] = {}
        self._scan_dunder_relationship_access()

    def _build_knowledge_base(self, base_class: type) -> None:
        """Build model knowledge base from SQLAlchemy mappers."""
        for mapper in base_class._sa_registry.mappers:
            cls = mapper.class_
            cls_name = cls.__name__
            self.model_relationships[cls_name] = {
                rel.key for rel in mapper.relationships
            }
            self.model_columns[cls_name] = {
                col.key for col in mapper.column_attrs
            }
            self.model_pk_columns[cls_name] = {
                col.key for col in mapper.primary_key
            }
            self.model_classes[cls_name] = cls

    def _discover_method_behaviors(self) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
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

        :returns: (commit_methods, model_returning_methods, sync_model_returning_methods)
        """
        # method_name -> (session_param_name, AST)
        method_infos: dict[str, tuple[str, ast.Module]] = {}
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

                    # Keep only the first method of the same name (MRO order, most specific subclass first)
                    if attr_name in method_infos:
                        continue

                    try:
                        source = textwrap.dedent(python_inspect.getsource(func))
                        tree = ast.parse(source)
                    except (OSError, TypeError, SyntaxError):
                        continue

                    method_infos[attr_name] = (session_param, tree)
                    method_owners[attr_name] = cls_name

                    # Record return type (prefer get_type_hints result, fall back to __annotations__)
                    return_hint = (
                        hints.get('return') if hints is not None
                        else getattr(func, '__annotations__', {}).get('return')
                    )
                    if return_hint is not None:
                        method_return_hints[attr_name] = return_hint

        # -------- Commit method discovery --------

        # Phase 1: methods that directly call session.commit() / session.rollback()
        commit_methods: set[str] = set()
        for method_name, (session_param, tree) in method_infos.items():
            if _ast_has_typed_commit(tree, session_param):
                commit_methods.add(method_name)

        # Phase 2: transitive closure -- methods that call commit methods passing the session
        changed = True
        while changed:
            changed = False
            for method_name, (session_param, tree) in method_infos.items():
                if method_name in commit_methods:
                    continue
                if _ast_calls_commit_method_with_session(
                    tree, session_param, frozenset(commit_methods),
                ):
                    commit_methods.add(method_name)
                    changed = True

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
            if origin is Union or origin is types.UnionType:
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

        logger.debug(f"Auto-discovered commit methods: {sorted(commit_methods)}")
        logger.debug(f"Auto-discovered model-returning methods: {sorted(model_returning)}")
        if sync_model_returning:
            logger.debug(f"Auto-discovered sync model-returning methods: {sorted(sync_model_returning)}")
        return frozenset(commit_methods), frozenset(model_returning), frozenset(sync_model_returning)

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
                    # if __bool__ exists (even without rel access), Python won't fall back to __len__
                    dunder_rels[dunder] = accessed_rels

            if dunder_rels:
                self.model_dunder_rels[model_name] = dunder_rels

        if self.model_dunder_rels:
            logger.debug(f"Found dunder relationship access: {self.model_dunder_rels}")

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
                            source_file = '<unknown>'
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

        return warnings

    def check_project_coroutines(
        self,
        project_root: str,
        skip_paths: list[str] | None = None,
        skip_third_party_attrs: bool = False,
    ) -> list[RelationLoadWarning]:
        """
        Scan all imported modules' async functions and async generators.

        Iterates sys.modules, analyzing coroutine functions and async generators
        from project source files. Also scans methods of non-model classes
        (e.g. command handlers, service classes).
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
                    # Module-level async function / async generator
                    func_module = getattr(attr, '__module__', None)
                    if func_module == module_name:
                        funcs_to_check.append((f"{module_name}.{attr_name}", attr))
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
                            funcs_to_check.append(
                                (f"{module_name}.{attr.__name__}.{method_name}", func),
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
        """Check whether obj is an async callable (coroutine function or async generator)."""
        return (python_inspect.iscoroutinefunction(obj)
                or python_inspect.isasyncgenfunction(obj))

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

        analyzer = _FunctionAnalyzer(
            model_relationships=self.model_relationships,
            model_columns=self.model_columns,
            model_pk_columns=self.model_pk_columns,
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
            source_file = '<unknown>'
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
        """Extract model class name from type annotation."""
        # Handle Annotated[Model, Depends(...)]
        origin = typing.get_origin(hint)
        if origin is Annotated:
            args = typing.get_args(hint)
            if args:
                return self._extract_model_from_hint(args[0])

        # Direct model class
        if isinstance(hint, type) and hint.__name__ in self.model_relationships:
            return hint.__name__

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
        if origin is types.UnionType or origin is Union:
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
            source_file = '<unknown>'

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


def _ast_calls_commit_method_with_session(
    tree: ast.Module,
    session_param: str,
    commit_methods: frozenset[str],
) -> bool:
    """
    Check if the AST calls a known commit method and passes the session parameter.

    Matches patterns:

    - ``await obj.save(session, ...)``
    - ``await cls.from_remote_url(session=session, ...)``
    """
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in commit_methods
        ):
            continue
        # Check if session is passed as a positional argument
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id == session_param:
                return True
        # Check if session is passed as a keyword argument
        for kw in node.keywords:
            if isinstance(kw.value, ast.Name) and kw.value.id == session_param:
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
        model_pk_columns: dict[str, set[str]],
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
    ) -> None:
        self.model_relationships: dict[str, set[str]] = model_relationships
        self.model_columns: dict[str, set[str]] = model_columns
        self.model_pk_columns: dict[str, set[str]] = model_pk_columns
        self.required_rels: dict[str, str] = required_rels
        self.source_file: str = source_file
        self.line_offset: int = line_offset
        self.path: str = path
        self.warnings: list[RelationLoadWarning] = []
        self._parent_map: dict[int, ast.AST] = {}
        self.commit_methods: frozenset[str] = commit_methods or frozenset()
        self.model_returning_methods: frozenset[str] = model_returning_methods or frozenset()
        self.noreturn_names: frozenset[str] = noreturn_names or frozenset()
        # Complete model-returning set for variable tracking (includes sync methods).
        # Sync methods are NOT added to safe_methods because calling sync methods
        # on expired objects is equally dangerous.
        _sync = sync_model_returning_methods or frozenset()
        self._all_model_returning: frozenset[str] = self.model_returning_methods | _sync
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

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        """Check assignment statement."""
        self._handle_attribute_writes(node.targets, node.value)
        self._check_assign(node.targets, node.value, node)
        self.generic_visit(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check annotated assignment statement."""
        if node.target and node.value:
            self._handle_attribute_writes([node.target], node.value)
            self._check_assign([node.target], node.value, node)
        self.generic_visit(node)

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
                elif method_name in self.commit_methods:
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
        if isinstance(value, ast.Await) and isinstance(value.value, ast.Call):
            self._check_expired_call_args(value.value, node)
        elif isinstance(value, ast.Call):
            self._check_expired_call_args(value, node)

        # Check await expression
        if isinstance(value, ast.Await) and isinstance(value.value, ast.Call):
            call = value.value
            method_name = self._get_method_name(call)
            loaded_rels = self._extract_load_from_call(call)

            # Auto-discovered commit methods (with assignment)
            # Includes save/update/delete/get_or_create/create_duplicate etc.
            # Tracks actual SQLAlchemy commit behavior:
            #   commit=True  -> session.commit() -> all objects expire
            #   refresh=True (default, save/update only) -> cls.get() -> column attrs restored
            #   load= -> cls.get(load=) -> specified rels loaded
            #   commit=False -> session.flush() -> no expiration
            if method_name in self.commit_methods:
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
                                new_rels: set[str] = set()
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
                        # Other commit methods (e.g. create_duplicate, get_or_create)
                        # Only model-returning methods' return values are model instances
                        # that need tracking. Non-model-returning methods (e.g. calculate_cost -> int)
                        # don't create tracking variables, avoiding false RLC010 positives
                        # from scalar return values being treated as ORM objects.
                        if method_name in self._all_model_returning:
                            self.tracked_vars[var_name] = _TrackedVar(
                                model_name=old_var.model_name,
                                loaded_rels=loaded_rels,
                                post_commit=False,
                                caller_provided=False,
                                line=self._abs_line(node),
                            )

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
        is_chained = '.' in var_key

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
                # RLC002: accessing unloaded relationship after save/update
                # Triggers regardless of caller_provided (post-commit expiration)
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
                # RLC003: accessing unloaded relationship
                # Only triggers for locally obtained vars, caller_provided skipped
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
            # PK columns don't expire (SQLAlchemy identity map retains PK values).
            # But chained access doesn't skip PK: objects inside containers may have
            # detached from the session identity map, so PK values are not guaranteed.
            if not is_chained:
                pk_cols = self.model_pk_columns.get(var_info.model_name, set())
                if attr_name in pk_cols:
                    self.generic_visit(node)
                    return
            # RLC007: column access on expired (post-commit) object.
            # After commit, ALL objects in the session are expired.
            # Accessing any column triggers a synchronous lazy load,
            # which causes MissingGreenlet in async context.
            # Typical scenario: obj_a.save() then obj_b.column (obj_b not refreshed)
            self.warnings.append(RelationLoadWarning(
                code='RLC007',
                file=self.source_file,
                line=self._abs_line(node),
                message=(
                    f"Accessing column '{var_key}.{attr_name}' on expired object "
                    f"after commit. The object was not refreshed and access will "
                    f"trigger synchronous lazy load -> MissingGreenlet. "
                    f"Suggestion: refresh with "
                    f"{var_key} = await Type.get(session, Type.id == {var_key}.id)"
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

        # return await obj.save(...)
        if isinstance(node.value, ast.Await):
            call = node.value.value
            if isinstance(call, ast.Call):
                method_name = self._get_method_name(call)
                loaded_rels = self._extract_load_from_call(call)

                # return await obj.save/update/create_duplicate/... (commit methods)
                if method_name in self.commit_methods:
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
        """Check if returned variable satisfies response_model requirements."""
        if var_name not in self.tracked_vars:
            return
        var_info = self.tracked_vars[var_name]
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
            # Python truthiness protocol: __bool__ first, then falls back to __len__
            # If __bool__ is recorded in dunder_rels (even with empty rel set),
            # it means that dunder is defined and Python won't fall back to __len__
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
        """
        if isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name):
                return call.func.value.id
        return None

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
        """Check if call has keyword=False argument."""
        for kw in call.keywords:
            if kw.arg == keyword:
                if isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    return True
        return False

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
        raise RuntimeError(
            f"Relation load static analysis found {len(warnings)} model method issues. "
            f"Fix them before restarting. See error log above for details."
        )
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
    """Warn at process exit if app check was missed."""
    if check_on_startup and _model_check_completed and not _app_check_completed:
        logger.warning(
            "Model method checks completed, but endpoint/coroutine checks were not run.\n"
            "Add the middleware:\n"
            "  from sqlmodel_ext.relation_load_checker import RelationLoadCheckMiddleware\n"
            "  app.add_middleware(RelationLoadCheckMiddleware)\n"
            "Or call manually:\n"
            "  checker = RelationLoadChecker(base_class)\n"
            "  checker.check_app(app)\n"
            "To disable:\n"
            "  import sqlmodel_ext.relation_load_checker as rlc\n"
            "  rlc.check_on_startup = False"
        )


atexit.register(_check_completion_warning)
