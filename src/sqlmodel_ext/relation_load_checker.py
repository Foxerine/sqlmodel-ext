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
import sys
import textwrap
import typing
from dataclasses import dataclass, field
from typing import Annotated, Any

logger = logging.getLogger(__name__)

# Conditional FastAPI import
try:
    from fastapi.params import Depends as _FastAPIDependsClass
    _HAS_FASTAPI = True
except ImportError:
    _FastAPIDependsClass = None  # type: ignore
    _HAS_FASTAPI = False


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
    """Rule code (RLC001-RLC007)."""
    file: str
    """File path."""
    line: int
    """Line number."""
    message: str
    """Warning details."""

    def __str__(self) -> str:
        return f"[{self.code}] {self.file}:{self.line} - {self.message}"


# save/update/delete methods that trigger commit
_COMMIT_METHODS = frozenset({'save', 'update', 'delete'})
# Query methods that return model instances
_QUERY_METHODS = frozenset({'get', 'get_exist_one', 'first'})
# save/update return refreshed object
_SAVE_METHODS = frozenset({'save', 'update'})
# Methods that may commit internally AND return a model instance (e.g. get_or_create may save)
_COMMIT_QUERY_METHODS = frozenset({'get_or_create'})


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
        # model class name -> actual class object
        self.model_classes: dict[str, type] = {}
        # analyzed function ids for dedup
        self._analyzed_func_ids: set[int] = set()

        self._build_knowledge_base(base_class)

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
            self.model_classes[cls_name] = cls

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
        defined async methods. Uses ``vars(cls)`` to get only class-own methods,
        avoiding duplicate analysis of inherited methods.

        For ``self`` parameter:
        - Marked as caller_provided (caller responsible for preloading, skips RLC003)
        - Parses ``@requires_relations`` decorator for self's loaded relations
        - save()/update() still triggers RLC002 (post-commit expiration)
        """
        warnings: list[RelationLoadWarning] = []

        for cls_name, cls in self.model_classes.items():
            for attr_name in vars(cls):
                if attr_name.startswith('__') and attr_name.endswith('__'):
                    continue

                raw_attr = vars(cls)[attr_name]
                # Unwrap staticmethod
                func = raw_attr.__func__ if isinstance(raw_attr, staticmethod) else raw_attr

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
                    logger.debug(f"Error analyzing model method {label}: {e}")

        return warnings

    def check_project_coroutines(
        self,
        project_root: str,
        skip_paths: list[str] | None = None,
    ) -> list[RelationLoadWarning]:
        """
        Scan all imported modules' async functions and async generators.

        Iterates sys.modules, analyzing coroutine functions and async generators
        from project source files. Also scans methods of non-model classes
        (e.g. command handlers, service classes).
        Automatically skips functions already analyzed by check_app/check_model_methods.

        :param project_root: absolute path to project root directory
        :param skip_paths: list of path fragments to skip (e.g. ['/base/', '/mixin/'])
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

                if self._is_async_callable(attr):
                    # Module-level async function / async generator
                    func_module = getattr(attr, '__module__', None)
                    if func_module == module_name:
                        funcs_to_check.append((f"{module_name}.{attr_name}", attr))
                elif python_inspect.isclass(attr):
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
                        if self._is_async_callable(func):
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

        # Detect instance method
        sig = python_inspect.signature(func)
        first_param = next(iter(sig.parameters), None)
        caller_provided_params: set[str] = set()

        if first_param == 'self':
            param_models['self'] = cls_name
            caller_provided_params.add('self')

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
    ) -> tuple[list[RelationLoadWarning], '_FunctionAnalyzer | None']:
        """
        Core AST analysis: parse function body and run _FunctionAnalyzer.

        :param caller_provided_params: set of caller-provided parameter names, skip RLC003
        :param pre_parsed: pre-parsed (source_file, tree, line_offset) to avoid re-parsing
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

        :returns: param_name -> model_class_name
        """
        param_models: dict[str, str] = {}

        try:
            hints = typing.get_type_hints(func, include_extras=True)
        except Exception:
            return param_models

        for param_name, hint in hints.items():
            model_name = self._extract_model_from_hint(hint)
            if model_name is not None:
                param_models[param_name] = model_name

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

        :returns: (source_file, ast_tree_or_None, line_offset)
        """
        try:
            source_file = python_inspect.getfile(func)
        except (TypeError, OSError):
            source_file = '<unknown>'

        try:
            source = python_inspect.getsource(func)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return source_file, None, 0

        try:
            line_offset = python_inspect.getsourcelines(func)[1] - 1
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


def _extract_load_value(node: ast.expr) -> set[str]:
    """
    Extract relationship names from a load= AST value node.

    Supports:
    - ``load=Model.rel`` -> ``{'rel'}``
    - ``load=[Model.r1, Model.r2]`` -> ``{'r1', 'r2'}``
    """
    result: set[str] = set()

    if isinstance(node, ast.Attribute):
        # load=Model.rel_name
        result.add(node.attr)
    elif isinstance(node, ast.List):
        # load=[Model.r1, Model.r2, ...]
        for elt in node.elts:
            if isinstance(elt, ast.Attribute):
                result.add(elt.attr)

    return result


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
    ) -> None:
        self.model_relationships = model_relationships
        self.model_columns = model_columns
        self.required_rels = required_rels
        self.source_file = source_file
        self.line_offset = line_offset
        self.path = path
        self.warnings: list[RelationLoadWarning] = []
        self._parent_map: dict[int, ast.AST] = {}

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

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check assignment statement."""
        self._check_assign(node.targets, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check annotated assignment statement."""
        if node.target and node.value:
            self._check_assign([node.target], node.value, node)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        """
        Check expression statement (await without assignment).

        Tracks actual SQLAlchemy commit behavior:
        - session.commit() expires ALL objects in the session
        - save()/update() internally calls commit(), so all tracked vars expire
        - commit=False only flushes, no expiration
        - get_or_create() may commit internally, expiring all objects
        - session.refresh(obj) restores column attrs (un-expires the object)
        """
        if isinstance(node.value, ast.Await):
            call = node.value.value
            if isinstance(call, ast.Call):
                method_name = self._get_method_name(call)

                # await obj.save(session) / await obj.update(session) -- no assignment
                if method_name in _SAVE_METHODS:
                    obj_name = self._get_call_object_name(call)
                    if obj_name and obj_name in self.tracked_vars:
                        commit_disabled = self._has_keyword_false(call, 'commit')
                        if not commit_disabled:
                            # commit=True: session.commit() expires all objects
                            self._expire_all_tracked_vars()
                            # save/update(refresh=True) refreshes the object in-place
                            # via session.refresh() -- column attrs restored, rels still not loaded
                            if not self._has_keyword_false(call, 'refresh'):
                                self.tracked_vars[obj_name].post_commit = False

                # await session.commit() -- direct commit call
                elif method_name == 'commit':
                    self._expire_all_tracked_vars()

                # await Model.get_or_create(session, ...) -- may commit internally
                elif method_name in _COMMIT_QUERY_METHODS:
                    if not self._has_keyword_false(call, 'commit'):
                        self._expire_all_tracked_vars()

                # await session.refresh(obj) -- restores column attrs
                elif method_name == 'refresh':
                    self._handle_session_refresh(call)

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

        # Check await expression
        if isinstance(value, ast.Await) and isinstance(value.value, ast.Call):
            call = value.value
            method_name = self._get_method_name(call)
            loaded_rels = self._extract_load_from_call(call)

            # Model.get(...) call
            if method_name in _QUERY_METHODS:
                model_name = self._get_call_class_name(call)
                if model_name and model_name in self.model_relationships:
                    # options= passes raw SQLAlchemy load options;
                    # cannot determine from AST which rels are loaded, assume all
                    if self._has_keyword(call, 'options'):
                        effective_rels = self.model_relationships[model_name].copy()
                    else:
                        effective_rels = loaded_rels
                    self.tracked_vars[var_name] = _TrackedVar(
                        model_name=model_name,
                        loaded_rels=effective_rels,
                        post_commit=False,
                        caller_provided=False,
                        line=self._abs_line(node),
                    )

            # var.save(...) / var.update(...) call
            # Tracks actual SQLAlchemy commit behavior:
            #   commit=True  -> session.commit() -> all objects expire
            #   refresh=True (default) -> session.refresh() -> column attrs restored
            #   load= -> cls.get(load=) -> specified rels loaded
            #   commit=False -> session.flush() -> no expiration
            elif method_name in _SAVE_METHODS:
                obj_name = self._get_call_object_name(call)
                if obj_name and obj_name in self.tracked_vars:
                    old_var = self.tracked_vars[obj_name]
                    commit_disabled = self._has_keyword_false(call, 'commit')
                    refresh_disabled = self._has_keyword_false(call, 'refresh')

                    if not commit_disabled:
                        # commit=True: session.commit() expires ALL objects
                        self._expire_all_tracked_vars()

                    if var_name == obj_name:
                        old_var.caller_provided = False
                        if commit_disabled:
                            # commit=False: no commit no expiration, preserve loaded rels
                            old_var.post_commit = False
                            old_var.loaded_rels |= loaded_rels
                        elif refresh_disabled:
                            # commit=True, refresh=False: expired and not refreshed
                            pass  # _expire_all already marked
                        else:
                            # commit=True, refresh=True: refreshed, load= rels available
                            old_var.post_commit = False
                            old_var.loaded_rels = loaded_rels
                        old_var.line = self._abs_line(node)
                    else:
                        # Return value assigned to different variable (result = await self.save(...))
                        # save/update returns self (same object), in-place refresh means
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
                        # Original object is also refreshed in-place (save returns self,
                        # refresh via session.refresh() restores column attrs in-place)
                        if not commit_disabled and not refresh_disabled:
                            old_var.post_commit = False
                            old_var.loaded_rels = loaded_rels

            # Model.get_with_count(...) etc
            elif method_name == 'get_with_count':
                model_name = self._get_call_class_name(call)
                if model_name and model_name in self.model_relationships:
                    self.tracked_vars[var_name] = _TrackedVar(
                        model_name=model_name,
                        loaded_rels=loaded_rels,
                        post_commit=False,
                        caller_provided=False,
                        line=self._abs_line(node),
                    )

            # Model.get_or_create(session, ...) -- may commit internally + return model
            elif method_name in _COMMIT_QUERY_METHODS:
                # commit=False means flush only, no expiration
                if not self._has_keyword_false(call, 'commit'):
                    self._expire_all_tracked_vars()
                # Return value is a fresh/existing model instance (columns loaded, rels not)
                model_name = self._get_call_class_name(call)
                if model_name and model_name in self.model_relationships:
                    self.tracked_vars[var_name] = _TrackedVar(
                        model_name=model_name,
                        loaded_rels=loaded_rels,
                        post_commit=False,
                        caller_provided=False,
                        line=self._abs_line(node),
                    )

    # ========================= Attribute access detection =========================

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Detect attribute access."""
        # Only check var.attr form (var is a Name node)
        if not isinstance(node.value, ast.Name):
            self.generic_visit(node)
            return

        var_name = node.value.id
        attr_name = node.attr

        # Skip method calls (e.g. obj.save())
        parent = self._get_parent(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            self.generic_visit(node)
            return

        # Skip assignment targets (self.attr = value is a write, does not trigger lazy load)
        if isinstance(parent, ast.Assign) and node in parent.targets:
            self.generic_visit(node)
            return

        # Check if it's a tracked variable's attribute access
        if var_name in self.tracked_vars:
            var_info = self.tracked_vars[var_name]
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
                            f"Accessing '{var_name}.{attr_name}' relationship after "
                            f"save()/update() without load= parameter. "
                            f"Suggestion: load={var_info.model_name}.{attr_name}"
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
                            f"Accessing '{var_name}.{attr_name}' relationship "
                            f"without load= parameter. "
                            f"Suggestion: load={var_info.model_name}.{attr_name}"
                        ),
                    ))
            elif var_info.post_commit and attr_name in cols:
                # RLC007: column access on expired (post-commit) object
                # After commit, ALL objects in the session are expired.
                # Accessing any column triggers a synchronous lazy load,
                # which causes MissingGreenlet in async context.
                # Typical scenario: obj_a.save() then obj_b.column (obj_b not refreshed)
                self.warnings.append(RelationLoadWarning(
                    code='RLC007',
                    file=self.source_file,
                    line=self._abs_line(node),
                    message=(
                        f"Accessing column '{var_name}.{attr_name}' on expired object "
                        f"after commit. The object was not refreshed and access will "
                        f"trigger synchronous lazy load -> MissingGreenlet. "
                        f"Suggestion: await session.refresh({var_name})"
                    ),
                ))

        self.generic_visit(node)

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

                # return await Model.get(..., load=...)
                if method_name in _QUERY_METHODS:
                    model_name = self._get_call_class_name(call)
                    if model_name:
                        self._check_return_loaded(model_name, loaded_rels, node)

                # return await obj.save(session, load=...)
                elif method_name in _SAVE_METHODS:
                    obj_name = self._get_call_object_name(call)
                    if obj_name and obj_name in self.tracked_vars:
                        model_name = self.tracked_vars[obj_name].model_name
                        self._check_return_loaded(model_name, loaded_rels, node)

                # return await Model.get_with_count(...)
                elif method_name == 'get_with_count':
                    model_name = self._get_call_class_name(call)
                    if model_name:
                        self._check_return_loaded(model_name, loaded_rels, node)

        self.generic_visit(node)

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

    def _handle_session_refresh(self, call: ast.Call) -> None:
        """
        Handle ``await session.refresh(obj)`` call.

        Refresh restores column attributes (un-expires the object),
        but does NOT load relationships.
        Matches pattern: ``await session.refresh(var_name)``
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
                # relationships are still NOT loaded (refresh doesn't load them)
                # loaded_rels remains empty

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
    """

    def __init__(
        self,
        app: Any,
        *,
        project_root: str | None = None,
        skip_paths: list[str] | None = None,
    ) -> None:
        self.app = app
        self.project_root = project_root or _PROJECT_ROOT
        self.skip_paths = skip_paths
        self._checked = False

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
        warnings.extend(
            checker.check_project_coroutines(
                self.project_root,
                skip_paths=self.skip_paths,
            )
        )

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
