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
    - RLC013: column access after ``yield`` in an async generator (consumer holds the same session and may commit during the yield, expiring the object)
    - RLC014: a FastAPI ``Depends`` commits inside its body, so ORM objects injected by
      *sibling* dependencies on the same session are already expired when the endpoint
      starts; accessing their columns raises MissingGreenlet

Expiry-modelling premises (shared by RLC007/008/010/013/014):

1. **Session parameters are detected by subclass, not identity.** A parameter
   annotated with any subclass of sqlmodel's ``AsyncSession`` (including
   :class:`sqlmodel_ext.session.AsyncSession`) is a session parameter
   (``_is_async_session_hint``). An identity check would make commit / detach /
   model-returning discovery blind to every method annotated with a subclass.
2. **Session identity.** A commit only expires tracked objects when the session
   passed to the commit method is one of the analyzed function's *own* session
   parameters. Passing a short-lived local session (``async with factory() as s``)
   does not expire objects tracked on the signature session. In functions with
   several session parameters, committing one session only expires the objects
   bound to that session (``_TrackedVar.session_name``).
3. **Two kinds of conditional commit** (different semantics, configured separately):

   a. :data:`conditional_commit_methods` -- methods that commit only on a cache
      miss (singleton creation, idempotent get-or-create) and are pure reads in
      the steady state. The call site cannot tell hit from miss, so the whole
      method is excluded from commit discovery: the analyzer *accepts* missing
      the rare miss-path commit in exchange for zero steady-state false positives.
   b. :data:`explicit_commit_methods` -- methods declared as
      ``commit: bool = False`` that only commit when the caller passes
      ``commit=True``. The call site *can* tell, so the method stays in the commit
      set and each call is judged by its own arguments: only a literal
      ``commit=True`` counts as a commit (``_ast_call_commits``). A dynamic value
      is treated as "no commit" because it cannot be proven true.

   Rule of thumb: if the call site can statically show whether this call
   commits, use (b); otherwise (a).
4. **Detach immunity.** After ``session.reset()`` (directly or through a
   detaching method) objects are detached: loaded column values stay readable
   and later commits can no longer expire them (``_TrackedVar.detached``).
5. **Test code is resolved per pytest fixture.** With
   ``check_project_coroutines(params_share_session=False)`` a model parameter of
   a test function is considered to share the test's session if and only if the
   fixture's transitive dependency closure consumes one of the session fixture
   *names* requested by the test itself. Transient constructions, fixtures that
   open their own session, and fixtures whose definition cannot be located are
   modelled as detached (see ``_fixture_param_shares_session``).

Invariant: every warning returned by a public ``check_*`` entry point
(``check_app`` / ``check_model_methods`` / ``check_project_coroutines`` /
``check_function``) has already been filtered through ``# noqa: RLCxxx``
suppressions (``_filter_noqa_suppressions``). The per-unit helpers
(``_check_endpoint`` / ``_check_model_method`` / ``_check_coroutine``) are only
aggregated by those entry points. Endpoint warnings are anchored on the line
returned by ``inspect.getsourcelines`` (which includes decorators), so an
endpoint ``# noqa`` comment must be placed on the **first decorator line**.

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

Commit-semantics configuration (module-level, read when a checker analyzes code)::

    rlc.conditional_commit_methods = frozenset({'get_or_create'})
    rlc.explicit_commit_methods = frozenset({'enqueue'})
    rlc.dependency_commit_methods = rlc.dependency_commit_methods | {'approve'}
"""
import atexit
import ast
import functools
import inspect as python_inspect
import logging
import os
import pathlib
import re
import symtable
import sys
import textwrap
import types
import typing
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Self, TypeVar, Union, override

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import QueryableAttribute
from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSession

from sqlmodel_ext._type_unwrap import (
    get_pydantic_generic_args,
    unwrap_generic_to_dto_class,
    unwrap_to_class,
)

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = '<unknown>'

_MAX_WRAPPER_DEPTH = 32
"""Maximum number of dependency wrapper layers (``functools.partial`` /
``__wrapped__``) unwrapped when extracting ``load=``.

Cycles are already stopped by a ``seen`` set; this bound is a second line of
defense so the checker is harder to break than the code it checks."""

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


# ========================= Commit-semantics configuration =========================

conditional_commit_methods: frozenset[str] = frozenset()
"""Method names that commit only conditionally and cannot be told apart at the call site.

Typical members are "commit on miss" helpers: lazily created singletons
(``get_instance``) or idempotent ``get_or_create`` style methods that are pure
reads in the steady state. Names listed here are **excluded from commit
discovery entirely** (they neither count as commit methods nor propagate commit
status to their callers through the transitive closure). The trade-off is
deliberate: the rare miss-path commit is no longer detected, in exchange for no
false RLC007/008/010 after every steady-state read.

Matching is by method name across all classes, so only list names that have
this semantics everywhere in your project. Defaults to empty: none of the
library's own methods commit conditionally.
"""

explicit_commit_methods: frozenset[str] = frozenset()
"""Method names declared as ``commit: bool = False`` that commit only when asked.

Unlike :data:`conditional_commit_methods`, these methods **stay** in the commit
set; each call is judged by its own arguments. A call counts as a commit only
when it passes the literal ``commit=True``; omitting the argument,
``commit=False`` or a dynamic value count as "no commit" (a dynamic value cannot
be proven true, and treating it as a commit would cascade false positives over
every later attribute access).

Methods whose ``commit`` parameter defaults to ``True`` (the library's own
``save`` / ``update`` / ``delete`` / ``add``) must **not** be listed here: for
them only an explicit ``commit=False`` suppresses the commit, which is the
default rule. Defaults to empty.
"""

dependency_commit_methods: frozenset[str] = frozenset({'add', 'save', 'update', 'delete'})
"""Method names whose call inside a FastAPI dependency triggers RLC014.

RLC014 flags endpoint parameters injected by a *sibling* dependency when some
dependency of the same endpoint commits the shared session. Only methods that
(almost) always commit belong here; "commit on miss" readers such as
``get_or_create`` are intentionally left out, otherwise every singleton-reading
dependency would be reported as poisoning its siblings.

A name only triggers RLC014 if it is also an auto-discovered commit method
(intersection with ``RelationLoadChecker.commit_methods``), which guards against
same-named non-committing methods. Defaults to the library's own always-commit
CRUD methods; extend it with your project's committing domain methods.
"""


@dataclass
class RelationLoadWarning:
    """Relation load static analysis warning."""
    code: str
    """Rule code (RLC001-RLC014)."""
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

# Model classmethod queries: go through cache -> DB and return a fresh instance,
# so they act as a refresh for an expired object. Only treated as a refresh when
# the receiver is a known model class (``Type.get(...)``), excluding ``dict.get()``.
_MODEL_REFRESH_CLASSMETHODS = frozenset({'get', 'get_one', 'get_instance'})


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
    expired_by_yield: bool = False
    """Whether the object was expired by a ``yield`` that handed control to a consumer
    (the consumer may commit on the shared session). Set together with ``post_commit``
    so downstream checks can distinguish RLC007 (commit) from RLC013 (yield) in the
    error message."""
    pre_committed_by_sibling_dep: bool = False
    """Whether the object is already expired at function entry because a sibling
    FastAPI dependency committed the shared session. Dependency resolution order is
    not guaranteed and ``expire_on_commit=True`` expires every object in the session,
    so any dependency-internal commit expires the other injected objects. Set together
    with ``post_commit`` to distinguish RLC007 (commit in the body) from RLC014
    (commit at the dependency boundary)."""
    detached: bool = False
    """Whether the object has been detached by ``session.reset()`` (directly or through
    a detaching method). ``reset`` = ``expunge_all`` + release the connection: loaded
    column values stay readable, and because the object is no longer in the identity
    map, **later commits cannot expire it**. Detached variables are therefore immune to
    expiry (including yield-expiry); otherwise the correct pattern "reset to release
    the connection -> persist through a short-lived session -> keep reading the
    detached object's columns" would be reported as RLC007/010."""
    session_name: str | None = None
    """Name of the **signature session parameter** the object belongs to (in a function
    with several sessions, committing one session only expires that session's objects).
    Taken from the session argument of the query that produced it, or, for parameters,
    from the function's only session parameter. ``None`` means unknown: the variable is
    hit by an expire/detach targeting **any** session (pessimistic; in single-session
    functions this is identical to expiring everything)."""
    line: int = 0
    """Definition/last-update line number."""


def _is_async_session_hint(hint: Any) -> bool:
    """
    Whether a parameter annotation denotes a database ``AsyncSession``.

    **Decided by subclass, not identity**: projects (and this library) annotate
    session parameters with subclasses of sqlmodel's ``AsyncSession`` such as
    :class:`sqlmodel_ext.session.AsyncSession`. An identity check (``hint is
    AsyncSession``) is always false for those, which would leave commit /
    detaching / model-returning discovery with only the methods that happen to
    annotate the upstream base class.

    Accepted forms:

    - a resolved type: ``AsyncSession`` or any subclass
    - ``Annotated[AsyncSession, ...]`` (FastAPI dependency aliases; only reached
      when falling back to raw ``__annotations__``)
    - ``AsyncSession | None`` / ``Optional[AsyncSession]``: any member matches
    - ``ForwardRef`` / unresolved string annotations (``TYPE_CHECKING`` imports):
      split on ``|``, and a segment must be **exactly** ``'AsyncSession'``. There is
      deliberately no fuzzy ``__name__`` / dotted-suffix matching, because unrelated
      HTTP clients are also commonly named ``AsyncSession``.
    """
    if isinstance(hint, str):
        return any(part.strip() == 'AsyncSession' for part in hint.split('|'))
    if isinstance(hint, typing.ForwardRef):
        return _is_async_session_hint(hint.__forward_arg__)
    origin = typing.get_origin(hint)
    if origin is Annotated:
        return _is_async_session_hint(typing.get_args(hint)[0])
    if origin is Union or origin is types.UnionType:  # pyright: ignore[reportDeprecated]
        return any(_is_async_session_hint(arg) for arg in typing.get_args(hint))
    return isinstance(hint, type) and issubclass(hint, _AsyncSession)


def _session_param_names(func: Any) -> list[str]:
    """
    Names of ``func``'s parameters annotated as ``AsyncSession``, in signature order
    (see ``_is_async_session_hint``).

    ``typing.get_type_hints`` may fail as a whole because of third-party
    ForwardRefs or ``TYPE_CHECKING`` imports; the raw ``__annotations__`` are used
    in that case. The ``return`` annotation is not a parameter and is excluded
    (otherwise ``-> AsyncSession | None`` would yield a parameter named ``return``).
    """
    try:
        hints: dict[str, Any] = typing.get_type_hints(func)
    except Exception:
        hints = getattr(func, '__annotations__', {})
    return [
        name for name, hint in hints.items()
        if name != 'return' and _is_async_session_hint(hint)
    ]


def _extract_model_from_hint(hint: Any, model_names: set[str]) -> str | None:
    """
    Extract a model class name from a type annotation.

    Covered forms:

    - a direct type ``ModelType``
    - ``ModelType | None`` / ``Optional[ModelType]``
    - ``ClassVar[T]`` -> recurse into ``T``
    - ``Annotated[T, ...]`` (already stripped by ``get_type_hints(include_extras=False)``)
    - containers (value / element type):

      * ``dict[K, V]`` / ``Mapping[K, V]`` / ``MutableMapping[K, V]`` -> ``V``
      * ``list[V]`` / ``Sequence[V]`` / ``Iterable[V]`` / ``set[V]`` /
        ``frozenset[V]`` / ``tuple[V, ...]`` -> ``V``

    Anything else (``int``, ``str``, unknown generics) returns ``None``.

    :param hint: resolved type annotation
    :param model_names: known model class names (used as a filter)
    :returns: the model class name if matched, otherwise ``None``
    """
    if isinstance(hint, type) and hint.__name__ in model_names:
        return hint.__name__

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is None:
        return None

    # ClassVar[T] -> recurse into T
    if origin is typing.ClassVar:
        return _extract_model_from_hint(args[0], model_names) if args else None

    # Union / Optional -> the first matching member wins
    if origin is Union or origin is types.UnionType:  # pyright: ignore[reportDeprecated]
        for a in args:
            if a is type(None):
                continue
            r = _extract_model_from_hint(a, model_names)
            if r is not None:
                return r
        return None

    # dict-like -> value type args[1]
    if origin is dict:
        if len(args) >= 2:
            return _extract_model_from_hint(args[1], model_names)
        return None

    # list / set / frozenset / tuple -> element type args[0]
    if origin in (list, set, frozenset, tuple):
        if args:
            return _extract_model_from_hint(args[0], model_names)
        return None

    # Abstract ABCs (typing.Mapping / Sequence etc. report collections.abc.* as origin)
    raw_origin_name = getattr(origin, '__name__', None)
    origin_name = raw_origin_name if raw_origin_name is not None else ''
    if origin_name in ('Mapping', 'MutableMapping') and len(args) >= 2:
        return _extract_model_from_hint(args[1], model_names)
    if origin_name in ('Sequence', 'MutableSequence', 'Iterable', 'Iterator',
                       'Collection', 'Set', 'MutableSet', 'AbstractSet') and args:
        return _extract_model_from_hint(args[0], model_names)

    return None


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
        # Detaching methods: methods that (transitively) call session.reset().
        # Even if such a method saves self, it finally detaches every object in
        # the session, so callers cannot rely on "a model-returning commit method
        # called on self refreshes self".
        self.detaching_methods: frozenset[str] = frozenset()
        # Per-class commit methods (MRO-aware, model class name -> effective commit method set)
        self._model_commit_methods: dict[str, frozenset[str]] = {}
        # Per-class detaching methods (same shape)
        self._model_detaching_methods: dict[str, frozenset[str]] = {}
        # Per-class **async model-returning** methods (same shape): a same-named
        # method on another class returning a scalar must not be tracked as a model.
        self._model_returning_by_class: dict[str, frozenset[str]] = {}
        # Intermediate state (for _discover_non_model_commit_methods to extend incrementally)
        self._method_asts: dict[str, list[tuple[str, str, ast.Module]]] = {}
        self._class_commit: dict[str, set[str]] = {}
        self._class_detaching: dict[str, set[str]] = {}
        # Per-class index "self.<attr> -> model class name". Lets
        # ``var = self.<attr>.get(...)`` / ``self.<attr>[key]`` be recognized as a
        # model lookup, so ORM objects taken out of instance-attribute containers
        # enter tracked_vars and are expired by later commits (RLC007).
        # Built from annotations shaped like ``dict[K, Model]`` / ``list[Model]`` /
        # ``Model | None`` / ``Model``.
        self._model_self_attr_models: dict[str, dict[str, str]] = {}

        self._build_knowledge_base(base_class)
        (
            self.commit_methods, self.model_returning_methods,
            self.sync_model_returning_methods, self.refreshing_commit_methods,
            self.detaching_methods,
        ) = self._discover_method_behaviors()
        # Extend commit method set: scan non-model project classes (e.g. Messages)
        # for transitive commit methods.
        self._discover_non_model_commit_methods()
        # model_name -> {dunder_name -> set of accessed relationship names}
        self.model_dunder_rels: dict[str, dict[str, set[str]]] = {}
        self._scan_dunder_relationship_access()
        # Build the self.<attr> -> model class name index (needs model_classes)
        self._build_self_attr_models()

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

    def _build_self_attr_models(self) -> None:
        """
        Build the ``{class_name: {attr_name: model_class_name}}`` index.

        Scans the annotations of every model class (including bases and mixins
        in its MRO) for instance attributes annotated as ``dict[K, Model]`` /
        ``list[Model]`` / ``Model | None`` / ``Model``. Rich-model methods often
        use such attributes as cached ORM containers; objects taken out of them
        must be tracked, otherwise they are an RLC007/RLC008 blind spot.

        The most specific declaration wins (subclass before base class).
        """
        model_class_names = set(self.model_classes.keys())
        for cls_name, cls in self.model_classes.items():
            attr_models: dict[str, str] = {}
            for klass in cls.__mro__:
                if klass is object:
                    continue
                try:
                    hints = typing.get_type_hints(klass, include_extras=False)
                except Exception:
                    hints = getattr(klass, '__annotations__', None) or {}
                for attr, hint in hints.items():
                    if attr in attr_models:
                        continue
                    model_name = _extract_model_from_hint(hint, model_class_names)
                    if model_name is not None:
                        attr_models[attr] = model_name
            if attr_models:
                self._model_self_attr_models[cls_name] = attr_models

    def _discover_method_behaviors(
            self,
    ) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
        """
        Single-pass discovery of commit, detaching and model-returning methods.

        Uses the type system as single source of truth:

        **Commit methods** (anchored on ``AsyncSession.commit()``):

        1. Scan all model classes and their bases for async methods accepting ``AsyncSession``
        2. Check if the method body calls ``.commit()`` / ``.rollback()`` on that parameter
        3. Transitive closure: methods that call step-2 methods passing the session are also commit methods

        Names in :data:`conditional_commit_methods` are skipped entirely.

        **Detaching methods** (anchored on ``session.reset()``):

        1. Same method set as above
        2. Check if the method body calls ``.reset()`` on that parameter
        3. Transitive closure, as for commit methods
        4. Detaching and commit are **orthogonal**: ``reset`` = ``expunge_all``;
           detached objects keep their loaded column values and can no longer be
           expired by later commits.

        When a method has **several** ``AsyncSession`` parameters, the discovery
        above runs for **each** of them (``method_asts`` holds one entry per
        session parameter).

        **Model-returning methods** (anchored on return type annotations):

        1. Inspect the method's return type annotation
        2. If it returns ``Self``, a concrete model class, ``Self | None``, ``T``, etc. -> model-returning
        3. Sync methods only used for variable tracking (not added to safe_methods)

        :returns: (commit_methods, model_returning_methods, sync_model_returning_methods,
                   refreshing_commit_methods, detaching_methods)
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
        # method_name -> every version's (owning class, return type hint)
        method_return_hints: dict[str, list[tuple[str, Any]]] = {}
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

                    # AsyncSession-typed parameters (subclasses / Optional / string forward
                    # refs are all handled by _session_param_names). Methods without a
                    # session parameter do not take part in behavior discovery.
                    session_params = _session_param_names(func)
                    if not session_params:
                        continue

                    # Type hints for return type analysis. get_type_hints may fail due to
                    # third-party ForwardRefs; fall back to __annotations__ in that case.
                    try:
                        hints = typing.get_type_hints(func)
                    except Exception:
                        hints = None

                    try:
                        source = textwrap.dedent(python_inspect.getsource(func))
                        tree = ast.parse(source)
                    except (OSError, TypeError, SyntaxError):
                        continue

                    # All versions are saved for commit-method discovery (Phase 1/2
                    # must check every version), one entry **per session parameter**:
                    # anchoring only the first one would miss a method that commits or
                    # resets only its second session.
                    for session_param in session_params:
                        method_asts.setdefault(attr_name, []).append(
                            (klass.__name__, session_param, tree),
                        )

                    # Metadata only keeps the first version (MRO order, most specific subclass first)
                    if attr_name not in method_owners:
                        method_owners[attr_name] = cls_name

                    # Record the return type of **every** version together with its
                    # defining class. Same-named methods on different classes can
                    # disagree (``cancel -> bool`` vs ``cancel -> Self``); keeping only
                    # the first version would let class scan order decide, and "any
                    # version returns a model" would track scalars as models. The
                    # decision must be per class (see _rebuild_model_returning_methods).
                    return_hint = (
                        hints.get('return') if hints is not None
                        else getattr(func, '__annotations__', {}).get('return')
                    )
                    if return_hint is not None:
                        method_return_hints.setdefault(attr_name, []).append(
                            (klass.__name__, return_hint),
                        )

        # -------- Commit method discovery --------

        # Phase 1: methods that directly call session.commit() / session.rollback()
        # Check every version: if any version contains a direct session.commit(),
        # mark the method as a commit method.
        commit_methods: set[str] = set()
        # Per-class commit tracking: defining_class -> set of commit method names
        class_commit: dict[str, set[str]] = {}
        for method_name, versions in method_asts.items():
            if method_name in conditional_commit_methods:
                continue  # steady-state read, see conditional_commit_methods
            for owning_cls, sp, tree in versions:
                if _ast_has_typed_commit(tree, sp):
                    commit_methods.add(method_name)
                    class_commit.setdefault(owning_cls, set()).add(method_name)

        # Phase 2: transitive closure -- methods that call commit methods passing the session
        # Per-class tracking: when the callee type can be resolved from the AST,
        # check that type's commit state to avoid false positives from same-named
        # methods on different classes.
        model_class_names = frozenset(self.model_classes.keys())
        changed = True
        while changed:
            changed = False
            for method_name, versions in method_asts.items():
                if method_name in conditional_commit_methods:
                    continue  # must not mark callers as committing via the closure either
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

        # -------- Detaching method discovery --------
        # A detaching method calls session.reset() in some version, or transitively
        # calls another detaching method passing the session. After such a call every
        # tracked object is detached, so "self was refreshed through the identity map"
        # no longer holds.
        detaching_methods: set[str] = set()
        class_detaching: dict[str, set[str]] = {}
        for method_name, versions in method_asts.items():
            for owning_cls, sp, tree in versions:
                if _ast_has_typed_reset(tree, sp):
                    detaching_methods.add(method_name)
                    class_detaching.setdefault(owning_cls, set()).add(method_name)

        # Transitive closure: reuse the "method call + session argument" detection of
        # _ast_calls_commit_method_with_session with the detaching method set.
        # The detaching closure deliberately does NOT use per-class resolution: an
        # abstract base method calling ``self._step(session, ...)`` would resolve to the
        # abstract stub and miss the subclasses' detaching overrides, breaking the
        # chain. Global name matching is slightly looser, but a detaching false
        # positive is far cheaper than a missed MissingGreenlet.
        changed = True
        while changed:
            changed = False
            for method_name, versions in method_asts.items():
                for owning_cls, sp, tree in versions:
                    if method_name in class_detaching.get(owning_cls, set()):
                        continue
                    if _ast_calls_commit_method_with_session(
                        tree, sp, frozenset(detaching_methods),
                        owning_class=owning_cls,
                    ):
                        class_detaching.setdefault(owning_cls, set()).add(method_name)
                        detaching_methods.add(method_name)
                        changed = True

        # Detaching is NOT a subset of commit. SQLAlchemy semantics:
        # - ``session.commit()`` (expire_on_commit=True) -> ``expire_all()``: column
        #   attributes are invalidated and the next access emits a SELECT (->
        #   MissingGreenlet without a greenlet context). That is what RLC007/008/010 catch.
        # - ``session.reset()`` -> ``expunge_all()`` + release the connection: objects
        #   leave the identity map but their loaded ``__dict__`` values are kept, so
        #   reading loaded columns does not lazy load. Only relationship access fails,
        #   which RLC003 already covers.
        # Merging the two would flag the correct "reset to release the connection
        # during long I/O, then keep reading cached columns" pattern.

        # Save intermediate state for _discover_non_model_commit_methods to extend incrementally
        self._method_asts = method_asts
        self._class_commit = class_commit
        self._class_detaching = class_detaching

        # -------- Per-model-class commit / detaching methods (MRO-aware) --------
        self._rebuild_model_commit_methods(class_commit)
        self._rebuild_model_detaching_methods(class_detaching)

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

        # Global name set (conservative fallback when the receiver type cannot be
        # resolved, and pre-filter for the refreshing closure) + per-defining-class
        # sets (the real criterion; see _rebuild_model_returning_methods).
        model_returning: set[str] = set()
        class_returning: dict[str, set[str]] = {}
        for method_name, return_versions in method_return_hints.items():
            for owning_cls, return_hint in return_versions:
                if _hint_returns_model(return_hint):
                    model_returning.add(method_name)
                    class_returning.setdefault(owning_cls, set()).add(method_name)
        self._rebuild_model_returning_methods(class_returning)

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
        # If a method's return comes from save/update(commit!=False), from another
        # refreshing method, or from an explicit ``Model.get/get_one/get_instance``
        # re-fetch, the return value has already been refreshed internally, and
        # callers may safely access its column attrs.
        # Transitive closure: fill_from_video_url -> fill_from_url -> fill_from_file_path -> save()
        model_class_names_frozen = frozenset(self.model_classes.keys())
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
                    if _method_returns_from_refreshing(
                            tree, frozenset(refreshing_commit), model_class_names_frozen):
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
        if detaching_methods:
            logger.debug(f"Auto-discovered detaching methods: {sorted(detaching_methods)}")
        return (
            frozenset(commit_methods), frozenset(model_returning),
            frozenset(sync_model_returning), frozenset(refreshing_commit),
            frozenset(detaching_methods),
        )

    def _effective_per_class(
            self,
            class_methods: dict[str, set[str]],
    ) -> dict[str, frozenset[str]]:
        """
        Resolve a "defining class -> method names" map into a per-model-class map (MRO-aware).

        For each model class, walk the MRO and let the most specific definition of
        every attribute decide whether that name belongs to the class.
        """
        per_class: dict[str, frozenset[str]] = {}
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
                    if attr_name in class_methods.get(klass_name, set()):
                        effective.add(attr_name)
            per_class[mcls_name] = frozenset(effective)
        return per_class

    def _rebuild_model_commit_methods(
            self,
            class_commit: dict[str, set[str]],
    ) -> None:
        """Build the per-model-class commit method set (MRO-aware)."""
        self._model_commit_methods = self._effective_per_class(class_commit)

    def _rebuild_model_returning_methods(
            self,
            class_returning: dict[str, set[str]],
    ) -> None:
        """
        Build the per-model-class async model-returning method set (MRO-aware).

        Same-named methods on different classes (``cancel -> bool`` vs
        ``cancel -> Self``) do not contaminate each other: when the receiver type
        is resolved, only that class's definition decides.
        """
        self._model_returning_by_class = self._effective_per_class(class_returning)

    def _rebuild_model_detaching_methods(
            self,
            class_detaching: dict[str, set[str]],
    ) -> None:
        """Build the per-model-class detaching method set (MRO-aware)."""
        self._model_detaching_methods = self._effective_per_class(class_detaching)

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

        Module objects and the classes found in them are treated as **external
        input**: lazy-import shims or hostile metaclasses can make ``__file__``,
        ``dir()``, ``__module__``, ``__name__`` or ``vars()`` raise arbitrary
        exceptions, or return ``str`` subclasses with hijacked dunders. A single
        module / class / member that cannot be inspected is skipped; it never
        aborts checker construction.
        """
        project_root = _PROJECT_ROOT.replace('\\', '/')

        # -------- Collect non-model class method ASTs --------
        non_model_asts: dict[str, list[tuple[str, str, ast.Module]]] = {}
        seen_func_ids: set[int] = set()

        for _raw_module_name, module in list(sys.modules.items()):
            # The isolation boundary covers the whole module-identity decision, not
            # just the attribute reads: the *values* (e.g. a ``str`` subclass
            # ``__file__``) can raise from replace / startswith / ``in`` as well.
            # Values are normalized with ``str.__str__`` (``str(x)`` would go through
            # a hijackable ``__str__``).
            try:
                if module is None:
                    continue
                _module_name = str.__str__(_raw_module_name)
                module_file = getattr(module, '__file__', None)
                if module_file is None:
                    continue
                module_file_normalized = str.__str__(module_file).replace('\\', '/')
                _skip_module = (
                    not module_file_normalized.startswith(project_root)
                    or '/site-packages/' in module_file_normalized
                )
            except Exception:
                continue
            if _skip_module:
                continue

            # dir() calls __dir__, which a hostile module proxy can hijack too.
            # Keep (raw name for getattr, normalized label) pairs.
            try:
                _module_attr_pairs: list[tuple[Any, str]] = [
                    (n, str.__str__(n)) for n in dir(module)
                ]
            except Exception:
                continue

            for _raw_attr_name, _attr_label in _module_attr_pairs:
                try:
                    attr = getattr(module, _raw_attr_name)
                except Exception:
                    continue
                if not python_inspect.isclass(attr):
                    continue
                # Class metadata reads may raise arbitrarily (a metaclass hijacking
                # __getattribute__); ``getattr(x, k, default)`` only swallows
                # AttributeError. The boundary covers the whole class-identity
                # decision including comparisons and iteration.
                try:
                    _attr_module = str.__str__(attr.__module__)
                    _attr_name_str = str.__str__(attr.__name__)
                    _attr_vars = vars(attr)
                    _skip_class = (_attr_module != _module_name
                                   or _attr_name_str in self.model_classes)
                    # Raw key for indexing (a custom mapping may key on identity),
                    # normalized label for comparison and formatting.
                    _member_pairs: list[tuple[Any, str]] = (
                        [] if _skip_class else [(k, str.__str__(k)) for k in _attr_vars]
                    )
                except Exception:
                    continue
                if _skip_class:
                    continue

                for _raw_key, method_name in _member_pairs:
                    if method_name.startswith('__') and method_name.endswith('__'):
                        continue

                    # The whole per-candidate discovery pipeline is one isolation
                    # boundary: candidates flow into introspection outside this file
                    # (get_type_hints, iscoroutinefunction, ...), which a boundary drawn
                    # around individual reads cannot see. getsource's
                    # (OSError, TypeError, SyntaxError) stays a narrow inner catch: that
                    # is "normal un-analyzable", not a hostile candidate.
                    try:
                        raw = _attr_vars[_raw_key]
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

                        # Non-model classes commonly import AsyncSession under
                        # TYPE_CHECKING (a string annotation at runtime);
                        # _session_param_names covers that fallback.
                        session_params = _session_param_names(func)
                        if not session_params:
                            continue

                        try:
                            source = textwrap.dedent(python_inspect.getsource(func))
                            tree = ast.parse(source)
                        except (OSError, TypeError, SyntaxError):
                            continue
                    except Exception:
                        continue

                    # One entry per session parameter (same as _discover_method_behaviors)
                    for session_param in session_params:
                        non_model_asts.setdefault(method_name, []).append(
                            (_attr_name_str, session_param, tree)
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
            if method_name in conditional_commit_methods:
                continue  # steady-state read, see conditional_commit_methods
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
                if method_name in conditional_commit_methods:
                    continue  # must not mark callers as committing via the closure either
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
        :returns: all detected warnings, already filtered by ``# noqa: RLCxxx``
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

        return self._filter_noqa_suppressions(warnings)

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

        :returns: all detected warnings, already filtered by ``# noqa: RLCxxx``
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

        Every public ``check_*`` entry point calls this before returning.

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
        params_share_session: bool = True,
    ) -> list[RelationLoadWarning]:
        """
        Scan all imported modules' async functions and async generators.

        Iterates sys.modules, analyzing coroutine functions and async generators
        from project source files. Includes module-level functions and methods
        of non-SQLModel classes (e.g. command handlers, service classes).
        Automatically skips functions already analyzed by check_app/check_model_methods.

        :param project_root: absolute path to project root directory
        :param skip_paths: list of path fragments to skip (e.g. ['/base/', '/mixin/'])
        :param skip_third_party_attrs: skip project-module attributes whose inspection
            raises. When imported third-party libs use lazy proxies (e.g.
            openai.AudioProxy), inspect operations may trigger client initialization
            and raise; module objects themselves (lazy-import shims) and hostile class
            metadata can raise as well. When enabled, those exceptions are caught and
            the module / attribute / member is skipped; when disabled they propagate
            (fail loud).
        :param params_share_session: whether a function's model parameters live in the
            same identity map as its session parameter. ``True`` (default) fits
            production code: FastAPI dependencies / callers assemble parameters on the
            same session, so a commit in the body expires them. **Pass ``False`` when
            scanning pytest test code** -- this is not a blanket "parameters are
            detached" exemption: each parameter is resolved through its pytest fixture.
            A fixture whose transitive dependency closure reaches a session fixture
            *name* requested by the test shares the session and is checked normally;
            transient constructions, fixtures opening their own session, and fixtures
            whose definition cannot be located are modelled as detached (see
            ``_fixture_param_shares_session``).
        :returns: all detected warnings, already filtered by ``# noqa: RLCxxx``
        """
        warnings: list[RelationLoadWarning] = []
        # Normalize path separators
        project_root_normalized = project_root.replace('\\', '/')

        default_skip = skip_paths or []

        for _raw_module_name, module in list(sys.modules.items()):
            # Module objects are external input (see _discover_non_model_commit_methods);
            # here the isolation obeys skip_third_party_attrs: True skips, False fails loud.
            try:
                if module is None:
                    continue
                module_name = str.__str__(_raw_module_name)
                module_file = getattr(module, '__file__', None)
                if module_file is None:
                    continue
                module_file_normalized = str.__str__(module_file).replace('\\', '/')
                _skip_module = (
                    not module_file_normalized.startswith(project_root_normalized)
                    # Skip third-party libraries (venv site-packages paths may start
                    # with the project root but are not project code)
                    or '/site-packages/' in module_file_normalized
                    # Skip configured paths
                    or any(skip in module_file_normalized for skip in default_skip)
                )
            except Exception:
                if skip_third_party_attrs:
                    continue
                raise
            if _skip_module:
                continue

            # Collect functions to analyze: module-level + class methods
            funcs_to_check: list[tuple[str, Any]] = []

            # dir() can be hijacked as well; its elements are external values too.
            try:
                _module_attr_pairs: list[tuple[Any, str]] = [
                    (n, str.__str__(n)) for n in dir(module)
                ]
            except Exception:
                if skip_third_party_attrs:
                    continue
                raise

            for _raw_attr_name, attr_name in _module_attr_pairs:
                try:
                    attr = getattr(module, _raw_attr_name)
                except Exception:
                    if skip_third_party_attrs:
                        continue
                    raise

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
                    # __annotations__/__globals__. The ownership check obeys the same
                    # skip_third_party_attrs contract (the reads and the comparison
                    # can both raise on hostile objects).
                    try:
                        actual_func = getattr(attr, '__wrapped__', attr)
                        _raw_func_module = getattr(actual_func, '__module__', None)
                        _func_module_matches = (
                            _raw_func_module is not None
                            and str.__str__(_raw_func_module) == module_name
                        )
                    except Exception:
                        if skip_third_party_attrs:
                            continue
                        raise
                    if _func_module_matches:
                        funcs_to_check.append((f"{module_name}.{attr_name}", actual_func))
                elif is_class:
                    # Non-model class methods (model classes already covered by check_model_methods)
                    try:
                        _attr_module_matches = str.__str__(attr.__module__) == module_name
                    except Exception:
                        if skip_third_party_attrs:
                            continue
                        raise
                    if not _attr_module_matches:
                        continue
                    try:
                        _attr_name_str = str.__str__(attr.__name__)
                        _attr_vars = vars(attr)
                        # Model classes are analyzed by check_model_methods
                        _skip_class = _attr_name_str in self.model_classes
                        _member_pairs: list[tuple[Any, str]] = (
                            [] if _skip_class else [(k, str.__str__(k)) for k in _attr_vars]
                        )
                    except Exception:
                        if skip_third_party_attrs:
                            continue
                        raise
                    if _skip_class:
                        continue
                    for _raw_key, method_name in _member_pairs:
                        if method_name.startswith('__') and method_name.endswith('__'):
                            continue
                        # One isolation boundary for the whole per-candidate pipeline
                        # (candidates flow into introspection outside this file).
                        try:
                            raw = _attr_vars[_raw_key]
                            func = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
                            is_func_async = self._is_async_callable(func)
                            # Keep the original semantics: is_func_async decides; a
                            # literal ``__wrapped__ = None`` must not drop a real
                            # async function.
                            actual_func = getattr(func, '__wrapped__', func) if is_func_async else func
                        except Exception:
                            if skip_third_party_attrs:
                                continue
                            raise
                        if is_func_async:
                            funcs_to_check.append(
                                (f"{module_name}.{_attr_name_str}.{method_name}", actual_func),
                            )

            for label, func in funcs_to_check:
                if id(func) in self._analyzed_func_ids:
                    continue
                self._analyzed_func_ids.add(id(func))

                try:
                    func_warnings = self._check_coroutine(
                        func=func, label=label,
                        resolve_fixture_params=not params_share_session,
                    )
                    warnings.extend(func_warnings)
                except Exception as e:
                    logger.debug(f"Error analyzing coroutine {label}: {e}")

        return self._filter_noqa_suppressions(warnings)

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
        :returns: detected warnings, already filtered by ``# noqa: RLCxxx``
        """
        return self._filter_noqa_suppressions(
            self._check_coroutine(func, label='<standalone>'),
        )

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

        # 3b. RLC014: find committing dependencies and the sibling parameters they expire.
        # With ``expire_on_commit=True`` a ``Model.save()`` inside any dependency expires
        # every ORM object of the shared session, so the ORM parameters injected by the
        # other dependencies are already stale when the endpoint starts.
        committing_dep_params = self._get_committing_dep_params(endpoint)
        pre_committed_params: set[str] = set()
        if committing_dep_params:
            # Every ORM parameter except the committing dependencies' own results is
            # affected. A committing dependency conventionally re-loads the object it
            # returns (``load=`` at the end), so its own parameter is exempt.
            pre_committed_params = {
                p for p in param_models
                if p not in committing_dep_params
            }

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
            pre_committed_params=pre_committed_params,
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

        All method parameters (``self`` plus other model-typed arguments) are marked
        as caller_provided — these objects are passed in by the caller, preloading
        is the caller's contract (typically declared via a docstring "preload before
        calling: ..." note), so the method body must not be flagged with RLC003 for
        "missing load=". This matches ``_check_coroutine``'s policy of marking all
        params as caller_provided.

        Post-commit relation access still triggers RLC002 / RLC007 / RLC008
        (caller_provided does not exempt those checks).
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

        # cls -> class alias (classmethod's cls parameter doesn't enter tracked_vars,
        # only used for resolving class-level calls)
        class_aliases: dict[str, str] = {}
        if first_param == 'self':
            param_models['self'] = cls_name
        elif first_param == 'cls':
            class_aliases['cls'] = cls_name

        # All method params are caller-provided (self + other model-typed args
        # are the caller's preload responsibility).
        caller_provided_params: set[str] = set(param_models.keys())

        # @requires_relations declared rels as self's dep_loads
        dep_loads: dict[str, set[str]] = {}
        if 'self' in param_models and decorator_loads:
            dep_loads['self'] = decorator_loads

        # self.<attr> -> model class name index (the class itself and its MRO, subclass first)
        self_attr_models = self._model_self_attr_models.get(cls_name, {})

        warnings, _ = self._analyze_function_body(
            func=func,
            param_models=param_models,
            required_rels={},
            dep_loads=dep_loads,
            label=label,
            caller_provided_params=caller_provided_params,
            pre_parsed=(source_file, tree, line_offset),
            class_aliases=class_aliases,
            self_attr_models=self_attr_models,
        )
        return warnings

    def _check_coroutine(
        self,
        func: Any,
        label: str,
        resolve_fixture_params: bool = False,
    ) -> list[RelationLoadWarning]:
        """Check a regular coroutine function (background tasks, stream handlers, etc.).

        :param resolve_fixture_params: when ``True`` (pytest test-code scan, see
            ``check_project_coroutines(params_share_session=False)``), every model
            parameter is resolved through its pytest fixture: if the fixture's
            transitive dependency closure reaches a session fixture name requested by
            the test, the object lives on the shared session and is tracked normally;
            otherwise (transient construction / fixture-owned session / definition not
            found) it is modelled as detached.
        """
        param_models = self._resolve_param_models(func)

        detached_params: frozenset[str] = frozenset()
        if resolve_fixture_params:
            detached_params = frozenset(
                p for p in param_models
                if not self._fixture_param_shares_session(func, p)
            )

        # All params are caller-provided, skip RLC003
        warnings, _ = self._analyze_function_body(
            func=func,
            param_models=param_models,
            required_rels={},
            dep_loads={},
            label=label,
            caller_provided_params=set(param_models.keys()),
            detached_params=detached_params,
        )
        return warnings

    @classmethod
    def _fixture_param_shares_session(cls, func: Any, param_name: str) -> bool:
        """
        Whether a pytest test function's model parameter (a fixture product) shares
        the identity map of the session injected into the test body.

        The criterion is the **name identity of the session fixture**, not the mere
        presence of a session-typed parameter: pytest instantiates a fixture name once
        per test, so the parameter shares the session if and only if its fixture's
        **transitive dependency closure** consumes one of the session fixture names
        requested by the test itself (the test's ``AsyncSession``-typed parameter
        names). Examples in both directions:

        - ``independent_user(independent_session)`` -- depends on a session-typed
          fixture, but a *different* one (opened from a session factory) => not shared;
        - ``wrapped_user(shared_user)`` -- no session parameter at this level, but
          ``shared_user(session)`` consumes the test's ``session`` => shared (the
          dependency graph is followed recursively, with a cycle guard).

        Fixture lookup mimics pytest: the test's own module, then package-level
        ``conftest`` modules from the deepest to the shallowest (closest override
        wins). A fixture whose definition cannot be found (third-party plugin,
        dynamic registration) is treated as "not shared": keeping full checks for an
        unknown source would reintroduce the false positives this mode avoids.
        """
        session_names = frozenset(_session_param_names(func))
        if not session_names:
            return False  # the test did not request a session fixture => nothing to share
        modules = cls._fixture_lookup_modules(func)
        if not modules:
            return False
        visited: set[tuple[str, int]] = set()
        return cls._fixture_reaches_test_session(
            param_name, modules, session_names, visited,
        )

    @staticmethod
    def _fixture_lookup_modules(func: Any) -> list[Any]:
        """Fixture resolution order: the function's module, then package ``conftest``
        modules from the deepest to the shallowest (closest first)."""
        module_name = getattr(func, '__module__', None)
        if not module_name:
            return []
        modules: list[Any] = []
        own_module = sys.modules.get(module_name)
        if own_module is not None:
            modules.append(own_module)
        parts = module_name.split('.')
        for depth in range(len(parts) - 1, 0, -1):
            conftest = sys.modules.get('.'.join(parts[:depth]) + '.conftest')
            if conftest is not None:
                modules.append(conftest)
        return modules

    @staticmethod
    def _resolve_fixture_func(
        name: str,
        modules: list[Any],
        start: int = 0,
    ) -> tuple[Any, int] | None:
        """Resolve a fixture name to ``(original function, index of the hit module)``,
        searching ``modules`` from ``start``.

        ``start`` supports pytest's same-name fixture override: a child fixture
        ``shared_user(shared_user)`` must skip its own definition and continue from
        the next (shallower) module. Unwraps the ``@pytest.fixture`` wrappers of
        different pytest versions (``__wrapped__`` / ``.func``).
        """
        for idx in range(start, len(modules)):
            fixture_obj = getattr(modules[idx], name, None)
            if fixture_obj is None:
                continue
            fixture_func = getattr(fixture_obj, '__wrapped__', None)
            if fixture_func is None:
                # Some pytest versions' FixtureFunctionDefinition expose the function as .func
                fixture_func = getattr(fixture_obj, 'func', None)
            if fixture_func is None and callable(fixture_obj):
                fixture_func = fixture_obj
            if callable(fixture_func):
                return fixture_func, idx
        return None

    @classmethod
    def _fixture_reaches_test_session(
        cls,
        name: str,
        modules: list[Any],
        session_names: frozenset[str],
        visited: set[tuple[str, int]],
        start: int = 0,
    ) -> bool:
        """
        Fixture dependency-graph reachability: does the transitive closure of ``name``
        (resolved from ``start``) consume one of ``session_names`` (the session fixture
        names requested by the test)?

        A name hit means shared -- pytest has a single instance per fixture name per
        test, so the annotation type does not matter (this also covers unannotated
        ``session`` parameters). Alias fixtures (``db(session) -> session``) are
        covered by the recursion.

        **Graph nodes are "name x resolution position", not bare names**: pytest lets
        a child ``shared_user(shared_user)`` request the parent conftest's fixture of
        the same name -- two different definitions, not a cycle. A same-named
        dependency is resolved from the *next* module (``start=idx+1``); a differently
        named one restarts from the head (the test's view of overrides). ``visited``
        de-duplicates on ``(name, hit index)``, so real cycles are still stopped.
        """
        resolved = cls._resolve_fixture_func(name, modules, start)
        if resolved is None:
            return False
        fixture_func, idx = resolved
        if (name, idx) in visited:
            return False
        visited.add((name, idx))
        try:
            dep_names = list(python_inspect.signature(fixture_func).parameters)
        except (TypeError, ValueError):
            return False
        for dep in dep_names:
            if dep in session_names:
                return True
            dep_start = idx + 1 if dep == name else 0
            if cls._fixture_reaches_test_session(
                dep, modules, session_names, visited, start=dep_start,
            ):
                return True
        return False

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
        self_attr_models: dict[str, str] | None = None,
        pre_committed_params: set[str] | None = None,
        detached_params: frozenset[str] | None = None,
    ) -> tuple[list[RelationLoadWarning], '_FunctionAnalyzer | None']:
        """
        Core AST analysis: parse function body and run _FunctionAnalyzer.

        :param caller_provided_params: set of caller-provided parameter names, skip RLC003
        :param pre_parsed: pre-parsed (source_file, tree, line_offset) to avoid re-parsing
        :param class_aliases: class alias mapping (e.g. cls -> UserFile) for resolving classmethod calls
        :param self_attr_models: the class's ``self.<attr>`` -> model class name index, used
            to track ``var = self.<attr>.get(key)`` style lookups as model queries
        :param pre_committed_params: parameter names already expired at entry by a
            sibling dependency's commit (endpoints only). Their tracked state starts
            with ``post_commit=True`` + ``pre_committed_by_sibling_dep=True`` (RLC014).
        :param detached_params: parameter names modelled as detached (not sharing the
            analyzed function's session); later commits do not expire them.
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
        session_param_names = frozenset(_session_param_names(func))

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
            session_param_names=session_param_names,
            model_commit_methods=self._model_commit_methods,
            model_rel_targets=self.model_rel_targets,
            refreshing_commit_methods=self.refreshing_commit_methods,
            detaching_methods=self.detaching_methods,
            model_detaching_methods=self._model_detaching_methods,
            self_attr_models=self_attr_models,
            pre_committed_params=pre_committed_params or set(),
            detached_params=detached_params,
            model_returning_by_class=self._model_returning_by_class,
            analyzed_func=func,
        )
        # Manually iterate ``func_node.body`` instead of ``analyzer.visit(func_node)``:
        # ``visit_FunctionDef`` / ``visit_AsyncFunctionDef`` are no-ops (used to skip
        # nested function bodies — see their docstrings). Calling ``visit(func_node)``
        # on the top-level entry would also hit that no-op and swallow the entire
        # function body. Manual iteration dispatches each top-level statement (Return /
        # Assign / Expr / ...) to its visit_* handler, while nested ``def`` statements
        # in the body still hit the no-op and are correctly skipped.
        for stmt in func_node.body:
            analyzer.visit(stmt)

        return list(analyzer.warnings), analyzer

    # ========================= RLC005 check =========================

    def _model_satisfies(self, actual: str, required: str) -> bool:
        """Whether a preload held on class ``actual`` satisfies a "``required``.<rel>" requirement.

        **Identity must be relaxed under STI/JTI**: the table class a response DTO is
        anchored on is often the *base* class, while a dependency returns a concrete
        *subclass*. The names differ, but the subclass's relationship is the same
        mapper attribute declared on the base -- ``selectinload`` on the subclass
        satisfies the base requirement. Comparing names only would flag correctly
        written endpoints.

        **The relation is one-directional**: ``actual`` must be the same as or more
        specific than ``required``. The reverse (dependency returns the base, the DTO
        needs a subclass-only relationship) does not hold -- that is RLC012's territory.

        :param actual: model class name that holds the preload (dependency return type / tracked var)
        :param required: table class name the response_model is anchored on
        """
        if actual == required:
            return True
        if actual not in self.model_classes or required not in self.model_classes:
            return False
        return issubclass(self.model_classes[actual], self.model_classes[required])

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

        # Relationship names that appear in **any** ``load=`` of the endpoint body.
        # ``tracked_vars`` only covers query results assigned to a variable; a list
        # endpoint typically does ``return await X.list_for(session, ..., load=rel(X.y))``
        # -- returned directly, and its return type is a response DTO rather than a
        # model, so no tracked var exists. Without this set such correct endpoints
        # would be flagged.
        endpoint_body_loads = analyzer.all_loaded_rel_names
        for rel_name, model_name in required_rels.items():
            loaded_anywhere = rel_name in endpoint_body_loads
            # Check if loaded in dependencies
            if not loaded_anywhere:
                for param_name, loaded_set in dep_loads.items():
                    if param_name in param_models and self._model_satisfies(
                        param_models[param_name], model_name,
                    ):
                        if rel_name in loaded_set:
                            loaded_anywhere = True
                            break
            # Check if loaded in function body
            if not loaded_anywhere:
                for var in analyzer.tracked_vars.values():
                    if (
                        self._model_satisfies(var.model_name, model_name)
                        and rel_name in var.loaded_rels
                    ):
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

        TYPE_CHECKING forward reference handling: passes ``self.model_classes``
        (all mapped ORM classes) as ``localns`` to ``get_type_hints()`` so that
        string annotations like ``llm: 'LLM'`` (where ``LLM`` is imported only
        under ``if TYPE_CHECKING:``) resolve to actual classes instead of being
        silently dropped — silent drops cause RLC007/RLC013 false negatives.

        :returns: param_name -> model_class_name
        """
        param_models: dict[str, str] = {}

        # Provide model_classes as localns so TYPE_CHECKING forward references resolve.
        localns: dict[str, Any] = dict(self.model_classes)
        try:
            hints = typing.get_type_hints(func, include_extras=True, localns=localns)
        except Exception:
            # Even with model_classes in scope, get_type_hints may still fail on
            # nested third-party ForwardRefs. Fall back to per-parameter __annotations__:
            # try to eval string annotations against globals + model_classes.
            hints = {}
            annotations: dict[str, Any] = getattr(func, '__annotations__', {})
            for param_name, annotation in annotations.items():
                if isinstance(annotation, str):
                    resolved = self._eval_string_annotation(annotation, func)
                    if resolved is not None:
                        hints[param_name] = resolved
                    continue
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
            actual_type = unwrap_to_class(hint)
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

    def _eval_string_annotation(self, annotation: str, func: Any) -> Any:
        """
        Evaluate a string annotation that ``typing.get_type_hints`` couldn't resolve.

        Typical case: a TYPE_CHECKING-only import is referenced as a forward reference
        (e.g. ``llm: 'LLM'`` where ``LLM`` is imported only under ``if TYPE_CHECKING:``).
        Evaluates against ``func.__globals__`` merged with the known ORM model classes.

        Supported forms: ``'LLM'``, ``'LLM | None'``, ``'list[LLM]'`` etc.
        Returns ``None`` on failure (silent skip, matches the original fallback).
        """
        globalns: dict[str, Any] = getattr(func, '__globals__', {}) or {}
        localns: dict[str, Any] = dict(self.model_classes)
        try:
            return eval(annotation, globalns, localns)  # noqa: S307 — source-code literal, scoped namespace
        except Exception:
            return None

    def _extract_model_from_hint(self, hint: Any) -> str | None:
        """
        Extract a model class name from a type annotation.

        Delegates to ``unwrap_to_class()`` to strip ``Annotated``, ``X | None``
        etc. wrappers, then checks whether the unwrapped class is a known ORM
        model.
        """
        cls = unwrap_to_class(hint)
        if cls is not None and cls.__name__ in self.model_relationships:
            return cls.__name__
        return None

    def _get_response_model_relationships(
        self,
        response_model: type | None,
    ) -> dict[str, str]:
        """
        Get the relationships a response_model depends on.

        - Unions (``A | B``, discriminated unions) are traversed member by member and merged.
        - Container models (``ListResponse[X]``, named bucket / page models whose
          fields hold lists of DTOs) are drilled into through their field annotations.
        - For a DTO, the nearest table model in its MRO decides; a field counts when
          (1) its name is a relationship of that table model, or (2) the table model
          implements it as a property / method whose body reads a relationship
          (``_relations_read_by_field``).

        :returns: relationship_name -> model_class_name
        """
        if response_model is None:
            return {}

        # ---- Union (``A | B`` and discriminated unions): traverse every member and merge ----
        # Looking at args[0] only would miss the other members, whose fields -- and
        # therefore relationship needs -- may differ.
        origin = typing.get_origin(response_model)
        if origin is not None:
            args = typing.get_args(response_model)
            if args:
                merged: dict[str, str] = {}
                for arg in args:
                    if arg is type(None):
                        continue
                    merged.update(self._get_response_model_relationships(arg))
                return merged

        if not hasattr(response_model, 'model_fields'):
            return {}

        # ---- Container fields: ``items: list[ItemDTO]`` / ``all: ListResponse[X] | None`` ----
        # Containers have no relationship fields of their own (only ``count`` /
        # ``items``); without drilling in, every list endpoint would be invisible to
        # RLC001/005. Drill through **field annotations** rather than
        # typing.get_origin/get_args: the latter return nothing for Pydantic concrete
        # generics, and named containers carry no generic metadata at all.
        container_rels = self._relations_from_container_fields(response_model)
        if container_rels:
            return container_rels

        required: dict[str, str] = {}

        # Find the corresponding table model in response_model's MRO
        for base in response_model.__mro__:
            base_name = base.__name__
            if base_name not in self.model_relationships:
                continue
            rels = self.model_relationships[base_name]
            field_names = set(response_model.model_fields) | set(
                getattr(response_model, 'model_computed_fields', {})
            )
            for field_name in field_names:
                # Criterion 1: the field IS a relationship (the response exposes the related object)
                if field_name in rels:
                    required[field_name] = base_name
                    continue
                # Criterion 2: the field is a property / method on the table class
                # whose implementation reads a relationship -- the field name has
                # nothing to do with the relationship name, so criterion 1 misses it.
                for read_rel in self._relations_read_by_field(base_name, field_name, rels):
                    required[read_rel] = base_name
            break  # Use only the nearest table model

        return required

    def _relations_from_container_fields(self, response_model: type) -> dict[str, str]:
        """Relationship needs derived from the **element types** of container fields.

        Named container classes (polymorphic ``ListResponse`` subclasses, bucket
        models) carry no ``__pydantic_generic_metadata__``; their element types only
        appear in field annotations. Each field annotation is stripped of
        ``list[...]`` / ``X | None`` / ``Annotated[...]`` and recursed into.
        Only annotations are inspected; no code is executed.
        """
        merged: dict[str, str] = {}
        for field_info in response_model.model_fields.values():
            annotation = field_info.annotation
            if annotation is None:
                continue
            for inner in self._unwrap_annotation(annotation):
                if inner is response_model:
                    continue  # self-reference guard
                if hasattr(inner, 'model_fields'):
                    merged.update(self._get_response_model_relationships(inner))
        return merged

    @classmethod
    def _unwrap_annotation(cls, annotation: Any, depth: int = 0) -> list[Any]:
        """Strip ``list[...]`` / ``Union`` / ``Annotated[...]`` shells and return the candidate types.

        A discriminated union (``Annotated[A | B | C, Discriminator(...)]``) expands
        to all of its members -- every concrete branch of polymorphic items is checked.
        """
        if depth > 5:
            return []
        results: list[Any] = []
        origin = typing.get_origin(annotation)
        if origin is None:
            return [annotation]
        for arg in typing.get_args(annotation):
            if arg is type(None) or isinstance(arg, (str, bytes)):
                continue
            if isinstance(arg, type) or typing.get_origin(arg) is not None:
                results.extend(cls._unwrap_annotation(arg, depth + 1))
        return results or [annotation]

    _FIELD_IMPL_SCAN_MAX_DEPTH: ClassVar[int] = 3
    """Maximum delegation depth followed when scanning a field implementation.

    Real code rarely delegates more than one level (``X.to_json`` ->
    ``SomeMixin.to_json(self)``); 3 leaves headroom."""

    def _relations_read_by_field(
        self,
        model_name: str,
        field_name: str,
        rel_names: set[str],
    ) -> set[str]:
        """If ``field_name`` is a property / method on the table class, the relationships its body reads.

        A response field often has nothing in common with the relationship it
        depends on (``supports_x`` reads ``self.provider``, ``price`` reads
        ``self.llm``). Such fields still lazy load during serialization: with an
        ``_is_relation_loaded`` guard they silently degrade to ``None``, without a
        guard ``raise_on_sql`` fails the whole response. "Field name == relationship
        name" cannot see this class of dependency.

        :param model_name: table model class name
        :param field_name: response_model field name
        :param rel_names: all relationship names of that model (to recognize hits)
        :returns: relationship names read by the implementation; empty if the field is
            not a property / method
        """
        cls = self.model_classes.get(model_name)
        if cls is None:
            return set()
        impl = getattr(cls, field_name, None)
        impl = getattr(impl, 'fget', impl)  # @property -> underlying function
        if impl is None or not callable(impl):
            return set()  # plain column attribute / descriptor: no body to scan
        return self._scan_field_impl_for_relations(impl, cls, rel_names, set(), 0)

    def _scan_field_impl_for_relations(
        self,
        fn: Any,
        cls: type,
        rel_names: set[str],
        seen: set[Any],
        depth: int,
    ) -> set[str]:
        """Scan a function body for relationship hits; follow ``SomeMixin.method(self)`` delegation.

        Delegation must be followed: a class may contain a one-line delegation while
        the code that actually reads the relationship lives in a mixin.

        Functions without source (C extensions / dynamically generated) and overly
        deep delegation chains are **skipped rather than reported**: this runs inside
        a startup checker, which must never prevent the application from starting.
        """
        if fn is None or not callable(fn) or fn in seen or depth > self._FIELD_IMPL_SCAN_MAX_DEPTH:
            return set()
        seen.add(fn)
        try:
            tree = ast.parse(textwrap.dedent(python_inspect.getsource(fn)))
        except (OSError, SyntaxError, TypeError):
            return set()
        hits: set[str] = set()
        for node in ast.walk(tree):
            # ``self.<relationship>``
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == 'self' and node.attr in rel_names):
                hits.add(node.attr)
            # A literal inside a guard: ``self._is_relation_loaded('<relationship>')``
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in rel_names):
                hits.add(node.value)
            # Delegation: ``SomeMixin.method(self)`` / ``super().method()``
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                delegated = node.func.attr
                is_delegation = any(
                    isinstance(a, ast.Name) and a.id == 'self' for a in node.args
                ) or isinstance(node.func.value, ast.Call)
                if delegated and is_delegation:
                    for base in cls.__mro__[1:]:
                        target = base.__dict__.get(delegated)
                        target = getattr(target, 'fget', target)
                        if target is not None and callable(target):
                            hits |= self._scan_field_impl_for_relations(
                                target, cls, rel_names, seen, depth + 1,
                            )
                            break
        return hits

    # ========================= RLC012: STI column compatibility check =========================

    def _unwrap_generic_to_orm_class(self, hint: Any) -> type | None:
        """
        Recursively extract an ORM model class from a type annotation.

        Handles generic containers (e.g. ``ListResponse[ImageGenerator]``),
        ``Annotated``, ``X | None`` wrappers, and Pydantic v2 concretized generics.
        Differs from ``unwrap_to_class``: when direct unwrap fails, it recurses
        into the generic arguments until it finds a registered ORM class.

        :returns: ORM model class or None
        """
        # Try direct unwrap first (Annotated/Union/direct class)
        cls = unwrap_to_class(hint)
        if cls is not None and cls.__name__ in self.model_classes:
            return cls

        # Recurse into generic arguments (including Pydantic concretized generics)
        for arg in get_pydantic_generic_args(hint):
            if arg is type(None):
                continue
            result = self._unwrap_generic_to_orm_class(arg)
            if result is not None:
                return result

        return None

    @staticmethod
    def _extract_discriminator_field(hint: Any) -> str | None:
        """Extract the discriminator **field name** from ``Annotated[A | B, Field(discriminator='x')]``.

        Read from Pydantic's own metadata; no field name is hard-coded. This is the
        starting point for binding each union member to its STI subclass.
        """
        if typing.get_origin(hint) is not Annotated:
            return None
        for meta in typing.get_args(hint)[1:]:
            discriminator = getattr(meta, 'discriminator', None)
            if isinstance(discriminator, str):
                return discriminator
        return None

    @staticmethod
    def _resolve_discriminated_host(
        member: type,
        discriminator: str | None,
        concrete_descendants: list[tuple[str, set[str]]],
        descendant_classes: dict[str, type],
    ) -> tuple[str, set[str]] | None:
        """Bind one member of a discriminated union to the STI subclass it will actually serialize.

        Binding chain (using the discriminator field name obtained at runtime, never
        a hard-coded one)::

            member.model_fields[discriminator].annotation  ->  Literal['a']  ->  'a'
            for each subclass: getattr(cls, 'kind') / getattr(cls, 'KIND')
                               / polymorphic_identity  ==  'a' ?

        The upper-case form is tried because discriminator values are commonly
        declared on the ORM side as ``ClassVar`` constants (``KIND: ClassVar[str] = 'a'``)
        -- a generic casing convention, not a domain-specific one.

        :returns: ``(subclass name, subclass model_fields)``; ``None`` when no unique
            subclass can be determined (the caller fails loud)
        """
        if discriminator is None:
            return None

        field_info = member.model_fields.get(discriminator)
        if field_info is None:
            return None
        literal_args = typing.get_args(field_info.annotation)
        if len(literal_args) != 1 or not isinstance(literal_args[0], str):
            return None
        discriminator_value = literal_args[0]

        matched: list[tuple[str, set[str]]] = []
        for sub_name, sub_fields in concrete_descendants:
            sub_cls = descendant_classes.get(sub_name)
            if sub_cls is None:
                continue
            declared: list[Any] = [
                getattr(sub_cls, attr, None)
                for attr in (discriminator, discriminator.upper())
            ]
            try:
                declared.append(sa_inspect(sub_cls).polymorphic_identity)
            except Exception:
                pass
            if any(value == discriminator_value for value in declared):
                matched.append((sub_name, sub_fields))

        # Exactly one match binds; zero (nobody claims the value) and several (value
        # collision) are both "cannot prove" and are left to the caller to report.
        return matched[0] if len(matched) == 1 else None

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
        while a sibling STI subclass that may be returned lacks those columns on
        its mapper. FastAPI serialization of the heterogeneous result triggers a
        deferred column load -> MissingGreenlet.

        For a **discriminated union** response_model, each member is instead checked
        against the one subclass its discriminator value binds to
        (``_resolve_discriminated_host``); a member that cannot be bound is reported
        rather than passed.
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
        # The discriminator binding asks each class object which value it claims,
        # so keep the classes as well (indexed by name).
        descendant_classes: dict[str, type] = {}
        for sub_mapper in mapper.self_and_descendants:
            if sub_mapper is mapper:
                continue
            if sub_mapper.polymorphic_identity is None:
                continue  # Abstract intermediate class
            sub_cls = sub_mapper.class_
            sub_name = sub_cls.__name__
            sub_fields = set(sub_cls.model_fields.keys()) if hasattr(sub_cls, 'model_fields') else set()
            concrete_descendants.append((sub_name, sub_fields))
            descendant_classes[sub_name] = sub_cls

        if len(concrete_descendants) < 2:
            return  # Only one concrete subclass, no heterogeneous risk

        source_file = path
        line = 0
        try:
            source_file = python_inspect.getfile(endpoint)
            _, start_line = python_inspect.getsourcelines(endpoint)
            line = start_line
        except (OSError, TypeError):
            pass

        # 4. Discriminated union (``Annotated[A | B, Field(discriminator=...)]``):
        # a different criterion. A single-DTO response_model must fit **every**
        # subclass the query may return. With a discriminated union Pydantic selects
        # exactly **one** member by the discriminator, so the correct criterion is
        # "each member is compatible with the subclass it binds to". It must NOT be
        # weakened to "some subclass happens to have these fields": a member that
        # mistakenly declares another subclass's field would pass, yet Pydantic
        # serializes it against its own subclass. When no binding can be proven the
        # member is reported (a gate that passes on "cannot prove" is no gate).
        member_dtos = [
            m for m in self._unwrap_annotation(response_model)
            if isinstance(m, type) and hasattr(m, 'model_fields')
        ]
        if len(member_dtos) > 1:
            discriminator = self._extract_discriminator_field(response_model)
            for member in member_dtos:
                member_orm_fields = {
                    f for f in member.model_fields
                    if any(f in sub_fields for _, sub_fields in concrete_descendants)
                }
                if not member_orm_fields:
                    continue  # member has no ORM fields (pure computed / DTO-only)

                bound = self._resolve_discriminated_host(
                    member, discriminator, concrete_descendants, descendant_classes,
                )
                if bound is None:
                    warnings.append(RelationLoadWarning(
                        code='RLC012',
                        file=source_file,
                        line=line,
                        message=(
                            f"Discriminated union member '{member.__name__}' cannot be bound "
                            f"to its STI subclass (discriminator={discriminator!r}); the "
                            f"checker cannot prove its fields are compatible with the subclass "
                            f"it actually serializes. Not passed: expose the discriminator "
                            f"value on the subclass under the discriminator field name (or "
                            f"its upper-case form), or make polymorphic_identity equal to it"
                        ),
                    ))
                    continue

                bound_name, bound_fields = bound
                missing = sorted(member_orm_fields - bound_fields)
                if missing:
                    warnings.append(RelationLoadWarning(
                        code='RLC012',
                        file=source_file,
                        line=line,
                        message=(
                            f"Discriminated union member '{member.__name__}' declares fields "
                            f"{missing} that are not in the model_fields of the subclass it "
                            f"binds to, {bound_name} (discriminator={discriminator!r}). The "
                            f"endpoint returns {return_model_name} (STI base class); Pydantic "
                            f"selects this member to serialize {bound_name} instances, and "
                            f"reading these fields triggers a ValidationError / deferred "
                            f"column load -> MissingGreenlet. Another subclass happening to "
                            f"have these fields does not count"
                        ),
                    ))
            return

        # 5. Single-DTO response_model: extract DTO fields
        dto_cls = unwrap_generic_to_dto_class(response_model)
        if dto_cls is None:
            return

        dto_fields = set(dto_cls.model_fields.keys())

        # 6. Build the union of model_fields across all concrete subclasses
        all_sub_fields: set[str] = set()
        for _, fields in concrete_descendants:
            all_sub_fields |= fields

        # 7. For each DTO field, check whether it exists on every subclass's model_fields
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

    def _iter_endpoint_dependencies(self, endpoint: Any) -> Iterator[tuple[str, Any]]:
        """
        Yield ``(parameter name, dependency callable)`` for every
        ``Annotated[..., Depends(dep)]`` parameter of an endpoint.

        Yields nothing when FastAPI is not installed or the endpoint's hints cannot
        be resolved.
        """
        if not _HAS_FASTAPI or _FastAPIDependsClass is None:
            return
        try:
            hints = typing.get_type_hints(endpoint, include_extras=True)
        except Exception:
            return

        for param_name, hint in hints.items():
            if typing.get_origin(hint) is not Annotated:
                continue
            for metadata in typing.get_args(hint)[1:]:
                if isinstance(metadata, _FastAPIDependsClass):
                    dep_func = metadata.dependency
                    if dep_func is not None:
                        yield param_name, dep_func
                    break

    def _analyze_dependencies(self, endpoint: Any) -> dict[str, set[str]]:
        """
        Analyze load= usage in endpoint dependency functions.

        :returns: param_name -> set of loaded relationship names
        """
        return {
            param_name: self._extract_loads_from_function(dep_func)
            for param_name, dep_func in self._iter_endpoint_dependencies(endpoint)
        }

    def _get_committing_dep_params(self, endpoint: Any) -> set[str]:
        """
        Endpoint parameters whose dependency commits the session (the basis of RLC014).

        A dependency counts as committing when some path of its body calls one of
        :data:`dependency_commit_methods` passing its session (see
        ``_dep_function_commits``). Conservative: any path that may commit is enough.

        Dependencies returned by factories (``Depends(require_x(...))``) are unwrapped
        through ``__wrapped__`` / ``functools.partial``; a dependency that cannot be
        analyzed is treated as non-committing (silence over false positives).

        :param endpoint: FastAPI endpoint function
        :returns: names of the endpoint parameters injected by committing dependencies
        """
        return {
            param_name
            for param_name, dep_func in self._iter_endpoint_dependencies(endpoint)
            if self._dep_function_commits(dep_func)
        }

    def _dep_function_commits(self, dep_func: Any) -> bool:
        """
        Whether a dependency function (``Depends(...).dependency``) **explicitly**
        calls a committing method on some path.

        - unwrap ``__wrapped__`` / ``functools.partial`` to the original function
        - parse its source
        - run ``_ast_calls_commit_method_with_session`` for every ``AsyncSession``
          parameter, restricted to :data:`dependency_commit_methods` intersected with
          the auto-discovered ``commit_methods`` (the double filter guards against
          name coincidences)

        Unparseable source / no function node / no session parameter / no commit
        call -> ``False``: "cannot decide" is treated as "does not commit" to avoid
        RLC014 false positives.
        """
        actual = dep_func
        if hasattr(actual, '__wrapped__'):
            actual = actual.__wrapped__
        if hasattr(actual, 'func'):
            actual = actual.func

        try:
            source = python_inspect.getsource(actual)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return False

        func_node = self._find_function_node(tree, getattr(actual, '__name__', ''))
        if func_node is None:
            return False

        session_params = _session_param_names(actual)
        if not session_params:
            return False

        # Only "definite commit" methods count (commit-on-miss readers would drag
        # every singleton-reading dependency into the false-positive set).
        strict_commit = frozenset(dependency_commit_methods) & self.commit_methods

        for session_param in session_params:
            if _ast_calls_commit_method_with_session(
                tree, session_param, strict_commit,
            ):
                return True
        return False

    def _extract_loads_from_function(self, func: Any) -> set[str]:
        """
        Relationship names referenced by ``load=`` in a dependency callable --
        analyzed **layer by layer**, never skipping a layer.

        **``load=<bare name>`` needs runtime resolution**: when the preload list is
        passed in as an argument, the relationship names do not appear in the
        analyzed function's source at all. Three real shapes:

        - ``Depends(require_x(X, load=rel(X.y)))`` -- the returned closure only says
          ``load=load``; the value lives in a **closure cell**;
        - ``Depends(partial(dep, load=rel(X.y)))`` -- ``load`` is just a parameter of
          ``dep``; the value lives in **``partial.keywords``**;
        - ``load=MODULE_CONST`` where the constant is defined in **another** module.

        **Why layer by layer**: ``functools.wraps`` copies metadata only; it does not
        make the wrapper behave transparently. A wrapper may itself call
        ``inner(load=<its own value>)``, in which case the wrapper decides the preload
        and jumping straight to ``__wrapped__`` would skip the real behavior. So each
        layer first analyzes its **own** source; if the layer determines ``load=`` it
        wins (even when it resolves to an empty set, which means it explicitly passes
        no preload); only a layer that does not decide ``load=`` is drilled through.

        ``functools.partial`` has no source; its ``keywords`` are collected (outer
        partial wins, matching call semantics: ``partial(partial(f, load=A), load=B)``
        calls ``f(load=B)``) before drilling in.

        **Honest boundary**: only ``partial.keywords`` are collected, not positional
        ``partial.args`` (binding them to parameter names needs full signature
        binding, which the checker does not do). This is safe **only because default
        values are read for keyword-only parameters exclusively**
        (``_parameter_default`` reads ``__kwdefaults__``): a keyword-only parameter
        can never be filled positionally, so the invisible ``partial.args`` cannot
        override a default the checker credits. Supporting positional defaults
        (``__defaults__``) would require implementing ``partial.args`` binding at the
        same time; adding only one half turns the gap into a false negative.

        Anything that cannot be resolved yields an empty set, i.e. the "report" direction.
        """
        bound_kwargs: dict[str, Any] = {}
        seen: set[int] = set()
        # Whether some layer already passed ``load=<expr>`` explicitly. From then on a
        # deeper ``partial``'s bound ``load`` is overridden by that keyword at call time
        # and must not enter ``bound_kwargs``.
        load_killed_by_explicit_kwarg = False
        # Explicit Any: each wrapper layer has a different type (partial / function /
        # arbitrary callable wrapper); narrowing to one would reject the next layer's
        # ``__wrapped__`` / ``func`` access.
        current: Any = func

        for _ in range(_MAX_WRAPPER_DEPTH):
            if id(current) in seen:
                break
            seen.add(id(current))

            if isinstance(current, functools.partial):
                # partial has no source: collect bound values (first seen = outer wins)
                for key, value in current.keywords.items():
                    if key == 'load' and load_killed_by_explicit_kwarg:
                        # An outer layer already passes load= explicitly; this bound
                        # value never reaches the inner function.
                        continue
                    bound_kwargs.setdefault(key, value)
                current = typing.cast(Any, current.func)
                continue

            owns_load, names, forwarded_load = self._extract_loads_from_layer(
                current, bound_kwargs,
                # No layer has passed load= explicitly so far => this layer's parameter
                # default is the effective value. Once someone passed it (even None),
                # the default is not used.
                allow_defaults=not load_killed_by_explicit_kwarg,
            )
            if owns_load:
                # This layer decides load=; the inner layers' load= is not the effective one
                return names
            if forwarded_load:
                # This layer forwards load= explicitly => deeper partial-bound loads are dead
                load_killed_by_explicit_kwarg = True

            if hasattr(current, '__wrapped__'):
                current = current.__wrapped__
                continue
            if hasattr(current, 'func'):
                # A non-partial callable wrapper exposing ``.func``
                current = current.func
                continue
            break

        return set()

    def _extract_loads_from_layer(
            self,
            func: Any,
            bound_kwargs: dict[str, Any],
            allow_defaults: bool,
    ) -> tuple[bool, set[str], bool]:
        """Analyze the source of **this layer only**; never drill in.

        **The source must be taken via ``func.__code__``, not ``getsource(func)``**:
        ``inspect.getsource`` unwraps ``__wrapped__`` first, so passing the function
        object would analyze the **innermost** layer and skip an outer wrapper's
        explicit ``load=None`` (a false negative). A code object has no
        ``__wrapped__`` and ``inspect.findsource`` locates it by ``co_filename`` +
        ``co_firstlineno``. ``ValueError`` is caught as well (``unwrap`` raises
        ``wrapper loop`` on a ``__wrapped__`` cycle before the caller's cycle check).
        Side effect: ``load=`` written in decorator arguments (``@deco(load=X)``) is
        not attributed to this layer, which is correct for "what this layer's body does".

        :param bound_kwargs: arguments already bound by outer ``functools.partial`` wrappers.
        :param allow_defaults: whether this layer's keyword-only parameter defaults are
            the effective values (no outer layer passed ``load=`` explicitly).
        :returns: ``(this layer gives a definite answer, relationship names, this layer
            explicitly forwards load=)``:

            ===============================  ========================  =============================
            layer writes                     returns                   caller action
            ===============================  ========================  =============================
            ``load=rel(X.y)`` / resolvable   ``(True, {'y'}, False)``  stop, use it
            ``load=None`` / ``load=[]``      ``(True, set(), False)``  stop, explicitly no preload
            ``load=kwargs.get('load')``      ``(False, set(), True)``  drill in; deeper partial ``load`` is dead
            no ``load=`` at all              ``(False, set(), False)`` drill in; deeper partial ``load`` still valid
            ===============================  ========================  =============================

            An explicit keyword always overrides a deeper ``partial``'s binding
            (``partial(f, load=A)(load=B)`` calls ``f(load=B)``, whatever ``B`` is), hence
            the third item. A layer whose source cannot be read also returns
            ``(False, set(), False)``.
        """
        if not hasattr(func, '__code__'):
            # No code object (callable instance etc.): nothing to analyze here, drill in
            return False, set(), False
        code: types.CodeType = func.__code__

        try:
            source = python_inspect.getsource(code)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, ValueError, SyntaxError, IndentationError):
            return False, set(), False

        if not tree.body or not isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return False, set(), False
        func_node = tree.body[0]

        # Names **bound locally** in this layer (decided by CPython's symbol table).
        # Once ``load`` is rebound here (``load = []``, ``def load()``, ``import ... as
        # load``, ``except ... as load``, walrus, ...), an outer partial's value is gone:
        # neither resolve it through ``bound_kwargs`` nor treat it as forwarding.
        # ``None`` = the symbol table could not be built (unknown); treat as "may have
        # been rebound" => this layer decides with an empty set => report.
        rebound = _locally_bound_names(source, code.co_name)
        if rebound is not None:
            # Add names mutated **in place** (``load.clear()`` rebinds nothing, but the
            # value is no longer the caller's). ``None`` stays ``None``.
            rebound = rebound | _locally_mutated_names(func_node)

        owns_load = False
        # Every ``load=`` occurrence forms its own group and the result is the
        # **intersection** of the groups, not the union. Correctness needs "every
        # reachable path preloads X", not "some path does":
        #
        #     if disabled:
        #         return await inner(load=None)   # this path preloads nothing
        #     return await inner(load=load)
        #
        # The intersection is conservative: two branches loading different
        # relationships yield an empty set (possible false positive, never a false
        # negative). The checker does no real control-flow analysis, so sequential
        # ``load=`` calls in one layer are intersected too.
        occurrence_loads: list[set[str]] = []
        forwarded_load = False
        # Not ``ast.walk``: that would descend into nested functions / lambdas that
        # this layer defines but may never call, crediting dead code. Whether a
        # nested function runs is statically undecidable, so it is never entered.
        #
        # A return path that omits ``load=`` entirely must contribute an empty set
        # too (otherwise "for all paths" silently shrinks to "for all occurrences"):
        #
        #     if omit:
        #         return await func()            # inner receives its default None
        #     return await func(load=load)
        #
        # Only **delegation shapes** count as such paths: the callee is a bare name
        # this layer received (closure free variable or parameter, e.g. ``func`` in an
        # ``@wraps(func)`` wrapper), or a query method call such as ``Model.get(...)``
        # (the registry discovered by the checker, not a second list). A call with
        # ``**kwargs`` cannot be proven to omit ``load`` and does not count.
        delegate_names = frozenset(code.co_freevars) | _parameter_names(code)
        query_methods = self.model_returning_methods | self.sync_model_returning_methods

        for node in _iter_layer_body_nodes(func_node):
            if isinstance(node, ast.keyword) and node.arg == 'load':
                names = _extract_load_value(node.value)
                if (
                    not names
                    and isinstance(node.value, ast.Name)
                    and rebound is not None          # unknown => never trust runtime values
                    and node.value.id not in rebound
                ):
                    names = _resolve_load_name_at_runtime(
                        func, node.value.id, bound_kwargs,
                        allow_defaults=allow_defaults,
                    )
                # "This layer decides" = relationship names were resolved, OR the value
                # is not a **provable direct forwarding** of the received ``load``. Any
                # computed value (e.g. ``choose_load(kwargs.get('load'))``) is judged
                # as deciding, with an empty set => report.
                forwarding = (
                    rebound is not None
                    and _is_direct_load_forwarding(node.value, code, rebound)
                )
                occurrence_loads.append(names)
                if names or not forwarding:
                    owns_load = True
                else:
                    # An explicit forwarding keyword reaches the inner layer and
                    # overrides a deeper partial's bound ``load``.
                    forwarded_load = True

        # Reaching-definitions analysis: a return path whose value comes from a query /
        # delegation call without ``load=`` (including "assign, then return the
        # variable") contributes an empty set. See _has_unsafe_return_path.
        if _has_unsafe_return_path(func_node, delegate_names, query_methods):
            occurrence_loads.append(set())

        loaded: set[str] = set(occurrence_loads[0]) if occurrence_loads else set()
        for occurrence in occurrence_loads[1:]:
            loaded &= occurrence

        return owns_load, loaded, forwarded_load

    # ========================= AST utilities =========================

    @staticmethod
    def _parse_function_source(func: Any) -> tuple[str, ast.Module | None, int]:
        """
        Parse function source code.

        Handles @wraps decorators: unwraps the __wrapped__ chain via inspect.unwrap()
        to get the original function's source (not the wrapper's).

        :returns: (source_file, ast_tree_or_None, line_offset)
        """
        # Unwrap @wraps decorator chain to get original function
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
        """Find the function node with the given name in the AST."""
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
    """Extract a relationship name from a single AST node, supporting both Model.rel and rel(Model.rel)."""
    if isinstance(node, ast.Attribute):
        # load=Model.rel_name
        return node.attr
    elif isinstance(node, ast.Call) and node.args:
        # load=rel(Model.rel_name) -- rel() is a type-conversion wrapper
        return _extract_load_single(node.args[0])
    return None


def _extract_load_value(node: ast.expr, module_tree: ast.Module | None = None) -> set[str]:
    """
    Extract relationship names from a load= AST value node.

    Supports:
    - load=Model.rel -> {'rel'}
    - load=rel(Model.rel) -> {'rel'}
    - load=[Model.r1, rel(Model.r2)] -> {'r1', 'r2'}
    - **load=MODULE_LEVEL_CONST** (requires ``module_tree``) -> the elements of that
      module-level list constant

    The last form is common when a preload list is long enough to be extracted into
    a module-level constant; leaving it unresolved would flag correct code.
    """
    result: set[str] = set()

    if isinstance(node, ast.Name) and module_tree is not None:
        # load=SOME_CONST -- find its top-level assignment and resolve the list elements
        for stmt in module_tree.body:
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if not any(isinstance(t, ast.Name) and t.id == node.id for t in targets):
                continue
            if stmt.value is not None:
                result |= _extract_load_value(stmt.value, module_tree)
        return result

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


def _relation_names_from_load_value(value: Any) -> set[str]:
    """Translate the **runtime value** of ``load=`` into relationship names.

    ``load`` accepts ``QueryableAttribute | list[QueryableAttribute] | None`` and
    ``rel()`` is an identity conversion, so the runtime value is an
    ``InstrumentedAttribute`` whose ``.key`` is the relationship name.

    :returns: recognized relationship names; **an empty set always means "nothing
        resolved"** (``load=None`` and unrecognized values both land here), which
        keeps callers on the "report" side -- this function never turns a real
        missing preload into silence.
    """
    if isinstance(value, QueryableAttribute):
        return {value.key}
    if isinstance(value, (list, tuple)):
        return {
            item.key for item in value
            if isinstance(item, QueryableAttribute)
        }
    return set()


def _iter_layer_body_nodes(
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Walk every node of **this layer's executed body**, never entering nested
    function / lambda scopes.

    ``ast.walk`` would descend into nested functions that the layer defines but
    may never call, crediting a ``load=`` in dead code. Whether a nested function
    runs is statically undecidable, so it is never entered (skip = no credit =
    report; the safe direction).

    Starting from ``func_node.body`` rather than ``func_node`` also excludes
    decorator arguments and parameter defaults, which are evaluated at definition
    time and are not part of the executed body.
    """
    # The skip check happens when a node is popped (not when children are pushed):
    # a nested ``def`` often sits directly in ``func_node.body`` and would otherwise
    # bypass a children-only filter.
    stack: list[ast.AST] = list(func_node.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _locally_bound_names(source: str, func_name: str) -> frozenset[str] | None:
    """Names **bound locally** in this layer, decided by CPython's symbol table (``symtable``).

    An outer ``partial``'s ``load`` is only the value this layer uses if the layer
    does not rebind it (``load = []``, ``def load()``, ``import x as load``, ...
    followed by ``func(load=load)`` passes the new value).

    Why not scan the AST: Python has too many binding forms (assignment, ``def``,
    ``class``, ``import``, ``except as``, ``with as``, ``for``, ``match`` captures,
    walrus, ...); a hand-written list would miss some, and every miss is a false
    negative. Why not bytecode: 3.13+ super-instructions fuse operations and their
    name tuples mix stored and loaded names. ``symtable`` is the compiler's own
    answer to "is this name bound in this scope".

    Note that "bound" is ``is_assigned() or is_imported()``: import bindings report
    ``is_assigned() == False``. Read-only parameters are not bindings produced by
    the layer's code.

    :param source: this layer's own (dedented) source, from ``getsource(code)``
    :param func_name: this layer's function name (``code.co_name``)
    :returns: a ``frozenset`` of names **confirmed** to be bound in this scope (an
        empty set means confirmed "no bindings"); ``None`` means the symbol table
        could not be built or located (unknown). ``None`` and the empty set have
        **opposite** meanings: callers must test ``is None`` explicitly and, on
        ``None``, neither trust closure / global values nor accept the
        direct-forwarding exemption.
    """
    try:
        top = symtable.symtable(source, '<rlc>', 'exec')
    except (SyntaxError, ValueError):
        # ``getsource(code)`` returns a fragment detached from its enclosing scope; a
        # ``nonlocal load`` inside it has no binding there and symtable raises. An
        # empty set here would claim "definitely not rebound", although
        # ``nonlocal load; load = []`` has in fact emptied the cell. Unknown => None.
        return None

    # The fragment contains exactly one function block, so normally there is exactly
    # one function-type child scope. Pick "the only one" instead of matching by name:
    # symtable names lambda scopes ``lambda`` while ``co_name`` is ``<lambda>``.
    functions = [c for c in top.get_children() if c.get_type() == 'function']
    scope = functions[0] if len(functions) == 1 else next(
        (c for c in functions if c.get_name() == func_name), None,
    )
    if scope is None:
        return None

    # Only this layer's symbol table; nested functions are their own child scopes
    # (while the name a nested ``def`` binds in this layer does appear here).
    return frozenset(
        symbol.get_name() for symbol in scope.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    )


def _locally_mutated_names(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Names **mutated in place** (not rebound) in this layer.

    ``symtable`` cannot answer this::

        async def wrapped(*, load=None):
            load.clear()                  # value emptied, name not rebound
            return await inner(load=load)

    ``_locally_bound_names`` correctly reports "not rebound", so ``load=load``
    would look like transparent forwarding while the checker (which runs before
    the request) still sees the un-emptied list. "What the name points to" and
    "what the pointed-to object is" are orthogonal; the union of both is needed.

    The criterion is **structural**, not a list of mutating method names:

    ==================================  ============================================
    form                                AST
    ==================================  ============================================
    ``x.anything(...)``                 ``Call(func=Attribute(value=Name(x)))``
    ``x[...] = ...`` / ``x.attr = ...`` ``Subscript`` / ``Attribute`` in ``Store``
    ``del x[...]`` / ``del x.attr``     same, in ``Del``
    ==================================  ============================================

    Deliberately conservative: any attribute call counts (read-only methods too),
    making the layer "decide" => report.

    **Alias propagation**: in ``alias = load; alias.clear()`` both names refer to the
    same object, so mutations of ``alias`` are propagated back to ``load`` through
    ``a = b`` / ``a, b = c, d`` / ``a: T = b`` edges (bounded fixed point; an alias
    may have several possible sources across branches).

    This is **not** a proof that the value is unchanged. Passing the object to
    another function (``mutate(load)``), boxing it in a container, storing it on
    an attribute, or aliasing it through a function return are not covered.
    Nested functions / lambdas are not entered (same rule as ``_iter_layer_body_nodes``).
    """
    direct: set[str] = set()
    # Alias edges ``alias = source`` (both bare names). The value is a set: the same
    # alias may have several possible sources on different branches, and AST walk
    # order is not runtime order, so every source must be kept.
    aliases: dict[str, set[str]] = {}

    for node in _iter_layer_body_nodes(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if isinstance(receiver, ast.Name):
                direct.add(receiver.id)
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            if isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Name):
                direct.add(node.value.id)
        elif isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Name):
                # ``a = b`` / ``a = c = b``
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.setdefault(target.id, set()).add(node.value.id)
            elif isinstance(node.value, ast.Tuple) and len(node.targets) == 1:
                # ``a, b = load, other`` -- paired by position (same arity only)
                target = node.targets[0]
                if isinstance(target, ast.Tuple) and len(target.elts) == len(node.value.elts):
                    for left, right in zip(target.elts, node.value.elts):
                        if isinstance(left, ast.Name) and isinstance(right, ast.Name):
                            aliases.setdefault(left.id, set()).add(right.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and isinstance(node.value, ast.Name):
                aliases.setdefault(node.target.id, set()).add(node.value.id)

    # Propagate backwards along alias edges: mutating an alias mutates every
    # possible source. Bounded fixed point: each round adds at least one name.
    mutated = set(direct)
    bound = len(aliases) + sum(len(sources) for sources in aliases.values()) + 1
    for _ in range(bound):
        grew = False
        for alias, sources in aliases.items():
            if alias not in mutated:
                continue
            for source in sources:
                if source not in mutated:
                    mutated.add(source)
                    grew = True
        if not grew:
            break
    return frozenset(mutated)


def _sub_statement_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """Every sub-statement body directly held by a statement, without enumerating statement types.

    Hard-coding ``(If, With, For, While, Try)`` would silently miss ``match``
    (``Match.cases[*].body``) and ``except*`` (``TryStar``), and every future
    compound statement. Discovery walks the node fields and takes:

    1. fields that are ``list[stmt]`` (``body`` / ``orelse`` / ``finalbody``);
    2. fields that are ``list[AST]`` whose elements themselves carry ``list[stmt]``
       fields (``ExceptHandler``, ``match_case``).

    Nested functions / classes are skipped by the caller.
    """
    bodies: list[list[ast.stmt]] = []
    for _, value in ast.iter_fields(stmt):
        if not isinstance(value, list) or not value:
            continue
        if all(isinstance(item, ast.stmt) for item in value):
            bodies.append(typing.cast('list[ast.stmt]', value))
            continue
        # ExceptHandler / match_case: intermediate nodes that carry their own body
        for item in value:
            if not isinstance(item, ast.AST):
                continue
            for _, sub in ast.iter_fields(item):
                if isinstance(sub, list) and sub and all(isinstance(s, ast.stmt) for s in sub):
                    bodies.append(typing.cast('list[ast.stmt]', sub))
    return bodies


def _has_unsafe_return_path(
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        delegate_names: frozenset[str],
        query_methods: frozenset[str],
) -> bool:
    """Whether this layer has a **return path** whose value comes from a query /
    delegation call **without** ``load=``.

    The syntactic shape of ``return <call>`` is not enough::

        result = await Model.get(session, cond)     # no load
        return result                               # same as ``return await Model.get(...)``

    Both are semantically identical and a routine refactoring swaps them; a static
    gate's verdict must not depend on it.

    The criterion is **reaching definitions**, not "any definition"::

        user = await User.from_token(session, token)        # definition 1: no load
        user = await User.get(..., load=User.scope)         # definition 2: load, kills 1
        return user                                          # only 2 reaches here

    Sequential assignments kill earlier ones; branch joins take the union. Nested
    functions / lambdas / classes are not entered.

    :returns: ``True`` if such a return path exists => the caller contributes an
        empty set to the intersection => the layer is not credited.
    """
    def _target_names(target: ast.expr) -> list[str]:
        """Every bare name in a target structure (nested and ``*starred`` included)."""
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Starred):
            return _target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            names: list[str] = []
            for element in target.elts:
                names.extend(_target_names(element))
            return names
        return []          # Attribute / Subscript targets bind no bare name

    def _value_is_unsafe(value: ast.expr | None, unsafe: set[str]) -> bool:
        """Whether this value may be an un-preloaded query result.

        Two sources: (1) the value itself is a query / delegation call without
        ``load=``; (2) the value is a name already marked unsafe (``b = a``).
        Value-selecting expressions (``x if c else y``, ``a or b``, tuples) propagate
        their operands' unsafety. Other expressions are deliberately **not** searched
        deeply: a ``Call``'s result is decided by the callee (``load=`` present means
        preloaded) and ``Attribute`` / ``Subscript`` are derived values (an id is not
        the un-preloaded model), so "contains an unsafe name somewhere" does not make
        the result unsafe.
        """
        if value is None:
            return False
        if isinstance(value, ast.Name):
            return value.id in unsafe
        if isinstance(value, ast.NamedExpr):
            return _value_is_unsafe(value.value, unsafe)
        if isinstance(value, ast.Starred):
            return _value_is_unsafe(value.value, unsafe)
        if isinstance(value, ast.IfExp):
            return _value_is_unsafe(value.body, unsafe) or _value_is_unsafe(value.orelse, unsafe)
        if isinstance(value, ast.BoolOp):
            return any(_value_is_unsafe(operand, unsafe) for operand in value.values)
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(_value_is_unsafe(element, unsafe) for element in value.elts)
        return _is_unsafe_call(value)

    def _rhs_unsafe_anywhere(value: ast.expr | None, unsafe: set[str]) -> bool:
        """RHS is unsafe if **any** element is (conservative, when elements cannot be paired)."""
        if value is None:
            return False
        if isinstance(value, (ast.Tuple, ast.List)):
            return any(_value_is_unsafe(element, unsafe) for element in value.elts)
        return _value_is_unsafe(value, unsafe)

    def _assign_effects(
            target: ast.expr,
            value: ast.expr | None,
            unsafe: set[str],
    ) -> list[tuple[str, bool]]:
        """Pair a target structure with the RHS following Python unpacking semantics
        => ``[(name, name is unsafe)]``.

        Elements are paired one by one (``result, marker = await Model.get(), 'm'``
        must not mark ``marker``). When pairing is impossible (``*starred``, arity
        mismatch, non-literal RHS) every name is marked if **any** RHS element is
        unsafe -- never "evaluate the whole RHS as one call", which would compute
        "safe" for a tuple and kill an earlier unsafe definition.
        """
        if isinstance(target, ast.Name):
            return [(target.id, _value_is_unsafe(value, unsafe))]
        if isinstance(target, (ast.Tuple, ast.List)):
            has_starred = any(isinstance(element, ast.Starred) for element in target.elts)
            if (
                isinstance(value, (ast.Tuple, ast.List))
                and not has_starred
                and len(target.elts) == len(value.elts)
            ):
                paired: list[tuple[str, bool]] = []
                for sub_target, sub_value in zip(target.elts, value.elts):
                    paired.extend(_assign_effects(sub_target, sub_value, unsafe))
                return paired
            unsafe_any = _rhs_unsafe_anywhere(value, unsafe)
            return [(name, unsafe_any) for name in _target_names(target)]
        # ``obj.attr = ...`` / ``d[k] = ...`` bind no bare name
        return []

    def _pattern_capture_names(pattern: ast.AST) -> list[str]:
        """Every capture name in a ``match`` case pattern.

        Every **string field** of a pattern node is a bound name (``MatchAs.name``,
        ``MatchStar.name``, ``MatchMapping.rest``); other fields are sub-patterns or
        expressions. This covers nested patterns and future binding forms (collecting
        extra names only marks more => report, the safe side).
        """
        names: list[str] = []
        for node in ast.walk(pattern):
            for _, value in ast.iter_fields(node):
                if isinstance(value, str):
                    names.append(value)
        return names

    def _expr_is_unsafe(node: ast.expr | None, unsafe: set[str]) -> bool:
        """Whether an expression contains, at any depth, an unsafe query call **or a
        reference to a name already marked unsafe**.

        ``for`` / ``with as`` / ``match`` subject bindings are indirect: the bound
        value may be wrapped in another call (``with nullcontext(await Model.get()) as
        row:``) or reference an earlier unsafe name. ``lambda`` bodies are not entered.
        """
        if node is None:
            return False
        stack: list[ast.AST] = [node]
        while stack:
            current = stack.pop()
            if isinstance(current, ast.Lambda):
                continue
            if isinstance(current, ast.Name) and current.id in unsafe:
                return True
            if isinstance(current, (ast.Call, ast.Await, ast.NamedExpr)):
                if _is_unsafe_call(typing.cast(ast.expr, current)):
                    return True
            stack.extend(ast.iter_child_nodes(current))
        return False

    def _own_expression_nodes(stmt: ast.stmt) -> list[ast.AST]:
        """The statement's **own** expression nodes (no sub-statement bodies, no
        ``lambda`` bodies) -- used to find definitions hidden in expressions (walrus)."""
        found: list[ast.AST] = []
        stack: list[ast.AST] = []
        for _, value in ast.iter_fields(stmt):
            items = value if isinstance(value, list) else [value]
            stack.extend(item for item in items if isinstance(item, ast.expr))
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Lambda):
                continue          # a lambda body is another scope
            found.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return found

    def _is_unsafe_call(node: ast.expr) -> bool:
        # A walrus is an expression itself: ``return (x := await Model.get())``
        if isinstance(node, ast.NamedExpr):
            node = node.value
        call = node.value if isinstance(node, ast.Await) else node
        if not isinstance(call, ast.Call):
            return False
        callee = call.func
        is_delegate = isinstance(callee, ast.Name) and callee.id in delegate_names
        is_query = isinstance(callee, ast.Attribute) and callee.attr in query_methods
        if not (is_delegate or is_query):
            return False
        has_load_kw = any(kw.arg == 'load' for kw in call.keywords)
        # ``**kwargs`` may carry load => its absence cannot be proven => not unsafe
        has_splat = any(kw.arg is None for kw in call.keywords)
        return not has_load_kw and not has_splat

    def _walk(stmts: list[ast.stmt], unsafe: set[str]) -> tuple[bool, set[str]]:
        hit = False
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            # A walrus also defines a name, hidden in an expression. Only **add**,
            # never discard: walruses often sit in short-circuit conditions and may not
            # execute, so a safe walrus cannot kill an earlier unsafe definition.
            for expr_node in _own_expression_nodes(stmt):
                if not isinstance(expr_node, ast.NamedExpr):
                    continue
                if isinstance(expr_node.target, ast.Name) and _value_is_unsafe(expr_node.value, unsafe):
                    unsafe.add(expr_node.target.id)
            if isinstance(stmt, ast.Return):
                # Must go through _value_is_unsafe (single source of truth), so that
                # inlining / extracting a temporary never changes the verdict.
                if _value_is_unsafe(stmt.value, unsafe):
                    hit = True
                continue
            if isinstance(stmt, ast.Assign):
                # Targets may be unpacking structures: pair element by element.
                for target in stmt.targets:
                    for name, is_unsafe in _assign_effects(target, stmt.value, unsafe):
                        if is_unsafe:
                            unsafe.add(name)
                        else:
                            unsafe.discard(name)           # overwrite = kill the earlier definition
                continue
            if isinstance(stmt, ast.AugAssign):
                # ``result += await Model.get()`` rewrites ``result`` too. Only add,
                # never discard: ``__iadd__`` may legally return the RHS, self or a
                # third object.
                if _expr_is_unsafe(stmt.value, unsafe):
                    for name in _target_names(stmt.target):
                        unsafe.add(name)
                continue
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value is not None:
                    if _value_is_unsafe(stmt.value, unsafe):
                        unsafe.add(stmt.target.id)
                    else:
                        unsafe.discard(stmt.target.id)
                continue
            if isinstance(stmt, ast.If):
                hit_body, state_body = _walk(stmt.body, set(unsafe))
                hit_else, state_else = _walk(stmt.orelse, set(unsafe))
                hit = hit or hit_body or hit_else
                unsafe = state_body | state_else            # join point: union
                continue
            # ``match`` capture patterns are bindings: a successful match binds the
            # subject (or a structural part of it). Only add: a failed guard / unmatched
            # case binds nothing, so a safe subject cannot kill earlier definitions.
            if isinstance(stmt, ast.Match):
                subject_unsafe = _expr_is_unsafe(stmt.subject, unsafe) or (
                    isinstance(stmt.subject, ast.Name) and stmt.subject.id in unsafe
                )
                if subject_unsafe:
                    for case in stmt.cases:
                        for name in _pattern_capture_names(case.pattern):
                            unsafe.add(name)

            # ``for`` / ``with as`` targets are definitions too (not ``Assign``); the
            # value comes from an iterable / context manager and is checked deeply.
            # No ``continue``: the sub-statement bodies are still walked below.
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                iterable_unsafe = _expr_is_unsafe(stmt.iter, unsafe)
                for name in _target_names(stmt.target):
                    if iterable_unsafe:
                        unsafe.add(name)
                    else:
                        unsafe.discard(name)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if item.optional_vars is None:
                        continue
                    context_unsafe = _expr_is_unsafe(item.context_expr, unsafe)
                    for name in _target_names(item.optional_vars):
                        if context_unsafe:
                            unsafe.add(name)
                        else:
                            unsafe.discard(name)

            # Every other compound statement goes through the generic sub-body
            # discovery (``match``, ``try`` / ``except*``, loops, ...), never a
            # hand-written list of statement types.
            for body in _sub_statement_bodies(stmt):
                hit_sub, state_sub = _walk(body, set(unsafe))
                hit = hit or hit_sub
                unsafe = unsafe | state_sub
        return hit, unsafe

    found, _ = _walk(func_node.body, set())
    return found


def _parameter_names(code: types.CodeType) -> frozenset[str]:
    """The layer's **parameter** names (``*args`` / ``**kwargs`` included), **no locals**.

    ``code.co_varnames`` as a whole also contains local variables, which would
    misclassify a purely local ``load`` as "forwarding the caller's value".
    CPython's ``co_varnames`` prefix layout is fixed:
    ``[positional-or-keyword co_argcount] [keyword-only co_kwonlyargcount] [*args?] [**kwargs?] [locals...]``.

    Equivalent to ``inspect.signature(func, follow_wrapped=False).parameters`` but
    read directly from the code object (no unwrapping, no extra failure modes).
    """
    count = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & python_inspect.CO_VARARGS:
        count += 1
    if code.co_flags & python_inspect.CO_VARKEYWORDS:
        count += 1
    return frozenset(code.co_varnames[:count])


def _varkeywords_name(code: types.CodeType) -> str | None:
    """Name of the layer's ``**kwargs`` parameter, or ``None`` if there is none.

    With the fixed ``co_varnames`` layout the ``**kwargs`` index is
    ``co_argcount + co_kwonlyargcount (+1 with *args)``, present only when
    ``co_flags`` has ``CO_VARKEYWORDS``.

    Needed because ``x['load']`` / ``x.get('load')`` only forwards the caller's
    ``load`` when ``x`` **is** this layer's ``**kwargs``; an ordinary dict parameter
    holding a ``'load'`` key is the wrapper's own configuration.
    """
    if not code.co_flags & python_inspect.CO_VARKEYWORDS:
        return None
    index = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & python_inspect.CO_VARARGS:
        index += 1
    if index >= len(code.co_varnames):
        return None
    return code.co_varnames[index]


def _is_direct_load_forwarding(
        node: ast.expr,
        code: types.CodeType,
        rebound: frozenset[str],
) -> bool:
    """Whether ``node`` **provably** forwards the ``load`` this layer received, unchanged.

    This is an allow-list, not "anything unresolvable is forwarding": a transformed
    value (``choose_load(kwargs.get('load'))`` returning ``[]``) is what really
    reaches the inner function, so drilling in and crediting the inner ``load=``
    would silence a real missing preload.

    Accepted shapes only, with nothing wrapped around them:

    ==============================  =================================================
    ``load``                        bare name ``load`` that is a parameter of this layer
    ``kwargs['load']``              subscript of the **same** key on this layer's ``**kwargs``
    ``kwargs.get('load')``          ``.get`` of the same key (one argument, no default)
    ==============================  =================================================

    :param code: this layer's code object, used to confirm the bare name is a
        parameter of this layer (a same-named module constant is a definite value,
        not forwarding) and to locate the ``**kwargs`` parameter.
    :param rebound: names bound / mutated locally in this layer; a rebound ``load``
        no longer carries the caller's value.
    """
    if isinstance(node, ast.Name):
        return (
            node.id == 'load'
            and node.id in _parameter_names(code)
            and node.id not in rebound
        )

    # The two remaining shapes require the receiver to be this layer's ``**kwargs``.
    varkw = _varkeywords_name(code)
    if varkw is None:
        return False

    if isinstance(node, ast.Subscript):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == varkw
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 'load'
        )

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'get'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == varkw
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 'load'
    ):
        return True

    return False


def _parameter_default(func: Any, name: str) -> tuple[bool, Any]:
    """Default value of the **keyword-only** parameter ``name`` (``def f(*, name=X)``).

    A default is the effective value only when the caller does not pass the
    argument; that decision belongs to the caller (see ``allow_defaults`` of
    ``_resolve_load_name_at_runtime``).

    **Only ``__kwdefaults__`` is read, never ``__defaults__``.** The wrapper chain
    collects ``partial.keywords`` but not positional ``partial.args``; with
    positional defaults the two would combine into a false negative::

        async def dependency(prefix, load=Model.relation): ...
        bound = partial(dependency, 'prefix', None)   # positional None overrides the default

    Keyword-only parameters can never be filled positionally, so the invisible
    ``partial.args`` is orthogonal to this lookup. A positional-or-keyword ``load``
    simply yields no default => empty set => report (possible false positive, never
    a false negative).

    :returns: ``(has default, default)``; ``(False, None)`` when ``name`` is not a
        keyword-only parameter or has no default -- distinct from "the default is
        ``None``", hence the tuple instead of a sentinel.
    """
    if not hasattr(func, '__kwdefaults__'):
        return False, None
    kwdefaults = func.__kwdefaults__
    if kwdefaults is not None and name in kwdefaults:
        return True, kwdefaults[name]
    return False, None


def _resolve_load_name_at_runtime(
        func: Any,
        name: str,
        bound_kwargs: dict[str, Any] | None = None,
        allow_defaults: bool = False,
) -> set[str]:
    """Resolve a bare-name ``load=<name>`` in ``func``'s **runtime namespace**.

    Covers shapes the AST cannot see:

    - **closure free variables of dependency factories**: ``require_x(X, load=rel(X.y))``
      returns a checker whose body says ``load=load``; the value is fixed when the
      factory is called;
    - **module constants defined in another module** (``load=PRELOAD`` imported
      from elsewhere; the analyzed module's AST only has an import).

    Name ownership follows CPython's own rules (``co_freevars`` / ``co_varnames``):
    a local name is **never** looked up in globals, otherwise a same-named global
    constant would silence a real missing preload.

    :param bound_kwargs: arguments bound by wrappers (``functools.partial``). Highest
        priority: they are the values actually used at call time and override
        parameter defaults.
    :param allow_defaults: whether a keyword-only parameter default may be used (no
        layer passed ``load=`` explicitly on the way in).
    :returns: resolved relationship names; an empty set when unresolvable (the
        caller keeps reporting).
    """
    # 1. Wrapper-bound values first: for the inner function ``load`` is only a
    #    parameter (the co_varnames branch below refuses it); the value lives here.
    if bound_kwargs is not None and name in bound_kwargs:
        return _relation_names_from_load_value(bound_kwargs[name])

    if not hasattr(func, '__code__'):
        return set()
    code = func.__code__

    if name in code.co_freevars:
        # Closure free variable: the cell holds the load value passed to the factory
        closure = func.__closure__
        if closure is None:
            return set()
        index = code.co_freevars.index(name)
        if index >= len(closure):
            return set()
        try:
            cell_value = closure[index].cell_contents
        except ValueError:
            # Empty cell (free variable not assigned yet): unknown at runtime
            return set()
        return _relation_names_from_load_value(cell_value)

    if name in code.co_varnames:
        # Parameter / local name: the runtime value varies per call. A parameter's
        # **default** is a definite value, though -- it is effective when the caller
        # omits the argument (``async def dep(*, load=rel(X.y))`` + ``await dep()``).
        # Only usable when ``allow_defaults`` (nobody passed load= explicitly).
        if allow_defaults:
            has_default, default_value = _parameter_default(func, name)
            if has_default:
                return _relation_names_from_load_value(default_value)
        # Pure local or parameter without default: statically unknown. Never fall
        # back to __globals__.
        return set()

    if not hasattr(func, '__globals__'):
        return set()
    globals_ns = func.__globals__
    if name not in globals_ns:
        return set()
    return _relation_names_from_load_value(globals_ns[name])


# ========================= Commit method auto-discovery =========================


def _ast_has_typed_commit(tree: ast.Module, session_param: str) -> bool:
    """
    Check whether the AST calls ``.commit()`` or ``.rollback()`` on the given session parameter.

    Only matches ``<session_param>.commit()`` / ``<session_param>.rollback()``,
    anchored on the parameter name to avoid false matches.
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


def _ast_has_typed_reset(tree: ast.Module, session_param: str) -> bool:
    """
    Check whether the AST calls ``.reset()`` on the given session parameter.

    ``session.reset()`` = ``expunge_all`` + release the connection: **every object
    in the session is detached from the identity map**, while its loaded **column
    values are kept and stay readable**; detached objects are immune to later
    commits (``_TrackedVar.detached``). What still fails: objects that were already
    expired *before* the detach (commit-then-reset => detached and expired, reading
    a column raises ``DetachedInstanceError``) and access to unloaded relationships
    (RLC003).

    Matched pattern (anchored on the session parameter name, so ``redis.reset()``
    etc. do not match)::

        await <session_param>.reset()
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == 'reset'
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


def _ast_has_keyword_true_static(call: ast.Call, keyword: str) -> bool:
    """Statically check whether an AST Call node has a literal ``keyword=True`` argument (not a dynamic value)."""
    for kw in call.keywords:
        if (kw.arg == keyword
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True):
            return True
    return False


def _ast_call_commits(call: ast.Call, method_name: str) -> bool:
    """
    Whether this **call** constitutes a commit (the criterion depends on the method).

    - :data:`explicit_commit_methods`: the default is ``commit=False``, so only a
      literal ``commit=True`` at the call site commits; omitted / ``False`` /
      dynamic values do not (a dynamic value cannot be proven true, and treating it
      as a commit would cascade false positives over every later access).
    - Other methods: a commit unless the call passes an explicit ``commit=False``.

    Unlike :data:`conditional_commit_methods` (whole method excluded), this is
    call-site sensitive: an explicit ``commit=True`` is still reported.

    This answers "does this call really commit" for the **transitive closure**
    (``_ast_calls_commit_method_with_session``). It must not be used to gate
    ``_FunctionAnalyzer._is_commit_for_call()``, which asks "does this call enter
    the commit-method state machine": a regular method's ``commit=False`` call must
    still enter it (its ``commit_disabled`` branch preserves ``loaded_rels``);
    filtering it out earlier would silently drop relationship state and report
    false RLC003.
    """
    if method_name in explicit_commit_methods:
        return _ast_has_keyword_true_static(call, 'commit')
    return not _ast_has_keyword_false_static(call, 'commit')


def _resolve_callee_type(
    value_node: ast.expr,
    owning_class: str | None,
    model_class_names: frozenset[str],
) -> str | None:
    """
    Resolve the model type of a call target from an AST node.

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


def _method_returns_from_refreshing(
    tree: ast.AST,
    refreshing_methods: frozenset[str],
    model_class_names: frozenset[str],
) -> bool:
    """Check whether a method's return value comes from a known refreshing call.

    Used to auto-discover "internally refreshing" model-returning commit methods
    (transitive closure): if a method's return statement returns a variable or
    expression produced by a method in ``refreshing_methods`` (and not explicitly
    ``commit=False``), the return value is already refreshed.

    Patterns:
    1. ``return await obj.method(...)`` -- directly returns a refreshing result
    2. ``var = await obj.method(...)`` + ``return var`` -- indirect return
    3. ``var = await Model.get(...)`` + ``return var`` -- the common "re-fetch via
       ``Type.get`` after commit" pattern (``Model.get/get_one/get_instance`` go
       through cache -> DB and return a fresh instance)

    :param model_class_names: known model class names, used to decide whether
        ``ClassName.get(...)`` is a model classmethod query (excluding ``dict.get()``
        and other same-named calls)
    """

    def _is_refreshing_call(call: ast.Call) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        # (1) instance method in refreshing_methods (save/update and their closure)
        if call.func.attr in refreshing_methods:
            for kw in call.keywords:
                if kw.arg == 'commit' and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    return False
            return True
        # (2) model classmethod query: Model.get/get_one/get_instance
        if (call.func.attr in _MODEL_REFRESH_CLASSMETHODS
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in model_class_names):
            return True
        return False

    # Collect every variable assigned from a refreshing call
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
    Check whether a method is a commit method for a resolved type (MRO-aware).

    Walks the MRO to find the most specific definition of the method and checks
    whether that defining class marks it as a commit method. This correctly handles
    overrides: if a subclass overrides a committing base-class method without
    committing, the subclass version wins.

    For non-model classes (not in model_classes), class_commit is checked directly
    (no MRO resolution).
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
            # Found the most specific definition -> check that class's commit state
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
    Check whether the AST calls a known commit method while passing the session parameter.

    Matched patterns:

    - ``await obj.save(session, ...)``
    - ``await cls.from_remote_url(session=session, ...)``

    Excluded patterns (not treated as a commit), see ``_ast_call_commits``:

    - ``await obj.save(session, commit=False)`` -- commit explicitly disabled
    - a method in :data:`explicit_commit_methods` called without a literal ``commit=True``

    Enhanced mode (when per-class arguments are provided):

    When the call target's type can be resolved from the AST (self/cls/explicit
    class name), per-class commit state is used to avoid false matches between
    same-named methods on different classes. Falls back to global name matching
    (conservative) when the type cannot be resolved.
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

        # Does this call constitute a commit? The closure only propagates real commits.
        if not _ast_call_commits(node, method_name):
            continue

        # Check whether the session is passed as an argument
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

        # Type not resolvable or per-class disabled -> conservative (global name match)
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
        detaching_methods: frozenset[str] | None = None,
        model_detaching_methods: dict[str, frozenset[str]] | None = None,
        self_attr_models: dict[str, str] | None = None,
        pre_committed_params: set[str] | None = None,
        detached_params: frozenset[str] | None = None,
        model_returning_by_class: dict[str, frozenset[str]] | None = None,
        analyzed_func: Any = None,
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
        # RLC013: does the function signature include an externally-provided
        # AsyncSession parameter? Only when True does an async generator's yield
        # trigger pessimistic expiration of tracked ORM vars ("the consumer may
        # commit on the shared session during the yield"). Local sessions created
        # via ``async with session_factory() as ...`` do not appear in the signature
        # and are therefore safe.
        self.has_session_param: bool = bool(self.session_param_names)
        # Complete model-returning set for variable tracking (includes sync methods).
        # Sync methods are NOT added to safe_methods because calling sync methods
        # on expired objects is equally dangerous.
        _sync = sync_model_returning_methods or frozenset()
        self._all_model_returning: frozenset[str] = self.model_returning_methods | _sync
        self._sync_model_returning: frozenset[str] = _sync
        # Per-class async model-returning sets: the real criterion when the receiver
        # type is resolved (the global name set is only the fallback).
        self.model_returning_by_class: dict[str, frozenset[str]] = (
            model_returning_by_class if model_returning_by_class is not None else {}
        )
        self.refreshing_commit_methods: frozenset[str] = refreshing_commit_methods or frozenset()
        self.detaching_methods: frozenset[str] = detaching_methods or frozenset()
        self.model_detaching_methods: dict[str, frozenset[str]] = model_detaching_methods or {}
        self.class_aliases: dict[str, str] = class_aliases or {}
        self.model_dunder_rels: dict[str, dict[str, set[str]]] = model_dunder_rels or {}
        # self.<attr> -> model class name index of the analyzed method's class
        self.self_attr_models: dict[str, str] = self_attr_models or {}

        # Parameters already expired at function entry by a sibling dependency's commit
        # (endpoints only). They start with post_commit=True and
        # pre_committed_by_sibling_dep=True, so any later column access is routed to
        # RLC014 in visit_Attribute.
        pre_committed: set[str] = pre_committed_params or set()
        self.tracked_vars: dict[str, _TrackedVar] = {}
        self.all_loaded_rel_names: set[str] = set()
        """Relationship names seen in **any** ``load=`` of the function body (whatever
        receives the result). ``tracked_vars`` only covers query results assigned to a
        variable; RLC005 needs this set to avoid flagging ``return await X.q(..., load=...)``."""
        self._module_tree: ast.Module | None = None
        """AST of the analyzed function's module, used to resolve ``load=SOME_CONST``
        to the elements of a module-level constant. ``None`` when unavailable."""
        self._analyzed_func: Any = analyzed_func
        """The analyzed function object, used for the runtime fallback of
        ``load=<bare name>`` (``_module_tree`` only sees this module's assignments)."""
        if source_file and source_file != UNKNOWN_LABEL:
            try:
                self._module_tree = ast.parse(
                    pathlib.Path(source_file).read_text(encoding='utf-8'),
                )
            except (OSError, SyntaxError, ValueError):
                self._module_tree = None
        _detached_params = detached_params or frozenset()
        # Session ownership of parameters: known only with a single signature session;
        # with several, it cannot be decided statically => None
        _sole_session = (
            next(iter(self.session_param_names))
            if len(self.session_param_names) == 1 else None
        )

        # Initialize tracked vars from parameter type annotations
        for param_name, model_name in param_models.items():
            loaded = dep_loads.get(param_name, set())
            is_pre_committed = param_name in pre_committed
            self.tracked_vars[param_name] = _TrackedVar(
                model_name=model_name,
                loaded_rels=loaded.copy(),
                post_commit=is_pre_committed,
                pre_committed_by_sibling_dep=is_pre_committed,
                caller_provided=param_name in caller_provided_params,
                # detached_params (pytest scan, resolved per fixture): fixture products
                # that do not share the test body's session are immune to its commits.
                # Production endpoints / model methods pass an empty set.
                detached=param_name in _detached_params,
                session_name=_sole_session,
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

        This asks "does the call enter the commit-method state machine", **not**
        "does this call really commit": a regular method's ``save(commit=False)``
        must still enter it, so that its ``commit_disabled`` branch keeps the
        variable's ``loaded_rels``. Only :data:`explicit_commit_methods` are filtered
        here (without a literal ``commit=True`` they are not a commit at all). The
        transitive closure uses ``_ast_call_commits`` for the other question.
        """
        if (
            method_name in explicit_commit_methods
            and not _ast_has_keyword_true_static(call, 'commit')
        ):
            return False

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

    def _returning_for_type(self, resolved_type: str | None) -> frozenset[str]:
        """
        Effective model-returning method names for a receiver type.

        Resolved type with a per-class entry => that class's (MRO-aware) async
        model-returning set plus the global sync set (sync methods are not per-class
        yet); unresolved type => the global name set (conservative, like
        ``_is_commit_for_call``).
        """
        if resolved_type is not None and resolved_type in self.model_returning_by_class:
            return self.model_returning_by_class[resolved_type] | self._sync_model_returning
        return self._all_model_returning

    def _bound_session(self, call: ast.Call) -> str | None:
        """
        Target session of a query / factory call: the signature session name when the
        arguments contain **exactly one**; zero (short-lived local session / no session
        argument) or more than one (ownership of the result undecidable -- guessing the
        first would turn "unknown" into "known") => ``None`` (hit by every expire /
        detach target).
        """
        passed = self._passed_session_names(call)
        return passed[0] if len(passed) == 1 else None

    def _is_detaching_for_call(self, call: ast.Call, method_name: str) -> bool:
        """
        Check whether a method call is a detaching call (per-class aware, symmetric
        with ``_is_commit_for_call``).
        """
        if not self.detaching_methods:
            return False
        if not self.model_detaching_methods:
            return method_name in self.detaching_methods

        obj_name = self._get_call_object_name(call)
        resolved_type: str | None = None
        if obj_name is not None:
            if obj_name in self.tracked_vars:
                resolved_type = self.tracked_vars[obj_name].model_name
            else:
                resolved_type = self._resolve_class_name(obj_name)

        if resolved_type is not None:
            class_detaching = self.model_detaching_methods.get(resolved_type, frozenset())
            return method_name in class_detaching

        return method_name in self.detaching_methods

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

                # session.commit() / session.rollback() -- only when the receiver is one
                # of the function's **signature** sessions. A short-lived local session
                # (``async with factory() as s: await s.commit()``) commits another
                # identity map. The expiry targets the receiver session only.
                if (method_name in {'commit', 'rollback'}
                        and (_recv := self._get_call_object_name(call)) is not None
                        and _recv in self.session_param_names):
                    self._expire_tracked_vars_for([_recv])

                # session.reset(): detach. Detaching is not expiring -- loaded column
                # values stay readable (why "reset to release the connection, then keep
                # reading cached columns" is correct), so nothing is expired or
                # reported; but detached objects are no longer in the identity map and
                # **later commits cannot expire them**, which the detached flag records.
                elif (method_name == 'reset'
                        and (_recv := self._get_call_object_name(call)) is not None
                        and _recv in self.session_param_names):
                    self._mark_tracked_vars_detached_for([_recv])

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
                        # The commit only happens on this function's session (and only
                        # expires its tracked objects) when a **signature** session is
                        # passed. A short-lived local session, or no session at all
                        # (an Optional session the method opens itself), commits
                        # another session. Expiry targets the sessions actually passed.
                        passed_sessions = self._passed_session_names(call)
                        if not commit_disabled and passed_sessions:
                            self._expire_tracked_vars_for(passed_sessions)
                            # save/update default refresh=True, refreshes the call object in-place
                            if (obj_name in self.tracked_vars
                                    and method_name in _REFRESH_METHODS
                                    and not self._has_keyword_false(call, 'refresh')):
                                self.tracked_vars[obj_name].post_commit = False
                            # Model-returning commit method called on self -> identity map refreshes self.
                            # Example: super().fill_from_file_path() internally calls self.save() -> self is refreshed.
                            # Not for detaching methods: they end by detaching everything,
                            # so "self was refreshed" does not hold.
                            elif (obj_name == 'self'
                                    and obj_name in self.tracked_vars
                                    and method_name in self._returning_for_type(
                                        self.tracked_vars[obj_name].model_name)
                                    and not self._is_detaching_for_call(call, method_name)):
                                self.tracked_vars[obj_name].post_commit = False
                    else:
                        # Untracked variable calling a commit method (e.g. loop var user_file.fill_from_image_url()).
                        # Only expire when a session parameter is passed (to distinguish
                        # from non-model calls like redis.delete(key)).
                        untracked_sessions = self._passed_session_names(call)
                        if (not self._has_keyword_false(call, 'commit')
                                and untracked_sessions):
                            self._expire_tracked_vars_for(untracked_sessions)

                # Detaching method (reaches this session's reset()): afterwards every
                # object of that session is detached -- same semantics as the direct
                # reset branch above.
                detach_sessions = self._passed_session_names(call)
                if self._is_detaching_for_call(call, method_name) and detach_sessions:
                    self._mark_tracked_vars_detached_for(detach_sessions)

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

                    # Same as visit_Expr: only a signature session passed to the commit
                    # method expires this function's tracked objects; expiry targets the
                    # sessions actually passed.
                    assign_sessions = self._passed_session_names(call)
                    if not commit_disabled and assign_sessions:
                        self._expire_tracked_vars_for(assign_sessions)

                    if method_name in _REFRESH_METHODS:
                        # save/update: returns self, supports refresh= and load= parameters
                        refresh_disabled = self._has_keyword_false(call, 'refresh')
                        if var_name == obj_name:
                            old_var.caller_provided = False
                            if commit_disabled:
                                old_var.post_commit = False
                                old_var.loaded_rels |= loaded_rels
                            elif refresh_disabled:
                                pass  # already marked by the expiry above
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
                                # save returns the same instance: session ownership follows the source
                                session_name=(
                                    self._bound_session(call) or old_var.session_name
                                ),
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
                        # Per-class decision: a same-named method returning a scalar on
                        # this class must not be tracked because another class's version
                        # returns a model (the scalar would later be reported as RLC008).
                        if method_name in self._returning_for_type(old_var.model_name):
                            is_refreshing = method_name in self.refreshing_commit_methods
                            is_detaching = self._is_detaching_for_call(call, method_name)
                            self.tracked_vars[var_name] = _TrackedVar(
                                model_name=old_var.model_name,
                                loaded_rels=loaded_rels,
                                post_commit=not commit_disabled and not is_refreshing,
                                caller_provided=False,
                                session_name=(
                                    self._bound_session(call) or old_var.session_name
                                ),
                                line=self._abs_line(node),
                            )
                            # Model-returning commit method called on self -> identity map
                            # refreshes self (not for detaching methods)
                            if obj_name == 'self' and not is_detaching:
                                old_var.post_commit = False
                                old_var.loaded_rels = loaded_rels

                # Class method call (Model.create_unique / cls.reserve / Model(...).save() /...)
                else:
                    resolved = self._resolve_class_name(class_name)
                    if resolved is not None:
                        # Same as the instance branch: only a commit on a signature
                        # session expires tracked objects, targeted by session name.
                        cls_sessions = self._passed_session_names(call)
                        if not self._has_keyword_false(call, 'commit') and cls_sessions:
                            self._expire_tracked_vars_for(cls_sessions)
                        # Only the class's model-returning methods (or save/update) return
                        # model instances (per-class decision; scalars are not tracked).
                        # Classmethod factories / constructor-chain saves return fresh
                        # objects (they end with save -> get); an explicit refresh=False
                        # means the caller accepts an expired object.
                        if (method_name in _REFRESH_METHODS
                                or method_name in self._returning_for_type(resolved)):
                            self.tracked_vars[var_name] = _TrackedVar(
                                model_name=resolved,
                                loaded_rels=loaded_rels,
                                post_commit=(
                                    method_name in _REFRESH_METHODS
                                    and self._has_keyword_false(call, 'refresh')
                                    and not self._has_keyword_false(call, 'commit')
                                ),
                                caller_provided=False,
                                session_name=self._bound_session(call),
                                line=self._abs_line(node),
                            )
                    else:
                        # Untracked variable calling a commit method
                        # (e.g. loop variable user_file.fill_from_image_url()).
                        # Only expire when session is passed (distinguishes from
                        # non-model calls like redis.delete(key)).
                        else_sessions = self._passed_session_names(call)
                        if not self._has_keyword_false(call, 'commit') and else_sessions:
                            self._expire_tracked_vars_for(else_sessions)

            # Auto-discovered model-returning methods (non-commit, pure query)
            # Includes get/find_by_content_hash/get_exist_one etc. (including sync methods,
            # e.g. get_tool_by_name)
            # Discovered via return type annotations (the global name set is only a
            # pre-filter; with a resolved receiver the per-class set decides)
            elif method_name in self._all_model_returning:
                obj_name = self._get_call_object_name(call)
                class_name = self._get_call_class_name(call)
                resolved = self._resolve_class_name(class_name)

                # Class method call (Model.get / cls.find_by_content_hash /...)
                if resolved is not None:
                    if method_name in self._returning_for_type(resolved):
                        if self._has_keyword(call, 'options'):
                            effective_rels = self.model_relationships[resolved].copy()
                        else:
                            effective_rels = loaded_rels
                        self.tracked_vars[var_name] = _TrackedVar(
                            model_name=resolved,
                            loaded_rels=effective_rels,
                            post_commit=False,
                            caller_provided=False,
                            session_name=self._bound_session(call),
                            line=self._abs_line(node),
                        )

                # Instance method call (tracked_var.some_query_method/...)
                elif obj_name and obj_name in self.tracked_vars:
                    old_var = self.tracked_vars[obj_name]
                    if method_name in self._returning_for_type(old_var.model_name):
                        self.tracked_vars[var_name] = _TrackedVar(
                            model_name=old_var.model_name,
                            loaded_rels=loaded_rels,
                            post_commit=False,
                            caller_provided=False,
                            session_name=(
                                self._bound_session(call) or old_var.session_name
                            ),
                            line=self._abs_line(node),
                        )

            # Detaching method (reaches this session's reset()): every object of that
            # session becomes detached -- same semantics as visit_Expr's reset branch.
            # Independent of the commit / model-returning split above (a method may
            # both commit and detach).
            assign_detach_sessions = self._passed_session_names(call)
            if (self._is_detaching_for_call(call, method_name)
                    and assign_detach_sessions):
                self._mark_tracked_vars_detached_for(assign_detach_sessions)

        # Instance-attribute container lookup (sync): var = self.<attr>.get(key) /
        # var = self.<attr>[key], where self.<attr> is annotated as ``dict[K, Model]`` /
        # ``list[Model]`` / ``Model | None``: the value is tracked as a Model instance.
        #
        # **post_commit=True (pessimistic)**: such caches are filled by code outside
        # the analyzed method, so nothing guarantees no commit happened between filling
        # and this access; with expire_on_commit=True a column access would then lazy
        # load -> MissingGreenlet. If the cache really is fresh, re-fetch explicitly
        # (``var = await Type.get(session, col(Type.id) == var.id)``) -- an explicit
        # freshness statement instead of a silent assumption.
        if self.self_attr_models:
            attr_model = self._resolve_self_container_value(value)
            if attr_model is not None:
                self.tracked_vars[var_name] = _TrackedVar(
                    model_name=attr_model,
                    loaded_rels=set(),
                    post_commit=True,
                    caller_provided=True,
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
                    # The child object lives in the parent's session: expiry / detach /
                    # RLC013 / RLC014 / session ownership are all inherited (copying only
                    # post_commit would lose detach immunity and the RLC014 marker here).
                    self.tracked_vars[var_name] = _TrackedVar(
                        model_name=target_model,
                        loaded_rels=set(),
                        post_commit=src_var.post_commit,
                        caller_provided=True,
                        expired_by_yield=src_var.expired_by_yield,
                        pre_committed_by_sibling_dep=src_var.pre_committed_by_sibling_dep,
                        detached=src_var.detached,
                        session_name=src_var.session_name,
                        line=self._abs_line(node),
                    )

        # Plain alias: var = tracked_var. Both names refer to the same ORM instance, so
        # **every** state field is copied (dropping ``detached`` would report
        # ``session.reset(); alias = user; session.commit(); alias.id`` as RLC007, and
        # dropping the sibling-dependency marker would turn RLC014 into RLC007).
        if (isinstance(value, ast.Name)
                and value.id in self.tracked_vars
                and value.id != var_name):
            src_var = self.tracked_vars[value.id]
            self.tracked_vars[var_name] = _TrackedVar(
                model_name=src_var.model_name,
                loaded_rels=src_var.loaded_rels.copy(),
                post_commit=src_var.post_commit,
                caller_provided=src_var.caller_provided,
                expired_by_yield=src_var.expired_by_yield,
                pre_committed_by_sibling_dep=src_var.pre_committed_by_sibling_dep,
                detached=src_var.detached,
                session_name=src_var.session_name,
                line=self._abs_line(node),
            )

    def _resolve_self_container_value(self, value: ast.expr) -> str | None:
        """
        Recognize the ``self.<attr>.get(key)`` / ``self.<attr>[key]`` patterns.

        Uses the ``self_attr_models`` index (built from the owning class's
        annotations) to map a value taken out of a known ORM container to its model
        class name.

        Covered: ``self.<attr>.get(...)`` (dict-like) and ``self.<attr>[<key>]``
        (dict-like / list-like). Not covered (skipped conservatively):
        ``self.<attr>.values()`` and other view-returning methods, direct
        ``self.<attr>`` assignment (handled by relationship extraction), and deeper
        chains (``self.x.y.get()``).

        :returns: the resolved model class name, or ``None``
        """
        if not self.self_attr_models:
            return None

        # Pattern 1: self.<attr>.get(...) -- dict-like
        if (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == 'get'
                and isinstance(value.func.value, ast.Attribute)
                and isinstance(value.func.value.value, ast.Name)
                and value.func.value.value.id == 'self'):
            return self.self_attr_models.get(value.func.value.attr)

        # Pattern 2: self.<attr>[<key>] -- Subscript
        if (isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Attribute)
                and isinstance(value.value.value, ast.Name)
                and value.value.value.id == 'self'):
            return self.self_attr_models.get(value.value.attr)

        return None

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
            # RLC007/RLC013: column access on expired object (including PK) either
            # after commit or after yield.
            # After commit/yield state.dict is cleared (including PK); accessing
            # any column triggers _load_expired -> synchronous SELECT ->
            # MissingGreenlet in async context. The identity map retains the PK
            # for object lookup, but the attribute descriptor still takes the
            # expired path.
            # Typical scenarios:
            #     - RLC007: obj_a.save() then obj_b.column (obj_b not refreshed)
            #     - RLC013: async generator yields then accesses obj.column
            #               (consumer may commit during the yield, expiring the
            #               object in the shared session)
            if var_info.expired_by_yield:
                self.warnings.append(RelationLoadWarning(
                    code='RLC013',
                    file=self.source_file,
                    line=self._abs_line(node),
                    message=(
                        f"Accessing column '{var_key}.{attr_name}' after yield "
                        f"in an async generator. Yield hands control to the "
                        f"consumer, which holds the same session reference and "
                        f"may commit during the yield, expiring the object; "
                        f"access will trigger MissingGreenlet. "
                        f"Suggestion: extract '{var_key}.{attr_name}' into a "
                        f"local variable BEFORE the yield (do not use "
                        f"Type.get() to reload after the yield -- the next "
                        f"yield will expire it again)"
                    ),
                ))
            elif var_info.pre_committed_by_sibling_dep:
                self.warnings.append(RelationLoadWarning(
                    code='RLC014',
                    file=self.source_file,
                    line=self._abs_line(node),
                    message=(
                        f"Accessing column '{var_key}.{attr_name}' in the endpoint, but "
                        f"'{var_key}' ({var_info.model_name}) is already expired when the "
                        f"endpoint starts: another Depends on the same session commits "
                        f"inside its body (e.g. a get-or-create dependency calling "
                        f"Model.save()), and with expire_on_commit=True every ORM object "
                        f"injected by the other dependencies becomes stale. The column "
                        f"access triggers a synchronous refresh -> MissingGreenlet. "
                        f"Suggestion: re-read it in the endpoint "
                        f"(``await {var_info.model_name}.get_one(session, id)``), or take "
                        f"scalar values from an object the committing dependency refreshed"
                    ),
                ))
            else:
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

                # RLC010: check whether ``return await Call(..., expired_arg)`` carries
                # post-commit expired objects as call args. Previously visit_Return only
                # checked relationship preloading (_check_return_loaded), leaving
                # ``return await callee(..., expired_var)`` paths uncovered. Mirrors the
                # symmetric checks in visit_Expr and _check_assign.
                # Example scenario:
                #     user_file = await cls._unchecked(...)  # commits → tracked vars expire
                #     return await user_file._moderate(session, config=config)
                #                                              ^^^^^^^^^^^^^
                #                                              ``config`` expired; callee's
                #                                              ``config.attr`` triggers
                #                                              MissingGreenlet.
                self._check_expired_call_args(call, node)

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

        else_returns = False
        post_else: dict[str, _TrackedVar] = {}
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
                # body may or may not execute: merge (trust an explicit refresh in the
                # body, stay pessimistic about side-effect commits)
                self._merge_tracked_vars(pre_if, post_body, prefer_refresh=True)

        # ---- is-None branch narrowing ----
        # On the None branch of ``if x is None:`` x does not point to an ORM object,
        # so a commit on that path cannot expire "the object x points to". As long as
        # the None branch does not rebind x, x's merged state is taken from the
        # non-None side (other variables keep the pessimistic merge above: the commit
        # really affects them). If the following code is only reachable through the
        # None path, x is None from here on and is no longer tracked (attribute access
        # on it is a None bug, not an expiry issue).
        #
        #     if existing is None:
        #         await session.commit()      # releases a row lock on the miss path
        #     else:
        #         dup = await existing.create_duplicate(session)   # must not be RLC008
        narrow = self._narrow_is_none_test(node.test)
        if narrow is not None:
            x_name, body_is_none = narrow
            none_stmts = node.body if body_is_none else list(node.orelse)
            if x_name in pre_if and not self._stmts_rebind_name(none_stmts, x_name):
                if body_is_none:
                    none_survives = not body_returns
                    nn_state = post_else if node.orelse else pre_if
                    nn_survives = (not else_returns) if node.orelse else True
                else:
                    none_survives = (not else_returns) if node.orelse else True
                    nn_state = post_body
                    nn_survives = not body_returns
                if none_survives:
                    if nn_survives and x_name in nn_state:
                        self.tracked_vars[x_name] = nn_state[x_name]
                    elif not nn_survives:
                        _ = self.tracked_vars.pop(x_name, None)

    @staticmethod
    def _narrow_is_none_test(test: ast.expr) -> tuple[str, bool] | None:
        """
        Recognize ``x is None`` / ``x is not None`` if-conditions (x a bare name).

        :returns: ``(name, body is the None branch)`` -- ``x is None`` gives
            ``(x, True)``, ``x is not None`` gives ``(x, False)``; any other shape ``None``.
        """
        if not (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and len(test.ops) == 1
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None):
            return None
        if isinstance(test.ops[0], ast.Is):
            return test.left.id, True
        if isinstance(test.ops[0], ast.IsNot):
            return test.left.id, False
        return None

    @staticmethod
    def _stmts_rebind_name(stmts: list[ast.stmt], name: str) -> bool:
        """
        Whether the statements (nested included) rebind ``name`` (assignment, annotated /
        augmented assignment, walrus, for-target, with-as ... -- any ``ctx=Store``).

        Safety precondition of the is-None narrowing: if the None branch rebinds x, x
        points to a new object at the branch exit and "x is None on that path" no
        longer holds.
        """
        for stmt in stmts:
            for node in ast.walk(stmt):
                if (isinstance(node, ast.Name)
                        and node.id == name
                        and isinstance(node.ctx, ast.Store)):
                    return True
        return False

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
        Extract class name from a Model.get() / Model(...).save() call.

        Matches: Model.get(...), Model.get_with_count(...), and the constructor chain
        Model(...).save(...) (the receiver is a constructor call of Model, so the
        receiver type is Model -- common in seed scripts:
        ``await ToolSet(name=...).save(session)``). ``super().method()`` is not
        matched here (``_get_call_object_name`` maps it to ``'self'``).
        """
        if isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name):
                return call.func.value.id
            if (isinstance(call.func.value, ast.Call)
                    and isinstance(call.func.value.func, ast.Name)
                    and call.func.value.func.id != 'super'):
                return call.func.value.func.id
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

    def _passed_session_names(self, call: ast.Call) -> list[str]:
        """
        Signature session parameter names passed as arguments of a call (in order,
        de-duplicated).

        Empty => the call does not hand over any of this function's sessions (a
        same-named non-model method such as ``redis.delete(key)``, or a short-lived
        local session) -- no expire / detach. Non-empty => expire / detach target
        these sessions (in a function with several sessions, committing one only
        expires that session's objects).
        """
        passed: list[str] = []
        for arg in call.args:
            if (isinstance(arg, ast.Name) and arg.id in self.session_param_names
                    and arg.id not in passed):
                passed.append(arg.id)
        for kw in call.keywords:
            if (isinstance(kw.value, ast.Name)
                    and kw.value.id in self.session_param_names
                    and kw.value.id not in passed):
                passed.append(kw.value.id)
        return passed

    def _expire_tracked_vars_for(self, target_sessions: list[str]) -> None:
        """
        Mark the tracked variables of the **target sessions** as post-commit.

        Simulates commit: a commit expires every object of **that session** (not
        just the one being saved). A variable is hit when:

        - its ``session_name`` is in ``target_sessions`` (it belongs to the committed
          session), or
        - its ``session_name`` is ``None`` (unknown ownership: pessimistically hit; in
          single-session functions this equals expiring everything).

        Detached variables are exempt: objects detached by ``session.reset()`` are no
        longer in the identity map, so a commit cannot expire them and their columns
        stay readable (see ``_TrackedVar.detached``).
        """
        targets = set(target_sessions)
        for var in self.tracked_vars.values():
            if var.detached:
                continue
            if var.session_name is not None and var.session_name not in targets:
                continue
            var.post_commit = True
            var.loaded_rels.clear()

    def _expire_all_tracked_vars_for_yield(self) -> None:
        """
        Mark all tracked variables as yield-expired (RLC013).

        Pessimistic assumption: after the function yields, control is handed to
        the consumer, which holds the same session reference and may commit on
        it during the yield, expiring every object in the session.

        Difference from ``_expire_tracked_vars_for``: also sets
        ``expired_by_yield=True`` so that ``visit_Attribute`` can distinguish
        RLC007 (commit-caused) from RLC013 (yield-caused) and emit a more
        targeted fix suggestion ("extract values before yield" vs.
        "refresh with Type.get() after commit"). Detached variables are exempt
        for the same reason (the consumer's commit cannot reach them either).
        """
        for var in self.tracked_vars.values():
            if var.detached:
                continue
            var.post_commit = True
            var.expired_by_yield = True
            var.loaded_rels.clear()

    def _mark_tracked_vars_detached_for(self, target_sessions: list[str]) -> None:
        """
        Mark the tracked variables of the **target sessions** as detached
        (``session.reset()`` = ``expunge_all``). Hit criterion as in
        ``_expire_tracked_vars_for`` (``session_name is None`` is hit pessimistically).

        Detaching does **not** clear an existing ``post_commit``: a commit-then-reset
        chain produces "detached and expired" objects whose column reads raise
        ``DetachedInstanceError``, so existing warnings must stay. Detaching only
        prevents **later** commits from expiring the object.
        """
        targets = set(target_sessions)
        for var in self.tracked_vars.values():
            if var.session_name is not None and var.session_name not in targets:
                continue
            var.detached = True

    @override
    def visit_Yield(self, node: ast.Yield) -> None:
        """
        Detect ``yield`` statements in async generators.

        When the function signature contains an externally-provided AsyncSession
        parameter, every tracked ORM variable is pessimistically marked as
        expired after yield (the consumer may commit on the shared session
        during the yield).

        Automatic exceptions (no special handling needed):
            - Function does not receive a session parameter -> ``has_session_param``
              is False, nothing expires.
            - Variables re-assigned after yield (e.g. ``llm = await LLM.get_one(...)``)
              -> ``_check_assign`` automatically rebuilds the ``_TrackedVar`` and
              clears the expiration flag.
        """
        # Visit the yielded expression's children in the pre-yield state first,
        # so that attribute access inside the yield value itself is not
        # incorrectly flagged as yield-expired.
        self.generic_visit(node)
        if self.has_session_param:
            self._expire_all_tracked_vars_for_yield()

    @override
    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """
        Detect ``yield from`` statements (sync generators).

        Semantics identical to ``visit_Yield``: once control has been handed to
        the consumer, objects in the shared session may be expired.
        """
        self.generic_visit(node)
        if self.has_session_param:
            self._expire_all_tracked_vars_for_yield()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Do not recurse into nested function definitions.

        AST default DFS would descend into nested ``def`` bodies and count
        their yield/await statements as part of the outer function's flow,
        producing false positives. Nested function bodies execute in their
        own scope when invoked, not as part of the outer flow.

        Typical false positive:
            ``async def outer(...):
                async def _inner():
                    yield chunk``
        The outer function has no yield, but ``_inner``'s yield was being
        counted as the outer's, triggering RLC013.

        Trade-off: nested closures are not analyzed under the current
        scope. Independently scanning them would require a second pass
        with closure capture analysis; in practice nested closures rarely
        access ORM directly, so we accept the gap.
        """

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Do not recurse into nested async function definitions (same as visit_FunctionDef)."""

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
                expired_by_yield=var.expired_by_yield,
                pre_committed_by_sibling_dep=var.pre_committed_by_sibling_dep,
                detached=var.detached,
                session_name=var.session_name,
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
        *,
        prefer_refresh: bool = False,
    ) -> None:
        """
        Pessimistic merge of two branches' tracked_vars state into self.tracked_vars.

        Used when if/else branches both don't return, merging visible state for
        subsequent code.
        Rules (either branch may execute, take worst case):

        - post_commit: OR (if either branch committed, object is expired)
        - loaded_rels: intersection (only rels loaded in both branches are certain)
        - caller_provided: AND (only if both branches mark it)
        - pre_committed_by_sibling_dep: OR
        - detached: AND (detach immunity only holds when both paths detached)
        - session_name: kept when both sides agree, otherwise ``None`` (unknown)

        Only variables existing in both states are preserved (branch-specific new
        variables don't exist in the other branch).

        :param prefer_refresh: used for an ``if`` without ``else`` (``state_a`` = pre-if,
            ``state_b`` = post-body). An object expired before the ``if`` and fresh after
            the body is treated as actively refreshed, and the body state is trusted
            (``if stale: obj = await Model.get(...)``). **Not** for yield-expiry: a
            conditional re-fetch only covers the true branch, and yield-expiry is
            exactly the kind that re-fetching cannot fix (the next yield expires it
            again), so trusting it would hide a real MissingGreenlet on the other path.
        """
        merged: dict[str, _TrackedVar] = {}
        for name in state_a.keys() & state_b.keys():
            a = state_a[name]
            b = state_b[name]
            session_name = a.session_name if a.session_name == b.session_name else None
            if (prefer_refresh and a.post_commit and not b.post_commit
                    and not a.expired_by_yield):
                merged[name] = _TrackedVar(
                    model_name=b.model_name,
                    loaded_rels=b.loaded_rels,
                    post_commit=False,
                    caller_provided=a.caller_provided and b.caller_provided,
                    expired_by_yield=False,
                    pre_committed_by_sibling_dep=(
                        a.pre_committed_by_sibling_dep or b.pre_committed_by_sibling_dep
                    ),
                    detached=a.detached and b.detached,
                    session_name=session_name,
                    line=b.line,
                )
            else:
                merged[name] = _TrackedVar(
                    model_name=a.model_name,
                    loaded_rels=a.loaded_rels & b.loaded_rels,
                    post_commit=a.post_commit or b.post_commit,
                    caller_provided=a.caller_provided and b.caller_provided,
                    expired_by_yield=a.expired_by_yield or b.expired_by_yield,
                    pre_committed_by_sibling_dep=(
                        a.pre_committed_by_sibling_dep or b.pre_committed_by_sibling_dep
                    ),
                    detached=a.detached and b.detached,
                    session_name=session_name,
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

    def _extract_load_from_call(self, call: ast.Call) -> set[str]:
        """Extract relationship names from the load= keyword argument of a Call node.

        The names are also accumulated into :attr:`all_loaded_rel_names`, which RLC005
        uses for results that are not assigned to a variable
        (``return await X.q(..., load=...)``).

        A ``load=<bare name>`` the module AST cannot resolve (typically a constant
        defined in **another** module) falls back to
        ``_resolve_load_name_at_runtime`` on the analyzed function's runtime namespace.
        """
        for kw in call.keywords:
            if kw.arg == 'load':
                names = _extract_load_value(kw.value, self._module_tree)
                if not names and isinstance(kw.value, ast.Name):
                    names = _resolve_load_name_at_runtime(
                        self._analyzed_func, kw.value.id,
                    )
                self.all_loaded_rel_names |= names
                return names
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
