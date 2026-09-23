"""
Relation load checker: ``load=`` extraction for dependency callables.

Covers how ``RelationLoadChecker._extract_loads_from_function`` decides which
relationships a (possibly wrapped) dependency preloads:

* wrapper chains (``functools.partial`` / ``functools.wraps`` / ``__wrapped__``),
* keyword-only defaults vs. explicit overrides,
* the direct-forwarding allow-list (``load`` / ``kwargs['load']`` / ``kwargs.get('load')``),
* local rebinding and in-place mutation of ``load``,
* reaching definitions: a query/delegation call without ``load=`` whose result
  reaches a ``return`` must contribute an empty set,
* runtime namespace resolution (closure cells, cross-module constants),
* the one-directional STI satisfaction relation used by RLC005.

Most tests are *differential*: the dependency is actually awaited (no database
involved -- the stand-in ``get`` / ``fetch`` just echo ``load``) and the
checker's static answer must agree with what the call really passed.

Specimen code lives in throw-away modules written to ``tmp_path`` (the checker
needs real source files for ``inspect.getsource``). Each specimen module owns a
private SQLAlchemy ``DeclarativeBase``, so its tables never touch
``SQLModel.metadata``.

NOTE: no ``from __future__ import annotations`` (keeps annotations real objects,
matching the rest of the suite).
"""
import ast
import importlib
import inspect
import sys
import textwrap
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from types import CodeType, FrameType, FunctionType, MethodType, ModuleType, TracebackType
from typing import Any, TypeAlias

import pytest
from fastapi import FastAPI
from sqlmodel.ext.asyncio.session import AsyncSession

import sqlmodel_ext.relation_load_checker as rlc_module
from sqlmodel_ext import SQLModelBase
from sqlmodel_ext.relation_load_checker import (
    RelationLoadChecker,
    RelationLoadWarning,
    _FunctionAnalyzer,  # pyright: ignore[reportPrivateUsage]
    _is_direct_load_forwarding,  # pyright: ignore[reportPrivateUsage]
    _varkeywords_name,  # pyright: ignore[reportPrivateUsage]
)


# =============================== specimen helpers ===============================

_ORM_HEADER = '''
from functools import wraps

from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class OrmBase(DeclarativeBase):
    pass


class Related(OrmBase):
    __tablename__ = 'rlc_lr_related'
    id: Mapped[int] = mapped_column(primary_key=True)


class Model(OrmBase):
    __tablename__ = 'rlc_lr_model'
    id: Mapped[int] = mapped_column(primary_key=True)
    related_id: Mapped[int] = mapped_column(ForeignKey(Related.id))
    relation: Mapped[Related] = relationship()
'''

_GET_ECHO = '''
    @classmethod
    async def get(cls, *, load=None):
        return load
'''
"""Stand-in query method: named ``get`` so the checker treats ``Model.get(...)``
as a model-returning query call; at runtime it just echoes ``load``."""


def _import_module(tmp_path: Path, module_name: str, source: str) -> ModuleType:
    module_path = tmp_path / f'{module_name}.py'
    _ = module_path.write_text(source, encoding='utf-8')
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(tmp_path))


def _specimen(tmp_path: Path, body: str, *, model_get: str = '') -> ModuleType:
    """Write ``header + optional Model.get + body`` to a unique module and import it."""
    module_name = f'rlc_lr_specimen_{tmp_path.name.replace("-", "_")}'
    # ``model_get`` stays indented: it continues the ``Model`` class body.
    source = _ORM_HEADER + model_get + '\n\n' + textwrap.dedent(body)
    return _import_module(tmp_path, module_name, source)


def _bare_checker() -> RelationLoadChecker:
    """Checker without ``__init__`` (no mapper discovery needed).

    ``_extract_loads_from_layer`` reads ``model_returning_methods`` /
    ``sync_model_returning_methods``, so they are set explicitly to empty sets.
    Consequently these tests do not exercise the "query call without load=" branch;
    they test scoping / binding decisions only.
    """
    checker = object.__new__(RelationLoadChecker)
    checker.model_returning_methods = frozenset()
    checker.sync_model_returning_methods = frozenset()
    return checker


def _loads(checker: RelationLoadChecker, dependency: object) -> set[str]:
    return checker._extract_loads_from_function(dependency)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope='module')
def checker() -> RelationLoadChecker:
    return RelationLoadChecker(SQLModelBase)


# ====================== missing load= on a reachable path ======================

_MISSING_LOAD_BODY = '''
async def dependency(*, load=None):
    return load


def conditional_wrapper(func):
    @wraps(func)
    async def wrapped(*, load=None, omit=False):
        if omit:
            return await func()
        return await func(load=load)
    return wrapped


def attribute_conditional_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, omit=False):
        if omit:
            return await Model.get()
        return await Model.get(load=load)
    return wrapped


def assigned_conditional_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, omit=False):
        if omit:
            result = await Model.get()
            return result
        return await Model.get(load=load)
    return wrapped


def explicit_none_wrapper(func):
    @wraps(func)
    async def wrapped(*, load=None, disabled=False):
        if disabled:
            return await func(load=None)
        return await func(load=load)
    return wrapped
'''


class TestMissingLoadPaths:
    """A reachable path that passes no preload must pull the result to empty."""

    @pytest.mark.asyncio
    async def test_path_without_load_keyword_counts_as_empty(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MISSING_LOAD_BODY, model_get=_GET_ECHO)
        bound = partial(
            module.conditional_wrapper(module.dependency),
            load=module.Model.relation, omit=True,
        )

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual is None
        assert extracted == set(), (
            "the executed path calls the inner function without load=; intersecting only "
            "the existing load= occurrences would miss this empty path"
        )

    @pytest.mark.asyncio
    async def test_attribute_query_call_without_load_counts_as_empty(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MISSING_LOAD_BODY, model_get=_GET_ECHO)
        bound = partial(
            module.attribute_conditional_dependency(module.Model.get),
            load=module.Model.relation, omit=True,
        )

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual is None
        assert extracted == set(), "Model.get() without load= must contribute an empty set"

    @pytest.mark.asyncio
    async def test_assigned_query_without_load_flowing_to_return_counts_empty(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MISSING_LOAD_BODY, model_get=_GET_ECHO)
        bound = partial(
            module.assigned_conditional_dependency(module.Model.get),
            load=module.Model.relation, omit=True,
        )

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual is None
        assert extracted == set(), (
            "the unloaded Model.get() result is returned through a local variable; it is "
            "still what this path hands out"
        )

    @pytest.mark.asyncio
    async def test_unexecuted_loaded_branch_does_not_cover_empty_branch(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MISSING_LOAD_BODY, model_get=_GET_ECHO)
        bound = partial(
            module.explicit_none_wrapper(module.dependency),
            load=module.Model.relation, disabled=True,
        )

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual is None
        assert extracted == set(), (
            "the executed branch passes load=None; the other branch's load=load must not "
            "credit the relationship"
        )


# ======================= reaching definitions to return ========================

_REACHING_BODY = '''
class LoadResult:
    def __init__(self, load):
        self.load = load

    def __iadd__(self, other):
        return other


def augassign_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        result = await Model.get(load=load)
        if unsafe:
            result += await Model.get()
        return result
    return wrapped


def unpacking_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        result = await Model.get(load=load)
        if unsafe:
            result, marker = await Model.get(), 'marker'
        return result
    return wrapped


def namedexpr_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        result = await Model.get(load=load)
        if unsafe and (result := await Model.get()) is None:
            pass
        return result
    return wrapped


def expression_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        safe_result = await Model.get(load=load)
        unsafe_result = await Model.get()
        result = unsafe_result if unsafe else safe_result
        return result
    return wrapped


def return_expression_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        safe_result = await Model.get(load=load)
        unsafe_result = await Model.get()
        return unsafe_result if unsafe else safe_result
    return wrapped


def match_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        result = await Model.get(load=load)
        match unsafe:
            case True:
                result = await Model.get()
        return result
    return wrapped


def trystar_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        result = await Model.get(load=load)
        try:
            if unsafe:
                raise ExceptionGroup('unsafe', [ValueError('missing load')])
        except* ValueError:
            result = await Model.get()
        return result
    return wrapped


def match_capture_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        result = await Model.get(load=load)
        if unsafe:
            subject = await Model.get()
        else:
            subject = 'safe'
        match subject:
            case captured:
                result = captured
        return result
    return wrapped


def try_finally_dependency(func):
    @wraps(func)
    async def wrapped(*, load=None, unsafe=False):
        if unsafe:
            try:
                result = await Model.get()
            finally:
                return result
        return await Model.get(load=load)
    return wrapped
'''

_GET_LOAD_RESULT = '''
    @classmethod
    async def get(cls, *, load=None):
        return LoadResult(load)
'''


class TestReachingDefinitions:
    """An unloaded query result that really overwrites the returned name must be seen."""

    @pytest.mark.asyncio
    async def test_augassign_unsafe_definition_reaches_return(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _REACHING_BODY, model_get=_GET_LOAD_RESULT)
        bound = partial(
            module.augassign_dependency(module.Model.get),
            load=module.Model.relation, unsafe=True,
        )

        actual = await bound()
        extracted = _loads(checker, bound)

        assert actual.load is None
        assert extracted == set(), (
            "augmented assignment writes an unloaded query result back into result; the "
            "earlier safe definition must not be credited"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'factory_name',
        [
            'unpacking_dependency',
            'namedexpr_dependency',
            'expression_dependency',
            'return_expression_dependency',
            'match_dependency',
            'trystar_dependency',
            'match_capture_dependency',
            'try_finally_dependency',
        ],
    )
    async def test_unsafe_definition_reaches_return(
        self, tmp_path: Path, checker: RelationLoadChecker, factory_name: str,
    ) -> None:
        module = _specimen(tmp_path, _REACHING_BODY, model_get=_GET_ECHO)
        factory = getattr(module, factory_name)
        bound = partial(
            factory(module.Model.get),
            load=module.Model.relation, unsafe=True,
        )

        actual = await bound()
        extracted = _loads(checker, bound)

        assert actual is None
        assert extracted == set(), (
            f"{factory_name}: the executed path returns an unloaded query result; the "
            "loaded definition must not be credited"
        )


# ========================= keyword-only defaults ==============================

_DEFAULTS_BODY = '''
async def fetch(*, load=None):
    return load


async def kwonly_dependency(*, load=Model.relation):
    return await fetch(load=load)


async def positional_dependency(prefix, load=Model.relation):
    return await fetch(load=load)


def defaulting_wrapper(func):
    @wraps(func)
    async def wrapped(*, load=Model.relation):
        return await func(load=load)
    return wrapped


def forwarding_wrapper(func):
    @wraps(func)
    async def wrapped(**kwargs):
        return await func(load=kwargs.get('load'))
    return wrapped


def none_default_wrapper(func):
    @wraps(func)
    async def wrapped(*, load=None):
        return await func(load=load)
    return wrapped
'''


class TestDefaultLoadRuntimeDifferential:
    """Parameter defaults must be credited exactly when they are really in effect."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('wrapped', [False, True])
    async def test_kwonly_default_matches_real_no_argument_call(
        self, tmp_path: Path, checker: RelationLoadChecker, wrapped: bool,
    ) -> None:
        module = _specimen(tmp_path, _DEFAULTS_BODY)
        dependency = (
            module.defaulting_wrapper(module.fetch) if wrapped else module.kwonly_dependency
        )

        actual = await dependency()
        extracted = _loads(checker, dependency)

        assert actual is module.Model.relation
        assert extracted == {'relation'}

    @pytest.mark.asyncio
    async def test_positional_default_is_deliberately_not_credited(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        """A positional default is really in effect, but deliberately not credited.

        Together with ``test_partial_positional_argument_kills_default`` this pins
        both sides of the boundary: ``partial.args`` binding is not modeled, so a
        positional default is never credited (a known false positive) -- crediting it
        without modeling ``partial.args`` would reopen the false negative below.
        """
        module = _specimen(tmp_path, _DEFAULTS_BODY)
        dependency = partial(module.positional_dependency, 'prefix')

        actual = await dependency()
        extracted = _loads(checker, dependency)

        assert actual is module.Model.relation
        assert extracted == set()

    @pytest.mark.asyncio
    async def test_partial_positional_argument_kills_default(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _DEFAULTS_BODY)
        dependency = partial(module.positional_dependency, 'prefix', None)

        actual = await dependency()
        extracted = _loads(checker, dependency)

        assert actual is None
        assert extracted == set(), "partial.args overrides the default; it must not be credited"

    @pytest.mark.asyncio
    @pytest.mark.parametrize('wrapper_name', ['forwarding_wrapper', 'none_default_wrapper'])
    async def test_explicit_none_kills_inner_default(
        self, tmp_path: Path, checker: RelationLoadChecker, wrapper_name: str,
    ) -> None:
        module = _specimen(tmp_path, _DEFAULTS_BODY)
        dependency = getattr(module, wrapper_name)(module.kwonly_dependency)

        actual = await dependency()
        extracted = _loads(checker, dependency)

        assert actual is None
        assert extracted == set(), "an explicit None overrides the inner default"

    @pytest.mark.asyncio
    async def test_outer_partial_overrides_wrapper_default(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _DEFAULTS_BODY)
        dependency = partial(
            module.none_default_wrapper(module.fetch), load=module.Model.relation,
        )

        actual = await dependency()
        extracted = _loads(checker, dependency)

        assert actual is module.Model.relation
        assert extracted == {'relation'}


# ========================= direct-forwarding allow-list =========================

def _expression(source: str) -> ast.expr:
    return ast.parse(source, mode='eval').body


def _actual_kwargs(**options: object) -> dict[str, object]:
    return options


def _ordinary_mapping(config: dict[str, object]) -> dict[str, object]:
    return config


class TestDirectLoadForwardingAllowList:
    """``x['load']`` / ``x.get('load')`` only forward when ``x`` is this layer's ``**kwargs``."""

    def test_varkw_subscript_is_direct_forwarding(self) -> None:
        assert _is_direct_load_forwarding(
            _expression("options['load']"), _actual_kwargs.__code__, frozenset(),
        )

    def test_varkw_get_is_direct_forwarding(self) -> None:
        assert _is_direct_load_forwarding(
            _expression("options.get('load')"), _actual_kwargs.__code__, frozenset(),
        )

    def test_arbitrary_mapping_subscript_is_not_forwarding(self) -> None:
        """A same-named key in a config dict is a value decided by this layer."""
        assert not _is_direct_load_forwarding(
            _expression("config['load']"), _ordinary_mapping.__code__, frozenset(),
        )

    def test_arbitrary_mapping_get_is_not_forwarding(self) -> None:
        assert not _is_direct_load_forwarding(
            _expression("config.get('load')"), _ordinary_mapping.__code__, frozenset(),
        )


# ============================ **kwargs code layout =============================

def _only_varkw(**kw: object) -> None:
    _ = kw


def _positional_varkw(a: object, b: object, **options: object) -> None:
    _ = a, b, options


def _kwonly_varkw(*, flag: bool, **payload: object) -> None:
    _ = flag, payload


def _varargs_varkw(a: object, *items: object, flag: bool, **extra: object) -> None:
    _ = a, items, flag, extra


def _without_varkw(a: object, *, flag: bool) -> None:
    _ = a, flag


class TestVarkeywordsLayout:
    """``_varkeywords_name`` must cover every CPython parameter layout."""

    def test_only_varkw_layout(self) -> None:
        assert _varkeywords_name(_only_varkw.__code__) == 'kw'

    def test_positional_then_varkw_layout(self) -> None:
        assert _varkeywords_name(_positional_varkw.__code__) == 'options'

    def test_kwonly_then_varkw_layout(self) -> None:
        assert _varkeywords_name(_kwonly_varkw.__code__) == 'payload'

    def test_varargs_kwonly_then_varkw_layout(self) -> None:
        assert _varkeywords_name(_varargs_varkw.__code__) == 'extra'

    def test_absent_varkw_returns_none(self) -> None:
        assert _varkeywords_name(_without_varkw.__code__) is None


# ===================== forwarding vs. real call semantics ======================

_FORWARDING_BODY = '''
async def fetch(*, load=None):
    return load


async def dependency(*, load=None):
    return await fetch(load=load)


def forwarding_without_duplicate_keywords(func):
    @wraps(func)
    async def wrapped(**kwargs):
        return await func(load=kwargs.get('load'))
    return wrapped


def forwarding_with_duplicate_keywords(func):
    @wraps(func)
    async def wrapped(**kwargs):
        return await func(load=kwargs.get('load'), **kwargs)
    return wrapped


def fixed_load_wrapper(func):
    @wraps(func)
    async def wrapped(**kwargs):
        return await func(load=Model.relation)
    return wrapped
'''


class TestForwardingRuntimeDifferential:
    """The forwarding criterion must agree with what the wrapper really passes."""

    @pytest.mark.asyncio
    async def test_valid_forwarding_credit_matches_real_call(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _FORWARDING_BODY)
        wrapped = module.forwarding_without_duplicate_keywords(module.dependency)
        bound = partial(wrapped, load=module.Model.relation)

        actual = await bound()
        extracted = _loads(checker, bound)

        assert actual is module.Model.relation
        assert extracted == {'relation'}

    @pytest.mark.asyncio
    async def test_inner_partial_is_overridden_by_forwarded_none(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _FORWARDING_BODY)
        inner = partial(module.dependency, load=module.Model.relation)
        wrapped = module.forwarding_without_duplicate_keywords(inner)

        actual = await wrapped()
        extracted = _loads(checker, wrapped)

        assert actual is None
        assert extracted == set(), (
            "the wrapper really passes None, overriding the inner partial's binding"
        )

    @pytest.mark.asyncio
    async def test_nested_forwarding_preserves_outer_partial_value(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _FORWARDING_BODY)
        wrapped_once = module.forwarding_without_duplicate_keywords(module.dependency)
        wrapped_twice = module.forwarding_without_duplicate_keywords(wrapped_once)
        bound = partial(wrapped_twice, load=module.Model.relation)

        actual = await bound()
        extracted = _loads(checker, bound)

        assert actual is module.Model.relation
        assert extracted == {'relation'}

    @pytest.mark.asyncio
    async def test_nested_forwarding_kills_all_deeper_partial_values(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _FORWARDING_BODY)
        inner = partial(module.dependency, load=module.Model.relation)
        wrapped_once = module.forwarding_without_duplicate_keywords(inner)
        wrapped_twice = module.forwarding_without_duplicate_keywords(wrapped_once)

        actual = await wrapped_twice()
        extracted = _loads(checker, wrapped_twice)

        assert actual is None
        assert extracted == set()

    @pytest.mark.asyncio
    async def test_deeper_fixed_load_can_override_forwarded_none(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _FORWARDING_BODY)
        fixed = module.fixed_load_wrapper(module.dependency)
        wrapped = module.forwarding_without_duplicate_keywords(fixed)

        actual = await wrapped()
        extracted = _loads(checker, wrapped)

        assert actual is module.Model.relation
        assert extracted == {'relation'}

    @pytest.mark.asyncio
    async def test_duplicate_keyword_fixture_is_not_a_valid_runtime_witness(
        self, tmp_path: Path,
    ) -> None:
        module = _specimen(tmp_path, _FORWARDING_BODY)
        wrapped = module.forwarding_with_duplicate_keywords(module.dependency)
        bound = partial(wrapped, load=module.Model.relation)

        with pytest.raises(TypeError, match="multiple values for keyword argument 'load'"):
            _ = await bound()


# ==================== mutation / aliasing of the received load ====================

_MUTATION_BODY = '''
async def dependency(*, load=None):
    return load


def clearing_wrapper(func):
    @wraps(func)
    async def wrapped(*, load=None):
        load.clear()
        return await func(load=load)
    return wrapped


def overwriting_kwargs_wrapper(func):
    @wraps(func)
    async def wrapped(**kwargs):
        kwargs['load'] = None
        return await func(load=kwargs.get('load'))
    return wrapped


def alias_clearing_wrapper(func):
    @wraps(func)
    async def wrapped(*, load=None):
        alias = load
        alias.clear()
        return await func(load=load)
    return wrapped


def branch_alias_wrapper(func):
    @wraps(func)
    async def wrapped(*, load=None, use_other=False):
        other = []
        if use_other:
            alias = other
        else:
            alias = load
        alias.clear()
        return await func(load=load)
    return wrapped
'''


class TestLoadMutation:
    """A ``load`` mutated before forwarding no longer carries the caller's value."""

    @pytest.mark.asyncio
    async def test_inplace_mutation_does_not_credit_outer_partial_value(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MUTATION_BODY)
        bound = partial(module.clearing_wrapper(module.dependency), load=[module.Model.relation])

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual == []
        assert extracted == set(), "the wrapper clears load in place before forwarding it"

    @pytest.mark.asyncio
    async def test_varkw_overwrite_is_not_forwarding(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MUTATION_BODY)
        bound = partial(
            module.overwriting_kwargs_wrapper(module.dependency), load=module.Model.relation,
        )

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual is None
        assert extracted == set(), (
            "the wrapper overwrote **kwargs; kwargs.get('load') is no longer transparent"
        )

    @pytest.mark.asyncio
    async def test_alias_mutation_does_not_credit_original_name(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MUTATION_BODY)
        bound = partial(
            module.alias_clearing_wrapper(module.dependency), load=[module.Model.relation],
        )

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual == []
        assert extracted == set(), "alias and load share one list; alias.clear() empties load"

    @pytest.mark.asyncio
    async def test_branch_alias_sources_do_not_overwrite_each_other(
        self, tmp_path: Path, checker: RelationLoadChecker,
    ) -> None:
        module = _specimen(tmp_path, _MUTATION_BODY)
        bound = partial(
            module.branch_alias_wrapper(module.dependency),
            load=[module.Model.relation], use_other=False,
        )

        extracted = _loads(checker, bound)
        actual = await bound()

        assert actual == []
        assert extracted == set(), (
            "the executed branch aliases load and clears it; the other branch's alias "
            "source must not hide this edge"
        )


# ======================= nested scopes and local rebinding =======================

_SCOPE_BODY = '''
async def fetch(*, load: object = None) -> Model:
    raise RuntimeError('static analysis specimen, never executed')


async def dependency(*, load: object = None) -> Model:
    return await fetch(load=load)


def wrapper_with_dead_nested_load(func):
    @wraps(func)
    async def wrapped() -> Model:
        async def never_called() -> Model:
            return await fetch(load=Model.relation)

        _ = never_called
        return await func(load=None)
    return wrapped


def wrapper_overwriting_local_load(func):
    @wraps(func)
    async def wrapped() -> Model:
        load = []
        return await func(load=load)
    return wrapped


def wrapper_rebinding_parameter_with_nested_def(func):
    @wraps(func)
    async def wrapped(*, load=None) -> Model:
        def load():
            return None

        return await func(load=load)
    return wrapped


def wrapper_rebinding_parameter_with_unpack(func):
    @wraps(func)
    async def wrapped(*, load=None) -> Model:
        load, other = [], None
        return await func(load=load)
    return wrapped


def wrapper_reading_load_in_comprehension(func):
    @wraps(func)
    async def wrapped(*, load=None) -> Model:
        _ = [load for item in ()]
        return await func(load=load)
    return wrapped


def wrapper_rebinding_with_import(func):
    @wraps(func)
    async def wrapped(*, load=None) -> Model:
        import math as load
        return await func(load=load)
    return wrapped


GLOBAL_LOAD = Model.relation


def wrapper_rebinding_global(func):
    @wraps(func)
    async def wrapped() -> Model:
        global GLOBAL_LOAD
        GLOBAL_LOAD = []
        return await func(load=GLOBAL_LOAD)
    return wrapped


def wrapper_rebinding_nonlocal(func):
    load = Model.relation

    @wraps(func)
    async def wrapped() -> Model:
        nonlocal load
        load = []
        return await func(load=load)
    return wrapped


def same_name_wrappers(func):
    async def wrapped(*, load=None) -> Model:
        load = []
        return await func(load=load)

    rebound = wrapped

    async def wrapped(*, load=None) -> Model:
        return await func(load=load)

    transparent = wrapped
    return rebound, transparent
'''


class TestNestedAndLocalLoadScope:
    """Per-layer scanning must ignore dead nested bodies and locally rebound ``load``."""

    def test_unexecuted_nested_function_load_does_not_count(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        inner = partial(module.dependency, load=module.Model.relation)
        wrapped = module.wrapper_with_dead_nested_load(inner)
        assert _loads(_bare_checker(), wrapped) == set()

    def test_overwritten_local_load_does_not_forward_outer_binding(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        inner = partial(module.dependency, load=module.Model.relation)
        wrapped = module.wrapper_overwriting_local_load(inner)
        assert _loads(_bare_checker(), wrapped) == set()

    def test_nested_def_rebinding_shadows_bound_parameter(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        wrapped = module.wrapper_rebinding_parameter_with_nested_def(module.dependency)
        assert _loads(_bare_checker(), partial(wrapped, load=module.Model.relation)) == set()

    def test_unpacking_rebinding_shadows_bound_parameter(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        wrapped = module.wrapper_rebinding_parameter_with_unpack(module.dependency)
        assert _loads(_bare_checker(), partial(wrapped, load=module.Model.relation)) == set()

    def test_comprehension_read_does_not_mark_parameter_rebound(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        wrapped = module.wrapper_reading_load_in_comprehension(module.dependency)
        assert _loads(_bare_checker(), partial(wrapped, load=module.Model.relation)) == {
            'relation',
        }

    def test_import_rebinding_shadows_partial_load(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        wrapped = module.wrapper_rebinding_with_import(module.dependency)
        assert _loads(_bare_checker(), partial(wrapped, load=module.Model.relation)) == set()

    def test_global_assignment_shadows_runtime_global_relation(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        wrapped = module.wrapper_rebinding_global(module.dependency)
        assert _loads(_bare_checker(), wrapped) == set()

    def test_nonlocal_assignment_shadows_runtime_closure_relation(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        wrapped = module.wrapper_rebinding_nonlocal(module.dependency)
        assert _loads(_bare_checker(), wrapped) == set()

    def test_same_name_sibling_scopes_are_selected_by_code_block(self, tmp_path: Path) -> None:
        module = _specimen(tmp_path, _SCOPE_BODY)
        checker = _bare_checker()
        rebound, transparent = module.same_name_wrappers(module.dependency)

        assert _loads(checker, partial(rebound, load=module.Model.relation)) == set()
        assert _loads(checker, partial(transparent, load=module.Model.relation)) == {'relation'}


# ============================ runtime load wrappers =============================

_WRAPPERS_SOURCE = '''
from functools import wraps

from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class OrmBase(DeclarativeBase):
    pass


class RelatedR(OrmBase):
    __tablename__ = 'rlc_lr_related_r'
    id: Mapped[int] = mapped_column(primary_key=True)


class RelatedS(OrmBase):
    __tablename__ = 'rlc_lr_related_s'
    id: Mapped[int] = mapped_column(primary_key=True)


class Model(OrmBase):
    __tablename__ = 'rlc_lr_wrapper_model'
    id: Mapped[int] = mapped_column(primary_key=True)
    related_r_id: Mapped[int] = mapped_column(ForeignKey(RelatedR.id))
    related_s_id: Mapped[int] = mapped_column(ForeignKey(RelatedS.id))
    r: Mapped[RelatedR] = relationship()
    s: Mapped[RelatedS] = relationship()


async def fetch(*, load: object = None) -> Model:
    raise RuntimeError('static analysis specimen, never executed')


async def dependency(*, load: object = None) -> Model:
    return await fetch(load=load)


def transparent_wrapper(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapped


def injecting_wrapper(func, *, load):
    @wraps(func)
    async def wrapped() -> Model:
        return await fetch(load=load)
    return wrapped


def disabling_wrapper(func):
    @wraps(func)
    async def wrapped() -> Model:
        return await fetch(load=None)
    return wrapped


def forwarding_load_wrapper(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        return await func(*args, load=kwargs.get('load'), **kwargs)
    return wrapped


def transform_to_empty_wrapper(func):
    def choose_load(value):
        return []

    @wraps(func)
    async def wrapped(*args, **kwargs):
        return await func(*args, load=choose_load(kwargs.get('load')), **kwargs)
    return wrapped


class CallableDependency:
    def __init__(self, load):
        self.load = load

    async def __call__(self) -> Model:
        return await fetch(load=self.load)
'''


@pytest.fixture
def wrappers_module(tmp_path: Path) -> ModuleType:
    return _import_module(
        tmp_path, f'rlc_lr_wrappers_{tmp_path.name.replace("-", "_")}', _WRAPPERS_SOURCE,
    )


class TestRuntimeLoadWrappers:
    """Runtime resolution must keep values bound by callable wrappers."""

    def test_partial_bound_load_is_not_lost(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        assert _loads(checker, partial(m.dependency, load=m.Model.r)) == {'r'}

    def test_outer_partial_overrides_inner_partial(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        inner = partial(m.dependency, load=m.Model.r)
        assert _loads(checker, partial(inner, load=m.Model.s)) == {'s'}

    def test_transparent_wraps_outside_partial_preserves_bound_load(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        bound = partial(m.dependency, load=m.Model.r)
        assert _loads(checker, m.transparent_wrapper(bound)) == {'r'}

    def test_partial_outside_transparent_wraps_preserves_outer_override(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        wrapped = m.transparent_wrapper(m.dependency)
        assert _loads(checker, partial(wrapped, load=m.Model.s)) == {'s'}

    def test_wraps_wrapper_owned_closure_is_not_skipped(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        assert _loads(checker, m.injecting_wrapper(m.dependency, load=m.Model.s)) == {'s'}

    def test_explicit_load_none_stops_before_inner_preload(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        inner = partial(m.dependency, load=m.Model.r)
        assert _loads(checker, m.disabling_wrapper(inner)) == set()

    def test_forwarding_wrapper_uses_outer_partial_binding(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        wrapped = m.forwarding_load_wrapper(m.dependency)
        assert _loads(checker, partial(wrapped, load=m.Model.s)) == {'s'}

    def test_transforming_wrapper_returning_empty_does_not_credit_inner_load(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        """A computed value may be a transformation, not forwarding."""
        m = wrappers_module
        inner = partial(m.dependency, load=m.Model.r)
        assert _loads(checker, m.transform_to_empty_wrapper(inner)) == set()

    def test_wrapper_cycle_terminates_without_false_load(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        first = m.transparent_wrapper(m.dependency)
        second = m.transparent_wrapper(first)
        first.__wrapped__ = second
        assert _loads(checker, first) == set()

    def test_wrapper_chain_beyond_depth_limit_fails_safe(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        m = wrappers_module
        wrapped: Any = partial(m.dependency, load=m.Model.r)
        for _ in range(33):
            wrapped = m.transparent_wrapper(wrapped)
        assert _loads(checker, wrapped) == set()

    def test_callable_instance_without_code_fails_safe(
        self, wrappers_module: ModuleType, checker: RelationLoadChecker,
    ) -> None:
        """An instance without ``__code__`` is unresolvable: keep reporting, never credit."""
        m = wrappers_module
        assert _loads(checker, m.CallableDependency(m.Model.r)) == set()


# ============ runtime namespace resolution (closures / cross-module) ============

def _load_runtime_probe_modules(tmp_path: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    token = tmp_path.name.replace('-', '_')
    models_name = f'rlc_lr_runtime_models_{token}'
    constants_name = f'rlc_lr_runtime_constants_{token}'
    endpoints_name = f'rlc_lr_runtime_endpoints_{token}'
    models = _import_module(tmp_path, models_name, textwrap.dedent('''
        from sqlalchemy import ForeignKey
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

        from sqlmodel_ext import SQLModelBase


        class OrmBase(DeclarativeBase):
            pass


        class Related(OrmBase):
            __tablename__ = 'rlc_lr_probe_related'
            id: Mapped[int] = mapped_column(primary_key=True)


        class SpecialRelated(OrmBase):
            __tablename__ = 'rlc_lr_probe_special_related'
            id: Mapped[int] = mapped_column(primary_key=True)


        class Required(OrmBase):
            __tablename__ = 'rlc_lr_probe_required'
            id: Mapped[int] = mapped_column(primary_key=True)
            related_id: Mapped[int] = mapped_column(ForeignKey(Related.id))
            r: Mapped[Related] = relationship()


        class Actual(Required):
            special_related_id: Mapped[int | None] = mapped_column(ForeignKey(SpecialRelated.id))
            special: Mapped[SpecialRelated | None] = relationship()


        class Unrelated(OrmBase):
            __tablename__ = 'rlc_lr_probe_unrelated'
            id: Mapped[int] = mapped_column(primary_key=True)
            related_id: Mapped[int] = mapped_column(ForeignKey(Related.id))
            r: Mapped[Related] = relationship()


        class RequiredDTO(SQLModelBase):
            r: int


        async def fetch(*, load: object = None) -> Required:
            raise RuntimeError('static analysis specimen, never executed')


        def require_required(*, load: object = None):
            async def checker() -> Actual:
                return await fetch(load=load)
            return checker


        GLOBAL_LOAD = Required.r


        async def local_name_collision(GLOBAL_LOAD: object) -> Actual:
            return await fetch(load=GLOBAL_LOAD)


        async def non_relation_value() -> Actual:
            garbage = 'r'
            return await fetch(load=garbage)
    '''))
    constants = _import_module(tmp_path, constants_name, textwrap.dedent(f'''
        from {models_name} import Required

        CROSS_MODULE_LOAD = [Required.r]
        EMPTY_CROSS_MODULE_LOAD = []
    '''))
    endpoints = _import_module(tmp_path, endpoints_name, textwrap.dedent(f'''
        from typing import Annotated

        from fastapi import Depends

        from {constants_name} import CROSS_MODULE_LOAD, EMPTY_CROSS_MODULE_LOAD
        from {models_name} import Actual, RequiredDTO, fetch, require_required

        LoadedDep = Annotated[Actual, Depends(require_required(load=Actual.r))]
        MissingDep = Annotated[Actual, Depends(require_required())]


        async def closure_loaded(item: LoadedDep) -> RequiredDTO:
            return RequiredDTO(r=1)


        async def closure_missing(item: MissingDep) -> RequiredDTO:
            return RequiredDTO(r=1)


        async def cross_module_loaded() -> RequiredDTO:
            return await fetch(load=CROSS_MODULE_LOAD)


        async def cross_module_missing() -> RequiredDTO:
            return await fetch(load=EMPTY_CROSS_MODULE_LOAD)
    '''))
    return models, constants, endpoints


def _probe_checker(models: ModuleType) -> RelationLoadChecker:
    """Fresh checker with the specimen mappers registered in its knowledge base."""
    probe = RelationLoadChecker(SQLModelBase)
    probe.model_classes.update({
        'Required': models.Required,
        'Actual': models.Actual,
        'Unrelated': models.Unrelated,
    })
    probe.model_relationships.update({
        'Required': {'r'},
        'Actual': {'r', 'special'},
        'Unrelated': {'r'},
    })
    probe.model_columns.update({
        'Required': {'id', 'related_id'},
        'Actual': {'id', 'related_id', 'special_related_id'},
        'Unrelated': {'id', 'related_id'},
    })
    return probe


def _rlc005_for_endpoint(
    probe: RelationLoadChecker,
    endpoint: Any,
    response_model: type[SQLModelBase],
    monkeypatch: pytest.MonkeyPatch,
) -> list[RelationLoadWarning]:
    """Run ``check_app`` on a one-route app and keep only RLC005.

    The response DTO is pinned to require ``Required.r``: the specimen DTO cannot
    have the SQLAlchemy table class in its MRO, which is how this library anchors a
    response model to its table model.
    """
    # FastAPI's built-in routes (/openapi.json, /docs, ...) have no response model.
    monkeypatch.setattr(
        probe,
        '_get_response_model_relationships',
        lambda rm: {'r': 'Required'} if rm is response_model else {},
    )
    app = FastAPI()
    _ = app.get('/probe', response_model=response_model)(endpoint)
    return [w for w in probe.check_app(app) if w.code == 'RLC005']


def _analyzer_with_loads(
    probe: RelationLoadChecker,
    *,
    endpoint_loads: set[str] | None = None,
    tracked_model: str | None = None,
    tracked_rels: set[str] | None = None,
) -> _FunctionAnalyzer:
    param_models = {'item': tracked_model} if tracked_model is not None else {}
    dep_loads = {'item': tracked_rels or set()} if tracked_model is not None else {}
    analyzer = _FunctionAnalyzer(
        model_relationships=probe.model_relationships,
        model_columns=probe.model_columns,
        param_models=param_models,
        dep_loads=dep_loads,
        required_rels={},
        source_file='<test>',
        line_offset=0,
        path='<test>',
        caller_provided_params=set(),
    )
    analyzer.all_loaded_rel_names = endpoint_loads or set()
    return analyzer


class TestRuntimeLoadResolution:
    """The runtime-namespace fallback must hold both the positive and the missing variant."""

    def test_factory_closure_load_is_recognized_and_missing_load_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        models, _, endpoints = _load_runtime_probe_modules(tmp_path)
        probe = _probe_checker(models)

        assert _rlc005_for_endpoint(
            probe, endpoints.closure_loaded, models.RequiredDTO, monkeypatch,
        ) == []
        warnings = _rlc005_for_endpoint(
            probe, endpoints.closure_missing, models.RequiredDTO, monkeypatch,
        )
        assert len(warnings) == 1
        assert 'Required.r' in warnings[0].message

    def test_cross_module_constant_is_recognized_and_empty_constant_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        models, _, endpoints = _load_runtime_probe_modules(tmp_path)
        probe = _probe_checker(models)

        assert _rlc005_for_endpoint(
            probe, endpoints.cross_module_loaded, models.RequiredDTO, monkeypatch,
        ) == []
        warnings = _rlc005_for_endpoint(
            probe, endpoints.cross_module_missing, models.RequiredDTO, monkeypatch,
        )
        assert len(warnings) == 1
        assert 'Required.r' in warnings[0].message

    def test_local_name_does_not_fall_back_to_same_named_global(self, tmp_path: Path) -> None:
        models, _, _ = _load_runtime_probe_modules(tmp_path)
        probe = _probe_checker(models)

        assert _loads(probe, models.local_name_collision) == set(), (
            "a parameter resolved as the same-named global would suppress a real miss"
        )

    def test_non_queryable_runtime_value_does_not_count_as_load(self, tmp_path: Path) -> None:
        models, _, _ = _load_runtime_probe_modules(tmp_path)
        probe = _probe_checker(models)

        assert _loads(probe, models.non_relation_value) == set()


class TestModelSatisfactionDirection:
    """STI relaxation only goes "concrete subclass satisfies base requirement"."""

    def test_subclass_satisfies_base_but_reverse_and_unrelated_do_not(
        self, tmp_path: Path,
    ) -> None:
        models, _, _ = _load_runtime_probe_modules(tmp_path)
        probe = _probe_checker(models)

        assert probe._model_satisfies('Actual', 'Required')  # pyright: ignore[reportPrivateUsage]
        assert not probe._model_satisfies('Required', 'Actual')  # pyright: ignore[reportPrivateUsage]
        assert not probe._model_satisfies('Unrelated', 'Required')  # pyright: ignore[reportPrivateUsage]

    def test_tracked_subclass_load_satisfies_base_requirement(self, tmp_path: Path) -> None:
        models, _, endpoints = _load_runtime_probe_modules(tmp_path)
        probe = _probe_checker(models)
        warnings: list[RelationLoadWarning] = []
        analyzer = _analyzer_with_loads(probe, tracked_model='Actual', tracked_rels={'r'})

        probe._check_rlc005(  # pyright: ignore[reportPrivateUsage]
            warnings, {'r': 'Required'}, {}, {}, analyzer,
            endpoints.closure_loaded, '/tracked',
        )

        assert warnings == []

    def test_base_load_does_not_satisfy_subclass_specific_requirement(
        self, tmp_path: Path,
    ) -> None:
        models, _, endpoints = _load_runtime_probe_modules(tmp_path)
        probe = _probe_checker(models)
        warnings: list[RelationLoadWarning] = []
        analyzer = _analyzer_with_loads(probe, tracked_model='Required', tracked_rels={'special'})

        probe._check_rlc005(  # pyright: ignore[reportPrivateUsage]
            warnings, {'special': 'Actual'}, {}, {}, analyzer,
            endpoints.closure_loaded, '/reverse',
        )

        assert len(warnings) == 1
        assert 'Actual.special' in warnings[0].message


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known RLC005 false negative: all_loaded_rel_names is keyed by relationship "
        "name only, so a same-named relationship on an unrelated model satisfies it"
    ),
)
def test_same_named_relation_on_unrelated_model_does_not_satisfy_requirement(
    tmp_path: Path,
) -> None:
    """Known gap: loading ``Unrelated.r`` must not satisfy ``Required.r``."""
    models, _, endpoints = _load_runtime_probe_modules(tmp_path)
    probe = _probe_checker(models)
    warnings: list[RelationLoadWarning] = []
    analyzer = _analyzer_with_loads(probe, endpoint_loads={'r'})

    probe._check_rlc005(  # pyright: ignore[reportPrivateUsage]
        warnings, {'r': 'Required'}, {}, {}, analyzer,
        endpoints.closure_loaded, '/same-name-hole',
    )

    assert len(warnings) == 1


# ================= constructor: per-candidate discovery isolation =================

_SourceObject: TypeAlias = (
    ModuleType | type | MethodType | FunctionType | TracebackType | FrameType | CodeType
)


class _HostileAnnotationsCallable:
    async def __call__(self, session: AsyncSession) -> None:
        await session.rollback()

    def __getattribute__(self, name: str) -> object:
        if name == '__annotations__':
            raise RuntimeError("hostile callable refuses __annotations__")
        return super().__getattribute__(name)


@pytest.fixture
def callable_pipeline_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """A fake project module whose classes are scanned by non-model commit discovery."""
    tests_dir = Path(__file__).resolve().parent
    monkeypatch.setattr(rlc_module, '_PROJECT_ROOT', str(tests_dir.parent))

    module_name = 'rlc_lr_callable_pipeline_module'
    module = ModuleType(module_name)
    module.__file__ = str(tests_dir / f'{module_name}.py')
    hostile_annotations = _HostileAnnotationsCallable()
    inspect.markcoroutinefunction(hostile_annotations)

    class AHostileAnnotationsWorker:
        hostile_method = hostile_annotations

    class BHostileSourceWorker:
        async def hostile_source_method(self, session: AsyncSession) -> None:
            await session.rollback()

    class ZDiscoverableWorker:
        async def rlc_lr_pipeline_rollback_probe(self, session: AsyncSession) -> None:
            await session.rollback()

    for cls in (AHostileAnnotationsWorker, BHostileSourceWorker, ZDiscoverableWorker):
        cls.__module__ = module_name
        module.__dict__[cls.__name__] = cls
    sys.modules[module_name] = module
    try:
        yield module
    finally:
        del sys.modules[module_name]


def test_constructor_isolates_annotations_and_source_failures_per_candidate(
    callable_pipeline_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One hostile candidate must not abort discovery of the remaining ones."""
    source_worker = callable_pipeline_module.__dict__['BHostileSourceWorker']
    hostile_source = vars(source_worker)['hostile_source_method']
    original_getsource = rlc_module.python_inspect.getsource

    def hostile_getsource(candidate: _SourceObject) -> str:
        if candidate is hostile_source:
            raise RuntimeError("hostile callable refuses source introspection")
        return original_getsource(candidate)

    monkeypatch.setattr(rlc_module.python_inspect, 'getsource', hostile_getsource)

    fresh = RelationLoadChecker(SQLModelBase)

    assert 'rlc_lr_pipeline_rollback_probe' in fresh.commit_methods
