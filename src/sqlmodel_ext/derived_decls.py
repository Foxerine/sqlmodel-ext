"""
**Experimental.** Expand ``partial=True`` classes into a throwaway copy of the source tree so a type checker can see them.

This module is the "derive" step Python does not have. It pairs with the
metaclass keyword ``partial=True`` and is driven by the
:mod:`sqlmodel_ext.check_derived` command line tool. It is experimental: the
API may change in any release, and it has an exit plan (see below).

**The problem**: ``partial=True`` rewrites fields **at runtime** (``T`` becomes
``Unset | T = Unset``), while basedpyright only analyses **source code**. To the
checker a derived DTO still has the base annotations, so
``class ArticleUpdate(ArticleBase, partial=True)`` has ``title: str``
statically although ``title`` may be :data:`~sqlmodel_ext.unset.Unset` at
runtime. A consumer that writes ``if dto.title is not None: dto.title.upper()``
(the correct tri-state check is ``is not Unset``) is silently wrong, and no
tool says so.

This is a language boundary, not a configuration problem: Python has neither
compile-time derive macros (Rust ``#[derive]``) nor type-level mapped types
(TypeScript ``Partial<T>``). `PEP 827 <https://peps.python.org/pep-0827/>`_
(type manipulation, draft) proposes the missing piece.

**Exit plan**: once PEP 827 (or an equivalent) is accepted and supported by
basedpyright, a PATCH DTO can be spelled as a type-level transformation of its
base, the checker sees the tri-state directly, and this module together with
:mod:`sqlmodel_ext.check_derived` will be deprecated and then removed. Nothing
it produces is ever committed, so removal leaves nothing behind.

**This module never writes into the working tree**: everything it writes goes
into a disposable copy that the caller creates and deletes. There are no
generated files in the repository, nothing to review and nothing to clean up.
(Reading runtime facts means the caller has imported the project's modules,
so their own import-time side effects are outside this guarantee -- see
:mod:`sqlmodel_ext.check_derived`, invariant 1.)

**Two things are expanded, and both are required**:

1. **Field declarations**: every field the metaclass turned into ``Unset | T``
   gets a declaration ``name: Unset | <original annotation text>`` inside an
   ``if TYPE_CHECKING:`` block of the derived class, so consumers see the
   tri-state.
2. **Methods expanded along the MRO**: every member the class *inherits* and
   that reads a tri-state field (``self.<field>``) is copied into the derived
   class. Its ``self`` then has the derived type, and an insufficient guard is
   reported by basedpyright natively. The same code is checked differently
   depending only on where it is defined::

       class ArticleBase(SQLModelBase):
           subtitle: Str256 | None = None

           @model_validator(mode='after')
           def _check(self) -> Self:
               if self.subtitle is not None:     # self is ArticleBase: correct guard
                   _ = self.subtitle.upper()
               return self

       class ArticleUpdate(ArticleBase, partial=True):
           pass                                  # subtitle is Unset | Str256 | None at runtime,
                                                 # the inherited validator runs on construction,
                                                 # and the guard above is no longer enough

   Left in the base class this reports nothing; copied into
   ``ArticleUpdate`` it reports ``Cannot access attribute "upper" for class
   "MISSING"`` (``MISSING`` is the object :data:`~sqlmodel_ext.unset.Unset`
   refers to).

**Why a type checker instead of a heuristic scan**: the check is about how the
value is *used* afterwards, not about how the guard is *written*. Whether the
guard is ``is None``, a truthiness test, ``in (None, ...)`` or missing
altogether, any later use of the value is caught. A heuristic has to
enumerate dangerous guard shapes, which is an open set; the type checker only
needs to know that the value may be ``MISSING``.

**Known blind spots**:

- ``getattr(self, 'x', None)`` and other string-based attribute access: the
  checker cannot infer the attribute from a string literal.
- Anything the checker cannot see in general: code outside the checked
  ``include`` set, ``Any``-typed values, ``# pyright: ignore`` comments.
- Classes whose source file lies outside the project root, and members
  inherited from sqlmodel-ext / SQLModel / Pydantic themselves, are not
  expanded.
- A member is expanded only if its source mentions ``self.<field>`` for a
  tri-state field; access through another name (``other = self; other.x``)
  is not detected.

**Scope**: this module only collects and expands. What ``partial=True`` turns
a field into at runtime is decided by the metaclass (the single source of
truth); this module follows it. Copying the tree, running basedpyright and
comparing against a baseline is done by :mod:`sqlmodel_ext.check_derived`.

**Invariants**:

1. **No file outside the destination directory is ever written.** The only
   write is :meth:`Expander.write_into`, whose ``root`` is chosen by the
   caller and must not lie inside the project (checked there). Violating this
   would silently modify the user's source.
2. **What gets expanded is decided by runtime facts**, never by re-deriving
   the metaclass rules: a field is expanded iff its runtime annotation in
   ``model_fields`` accepts :data:`~sqlmodel_ext.unset.Unset`; a member is
   expanded iff the class does not define it, an ancestor on the MRO does,
   and it reads a tri-state field. Violating this lets this module and the
   metaclass drift apart while the output still looks plausible.
3. **Annotation text is copied verbatim from the ancestor's source**, never
   reconstructed from the runtime type object -- that would expand
   ``Str64`` into ``Annotated[str, MaxLen(64), ...]`` and lose the alias,
   which is the single source of truth for the constraint.
"""
import ast
import builtins
import collections
import functools
import inspect
import pathlib
import re
import textwrap
import typing
from dataclasses import dataclass, field

from pydantic import BaseModel

from sqlmodel_ext.base import optional_dto_registry
from sqlmodel_ext.unset import Unset

HEADER: typing.Final = (
    '# --- sqlmodel-ext check_derived: expanded in a throwaway copy, never committed ---'
)
"""Comment line written above every generated block; only for humans reading a kept copy (``--keep``)."""

GENERATOR_ARTIFACT_RULES: typing.Final = frozenset({
    'reportIncompatibleVariableOverride',
    'reportIncompatibleMethodOverride',
})
"""
Diagnostic rules that, **inside generated code**, describe the expansion itself rather than the checked code.

Re-declaring ``title: Unset | Str64`` in a subclass widens the base type
(``Str64``), and a copied method shadows the base's decorated member -- both
are overrides by construction. Outside generated ranges these rules are
reported normally.
"""

SKIPPED_PACKAGES: typing.Final = frozenset({'sqlmodel_ext', 'sqlmodel', 'pydantic'})
"""Top-level packages whose members are never expanded: they are the machinery, not user code."""


class DerivedDeclError(Exception):
    """
    A class cannot be expanded without guessing -- always raised, never skipped silently.

    Skipping silently would make "this class produced nothing" indistinguishable
    from "this class needs nothing", and the two call for opposite actions.
    """


def _accepts_unset(annotation: object) -> bool:
    """
    Whether a runtime annotation accepts :data:`Unset`, i.e. the metaclass really made it tri-state.

    The criterion is the runtime fact, not a re-implementation of the
    metaclass's skip rules (``Literal`` discriminators, fields the class
    declares itself, ...) -- that would be a second implementation drifting
    from the first.

    Descends into ``Annotated`` and nested unions: a constrained field is
    wrapped in ``Annotated`` and a one-level ``get_args`` would miss it.
    """
    pending: list[object] = [annotation]
    while pending:
        node = pending.pop()
        if node is Unset:
            return True
        pending.extend(typing.get_args(node))
    return False


def _names_in_annotation(node: ast.expr | None) -> set[str]:
    """
    Names referenced by an annotation expression, **including string forward references**.

    ``'Foo | None'`` is an ``ast.Constant``, not an ``ast.Name``; basedpyright
    does evaluate it, so collecting only ``Name`` nodes would miss the import
    and the expanded copy would report ``"Foo" is not defined``.

    Only used in annotation positions: an ordinary string such as
    ``"session"`` parses as a name too, and would produce a bogus import.
    """
    if node is None:
        return set()
    names: set[str] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name):
            names.add(inner.id)
        elif isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            try:
                names |= _names_in_annotation(ast.parse(inner.value, mode='eval').body)
            except SyntaxError:
                continue  # a non-type string in annotation position (e.g. Annotated metadata)
    return names


def _accepts_unset_in_text(annotation_text: str) -> bool:
    """
    Whether the source annotation already spells the tri-state itself.

    A field the author wrote as ``Unset | UUID = Unset`` is already honest in
    source, so the checker already sees it; generating another declaration
    would only add noise. The fields that need a declaration are exactly the
    ones that are tri-state at runtime but not in source.

    Uses the AST rather than ``'Unset' in text``, which would match names such
    as ``UnsetSomething`` and string literals.
    """
    try:
        node = ast.parse(annotation_text, mode='eval').body
    except SyntaxError:
        return False
    return 'Unset' in _names_in_annotation(node)


def _source_of(cls: type) -> str | None:
    """Absolute path of the class's source file; ``None`` for builtins and dynamically created classes."""
    try:
        return inspect.getsourcefile(cls)
    except TypeError:
        return None  # ``object`` and other builtins -- always reached when walking an MRO


@functools.cache
def _parse(path: str) -> ast.Module:
    """Parse a source file once per process."""
    return ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))


def _unwrap(member: object) -> object:
    """
    Peel descriptor and decorator wrappers until :func:`inspect.getsource` accepts the object.

    Layers are peeled repeatedly: a validator may be a ``classmethod`` around a
    function, and stopping after one layer hands ``getsource`` a
    ``classmethod`` object (``TypeError``).

    ``fget`` is the entry for properties: a ``property`` object is not
    callable, so filtering members with ``callable`` would drop every property,
    although a property reading a tri-state field fails exactly like a method.
    """
    for attr in ('__func__', 'fget', 'func', 'wrapped', '__wrapped__'):
        inner = getattr(member, attr, None)
        if inner is not None and inner is not member:
            return _unwrap(inner)
    return member


PYDANTIC_DECORATORS: typing.Final = frozenset({
    'model_validator', 'field_validator', 'model_serializer',
    'field_serializer', 'computed_field',
})
"""
Decorators removed from copied members.

They are removed because they produce false positives, not for convenience:
``@model_validator(mode='after')`` is typed ``(Self) -> Self``. In the base
class an author may return ``-> 'ArticleBase'`` (correct there); copied into a
subclass it becomes ``(Self@ArticleUpdate) -> ArticleBase`` and the decorator
call itself fails to type-check.

Removing them loses nothing: what is checked is whether the **body** holds up
under the tri-state. The decorator only registers the function with Pydantic,
and the copy is never executed. Whether the member runs on construction is
read from ``__pydantic_decorators__`` (see :attr:`Member.runs_on_construction`),
not from the decorators in the source.

Other decorators (``@property``, ``@override``, ``@overload``) are kept: they
change what the type checker checks.
"""


def _strip_validator_decorators(source: str) -> str:
    """
    Remove the :data:`PYDANTIC_DECORATORS` lines and keep everything else verbatim.

    Lines are deleted by range instead of re-rendering with ``ast.unparse``,
    which would drop comments and reformat strings and line breaks -- a
    diagnostic in the copy could then no longer be matched to the real source.
    """
    body = textwrap.dedent(source)
    try:
        node = ast.parse(body).body[0]
    except (SyntaxError, IndexError):
        return body
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return body

    drop: set[int] = set()
    for decorator in node.decorator_list:
        root = decorator.func if isinstance(decorator, ast.Call) else decorator
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in PYDANTIC_DECORATORS:
            drop |= set(range(decorator.lineno, (decorator.end_lineno or decorator.lineno) + 1))
    if not drop:
        return body
    return '\n'.join(
        line for number, line in enumerate(body.split('\n'), start=1)
        if number not in drop
    )


def _free_names(code: str) -> set[str]:
    """
    Names the code reads but does not bind -- the ones it needs from elsewhere.

    The code is dedented first: :func:`inspect.getsource` returns a method with
    its class-body indentation, and parsing that raises ``IndentationError``,
    which would silently produce no imports at all.

    Collecting every ``ast.Name`` is wrong: parameters, locals, loop targets
    and comprehension variables are ``Name`` nodes too, and would become
    nonsensical imports. :mod:`symtable` is wrong as well: module-level bindings
    report ``is_global()``, so the expanded method's own name would be imported.

    A name that is both read and written (``x = x + 1``) counts as bound and gets
    no import. That direction is safe: a missing import shows up as
    ``reportUndefinedVariable`` right away, whereas a wrong import would change
    the meaning of the checked code.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return set()

    loaded: set[str] = set()
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
            loaded |= _names_in_annotation(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            loaded |= _names_in_annotation(node.annotation)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            bound.add(node.name)
            loaded |= _names_in_annotation(node.returns)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            bound |= {(alias.asname or alias.name).split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            bound.add(node.name)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            bound |= set(node.names)
    return loaded - bound


def _reads_any_field(source: str, fields: set[str]) -> bool:
    """Whether ``source`` contains ``self.<field>`` for one of ``fields`` (whole identifiers only)."""
    pattern = re.compile(r'\bself\.(' + '|'.join(re.escape(name) for name in sorted(fields)) + r')\b')
    return pattern.search(source) is not None


@dataclass(frozen=True)
class Member:
    """A member copied into a derived class by MRO expansion."""

    name: str
    """Attribute name."""

    source: str
    """Source of the defining class's member, dedented, Pydantic decorators removed."""

    runs_on_construction: bool
    """
    Whether it runs automatically when an instance is constructed.

    This is the **only** basis for severity:

    - ``True`` (``model_validator``, ``model_post_init``): if the body does not
      hold under the tri-state it fails at construction -- and constructing a
      PATCH DTO with a single field is its normal path.
    - ``False`` (an inherited domain method): it only fails when called. PATCH
      DTOs usually never call them, so these are latent problems: real, but not
      urgent.

    Read from ``__pydantic_decorators__.model_validators``, not from
    ``type(member)``: after collecting decorators Pydantic puts plain
    functions back into ``vars(cls)``, so a type-based test would always say
    "not a validator", which looks exactly like "has no validators".
    """


@dataclass(frozen=True)
class Expansion:
    """Everything to expand into one class."""

    model: type[BaseModel]
    """The derived class."""

    host_path: str
    """Absolute path of its source file; the same relative file is rewritten in the copy."""

    decls: tuple[tuple[str, str], ...] = ()
    """Field declarations: ``(field name, annotation text copied verbatim from the ancestor)``."""

    methods: tuple[Member, ...] = ()
    """Members expanded along the MRO."""

    imports: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    """Names the expansion references that the host file cannot resolve: ``(module, name)``."""

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to expand."""
        return not self.decls and not self.methods


@dataclass(frozen=True)
class Span:
    """Which generated member a range of lines in the copy belongs to -- used to grade diagnostics."""

    path: str
    """Path relative to the project root (POSIX separators)."""

    start: int
    """0-based first line (same base as pyright's JSON ``range.start.line``)."""

    end: int
    """0-based last line, inclusive."""

    label: str
    """Human-readable origin, e.g. ``ArticleUpdate <- _check``."""

    blocking: bool
    """Whether a diagnostic in this range is blocking -- see :attr:`Member.runs_on_construction`."""


@functools.cache
def _annotations_in(owner: type) -> dict[str, str]:
    """Annotation text the class's **own source** writes, field name -> ``ast.unparse`` of the annotation."""
    path = _source_of(owner)
    if path is None:
        return {}
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ClassDef) and node.name == owner.__name__:
            return {
                stmt.target.id: ast.unparse(stmt.annotation)
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    return {}


@functools.cache
def _resolvable_in(path: str) -> frozenset[str]:
    """Top-level names a file can resolve: imported names plus names it defines."""
    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(alias.asname or alias.name).split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
    return frozenset(names)


def _annotation_of(model: type[BaseModel], field_name: str) -> tuple[str, type] | None:
    """
    Find the field's real source annotation along the MRO: ``(annotation text, that ancestor)``.

    Walks the whole MRO, not only the direct bases: the field may come from a
    grandparent.

    The ``field_name in ancestor.model_fields`` filter is required: a mixin
    often declares ``if TYPE_CHECKING: x: T | None`` so the checker sees a
    host field. That is not a field of the mixin, yet the mixin can come
    before the real owner in the MRO -- without the filter the stale
    declaration would be copied.
    """
    for ancestor in model.__mro__:
        # The MRO also contains non-model classes (``object``, behavior mixins).
        if not issubclass(ancestor, BaseModel):
            continue
        if field_name not in ancestor.model_fields:
            continue
        text = _annotations_in(ancestor).get(field_name)
        if text is not None:
            return text, ancestor
    return None


def _names_needing_import(code: str, host_path: str, origin: type) -> frozenset[tuple[str, str]]:
    """
    Names ``code`` references that the host file cannot resolve, imported from ``origin``'s module.

    They are imported from the module of the class that defines the code: the
    name is resolvable there by construction (defined or imported), whereas
    locating its true definition would need reflection on type aliases, and
    ``Str64 = Annotated[...]`` has no ``__module__``.

    Builtins are excluded by checking the :mod:`builtins` module. ``__builtins__``
    is a ``dict`` outside ``__main__``, so ``hasattr(__builtins__, ...)`` is
    always false and would import ``bool`` / ``int`` / ``str`` from user
    modules, shadowing the builtins.
    """
    known = _resolvable_in(host_path)
    return frozenset(
        (origin.__module__, name) for name in _free_names(code) - known
        if not hasattr(builtins, name)
    )


def _inherited_members(model: type[BaseModel], tri_fields: set[str]) -> list[tuple[Member, type]]:
    """
    Members the class does not define, an ancestor does, and that read a tri-state field.

    The criterion is "reads a tri-state field", not "is a validator": domain
    methods, ``model_post_init`` and properties fail on :data:`Unset` just as
    well, and carry no validator decorator.

    Walks ``vars()`` along the MRO instead of :func:`inspect.getmembers`: the
    latter needs ``predicate=callable`` to skip fields, and ``property`` objects
    are not callable. ``vars()`` also yields the owner for free.

    Members that call ``super()`` are expanded too. ``super()`` then points
    elsewhere, but the copy is only type-checked, never executed; skipping
    them would produce false negatives, which is what this module exists to
    remove.
    """
    own = set(vars(model))
    running =set(model.__pydantic_decorators__.model_validators) | {'model_post_init'}
    seen: set[str] = set()
    out: list[tuple[Member, type]] = []
    for ancestor in model.__mro__[1:]:
        if ancestor is BaseModel or ancestor is object:
            break  # framework internals above this point
        if ancestor.__module__.split('.')[0] in SKIPPED_PACKAGES:
            continue
        for name, member in vars(ancestor).items():
            if name.startswith('__') or name in own or name in seen:
                continue
            seen.add(name)
            try:
                source = inspect.getsource(_unwrap(member))  # pyright: ignore[reportArgumentType]  # getsource raises TypeError for non-code objects, handled below
            except (TypeError, OSError):
                continue  # ClassVar, constant, or a C extension -- no source to expand
            if not _reads_any_field(source, tri_fields):
                continue
            out.append((
                Member(name, _strip_validator_decorators(source), name in running),
                ancestor,
            ))
    return out


def collect(model: type[BaseModel]) -> Expansion:
    """
    Everything to expand into ``model``.

    Reads only; writes nothing.

    :raises DerivedDeclError: the source file or a field's source annotation cannot be found
    """
    host = _source_of(model)
    if host is None:
        raise DerivedDeclError(f"{model.__qualname__}: source file not found")

    decls: list[tuple[str, str]] = []
    imports: set[tuple[str, str]] = set()
    tri_fields: set[str] = set()
    for field_name, info in model.model_fields.items():
        if not _accepts_unset(info.annotation):
            continue  # not made tri-state by the metaclass (Literal discriminator, own declaration)
        found = _annotation_of(model, field_name)
        if found is None:
            raise DerivedDeclError(
                f"{model.__qualname__}.{field_name}: no source annotation found along the MRO"
            )
        text, origin = found
        if _accepts_unset_in_text(text):
            continue  # the author wrote the tri-state by hand -- source is already honest
        decls.append((field_name, text))
        tri_fields.add(field_name)
        imports |= _names_needing_import(f'_: {text}', host, origin)

    methods: list[Member] = []
    if tri_fields:
        for member, owner in _inherited_members(model, tri_fields):
            methods.append(member)
            imports |= _names_needing_import(member.source, host, owner)

    return Expansion(
        model=model,
        host_path=host,
        decls=tuple(decls),
        methods=tuple(methods),
        imports=frozenset(imports),
    )


def _in_scope() -> list[type[BaseModel]]:
    """
    Classes to check: every registered ``partial`` class **and all of its subclasses**.

    Taking only the registry misses subclasses: the tri-state annotations are
    inherited by subclasses that do not say ``partial=True`` themselves, so
    they are not registered, yet their fields are just as tri-state.
    """
    seen: dict[int, type[BaseModel]] = {}
    pending: list[type] = list(optional_dto_registry)
    while pending:
        model = pending.pop()
        if id(model) in seen or not issubclass(model, BaseModel):
            continue
        seen[id(model)] = model
        pending.extend(model.__subclasses__())
    return list(seen.values())


def collect_all(project_root: pathlib.Path) -> tuple[list[Expansion], list[str]]:
    """
    Collect every in-scope class whose source file lies under ``project_root``.

    Classes defined elsewhere (installed packages) are not part of the copy and
    are skipped. Errors are accumulated and returned, not raised on the first
    one, so a single run shows how many classes need attention.

    Call it after importing the modules that define the models: the registry is
    filled as a side effect of class creation.

    :param project_root: the directory that will be copied
    :returns: ``(non-empty expansions, error messages)``
    """
    root = project_root.resolve()
    expansions: list[Expansion] = []
    errors: list[str] = []
    for model in _in_scope():
        path = _source_of(model)
        if path is None or not pathlib.Path(path).resolve().is_relative_to(root):
            continue
        try:
            expansion = collect(model)
        except DerivedDeclError as exc:
            errors.append(str(exc))
            continue
        if not expansion.is_empty:
            expansions.append(expansion)
    return expansions, errors


def _class_anchor(expansion: Expansion) -> tuple[int, int]:
    """
    ``(insertion line index, 0-based; class body indentation)`` -- the end of the class body.

    In a throwaway copy the block can simply be appended to the class body;
    there is no code of anyone else's to keep in place.
    """
    for node in ast.walk(_parse(expansion.host_path)):
        if isinstance(node, ast.ClassDef) and node.name == expansion.model.__name__:
            if node.end_lineno is None or not node.body:
                break
            return node.end_lineno, node.body[0].col_offset
    raise DerivedDeclError(
        f"{expansion.model.__qualname__}: class definition not found in {expansion.host_path}"
    )


def _import_anchor(host_path: str) -> int:
    """
    Line index after which generated imports may go: after the module docstring and any ``from __future__`` imports.

    Anything placed before a ``from __future__`` import is a syntax error.
    """
    anchor = 0
    for index, stmt in enumerate(_parse(host_path).body):
        is_docstring = (
            index == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
        is_future = isinstance(stmt, ast.ImportFrom) and stmt.module == '__future__'
        if not (is_docstring or is_future):
            break
        anchor = stmt.end_lineno or anchor
    return anchor


def _render_block(expansion: Expansion, indent: int) -> tuple[list[str], list[tuple[int, int, str, bool]]]:
    """
    The block appended to the class body, plus each member's line range inside the block.

    The block sits under ``if TYPE_CHECKING:``: written directly in the class
    body, basedpyright's ``dataclass_transform`` would treat the declarations as
    fields without defaults and report "fields without default values cannot
    appear after fields with default values", unrelated to the tri-state.

    Returns a list of lines (not one string): the caller does line arithmetic,
    and a multi-line string is a single element to ``list.insert``.
    """
    pad = ' ' * indent
    inner = pad + ' ' * 4
    lines = [f'{pad}if TYPE_CHECKING:\n', f'{inner}{HEADER}\n']
    for name, text in expansion.decls:
        lines.append(f'{inner}{name}: Unset | {text}\n')

    spans: list[tuple[int, int, str, bool]] = []
    if expansion.decls:
        spans.append((2, len(lines) - 1, f'{expansion.model.__qualname__} (field declarations)', True))
    for member in expansion.methods:
        body = textwrap.indent(textwrap.dedent(member.source), inner).rstrip('\n')
        start = len(lines)
        lines.extend(f'{line}\n' for line in body.split('\n'))
        spans.append((
            start, len(lines) - 1,
            f'{expansion.model.__qualname__} <- {member.name}',
            member.runs_on_construction,
        ))
    return lines, spans


def _render_imports(expansions: list[Expansion], host_path: str) -> list[str]:
    """
    Import lines for names the expansions reference and the host file cannot resolve.

    Every import uses the ``import X as X`` form, which pyright treats as an
    explicit re-export, so it does not report ``reportUnusedImport`` (most
    generated imports are only used in annotations).

    Names the host already resolves are skipped: importing them again would be
    reported as ``reportRedeclaration``, noise produced by the generator.
    """
    known = _resolvable_in(host_path)
    needed: dict[str, str] = {}
    for expansion in expansions:
        for module, name in sorted(expansion.imports):
            _ = needed.setdefault(name, module)

    statements: list[str] = []
    if 'TYPE_CHECKING' not in known:
        statements.append('from typing import TYPE_CHECKING as TYPE_CHECKING')
    if any(expansion.decls for expansion in expansions) and 'Unset' not in known:
        statements.append('from sqlmodel_ext.unset import Unset as Unset')
    for name, module in sorted(needed.items()):
        # A private name imported across modules always triggers reportPrivateUsage;
        # that is an artefact of the generator, not of the checked code.
        suffix = '  # pyright: ignore[reportPrivateUsage]' if name.startswith('_') else ''
        statements.append(f'from {module} import {name} as {name}{suffix}')
    if not statements:
        return []
    # Newlines are added here, in one place: a missing newline glues two imports
    # together, and every resulting diagnostic looks like a genuine type error.
    return [f'{line}\n' for line in (HEADER, *statements)]


def write_into(
        root: pathlib.Path,
        project_root: pathlib.Path,
        expansions: list[Expansion],
) -> list[Span]:
    """
    Write the expansions into the copy at ``root``; return the final line range of every generated member.

    ``root`` must already contain a copy of ``project_root`` (same relative
    layout). This is the only function in the module that writes files.

    All classes of one file are rewritten in one pass: rewriting class by class
    would let the second rewrite overwrite the first.

    Insertions are applied from the top of the file downwards with an
    accumulated offset, so every recorded range is final. Severity grading
    depends on these ranges; a wrong range turns a blocking finding into a
    latent one.

    :param root: destination copy; must not be ``project_root`` or lie inside it
    :param project_root: the project the copy was taken from
    :param expansions: from :func:`collect_all`
    :raises ValueError: ``root`` is inside ``project_root`` (invariant 1)
    """
    root = root.resolve()
    project_root = project_root.resolve()
    if root.is_relative_to(project_root):
        raise ValueError(f"refusing to write into the project itself: {root} is inside {project_root}")

    grouped: dict[str, list[Expansion]] = collections.defaultdict(list)
    for expansion in expansions:
        grouped[expansion.host_path].append(expansion)

    spans: list[Span] = []
    for host_path, group in grouped.items():
        relative = pathlib.Path(host_path).resolve().relative_to(project_root).as_posix()
        target = root / relative
        lines = target.read_text(encoding='utf-8').splitlines(keepends=True)

        inserts: list[tuple[int, list[str], list[tuple[int, int, str, bool]]]] = []
        for expansion in group:
            anchor, indent = _class_anchor(expansion)
            chunk, chunk_spans = _render_block(expansion, indent)
            inserts.append((anchor, chunk, chunk_spans))
        imports = _render_imports(group, host_path)
        if imports:
            inserts.append((_import_anchor(host_path), imports, []))

        offset = 0
        for anchor, chunk, chunk_spans in sorted(inserts, key=lambda item: item[0]):
            at = anchor + offset
            if at > 0 and not lines[at - 1].endswith('\n'):
                lines[at - 1] += '\n'
            lines[at:at] = chunk
            offset += len(chunk)
            spans.extend(
                Span(relative, at + start, at + end, label, blocking)
                for start, end, label, blocking in chunk_spans
            )
        _ = target.write_text(''.join(lines), encoding='utf-8', newline='\n')
    return spans
