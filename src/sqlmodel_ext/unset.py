"""
Tri-state field semantics: separating "not provided" from ``null``.

This module exports Pydantic's official
:data:`~pydantic.experimental.missing_sentinel.MISSING` under the name
:data:`Unset`, plus the *shape constants* of the optional wire protocol
(:data:`OMITTED_SENTINEL`, :data:`SENTINEL_SCHEMA_BRANCH`,
:data:`SCHEMA_ANNOTATION_KEYS`). Behavior and serialization semantics follow
Pydantic exactly -- this module does not define its own sentinel, and it does
not decide when the wire value is enabled (that is a per-model config switch,
see ``SQLModelExtConfig.omitted_sentinel``).

This module holds **data only**. Normalizing inbound payloads, injecting the
schema branch and deciding whether a field may be omitted are behaviors of the
*model*, so they live on :class:`~sqlmodel_ext.base.SQLModelBase`
(``annotation_is_omissible`` / ``field_is_omissible`` /
``_normalise_omitted_sentinel`` / ``model_json_schema``).

**The problem it solves**: ``field: T | None = None`` expresses two
incompatible intents with the same spelling:

====================================  ==========================
 What the author means                  What the code looks like
====================================  ==========================
 "may be omitted; leave it alone"      ``field: T | None = None``
 "null is a meaningful value here"     ``field: T | None = None``
====================================  ==========================

The two are indistinguishable, yet they must be handled in opposite ways.
Moving "not provided" onto a carrier that is **not** ``None`` turns ``None``
back into an ordinary value.

**Why the name ``Unset`` instead of ``MISSING``**: "missing" suggests
something that *should* be present but is absent, whereas here the caller
*chose* not to provide the field. It also appears in type position
(``Unset | T``), where an ALL_CAPS constant name reads oddly. Import
``Unset`` from this module rather than ``MISSING`` from Pydantic so that a
codebase has a single name for the concept.

**Spellings and their meaning**:

========================================  =========================================
 Annotation                                 Meaning
========================================  =========================================
 ``Unset | T = Unset``                      may be omitted; ``null`` is rejected
 ``Unset | T | None = Unset``               may be omitted; ``null`` is a real value
 ``T | None``                               must be provided; may be ``null``
 ``T = <default>``                          may be omitted, has a natural default,
                                            ``null`` is rejected
========================================  =========================================

Whether the annotation includes ``None`` should depend on whether ``null``
maps to a **real value** for the field (e.g. "clear this column" on write,
"filter rows where the column IS NULL" on read) -- not on whether the field
may be left out. Omission is carried by :data:`Unset`.

The order of union members does not change *which* inputs are accepted or
rejected, but it does change the *order* of entries in
``ValidationError.errors()``; do not rely on that order downstream.

Check for omission with ``if x is Unset:`` -- type checkers narrow the other
branch to ``T``.

.. _omitted-wire-value:

**The wire value: for callers that cannot omit keys (opt-in, per model)**

``MISSING`` assumes the caller can omit a key, so its branch never appears in
the JSON Schema. Some schema consumers, however, require every key to be
present (for example LLM function calling in strict mode). With the plain
official behavior such a caller has no way to say "leave this field alone":
the schema only offers ``T``, and ``null`` means "clear it". Omission and
clearing collapse into one.

Whether a caller can omit keys is a property of the *caller*, and callers are
distinguished by *model* (a REST body vs. a tool-call argument model), not by
field. Hence the switch is a ``model_config`` key, **off by default**, enabled
by the model families that need it. When it is on:

- inbound payloads may use :data:`OMITTED_SENTINEL` for any omissible field
  (at any nesting depth); it is normalized to :data:`Unset` before
  validation;
- ``model_json_schema()`` adds a ``const`` branch for the sentinel to every
  omissible field (including nested models in ``$defs``).

The same domain DTO therefore yields a clean schema when inherited by a REST
model, and a sentinel-aware schema when inherited by a strict-mode model.

**Invariants**:

1. Fields whose value is :data:`Unset` never appear in ``model_dump()`` /
   ``model_dump_json()`` output -- no ``exclude_unset`` required. Guaranteed
   by Pydantic's ``MISSING`` implementation.
2. The wire value only exists in JSON Schema and inbound payloads; it is
   normalized to :data:`Unset` on entry. Application code should never
   compare against ``'__omitted__'``.
3. :data:`Unset` and ``None`` are different things and never substitute for
   each other.

**Known limitations**:

- ``typing_extensions.Sentinel`` does not support ``copy.deepcopy`` natively
  (its ``__getstate__`` raises). Frameworks deep-copy defaults (FastAPI does
  so for missing query parameters), so a bare ``Unset`` default would fail at
  request time. This module registers a ``copyreg`` reducer that makes
  ``deepcopy(Unset)`` return ``Unset`` itself; see :func:`_reduce_unset`.
- Pydantic still marks the feature experimental (module
  ``pydantic.experimental.missing_sentinel``, available since Pydantic 2.12).
  It builds on PEP 661.
- Static narrowing of ``Unset | T`` requires a type checker with PEP 661
  support (e.g. pyright >= 1.1.414 / basedpyright >= 1.40.1).
- In models with the wire value enabled, string fields can no longer hold the
  literal ``'__omitted__'`` itself.
"""
import copyreg
import typing

from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import Sentinel

Unset: typing.Final = MISSING
"""
"This field was not provided" -- Pydantic's official ``MISSING`` sentinel.

Check with ``if x is Unset:``.

The ``typing.Final`` annotation is required and must not be replaced by
``TypeAlias``: the name has to work both in type position (``Unset | T``) and
in value position (``= Unset``). A bare assignment is rejected in type
position ("Variable not allowed in type expression"); ``TypeAlias`` is
rejected in value position.
"""


def _reduce_unset(sentinel: object) -> str:
    """
    Keep :data:`Unset` a singleton under ``copy.deepcopy``; other sentinels are unaffected.

    ``typing_extensions.Sentinel.__getstate__`` raises ``TypeError`` to stop a
    singleton from being cloned (a clone would break ``is`` checks). But
    ``deepcopy`` falls back to the same protocol when there is no
    ``__deepcopy__``, so even the semantically correct copy -- returning the
    object itself -- is refused. Frameworks deep-copy field defaults (FastAPI
    does it for missing query parameters), which would make ``Unset`` unusable
    as a default there.

    The reducer is registered for the ``Sentinel`` type but only acts on the
    single :data:`Unset` object; every other sentinel still raises the
    upstream ``TypeError`` (re-creating them by name would break *their*
    identity checks). Patching the ``MISSING`` instance directly would be a
    monkey-patch of an upstream object.

    Returning a ``str`` makes ``copy.deepcopy`` return the original object. A
    real ``pickle`` round-trip still fails (pickle looks the name up as a
    global in ``typing_extensions``), which intentionally preserves upstream
    behavior: sentinels should not be revived across processes.

    :param sentinel: any ``Sentinel`` instance
    :raises TypeError: if ``sentinel`` is not :data:`Unset` (same as upstream)
    """
    if sentinel is Unset:
        return 'Unset'
    raise TypeError(f"Cannot pickle {type(sentinel).__name__!r} object")


# The copyreg stub declares reducers as returning ``tuple``; ``copy`` and
# ``pickle`` both accept ``str`` at runtime (the idiomatic singleton return).
copyreg.pickle(Sentinel, _reduce_unset)  # pyright: ignore[reportArgumentType]

OMITTED_SENTINEL: typing.Final = '__omitted__'
"""
Wire-protocol value meaning "do not touch this field" -- never a server-side value.

It only lives in (1) the JSON Schema of models that enable
``omitted_sentinel`` (as a ``const`` branch and as ``default``) and (2)
inbound JSON. It is normalized to :data:`Unset` on entry, so application code
should never compare against it.
"""

SENTINEL_SCHEMA_BRANCH: typing.Final[dict[str, typing.Any]] = {
    'const': OMITTED_SENTINEL, 'type': 'string',
}
"""JSON Schema fragment for the sentinel branch. Shared constant -- do not mutate in place."""

SCHEMA_ANNOTATION_KEYS: typing.Final = ('title', 'description', 'deprecated', 'examples')
"""
Annotation keywords that describe the field as a whole rather than constrain its type.

When the sentinel branch is injected these stay on the outer schema object
instead of moving into one ``anyOf`` branch.
"""
