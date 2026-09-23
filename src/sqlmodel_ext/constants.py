"""
Shared infrastructure constants (metaclass / mixins / serialization protocol).

Invariant: this module imports **no** ``sqlmodel_ext`` module. The metaclass
in ``sqlmodel_ext.base`` imports it very early during package initialization;
any import back into the package would create a cycle.
"""
from typing import Final

from pydantic import Field as _pydantic_field
from pydantic.fields import FieldInfo

EXCLUDE_IF_NONE: Final[FieldInfo] = _pydantic_field(exclude_if=lambda value: value is None)
"""
Field metadata marker: when the value is ``None``, the **whole key** is left out of every serialized output.

Put it in the metadata position of ``Annotated``; the field type and default
are written as usual::

    class Event(SQLModelBase):
        marker: Annotated[bool | None, EXCLUDE_IF_NONE] = None

All three parts are required: (1) ``| None``, (2) this marker, (3) the
``= None`` default. The marker removes the key from the output, and an absent
key can only be read back if the field has a default. Without (3), Pydantic
treats the field as "required but nullable" and re-reading the dumped JSON
fails with a missing-field ``ValidationError``.

**Why ``exclude_none`` is not enough**: ``exclude_none`` is a switch the
*caller* passes to ``model_dump``; the field itself cannot control it, and not
every serialization path passes it (message-bus publishers, framework
response serialization, hand-written ``model_dump_json()`` calls). This
marker makes the behavior a property of the field.

A typical use is adding a nullable field to a structure that older consumers
deserialize strictly (``extra='forbid'``): during a rolling deployment, a new
producer emitting ``"new_field": null`` would make old consumers reject the
whole message; with this marker the key is simply absent while its value is
``None``.

**Why ``Annotated`` metadata and not a ``Field()`` parameter**: wrapping
``sqlmodel.Field`` cannot match its disjoint overloads,
``FieldInfo.merge_field_infos`` is deprecated in Pydantic 2.12, and assigning
``.exclude_if`` to an existing ``FieldInfo`` after the fact has no effect
(Pydantic builds the serialization schema from ``_attributes_set``).
``Annotated`` metadata is Pydantic's documented, non-deprecated merge path.

Use it on **non-table** models. Table models that also need database
metadata in the same ``Annotated`` must be verified separately.

Requires Pydantic >= 2.12 (``exclude_if``).
"""

# Optimistic-lock protocol names shared by the metaclass (reserved-name guard,
# ``version_id_col`` wiring) and ``OptimisticLockMixin``. Defined here, not in
# the mixin package, because the metaclass must import them and importing from
# ``sqlmodel_ext.mixins`` would create an import cycle.
OPTIMISTIC_LOCK_ENABLED_FLAG: Final = '_has_optimistic_lock'
"""
Name of the ``ClassVar`` flag through which ``OptimisticLockMixin`` marks a class as optimistically locked.

The mixin declares the attribute with a literal name (Python attribute names
cannot be computed); this constant is for ``getattr`` lookups.
"""

OPTIMISTIC_LOCK_VERSION_COLUMN: Final = 'oplock_version'
"""
Column name of the optimistic-lock version counter (``version_id_col``).

Deliberately unusual so that it does not collide with a domain field such as
``version``. The name is **globally reserved**: any ``SQLModelBase`` subclass
that declares a field with this name in its own class body raises
``TypeError`` at class creation, whether or not it enables optimistic
locking. Only ``OptimisticLockMixin`` (a plain class that does not go through
the metaclass) defines it.
"""
