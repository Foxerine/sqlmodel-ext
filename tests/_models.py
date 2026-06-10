"""
Synthetic STI hierarchy used by regression tests.

Intentionally placed outside any test function so the polymorphic metaclass
queues the classes onto ``_sti_subclasses_to_register`` at import time. The
conftest fixture drives the two-phase registration and schema creation.

Hierarchy:
    Tool (STI root, has its own table)
    +-- FunctionA   - declares ``max_files`` with default 100
    +-- FunctionB   - declares ``timeout`` with default 30
    +-- FunctionC   - declares nothing extra (sibling with no own fields)

The STI bug under regression: FunctionA's Pydantic default 100 would leak
into the shared ``tool.max_files`` SA Column.default. Inserting a FunctionB
or FunctionC row would then silently materialise ``max_files = 100`` in
the sibling row because the ORM falls back to ``Column.default`` when the
attribute is absent from the sibling's ``__dict__``.
"""
from sqlmodel_ext import (
    AutoPolymorphicIdentityMixin,
    DeferredIndex,
    PolymorphicBaseMixin,
    SQLModelBase,
    Str64,
    Text1K,
    UUIDTableBaseMixin,
)


class Tool(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    """STI root: all subclasses share this single ``tool`` table."""
    name: Str64


class FunctionA(Tool, AutoPolymorphicIdentityMixin, table=True):
    """Sibling with a field carrying a non-None Pydantic default."""
    max_files: int = 100


class FunctionB(Tool, AutoPolymorphicIdentityMixin, table=True):
    """Sibling with its own non-overlapping field + default."""
    timeout: int = 30


class FunctionC(Tool, AutoPolymorphicIdentityMixin, table=True):
    """Sibling with no additional fields - the purest victim of default leaks."""
    pass


class Node(
    SQLModelBase,
    UUIDTableBaseMixin,
    PolymorphicBaseMixin,
    table=True,
    table_args=(
        # References ``payload`` which only the TextNode subclass registers --
        # a plain Index here would raise ConstraintColumnNotFoundError at
        # class-creation time. DeferredIndex materializes after STI phase 1.
        DeferredIndex('ix_node_payload', 'payload'),
    ),
):
    """STI root used by the DeferredIndex regression tests."""
    title: Str64


class TextNode(Node, AutoPolymorphicIdentityMixin, table=True):
    """Subclass declaring the column targeted by the base-class DeferredIndex."""
    payload: Text1K | None = None
