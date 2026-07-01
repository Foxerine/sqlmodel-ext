"""PEP 604 nullable relationship annotation support.

``SQLModelBase``'s metaclass normalizes a flat-string / ForwardRef nullable
relationship annotation such as ``'Child | None'`` into a structured
``ForwardRef('Child')`` before handing it to SQLModel's ``get_relationship_to``.
Upstream can only strip ``None`` from an already-evaluated ``typing.Union``; a
whole-string PEP 604 annotation would otherwise be treated as a class literally
named ``'Child | None'`` and blow up during mapper configuration.

This lets relationship fields use the pyright-friendly ``'X | None'`` forward
reference instead of ``Optional['X']`` (which triggers ``reportDeprecated``).

NOTE: no ``from __future__ import annotations`` here -- PEP 563 string
annotations interfere with SQLAlchemy's resolution of quoted Relationship
annotations.
"""
import ast
import uuid

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlmodel import Field, Relationship

from sqlmodel_ext import SQLModelBase, UUIDTableBaseMixin
from sqlmodel_ext.base import _is_none_node, _relationship_target_node


def _target(expr: str) -> str | None:
    """Parse a type expression and return the unparsed single target, or None."""
    node = _relationship_target_node(ast.parse(expr.strip(), mode='eval').body)
    return ast.unparse(node) if node is not None else None


class TestRelationshipTargetNode:
    """Unit coverage for the AST normalization helper across every shape a
    relationship annotation can take."""

    def test_bare_name(self) -> None:
        assert _target('Child') == 'Child'

    def test_dotted_name(self) -> None:
        assert _target('pkg.mod.Child') == 'pkg.mod.Child'

    def test_pep604_trailing_none(self) -> None:
        assert _target('Child | None') == 'Child'

    def test_pep604_leading_none(self) -> None:
        assert _target('None | Child') == 'Child'

    def test_optional(self) -> None:
        assert _target('Optional[Child]') == 'Child'

    def test_optional_requoted_inner(self) -> None:
        assert _target("Optional['Child']") == 'Child'

    def test_union_with_none(self) -> None:
        assert _target('Union[Child, None]') == 'Child'

    def test_multi_member_union_is_ambiguous(self) -> None:
        """A genuine multi-target union cannot be reduced to one -> None."""
        assert _target('Foo | Bar') is None

    def test_union_all_none_is_ambiguous(self) -> None:
        assert _target('None | None') is None

    def test_plain_none(self) -> None:
        assert _target('None') is None


class TestIsNoneNode:
    def test_none_constant(self) -> None:
        assert _is_none_node(ast.parse('None', mode='eval').body) is True

    def test_none_type_name(self) -> None:
        assert _is_none_node(ast.parse('NoneType', mode='eval').body) is True

    def test_non_none(self) -> None:
        assert _is_none_node(ast.parse('Child', mode='eval').body) is False


class ParentNode(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "parent"
    children: list["ChildNode"] = Relationship(back_populates="parent")


class ChildNode(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: str = "child"
    parent_id: uuid.UUID | None = Field(default=None, foreign_key="parentnode.id")
    # The point of this test: a forward-reference nullable relationship written
    # in flat PEP 604 form. Without the metaclass normalization this whole
    # string would reach SQLAlchemy as a class named ``"ParentNode | None"``.
    parent: "ParentNode | None" = Relationship(back_populates="children")


class TestForwardRefPep604Relationship:
    def test_mapper_resolves_target_class(self) -> None:
        rel = sa_inspect(ChildNode).relationships["parent"]
        assert rel.mapper.class_ is ParentNode

    def test_back_populates_wired_both_ways(self) -> None:
        child_rel = sa_inspect(ChildNode).relationships["parent"]
        parent_rel = sa_inspect(ParentNode).relationships["children"]
        assert child_rel.back_populates == "children"
        assert parent_rel.back_populates == "parent"

    def test_to_one_side_is_not_a_collection(self) -> None:
        """The nullable side resolves to a scalar (uselist=False), confirming the
        ``| None`` was interpreted as optionality rather than swallowed into the
        target class name."""
        rel = sa_inspect(ChildNode).relationships["parent"]
        assert rel.uselist is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
