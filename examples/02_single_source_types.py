"""
02 -- One declaration, three consumers: constraint aliases and ``max_length_of()``.

Run::

    python examples/02_single_source_types.py

A field typed ``Str64`` produces, from that single annotation:

* Pydantic validation (``max_length=64``, no NUL bytes),
* the database column type (``VARCHAR(64)``),
* the OpenAPI / JSON Schema (``maxLength: 64``).

Code that needs the number ("truncate a generated title to the column width")
asks the alias with ``max_length_of(Str64)`` instead of writing ``64`` a second
time. Decimal aliases work the same way: ``NonNegativeWriteDecimal38_18`` is a
``NUMERIC(38, 18)`` column that rejects floats, serializes as a JSON string and
publishes ``type: string`` in the request schema.
"""
import json
from decimal import Decimal

from pydantic import ValidationError
from sqlalchemy import Numeric
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql
from sqlmodel import SQLModel
from sqlmodel.sql.sqltypes import AutoString

from sqlmodel_ext import (
    NonNegativeWriteDecimal38_18,
    SQLModelBase,
    Str64,
    Text1K,
    UUIDTableBaseMixin,
    max_length_of,
)


class ProductBase(SQLModelBase):
    name: Str64
    """Display name."""

    description: Text1K | None = None
    """Long description; null = none."""

    price: NonNegativeWriteDecimal38_18
    """Unit price. Decimal only -- floats are rejected."""


class Product(ProductBase, UUIDTableBaseMixin, table=True):
    """The table columns are derived from the same aliases."""


def make_title(raw: str) -> str:
    """Truncate to what the column can hold -- the bound comes from the type, not a literal."""
    return raw[:max_length_of(Str64)]


def main() -> None:
    # --- 1. Validation ------------------------------------------------------
    try:
        _ = ProductBase(name="x" * 65, price=Decimal("1"))
    except ValidationError:
        pass
    else:
        raise AssertionError("65 characters must be rejected")

    try:
        _ = ProductBase.model_validate({'name': "pen", 'price': 0.1})
    except ValidationError:
        pass
    else:
        raise AssertionError("a float price must be rejected (binary floats lose precision)")

    product = ProductBase.model_validate({'name': "pen", 'price': "1.50"})
    assert product.price == Decimal("1.50")
    assert json.loads(product.model_dump_json())['price'] == "1.5", "Decimal goes out as a JSON string"

    # --- 2. Database column -------------------------------------------------
    table = SQLModel.metadata.tables['product']  # table name derived from the class name
    name_type = table.c.name.type
    assert isinstance(name_type, AutoString) and name_type.length == 64
    price_type = table.c.price.type
    assert isinstance(price_type, Numeric) and (price_type.precision, price_type.scale) == (38, 18)
    ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    assert "VARCHAR(64)" in ddl and "NUMERIC(38, 18)" in ddl

    # --- 3. JSON Schema / OpenAPI -------------------------------------------
    schema = ProductBase.model_json_schema()
    assert schema['properties']['name']['maxLength'] == 64
    assert schema['properties']['price']['type'] == 'string', "request schema never advertises floats"
    print(json.dumps(schema['properties'], indent=2))

    # --- 4. Code that needs the number asks the type ------------------------
    assert max_length_of(Str64) == 64
    assert max_length_of(Text1K | None) == 1000
    assert len(make_title("t" * 500)) == 64

    print("[OK] 02_single_source_types: validation, column and schema come from one annotation")


if __name__ == "__main__":
    main()
