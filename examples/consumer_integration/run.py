"""
Consumer-side integration test.

Runs from a clean virtualenv that has ONLY the built sqlmodel-ext wheel
installed (plus aiosqlite for the async SQLite driver). Simulates what an
external PyPI consumer experiences: if this script passes, the wheel is
well-formed and the STI default-isolation fix behaves end-to-end.

Exits with status 0 on success, non-zero on any failed assertion. Intended
to be invoked directly by CI:

    python -m pip install dist/*.whl aiosqlite
    python examples/consumer_integration/run.py
"""
from __future__ import annotations

import asyncio
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import configure_mappers
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlmodel_ext import (
    AutoPolymorphicIdentityMixin,
    PolymorphicBaseMixin,
    SQLModelBase,
    Str64,
    UUIDTableBaseMixin,
    register_sti_column_properties_for_all_subclasses,
    register_sti_columns_for_all_subclasses,
)


# ---------- STI model hierarchy ----------

class Tool(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    """Shared STI parent table."""
    name: Str64


class ExportFunction(Tool, AutoPolymorphicIdentityMixin, table=True):
    """Declares ``max_size`` with a large realistic default."""
    # 5 GiB chosen because the real-world production bug used exactly this value.
    max_size: int = 5 * 1024 * 1024 * 1024


class UploadFunction(Tool, AutoPolymorphicIdentityMixin, table=True):
    """Declares a different default for a different field."""
    allow_overwrite: bool = True


class NoOpFunction(Tool, AutoPolymorphicIdentityMixin, table=True):
    """The sibling with zero own fields - the purest victim of default leaks."""
    pass


# ---------- Integration scenario ----------

FIVE_GIB = 5 * 1024 * 1024 * 1024


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


async def main() -> None:
    # STI two-phase registration (public API contract).
    register_sti_columns_for_all_subclasses()
    configure_mappers()
    register_sti_column_properties_for_all_subclasses()

    # --- Layer 1: metadata assertions ---
    shared = Tool.__table__.columns
    if shared["max_size"].default is not None:
        fail(
            f"tool.max_size.default should be None (fix cleared the shared SA "
            f"default). Got: {shared['max_size'].default!r}"
        )
    if shared["max_size"].server_default is not None:
        fail(f"tool.max_size.server_default should be None")
    if shared["allow_overwrite"].default is not None:
        fail(
            f"tool.allow_overwrite.default should be None. "
            f"Got: {shared['allow_overwrite'].default!r}"
        )

    # Pydantic field-level defaults must still be intact.
    if ExportFunction.model_fields["max_size"].default != FIVE_GIB:
        fail(
            f"Pydantic default for ExportFunction.max_size was wiped by the fix: "
            f"{ExportFunction.model_fields['max_size'].default!r}"
        )

    # --- Layer 2: behavioural assertions via real async SQLite ---
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        async with AsyncSession(engine) as session:
            # NoOpFunction does not declare max_size; the shared column must be NULL.
            noop = NoOpFunction(name="noop")
            noop = await noop.save(session)
            result = await session.execute(
                select(Tool.__table__.c.max_size).where(Tool.__table__.c.id == noop.id)
            )
            stored = result.scalar_one()
            if stored is not None:
                fail(
                    f"NoOpFunction.max_size should be NULL (bug: sibling default "
                    f"leaked from ExportFunction). Got: {stored!r}"
                )

            # UploadFunction also does not declare max_size.
            upload = UploadFunction(name="upload")
            upload = await upload.save(session)
            result = await session.execute(
                select(Tool.__table__.c.max_size).where(Tool.__table__.c.id == upload.id)
            )
            stored = result.scalar_one()
            if stored is not None:
                fail(
                    f"UploadFunction.max_size should be NULL. Got: {stored!r}"
                )

            # Happy path: declaring subclass still persists its own Pydantic default.
            export = ExportFunction(name="export")
            export = await export.save(session)
            result = await session.execute(
                select(Tool.__table__.c.max_size).where(Tool.__table__.c.id == export.id)
            )
            stored = result.scalar_one()
            if stored != FIVE_GIB:
                fail(
                    f"ExportFunction.max_size should be {FIVE_GIB}. Got: {stored!r} "
                    f"- fix may have broken the declaring subclass's default path."
                )
    finally:
        await engine.dispose()

    print("[OK] sqlmodel-ext consumer integration test passed")


if __name__ == "__main__":
    asyncio.run(main())
