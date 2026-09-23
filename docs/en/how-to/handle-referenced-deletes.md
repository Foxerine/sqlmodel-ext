# Handle deletes of still-referenced rows

**Goal**: when deleting a record that other rows still reference via a foreign key (`RESTRICT` / `NO ACTION`), return an actionable 409 message to the user instead of a 500 or a database error that leaks the constraint name.

**Prerequisites**:

- PostgreSQL (the translation relies on SQLSTATE `23503`)
- The foreign key's `ondelete` is `RESTRICT` or the default `NO ACTION` (for cascade delete / set-null configuration see [Configure Cascade Delete](./configure-cascade-delete))

## 1. What happens

```python
await Owner.delete(session, owner)
# → sqlmodel_ext.mixins.ResourceReferencedError: Cannot delete: this resource is still referenced by other resources
```

A foreign key violation has **two directions**, and the driver raises exactly the same exception for both (same SQLSTATE, same constraint name):

| Direction | Trigger | Meaning |
|------|------|------|
| Points to a nonexistent parent row | INSERT/UPDATE of a child row whose FK target no longer exists | 404: the referenced resource does not exist |
| Still referenced | DELETE of a parent row that child rows still point to | **409**: the resource exists but can't be deleted right now |

So the direction can only be decided by the **call site**: `delete()` translates an `IntegrityError` into `ResourceReferencedError` only when both conditions hold — the error is caught inside `delete()`, **and** `IntegrityError.statement` is a `DELETE`. The first condition alone is not enough: commit flushes every pending operation in the session, so a faulty `INSERT` you queued earlier would also surface inside `delete()` — that one is re-raised unchanged.

## 2. Register the user-visible message next to the parent model

```python
from sqlmodel_ext import TableBaseMixin

class Owner(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64

TableBaseMixin.register_fk_delete_restrict_message(
    'project_owner_id_fkey',
    "This owner still has projects; delete or transfer those projects first",
)
```

- The convention is to write it next to the **referenced parent model** (the same convention as `register_unique_violation_message`: register the message where the constraint is declared).
- The message should explain **what the user can do next**, and must not contain table / column names.
- Unregistered constraints fall back to `FK_DELETE_RESTRICT_FALLBACK_MESSAGE` ("Cannot delete: this resource is still referenced by other resources").

::: warning The constraint name must match the database **exactly**
A typo doesn't raise an error; it just silently falls back to the generic message. Confirm the name from the actually running database (e.g. the `constraint=` field in the warning log that `delete()` records); don't copy it from a database built by `create_all` — migrations may have given the constraint a different name. PostgreSQL's default name is `<table>_<column>_fkey`.
:::

This registry and `register_foreign_key_violation_message` (the "referenced resource does not exist" direction) are keyed by the same constraint name but serve **opposite directions** and don't interfere with each other: `lookup_integrity_violation_message()` only looks up the latter, and only `delete()` looks up the former.

## 3. Map to 409 in FastAPI

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlmodel_ext.mixins import ResourceReferencedError

@app.exception_handler(ResourceReferencedError)
async def resource_referenced_handler(request: Request, exc: ResourceReferencedError) -> JSONResponse:
    # exc.friendly_message is safe to return to the client; exc.original_error is for diagnostics only, don't expose it
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.friendly_message})
```

`ResourceReferencedError.status_code` is `409` (not 404 — the row not only exists, it is in use). It also carries `constraint_name` (`None` when the driver didn't provide one) and `original_error`.

## 4. When you need "referenced by what, and how many": check explicitly first

The generic message deliberately doesn't say "referenced by whom". If an endpoint wants to tell the user "there are still 3 projects", it should query **before** deleting and raise its own error; `ResourceReferencedError` is only the fallback for this path (e.g. losing the TOCTOU race between the check and the delete):

```python
n = await Project.count(session, col(Project.owner_id) == owner.id)
if n:
    raise HTTPException(409, detail=f"This owner still has {n} projects")
await Owner.delete(session, owner)     # a concurrent insert can still make this raise ResourceReferencedError
```

## 5. Boundaries

- **`commit=False` is not covered**: in instance mode with `delete(..., commit=False)`, the actual `DELETE` is issued by your later flush / commit, outside `delete()`, and is not translated.
- **Don't raise `ResourceReferencedError` elsewhere**: it doesn't validate its own usage. Raising it from a generic integrity-error handler would misreport "referenced target does not exist" as "still referenced".
- **Other databases**: no SQLSTATE, no translation; the original `IntegrityError` is raised unchanged.

## Related reference

- [Exceptions of `delete()`](/en/reference/crud-methods#delete)
- [IntegrityError friendly-message registry](/en/reference/crud-methods#integrityerror-friendly-message-registry)
- [Configure Cascade Delete](./configure-cascade-delete)
