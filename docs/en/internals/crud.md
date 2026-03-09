# CRUD Implementation

::: tip Tip
`src/sqlmodel_ext/mixins/table.py` — `TableBaseMixin` and `UUIDTableBaseMixin`
:::

## `TableBaseMixin` Basics

```python
class TableBaseMixin(AsyncAttrs):
    _has_table_mixin: ClassVar[bool] = True   # Lets the metaclass identify "this is a table class"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(
        sa_type=DateTime,
        sa_column_kwargs={'default': now, 'onupdate': now},
        default_factory=now,
    )
```

Inheriting `AsyncAttrs` enables `await obj.awaitable_attrs.some_relation` syntax on model objects, providing additional async safety.

`_has_table_mixin = True` is a marker that lets the metaclass automatically add `table=True` in `__new__`.

## `save()` Implementation

`save()` is the most core method, containing optimistic lock retry logic:

```python
async def save(self, session, ..., optimistic_retry_count=0):
    cls = type(self)
    instance = self
    retries_remaining = optimistic_retry_count
    current_data = None

    while True:
        session.add(instance)
        try:
            await session.commit() # [!code focus]
            break                              # Success, exit // [!code focus]
        except StaleDataError as e:            # Version conflict! // [!code error]
            await session.rollback()

            if retries_remaining <= 0:
                raise OptimisticLockError(
                    message=f"optimistic lock conflict",
                    model_class=cls.__name__,
                    record_id=str(instance.id),
                    expected_version=instance.version,
                    original_error=e,
                ) from e

            retries_remaining -= 1

            # Save current modifications (excluding metadata fields)
            if current_data is None:
                current_data = self.model_dump(
                    exclude={'id', 'version', 'created_at', 'updated_at'}
                )

            # Get the latest record from the database
            fresh = await cls.get(session, cls.id == self.id) # [!code focus]
            if fresh is None:
                raise OptimisticLockError("record has been deleted") from e

            # Re-apply my changes to the latest record
            for key, value in current_data.items(): # [!code focus]
                if hasattr(fresh, key): # [!code focus]
                    setattr(fresh, key, value) # [!code focus]
            instance = fresh

    await session.refresh(instance) # [!code highlight]
    return instance
```

### `session.add()` Behavior

`session.add()` **does not execute SQL**. SQLAlchemy automatically decides during `commit()` or `flush()`:
- Object is new → `INSERT`
- Object is already in Session and has changes → `UPDATE`

### Why Must You Use the Return Value?

::: danger Danger
`session.commit()` expires **all objects in the Session**. The original `user` object's attributes become "expired", triggering implicit queries on access. `save()` returns a fresh object after `refresh()`.
:::

## `update()` Implementation

```python
async def update(self, session, other, extra_data=None,
                 exclude_unset=True, exclude=None, ...):
    update_data = other.model_dump(exclude_unset=exclude_unset, exclude=exclude) # [!code focus]
    instance.sqlmodel_update(update_data, update=extra_data)
    session.add(instance)
    await session.commit()
```

::: tip Tip
The key is `exclude_unset=True`: only explicitly set fields are updated; unset fields retain their original values.
:::

## `get()` Implementation

This is the longest method (~200 lines), handling multiple scenarios in layers.

### Layer 1: Basic Query

```python
statement = select(cls)
if condition is not None:
    statement = statement.where(condition)
```

### Layer 2: Pagination + Sorting

```python
if table_view:
    order_column = cls.created_at if table_view.order == "created_at" else cls.updated_at
    order_by = [desc(order_column) if table_view.desc else asc(order_column)]
    statement = statement.order_by(*order_by).offset(table_view.offset).limit(table_view.limit)
```

### Layer 3: Time Filtering

```python
@classmethod
def _build_time_filters(cls, created_before, created_after, ...):
    filters = []
    if created_after is not None:
        filters.append(cls.created_at >= created_after)
    if created_before is not None:
        filters.append(cls.created_at < created_before)
    ...
    return filters
```

### Layer 4: Relation Preloading

```python
if load:
    load_list = load if isinstance(load, list) else [load]
    load_chains = cls._build_load_chains(load_list) # [!code focus]

    for chain in load_chains:
        loader = selectinload(chain[0]) # [!code focus]
        for rel in chain[1:]:
            loader = loader.selectinload(rel) # [!code focus]
        statement = statement.options(loader)
```

`_build_load_chains` automatically detects relation dependencies and builds nested loading chains. For example, `load=[User.profile, Profile.avatar]` → `selectinload(User.profile).selectinload(Profile.avatar)`.

### Layer 5: Polymorphic Queries

```python
if is_jti:
    polymorphic_cls = with_polymorphic(cls, '*')
    statement = select(polymorphic_cls)   # Auto-JOINs all sub-tables

if is_sti:
    descendant_identities = [m.polymorphic_identity for m in mapper.self_and_descendants]
    statement = statement.where(poly_on.in_(descendant_identities))
```

JTI uses `with_polymorphic` to auto-JOIN sub-tables. STI requires manually adding a `WHERE _polymorphic_name IN (...)` filter.

### Layer 6: `fetch_mode` Determines Return Value

```python
result = await session.exec(statement)

if fetch_mode == "first":   return result.first()
elif fetch_mode == "one":   return result.one()
elif fetch_mode == "all":   return list(result.all())
```

## `rel()` and `cond()` — Type-safe Helper Functions

```python
def rel(relationship: object) -> QueryableAttribute[Any]:
    """Cast Relationship field to QueryableAttribute, fixing basedpyright inference"""
    if not isinstance(relationship, QueryableAttribute):
        raise AttributeError(...)
    return relationship

def cond(expr: ColumnElement[bool] | bool) -> ColumnElement[bool]:
    """Narrow column comparison expression to ColumnElement[bool], fixing & | operator type errors"""
    return cast(ColumnElement[bool], expr)
```

These two functions are similar to SQLModel's `col()` — they perform type assertions/casts at runtime to satisfy static type checkers (basedpyright).

## `get_one()` Implementation

```python
@classmethod
async def get_one(cls, session, id, *, load=None, with_for_update=False):
    return await cls.get(
        session, col(cls.id) == id,
        fetch_mode='one', load=load, with_for_update=with_for_update,
    )
```

Essentially a shortcut for `get(fetch_mode='one')`. `UUIDTableBaseMixin` provides a more precisely typed override (accepting only `uuid.UUID`).

## `get_exist_one()` FastAPI Integration

```python
@classmethod
async def get_exist_one(cls, session, id, load=None):
    instance = await cls.get(session, col(cls.id) == id, load=load)
    if not instance:
        if _HAS_FASTAPI:
            raise _FastAPIHTTPException(status_code=404, detail="Not found") # [!code highlight]
        raise RecordNotFoundError("Not found") # [!code highlight]
    return instance
```

::: info Info
At **module import time**, it checks whether FastAPI is installed. If so, it raises `HTTPException(404)`; otherwise, it raises `RecordNotFoundError`.
:::

## `sanitize_integrity_error()` Implementation

```python
@staticmethod
def sanitize_integrity_error(e: IntegrityError, default_message: str = "...") -> str:
    orig = e.orig
    # SQLSTATE 23514 (check_violation): PostgreSQL trigger's RAISE EXCEPTION
    if orig is not None and getattr(orig, 'sqlstate', None) == '23514':
        error_msg = str(orig)
        if '\n' in error_msg:
            error_msg = error_msg.split('\n')[0]  # Take the first line
        if error_msg.startswith('ERROR:'):
            error_msg = error_msg[6:].strip()
        return error_msg
    return default_message
```

PostgreSQL triggers can produce business-semantic error messages via `RAISE EXCEPTION ... USING ERRCODE = 'check_violation'`, which are safe to display to users. Other constraint errors (FK, unique, etc.) might leak table structure, so the default message is returned.

## FOR UPDATE Tracking

In the `get()` method, when `with_for_update=True`, the locked instance's `id()` is recorded in `session.info`:

```python
SESSION_FOR_UPDATE_KEY = '_for_update_locked'

# In get():
if with_for_update:
    locked: set[int] = session.info.setdefault(SESSION_FOR_UPDATE_KEY, set())
    locked.add(id(instance))
```

This is used by the `@requires_for_update` decorator for runtime checking.

## `count()` Implementation

```python
@classmethod
async def count(cls, session, condition=None, ...):
    statement = select(func.count()).select_from(cls)
    if condition is not None:
        statement = statement.where(condition)
    result = await session.scalar(statement)
    return result or 0
```

Uses database-level `COUNT(*)` rather than Python's `len()`.

## `get_with_count()` Implementation

```python
@classmethod
async def get_with_count(cls, session, condition=None, *, table_view=None, ...):
    total_count = await cls.count(session, condition, ...)
    items = await cls.get(session, condition, fetch_mode="all", table_view=table_view, ...)
    return ListResponse(count=total_count, items=items)
```

Essentially a combination of `count()` + `get(fetch_mode="all")`.
