# Relation preloading mechanism

::: tip Source location
`src/sqlmodel_ext/mixins/relation_preload.py` — `RelationPreloadMixin` and `@requires_relations`
:::

## Why this exists

In async SQLAlchemy, accessing an unloaded relation triggers an implicit synchronous query → `MissingGreenlet`. The conventional fix is to `load=` the relations at the call site, but **that requires the caller to know which relations the method touches internally** — a fragile contract.

`@requires_relations` declares "I need these relations" on the method itself, so callers don't have to know anything. This chapter explains **how** that magic works; for usage steps, see [Prevent MissingGreenlet errors](/en/how-to/prevent-missing-greenlet).

## Decorator implementation

```python
def requires_relations(*relations):
    def decorator(func):
        is_async_gen = python_inspect.isasyncgenfunction(func)

        if is_async_gen:
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                session = _extract_session(func, args, kwargs) # [!code focus]
                if session is not None:
                    await self.ensure_relations_loaded(session, relations) # [!code focus]
                async for item in func(self, *args, **kwargs):
                    yield item
        else:
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                session = _extract_session(func, args, kwargs) # [!code focus]
                if session is not None:
                    await self.ensure_relations_loaded(session, relations) # [!code focus]
                return await func(self, *args, **kwargs)

        wrapper._required_relations = relations # [!code highlight]
        return wrapper
    return decorator
```

Logic:
1. **Auto-extract `session`** from method arguments
2. Call `ensure_relations_loaded()` to ensure relations are loaded (skipped when no session can be found — callers of such a method must call it themselves first)
3. Execute the original method

Supports both regular async methods and async generators. The `_required_relations` attribute stores declaration info for import-time validation.

## `_extract_session()` — auto-finding the session

```python
def _extract_session(func, args, kwargs):
    # 1. Look in kwargs first
    if 'session' in kwargs:
        return kwargs['session']

    # 2. Find by positional 'session' parameter position
    sig = python_inspect.signature(func)
    param_names = list(sig.parameters.keys())
    if 'session' in param_names:
        idx = param_names.index('session') - 1   # Subtract self
        if 0 <= idx < len(args):
            return args[idx]

    # 3. Find AsyncSession type values in kwargs
    for value in kwargs.values():
        if isinstance(value, AsyncSession):
            return value

    return None
```

Three strategies ensure the session is found regardless of how it's passed.

## `RelationPreloadMixin` core logic

### Import-time validation

```python
class RelationPreloadMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        all_available_names = all_annotations | sqlmodel_relationships

        for method_name in dir(cls):
            method = getattr(cls, method_name, None)
            if method and hasattr(method, '_required_relations'):
                for spec in method._required_relations:
                    if isinstance(spec, str):
                        if spec not in all_available_names: # [!code focus]
                            raise AttributeError( # [!code focus]
                                f"{cls.__name__}.{method_name} declares '{spec}', " # [!code focus]
                                f"but {cls.__name__} has no such attribute" # [!code focus]
                            ) # [!code focus]
```

::: tip Import-time validation
Checks whether relation names exist at class definition time (import time). Typos error immediately, not at runtime.
:::

### `_is_relation_loaded()` — checking load state

```python
def _is_relation_loaded(self, rel_name):
    state = sa_inspect(self)
    return rel_name not in state.unloaded
```

Uses SQLAlchemy's `inspect()` to get the object's internal state. `state.unloaded` contains all unloaded relation names.

### `ensure_relations_loaded()` — incremental loading (public entry point)

```python
async def ensure_relations_loaded(self, session, relations):
    to_load = []

    for rel in relations:
        if isinstance(rel, str):
            if not self._is_relation_loaded(rel):
                to_load.append(rel)
        else:
            # Nested relation (e.g., Generator.config)
            parent_attr = _find_relation_to_class(self.__class__, rel.parent.class_)

            if not self._is_relation_loaded(parent_attr):
                to_load.append(parent_attr)
                to_load.append(rel)
            else:
                parent_obj = getattr(self, parent_attr)
                if not _is_obj_relation_loaded(parent_obj, rel.key):
                    to_load.append(parent_attr)
                    to_load.append(rel)

    if not to_load:
        return    # All already loaded

    # Execute a single query with selectinload
    fresh = await self.__class__.get(
        session, self.__class__.id == pk_value,
        load=load_options,
    )

    # Copy loaded relation objects onto self
    for key in all_direct_keys:
        value = getattr(fresh, key, None)
        object.__setattr__(self, key, value)
```

Key features:
1. **Incremental loading** — already loaded relations are not re-queried
2. **Nesting-aware** — when loading `Generator.config`, if `generator` itself isn't loaded, both are loaded together
3. **In-place update** — uses `object.__setattr__` to directly modify `self`, no instance replacement needed
4. **Public entry point** — it is a public method, not just an internal detail of the decorator: `@requires_relations` skips preloading when it can't find a session among the arguments, and callers of such a method must call `ensure_relations_loaded()` themselves first — correctness must not depend on that "skip" branch

### `ensure_relations_loaded_bulk()` — bulk preloading

Calling `ensure_relations_loaded()` on each of a batch of (possibly heterogeneous) instances costs N queries. `ensure_relations_loaded_bulk(session, instances, specs_by_class)` reduces this to **0–1 owner refresh + 1 per target inheritance-tree root**:

1. **Owner refresh (on demand)**: only owners whose needed foreign-key columns are unloaded / expired are refreshed, with a single `id IN (...)` (fresh instances cost zero extra queries).
2. **Grouping by target tree root**: target ids are collected from the owners' foreign-key columns, and each target inheritance tree's root is queried only once (several relations into the same STI/JTI tree are merged into one polymorphic `IN`), then attached with SQLAlchemy's public manual-preload API `set_committed_value` — afterwards the relation no longer appears in `inspect().unloaded`, the same effect as loading it through a loader.

The capability boundary is decided by the single source of truth `bulk_preload_unsupported_reason(spec)`: only relations that are "a direct string relation name, many-to-one, a single-column foreign key pointing at the target's `id`, with a `primaryjoin` that is a pure foreign-key equality" are handled in bulk. Extra predicates (such as `AND target.is_enabled`) filter rows in the ORM loader, while manual assembly by id would bypass them — and after assembly the relation counts as "loaded", so a per-instance fallback could not correct it — which is why such relations are excluded.

**When the foreign key is non-null but the target row doesn't exist, assembly is deliberately skipped**: the relation stays unloaded, and access under `lazy='raise_on_sql'` fails loudly instead of pretending "there is no association".

This is **a query-count optimization, not a source of correctness**: you should still call `ensure_relations_loaded()` on each instance afterwards (a no-op fast path when already loaded).

### `_find_relation_to_class()` — finding relation paths

```python
def _find_relation_to_class(from_class, to_class):
    """Find the relation attribute name from from_class that points to to_class"""
    for attr_name in dir(from_class):
        attr = getattr(from_class, attr_name, None)
        if hasattr(attr, 'property') and hasattr(attr.property, 'mapper'):
            target_class = attr.property.mapper.class_
            if target_class == to_class:
                return attr_name
    return None
```

The problem it solves: when you write `@requires_relations(Generator.config)`, the decorator knows it needs `Generator`'s `config` relation, but needs to know which attribute on `self` points to `Generator`.

## `requires_for_update` decorator implementation

```python
def requires_for_update(func: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        cls_name = type(self).__name__
        session = _extract_session(func, args[1:], kwargs)
        if session is None: # [!code focus]
            raise RuntimeError("... cannot extract an AsyncSession ...; the row-lock guard cannot verify") # [!code error]
        locked: set[int] = session.info.get(SESSION_FOR_UPDATE_KEY, set()) # [!code focus]
        if id(self) not in locked: # [!code focus]
            raise RuntimeError( # [!code error]
                f"{cls_name}.{func.__name__}() requires a FOR UPDATE locked instance. "
                f"Call {cls_name}.get(session, ..., with_for_update=True) first."
            )
        return await func(*args, **kwargs)

    wrapper._requires_for_update = True
    return wrapper
```

How it works:
1. Extract session from arguments (reuses `_extract_session()`)
2. **No session found → `RuntimeError` (fail-closed)**. A failed extraction is precisely the signal that "the guard has already stopped working" (e.g. an inner decorator didn't use `@wraps`, so `inspect.signature` can't see the `session` parameter); it must not look like "the guard passed"
3. Check whether `session.info[SESSION_FOR_UPDATE_KEY]` contains `id(self)`; if not → `RuntimeError`
4. Declared with `ParamSpec`, so the decorated method's signature stays visible to type checkers (`Callable[..., Any]` would erase the parameters)
5. Sets `_requires_for_update = True` metadata for the static analyzer to detect unlocked calls

`SESSION_FOR_UPDATE_KEY` is written by `get()` when `with_for_update=True`, and its lifecycle is maintained by module-level session event listeners (cleared on outermost commit / rollback, snapshot restored on savepoint rollback); see [CRUD pipeline](./crud-pipeline#for-update-tracking).

## Other contract decorators in the same family

| Decorator | Contract | How the session is located |
|-----------|----------|----------------------------|
| `@requires_locked_param(name)` | the instances in parameter `name` are all locked; an empty sequence counts as a violation | `Signature.bind` binds the full call arguments |
| `@requires_repeatable_read` | the session has been **verified** to be in REPEATABLE READ (the `session.info` marker is written only by `enter_repeatable_read()` after read-back confirmation) | `Signature.bind` |
| `@requires_read_committed` | the transaction is in READ COMMITTED (a live `SHOW transaction_isolation` every time; the marker is not trusted) | `Signature.bind` |

These three use `Signature.bind` instead of `_extract_session`'s "`args[0]` is `self`" heuristic: letting Python do the binding itself handles instance methods, classmethods, plain functions, keyword arguments and defaults correctly. They are fail-closed as well. For usage see [Enforce row locks and isolation levels](/en/how-to/enforce-locking-and-isolation).
