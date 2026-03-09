# Relation Preloading

::: tip Tip
`src/sqlmodel_ext/mixins/relation_preload.py` — `RelationPreloadMixin` and `@requires_relations`
:::

## Decorator Implementation

```python
def requires_relations(*relations):
    def decorator(func):
        is_async_gen = python_inspect.isasyncgenfunction(func)

        if is_async_gen:
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                session = _extract_session(func, args, kwargs) # [!code focus]
                if session is not None:
                    await self._ensure_relations_loaded(session, relations) # [!code focus]
                async for item in func(self, *args, **kwargs):
                    yield item
        else:
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                session = _extract_session(func, args, kwargs) # [!code focus]
                if session is not None:
                    await self._ensure_relations_loaded(session, relations) # [!code focus]
                return await func(self, *args, **kwargs)

        wrapper._required_relations = relations # [!code highlight]
        return wrapper
    return decorator
```

Logic:
1. **Auto-extract `session`** from method arguments
2. Call `_ensure_relations_loaded()` to ensure relations are loaded
3. Execute the original method

Supports both regular async methods and async generators. The `_required_relations` attribute stores declaration info for import-time validation.

## `_extract_session()` — Auto-finding the Session

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

## `RelationPreloadMixin` Core Logic

### Import-time Validation

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

::: tip Tip
Checks whether relation names exist at class definition time (import time). Typos error immediately, not at runtime.
:::

### `_is_relation_loaded()` — Checking Load State

```python
def _is_relation_loaded(self, rel_name):
    state = sa_inspect(self)
    return rel_name not in state.unloaded
```

Uses SQLAlchemy's `inspect()` to get the object's internal state. `state.unloaded` contains all unloaded relation names.

### `_ensure_relations_loaded()` — Incremental Loading

```python
async def _ensure_relations_loaded(self, session, relations):
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

### `_find_relation_to_class()` — Finding Relation Paths

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

## `requires_for_update` Decorator Implementation

```python
def requires_for_update(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        session = _extract_session(func, args, kwargs)
        if session is not None:
            locked: set[int] = session.info.get(SESSION_FOR_UPDATE_KEY, set()) # [!code focus]
            if id(self) not in locked: # [!code focus]
                cls_name = type(self).__name__
                raise RuntimeError( # [!code error]
                    f"{cls_name}.{func.__name__}() requires a FOR UPDATE locked instance. "
                    f"Call {cls_name}.get(session, ..., with_for_update=True) first."
                )
        return await func(self, *args, **kwargs)

    wrapper._requires_for_update = True
    return wrapper
```

How it works:
1. Extract session from arguments (reuses `_extract_session()`)
2. Check if `session.info[SESSION_FOR_UPDATE_KEY]` contains `id(self)`
3. Not in the locked set → immediate `RuntimeError`
4. Sets `_requires_for_update = True` metadata for the static analyzer to detect unlocked calls

`SESSION_FOR_UPDATE_KEY` is written by the `get()` method when `with_for_update=True`.
