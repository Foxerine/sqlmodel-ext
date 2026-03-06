# 关系预加载机制

::: tip 源码位置
`src/sqlmodel_ext/mixins/relation_preload.py` — `RelationPreloadMixin` 和 `@requires_relations`
:::

## 装饰器实现

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

逻辑：
1. 从方法参数中**自动提取 `session`**
2. 调用 `_ensure_relations_loaded()` 确保关系已加载
3. 执行原方法

同时支持普通异步方法和异步生成器。`_required_relations` 属性存储声明信息，供导入时验证使用。

## `_extract_session()` — 自动找 session

```python
def _extract_session(func, args, kwargs):
    # 1. 先从 kwargs 找
    if 'session' in kwargs:
        return kwargs['session']

    # 2. 从位置参数的 'session' 参数位置找
    sig = python_inspect.signature(func)
    param_names = list(sig.parameters.keys())
    if 'session' in param_names:
        idx = param_names.index('session') - 1   # 减去 self
        if 0 <= idx < len(args):
            return args[idx]

    # 3. 从 kwargs 中找 AsyncSession 类型的值
    for value in kwargs.values():
        if isinstance(value, AsyncSession):
            return value

    return None
```

三种策略确保无论 session 以何种方式传入都能找到。

## `RelationPreloadMixin` 核心逻辑

### 导入时验证

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
                                f"{cls.__name__}.{method_name} 声明了 '{spec}'，" # [!code focus]
                                f"但 {cls.__name__} 没有这个属性" # [!code focus]
                            ) # [!code focus]
```

::: tip 导入时验证
在类定义时（导入时）就检查关系名是否存在。拼写错误立刻报错，不等到运行时。
:::

### `_is_relation_loaded()` — 检查加载状态

```python
def _is_relation_loaded(self, rel_name):
    state = sa_inspect(self)
    return rel_name not in state.unloaded
```

使用 SQLAlchemy 的 `inspect()` 获取对象内部状态。`state.unloaded` 包含所有未加载的关系名。

### `_ensure_relations_loaded()` — 增量加载

```python
async def _ensure_relations_loaded(self, session, relations):
    to_load = []

    for rel in relations:
        if isinstance(rel, str):
            if not self._is_relation_loaded(rel):
                to_load.append(rel)
        else:
            # 嵌套关系（如 Generator.config）
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
        return    # 全部已加载

    # 执行一次带 selectinload 的查询
    fresh = await self.__class__.get(
        session, self.__class__.id == pk_value,
        load=load_options,
    )

    # 把加载好的关系对象复制到 self 上
    for key in all_direct_keys:
        value = getattr(fresh, key, None)
        object.__setattr__(self, key, value)
```

关键特性：
1. **增量加载** — 已加载的关系不重复查询
2. **嵌套感知** — 加载 `Generator.config` 时，如果 `generator` 本身也没加载，会一起加载
3. **原地更新** — 用 `object.__setattr__` 直接修改 `self`，不需要替换实例

### `_find_relation_to_class()` — 查找关系路径

```python
def _find_relation_to_class(from_class, to_class):
    """从 from_class 找到指向 to_class 的关系属性名"""
    for attr_name in dir(from_class):
        attr = getattr(from_class, attr_name, None)
        if hasattr(attr, 'property') and hasattr(attr.property, 'mapper'):
            target_class = attr.property.mapper.class_
            if target_class == to_class:
                return attr_name
    return None
```

解决的问题：当你写 `@requires_relations(Generator.config)` 时，装饰器知道需要 `Generator` 的 `config` 关系，但需要知道 `self` 上哪个属性指向 `Generator`。

## `requires_for_update` 装饰器实现

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

工作原理：
1. 从参数中提取 session（复用 `_extract_session()`）
2. 检查 `session.info[SESSION_FOR_UPDATE_KEY]` 中是否包含 `id(self)`
3. 不在锁定集合中 → 立即 `RuntimeError`
4. 设置 `_requires_for_update = True` 元数据，供静态分析器检测未锁定的调用

`SESSION_FOR_UPDATE_KEY` 由 `get()` 方法在 `with_for_update=True` 时写入。
