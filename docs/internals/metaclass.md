# 元类与 SQLModelBase

::: tip 源码位置
`src/sqlmodel_ext/base.py` — `SQLModelBase` 和 `__DeclarativeMeta` 元类

`src/sqlmodel_ext/_sa_type.py` — 从 Annotated 元数据提取 SQLAlchemy 列类型

`src/sqlmodel_ext/_compat.py` — Python 3.14 兼容性补丁
:::

这是整个项目的**基石**。所有模型类都继承 `SQLModelBase`，而 `SQLModelBase` 的元类 `__DeclarativeMeta` 在类创建时自动完成一系列配置。

## 用户写的代码 vs 元类做的事

```python
class UserBase(SQLModelBase):
    name: Str64
    email: str

class User(UserBase, UUIDTableBaseMixin, table=True):
    pass
```

| 类 | 是否建数据库表 | 角色 |
|---|---|---|
| `UserBase` | 否 | 纯数据模型，只定义字段 |
| `User` | 是 | 继承字段 + CRUD 能力，对应数据库表 |

SQLModel 通过 `table=True` 关键字判断是否建表。**元类就是处理这个参数的地方。**

## `__DeclarativeMeta.__new__` 逐步拆解

`__new__` 在类对象被创建的**那一瞬间**执行。

### 第一步：自动 `table=True`

```python
# base.py:113-116
is_intended_as_table = any(getattr(b, '_has_table_mixin', False) for b in bases)
if is_intended_as_table and 'table' not in kwargs:
    kwargs['table'] = True # [!code focus]
```

遍历父类列表，如果发现 `_has_table_mixin = True` 标记（`TableBaseMixin` 上定义），自动加上 `table=True`。

### 第二步：检测继承类型（JTI vs STI）

```python
# base.py:119-143
parent_tablename = None
for base in bases:
    if is_table_model_class(base) and hasattr(base, '__tablename__'):
        parent_tablename = base.__tablename__
        break

# 检查是否有指向父表的外键 → JTI 的特征
has_fk_to_parent = False
if parent_tablename is not None and will_be_table:
    for base in bases:
        for field_name, field_info in base.model_fields.items():
            fk = getattr(field_info, 'foreign_key', None)
            if fk and parent_tablename in fk:
                has_fk_to_parent = True

# STI：没有外键到父表，共用父表
if parent_tablename and will_be_table and not has_own_tablename and not has_fk_to_parent:
    attrs['__tablename__'] = parent_tablename
```

当 table 子类继承 table 父类时：
- **有外键指向父表** → JTI，子类有自己的表
- **没有外键** → STI，子类共用父类的 `__tablename__`

### 第三步：合并 `__mapper_args__`

```python
# base.py:146-158
collected_mapper_args = {}

if 'mapper_args' in kwargs:
    collected_mapper_args.update(kwargs.pop('mapper_args'))

for key in cls._KNOWN_MAPPER_KEYS:  # polymorphic_on, polymorphic_identity, ...
    if key in kwargs:
        collected_mapper_args[key] = kwargs.pop(key)

if collected_mapper_args:
    existing = attrs.get('__mapper_args__', {}).copy()
    existing.update(collected_mapper_args)
    attrs['__mapper_args__'] = existing
```

把 `polymorphic_on`、`polymorphic_abstract` 等关键字参数从 `kwargs` 中取出，合并到 `__mapper_args__` 字典中。这让用户可以用简洁语法：

```python
# sqlmodel-ext（简洁）
class Tool(SQLModelBase, polymorphic_on="_polymorphic_name", polymorphic_abstract=True): # [!code ++]
    pass

# 等价于原生 SQLAlchemy（繁琐）
class Tool(SQLModel, table=True): # [!code --]
    __mapper_args__ = { # [!code --]
        "polymorphic_on": "_polymorphic_name", # [!code --]
        "polymorphic_abstract": True, # [!code --]
    } # [!code --]
```

`_KNOWN_MAPPER_KEYS` 支持的快捷关键字：`polymorphic_on`、`polymorphic_identity`、`polymorphic_abstract`、`version_id_col`、`concrete`。

### 第四步：从类型注解中提取 `sa_type`

这是元类**最精巧的部分**。

```python
# base.py:169-202
annotations, ..., eval_globals, eval_locals = _resolve_annotations(attrs)

for field_name, field_type in annotations.items():
    sa_type = _extract_sa_type_from_annotation(field_type) # [!code focus]

    if sa_type is not None:
        field_value = attrs.get(field_name, Undefined)

        if field_value is Undefined:
            attrs[field_name] = Field(sa_type=sa_type) # [!code focus]
        elif isinstance(field_value, FieldInfo):
            if not hasattr(field_value, 'sa_type') or field_value.sa_type is Undefined:
                field_value.sa_type = sa_type # [!code focus]
```

#### `_extract_sa_type_from_annotation()` 的三种提取方式

在 `_sa_type.py` 中，从类型注解中用三种方式寻找 SQLAlchemy 列类型：

```python
def _extract_sa_type_from_annotation(annotation):
    # 方式 1：类型本身有 __sqlmodel_sa_type__ 属性
    if hasattr(annotation, '__sqlmodel_sa_type__'):
        return annotation.__sqlmodel_sa_type__

    # 方式 2：Annotated 的 metadata 中有
    if get_origin(annotation) is Annotated:
        for item in get_args(annotation)[1:]:
            if hasattr(item, '__sqlmodel_sa_type__'):
                return item.__sqlmodel_sa_type__
            schema = item.__get_pydantic_core_schema__(...)
            if 'sa_type' in schema.get('metadata', {}):
                return schema['metadata']['sa_type']

    # 方式 3：类型本身的 __get_pydantic_core_schema__ 返回 metadata
    schema = annotation.__get_pydantic_core_schema__(...)
    return schema.get('metadata', {}).get('sa_type')
```

以 `Array[str]` 为例：`__class_getitem__` 返回 `Annotated[list[str], _ArrayTypeHandler(str)]`，而 `_ArrayTypeHandler.__get_pydantic_core_schema__` 的 schema 中带有 `metadata={'sa_type': ARRAY(String)}`。元类找到后自动注入到 `Field(sa_type=ARRAY(String))` 中。

### 第五步：调用父类创建类

```python
result = super().__new__(cls, name, bases, attrs, **kwargs)
```

经过前四步预处理，把配置好的 `attrs` 和 `kwargs` 传给 SQLModel 原本的元类。

### 第六~八步：修复继承中的关系字段

```python
# 第六步：JTI 子类继承父类的 Relationship
for base in bases:
    if hasattr(base, '__sqlmodel_relationships__'):
        for rel_name, rel_info in base.__sqlmodel_relationships__.items():
            if rel_name not in result.__sqlmodel_relationships__:
                result.__sqlmodel_relationships__[rel_name] = rel_info

# 第七步：禁止子类重新定义父类的 Relationship
for base in bases:
    parent_relationships = getattr(base, '__sqlmodel_relationships__', {})
    for rel_name in parent_relationships:
        if rel_name in attrs:
            raise TypeError(f"不能重新定义父类的 Relationship '{rel_name}'")

# 第八步：从 model_fields 中移除 Relationship 字段
for rel_name in relationships:
    if rel_name in model_fields:
        del model_fields[rel_name]
if fields_removed:
    result.model_rebuild(force=True)
```

修复 SQLModel/SQLAlchemy 在处理继承 + 关系时的 bug：Relationship 被当 Pydantic 字段处理、JTI 子类丢失父类 Relationship、子类重定义导致歧义。

## `__DeclarativeMeta.__init__` — JTI 表创建

`__new__` 创建类后，`__init__` 做后续初始化。核心任务：**处理 JTI 子表的创建**。

```python
def __init__(cls, classname, bases, dict_, **kw):
    if not is_table_model_class(cls):
        ModelMetaclass.__init__(...)
        return

    base_is_table = any(is_table_model_class(base) for base in bases)

    if not base_is_table:
        # 第一个 table 类，正常流程
        cls._setup_relationships()
        DeclarativeMeta.__init__(...)
        return

    # 父类也是 table → 继承场景
    is_joined_inheritance = has_different_tablename and has_fk_to_parent

    if is_joined_inheritance:
        # JTI：创建子表
        # 1. 收集祖先表的列名
        # 2. 找到子类自有的字段
        # 3. 重建外键列
        # 4. 移除从祖先继承但不属于子表的列
        # 5. 设置子类自有的 Relationship
        DeclarativeMeta.__init__(...)

    else:
        # STI：子类共用父表
        ModelMetaclass.__init__(...)
        registry.map_imperatively(...)
```

::: info 为什么需要手动处理？
SQLModel 原本的逻辑是：如果父类已经是 table 模型，子类就**跳过** `DeclarativeMeta.__init__`。但 JTI 需要子类有自己的表！sqlmodel-ext 检测到 JTI 场景后，手动调用来创建子表。

对于 STI，使用 `registry.map_imperatively()` 把子类映射到父表，同时处理子类的 Relationship 和外键解析。
:::

### 第 1.5 步：`cache_ttl` 关键字

```python
# base.py:121-126
if 'cache_ttl' in kwargs:
    ttl = kwargs.pop('cache_ttl')
    if not isinstance(ttl, int) or ttl <= 0:
        raise ValueError(f"{name}: cache_ttl must be a positive integer, got: {ttl!r}")
    attrs['__cache_ttl__'] = ttl
```

`CachedTableBaseMixin` 使用 `__cache_ttl__` 控制缓存 TTL。元类把 `cache_ttl` 关键字参数转为类属性，让用户可以写 `class Foo(..., table=True, cache_ttl=1800):`。

## `SQLModelBase` 本身

```python
class SQLModelBase(SQLModel, metaclass=__DeclarativeMeta):
    model_config = ConfigDict(
        use_attribute_docstrings=True,  # 属性 docstring 作为字段描述
        validate_by_name=True,          # 允许通过字段名验证
        extra='forbid',                 # 禁止传入未定义的字段
    )

    @classmethod
    def get_computed_field_names(cls) -> set[str]:
        fields = cls.model_computed_fields
        return set(fields.keys()) if fields else set()
```

## `ExtraIgnoreModelBase` — 外部数据基类

```python
class ExtraIgnoreModelBase(SQLModelBase):
    model_config = ConfigDict(
        use_attribute_docstrings=True, validate_by_name=True, extra='ignore',
    )

    @model_validator(mode='before')
    @classmethod
    def _warn_unknown_fields(cls, data):
        if not isinstance(data, dict):
            return data
        accepted = {name for name, fi in cls.model_fields.items()}
        # 也包含 alias 和 validation_alias
        unknown = set(data.keys()) - accepted
        if unknown:
            logger.warning("External input contains unknown fields | model=%s ...", cls.__name__)
        return data
```

与 `SQLModelBase`（`extra='forbid'`）不同，`ExtraIgnoreModelBase` 使用 `extra='ignore'` 静默忽略未知字段，但会**记录 WARNING 日志**帮助开发者发现第三方 API 变更。

适用场景：第三方 API 响应、客户端 WebSocket 消息、外部 JSON 输入。

## `_compat.py` — Python 3.14 补丁

Python 3.14 引入 PEP 649（延迟求值注解），导致 SQLModel 内部函数出错。`_compat.py` 通过猴子补丁修复：

### 补丁 1：`get_sqlalchemy_type`

原函数遇到 `ForwardRef`、`ClassVar`、`Literal[StrEnum.MEMBER]` 等类型时调用 `issubclass()` 导致 `TypeError`。补丁在调用前拦截这些特殊情况。

### 补丁 2：`sqlmodel_table_construct`

多态继承的 table 子类中，继承的 Relationship 字段默认值可能被 SQLAlchemy 替换为 `InstrumentedAttribute` 对象。补丁跳过这些"被污染"的默认值。

::: info
两个补丁只在 Python >= 3.14 时激活。
:::

## 小结

| 元类步骤 | 解决的问题 |
|---------|-----------|
| 自动 `table=True` | 省去手写 |
| 检测 JTI/STI | 自动处理两种继承模式 |
| 合并 `__mapper_args__` | 简化多态配置语法 |
| 提取 `sa_type` | 自定义类型自动映射数据库列 |
| 修复继承关系字段 | 绕过 SQLModel/SQLAlchemy 的 bug |
| JTI 子表创建 | 让 SQLModel 支持联表继承 |

**核心设计理念**：用户只管声明式地写模型定义，元类在幕后处理所有 SQLAlchemy 配置细节。
