# 元类与 SQLModelBase

::: tip 源码位置
`src/sqlmodel_ext/base.py` — `SQLModelBase`、`SQLModelExtConfig` 和 `__DeclarativeMeta` 元类

`src/sqlmodel_ext/_sa_type.py` — 从 Annotated 元数据提取 SQLAlchemy 列类型

`src/sqlmodel_ext/_compat.py` — Python 3.14 兼容性补丁
:::

这是整个项目的**基石**，也是"[单点真相](./single-source-of-truth)"得以成立的地方：你只在一处声明事实，元类在**类创建的那一瞬间**把它派生成 SQLAlchemy 需要的一切配置。所有模型类都继承 `SQLModelBase`，而它的元类 `__DeclarativeMeta` 完成这些工作。

::: info 关于本页的代码片段
下面的片段是对 `base.py` 的**简化摘录**，用来说明每一步的意图；步骤编号与源码注释中的编号一致。准确实现以源码为准。
:::

## 用户写的代码 vs 元类做的事

```python
class UserBase(SQLModelBase):
    name: NonEmptyStrippedStr64
    email: Str255

class User(UserBase, UUIDTableBaseMixin):   # 没写 table=True
    pass

class UserUpdate(UserBase, partial=True):   # 没重写任何字段
    pass
```

| 类 | 是否建数据库表 | 角色 |
|---|---|---|
| `UserBase` | 否 | 纯数据模型，只定义字段 |
| `User` | 是 | 继承字段 + CRUD 能力，对应数据库表 |
| `UserUpdate` | 否 | PATCH 请求体，每个字段都变成 `Unset \| T` |

## `__DeclarativeMeta.__new__` 逐步拆解

### 第 1 步：自动 `table=True`

```python
is_intended_as_table = any(getattr(b, '_has_table_mixin', False) for b in bases)
if is_intended_as_table and 'table' not in kwargs:
    kwargs['table'] = True
```

父类里有 `_has_table_mixin = True`（`TableBaseMixin` 上定义）就自动加上 `table=True`。

### 第 1.5 步：`cache_ttl` 关键字

```python
if 'cache_ttl' in kwargs:
    ttl = kwargs.pop('cache_ttl')
    if not isinstance(ttl, int) or ttl <= 0:
        raise ValueError(f"{name}: cache_ttl must be a positive integer, got: {ttl!r}")
    attrs['__cache_ttl__'] = ttl
```

让你写 `class Foo(..., cache_ttl=1800):`，`CachedTableBaseMixin` 读取 `__cache_ttl__`。

### 第 2 步：检测继承类型（JTI vs STI）

```python
parent_tablename = None
for base in bases:
    if is_table_model_class(base) and hasattr(base, '__tablename__'):
        parent_tablename = base.__tablename__
        break

# 父类字段里有指向父表的外键 → JTI 的特征
...
if parent_tablename is not None and will_be_table and not has_own_tablename and not has_fk_to_parent:
    attrs['__tablename__'] = parent_tablename   # STI：共用父表
```

table 子类继承 table 父类时：**有外键指向父表** → JTI，子类有自己的表；**没有外键** → STI，共用父表。

### 第 3 步：合并 `__mapper_args__`

```python
collected_mapper_args = {}
if 'mapper_args' in kwargs:
    collected_mapper_args.update(kwargs.pop('mapper_args'))
for key in cls._KNOWN_MAPPER_KEYS:  # polymorphic_on, polymorphic_identity, ...
    if key in kwargs:
        collected_mapper_args[key] = kwargs.pop(key)
```

让用户可以用简洁语法：

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

`_KNOWN_MAPPER_KEYS`：`polymorphic_on`、`polymorphic_identity`、`polymorphic_abstract`、`version_id_col`、`concrete`。

### 第 3.5 步：乐观锁接线

```python
if will_be_table and any(getattr(b, '_has_optimistic_lock', False) for b in bases):
    if not _is_inheriting_table and 'version_id_col' not in attrs.get('__mapper_args__', {}):
        def _mapper_args_with_version_col(target_cls, _static=...):
            merged = dict(_static)
            merged['version_id_col'] = target_cls.__table__.c[OPTIMISTIC_LOCK_VERSION_COLUMN]
            return merged
        attrs['__mapper_args__'] = declared_attr.directive(_mapper_args_with_version_col)
```

混入 `OptimisticLockMixin` 的根表类，会把 `oplock_version` 列注册为 SQLAlchemy 的 `version_id_col`，于是每次 UPDATE 都带上 `WHERE ... AND oplock_version = :current`。`Column` 对象只有在表建好之后才存在，所以这里必须用延迟求值的 `declared_attr`。STI/JTI 子类通过 mapper 继承共享它。——**能力写在哪里，策略就在哪里**：用户只混入一个 mixin。

### 第 3.6 步：`table_args` 中的 `CustomTableArg`、`table_name`、`abstract`

处理 `table_args` 时，元类把继承 `CustomTableArg` 的标记对象**拆出来**，不传给 SQLAlchemy：

```python
real_table_args, custom_table_args = [], []
for arg in raw_table_args:
    (custom_table_args if isinstance(arg, CustomTableArg) else real_table_args).append(arg)
attrs['__table_args__'] = tuple(real_table_args)
# super().__new__ 之后追加到模块级队列 classes_with_custom_table_args
```

**为什么**：SQLAlchemy 的 `Table.__init__` 会立即消费 `__table_args__` 的每个元素——引用尚不存在的列的 `Index` 当场抛错。`CustomTableArg` 是通用的"延迟处理"标记：元类只负责拦截 + 入队，不知道具体语义；目前的消费者是 `mixins.polymorphic.DeferredIndex`（STI 子类列的延迟索引）。`table_name=` / `abstract=` 分别转成 `__tablename__` / `__abstract__`。

### 第 4 步：解析注解，记录"本类自己声明了哪些字段"

```python
annotations, annotation_strings, eval_globals, eval_locals = resolve_annotations(attrs)
_own_annotation_names = frozenset(annotations)
```

这份快照必须在任何注解注入**之前**拍下：之后的步骤会把继承来的字段注入 `annotations`，那时"本类声明的"与"继承后注入的"就分不清了。第 4.5.b 步与第 4.6 步都依赖它。

### 第 4.5 步：恢复 `Annotated[T, Field(...)]` 中的 SQLModel 属性

Pydantic v2 处理 `Annotated` 元数据时会把 `sqlmodel.main.FieldInfo` 换成 `pydantic.fields.FieldInfo`，后者不认识 `foreign_key`、`sa_type` 等 SQLModel 属性。`_recover_annotated_sqlmodel_fields()` 对 **table 类**（包括从父类继承来的 `Annotated` 字段）把它们还原成 `= Field(...)` 形式；非 table 类保留原样，供子 table 类继承。合并多个 `FieldInfo` 时，显式的 `default=None` 被当作真实的值（而不是"未设置"）保留——否则字段会静默变成必填。带右侧 `Field` 的字段（`name: Alias = Field(...)`）以 Pydantic 对该声明的解析结果为准重建——table 类自己声明时用 `FieldInfo.from_annotated_attribute()`，table 类**继承**（未在类体里重新声明）时用基类已解析的字段 `Base.model_fields[name]`。两者得到同一个字段：Pydantic 层属性原样采用（显式的 `alias=None` 仍是 `None`，与纯 SQLModel 一致），只补全列属性——别名的 `Field` 与右侧 `Field` 各自的 `FieldInfoMetadata` 载体被折叠成一个，因为 SQLModel 只读第一个。仍留在类注解里的元数据（例如与别名 `Field` 并列的 `AfterValidator`）不会再复制进字段，因此只执行一次。

### 第 4.5.b 步：`oplock_version` 是保留名

```python
if OPTIMISTIC_LOCK_VERSION_COLUMN in _own_annotation_names:
    raise TypeError(f"{name}: 'oplock_version' is reserved for OptimisticLockMixin's version_id_col ...")
```

任何类在**自己的类体**里声明 `oplock_version` 都会失败——不论是否启用乐观锁。只对启用了乐观锁的类保留是不够的：一个没启用的类可以声明同名的领域字段，而它的子类重新启用乐观锁时，这个字段会被静默接成 `version_id_col`。

### 第 4.6 步：`partial=True`

```python
if 'all_fields_optional' in kwargs:
    raise TypeError(f"{name}: the 'all_fields_optional' class keyword was removed in sqlmodel-ext 0.5.0. ...")
is_partial = kwargs.pop('partial', False)
if is_partial:
    if will_be_table:
        raise TypeError(f"{name}: 'partial=True' cannot be combined with 'table=True' ...")
    _apply_partial(annotations, attrs, bases, _own_annotation_names)
```

`_apply_partial()` 从基类的 `model_fields` 收集字段名，再沿 MRO 取回**原始注解**（保留 `Annotated` 元数据），然后：

- `T` → `Unset | T`，默认值设为 `Unset`（可空字段自然得到 `Unset | T | None`）；
- 字段写成 `field: T = Field(gt=..., le=...)`（非 `Annotated` 形式）时，约束在右侧，元类从基类 `model_fields[name].metadata` 把它们取回并重新包进 `Annotated`；
- `exclude` / `alias` / `validation_alias` / `serialization_alias` / `repr` / `frozen` / `deprecated` 这些**字段级**属性放在联合成员上会被 Pydantic 静默丢弃，所以由 `_hoist_field_metadata()` 从基类已解析的字段读取并提升到联合外层（`alias_generator` 生成的别名不提升，与普通继承一样由派生类重新生成），同时由 `_union_member_annotation()` 从联合成员中移除；约束与 `discriminator` 留在内层；
- 跳过本类自己声明的字段（`_own_annotation_names`）与 `Literal` 字段。

生成的注解是运行时的，静态类型检查器看到的仍是基类注解。需要静态强制时，显式声明 `Unset | T = Unset`，或用实验性的 `python -m sqlmodel_ext.check_derived`（见 [检查 partial DTO 的误用](/how-to/check-partial-dtos)）。详见 [Unset 三态](./unset-three-state)。

### 第 4.7 步：从类型注解中提取 `sa_type`

```python
for field_name, field_type in annotations.items():
    sa_type = extract_sa_type_from_annotation(field_type)
    if sa_type is not None:
        field_value = attrs.get(field_name, Undefined)
        if field_value is Undefined:
            # 没有 "= Field(...)"：优先从 Annotated 里取回用户的 FieldInfo，
            # 保住 default_factory / max_length 等，只把 sa_type 加进去
            annotated_fi = _find_field_info_in_annotated(field_type)
            attrs[field_name] = annotated_fi if annotated_fi is not None else Field(sa_type=sa_type)
        elif isinstance(field_value, FieldInfo):
            _durably_set_sa_type(field_value, sa_type)
        else:
            # 裸默认值（如 fpath: FilePathType = Path("a.txt")）
            attrs[field_name] = Field(default=field_value, sa_type=sa_type)
```

`_durably_set_sa_type()` 把 `sa_type` 写进 `FieldInfo.metadata` 里的 `FieldInfoMetadata` 条目——这正是 SQLModel 自己的 `Field(sa_type=...)` 使用、并且能撑过 Pydantic 重建 `model_fields` 的通道；直接 `setattr` 会在列构建之前丢失。已经显式设置的 `sa_type` 不会被覆盖。

#### `extract_sa_type_from_annotation()` 的三种提取方式

```python
def extract_sa_type_from_annotation(annotation):
    # 方式 1：类型本身有 __sqlmodel_sa_type__ 属性
    # 方式 2：Annotated 的元数据项有 __sqlmodel_sa_type__，或其 __get_pydantic_core_schema__
    #         返回的 schema 的 metadata 里有 'sa_type'
    # 方式 3：类型本身的 __get_pydantic_core_schema__ 返回的 metadata 里有 'sa_type'
    ...
```

以 `Array[str]` 为例：`__class_getitem__` 返回 `Annotated[list[str], ArrayTypeHandler(str)]`，而 `ArrayTypeHandler` 的 schema 带有 `metadata={'sa_type': ARRAY(String)}`；`JSON100K` 的 schema 带有 `metadata={'sa_type': JSONB}`。**类型自己声明它的列类型**，元类负责把它送到列构建器。

### 第 5–7 步：保存 SQLModel 的 `FieldInfo`，调用父类，再恢复

```python
_saved_sqlmodel_fis = {fn: attrs[fn] for fn in annotations if isinstance(attrs.get(fn), SQLModelFieldInfo)}  # 第 5 步（仅 table 类）
result = super().__new__(cls, name, bases, attrs, **kwargs)                                                  # 第 6 步
# 第 6.5 步：把拦截到的 CustomTableArg 追加到模块级队列
# 第 7 步：Pydantic 重建 model_fields 时丢掉了 SQLModel 专属属性（unique / index / foreign_key / sa_type ...），
#         用保存的 SQLModelFieldInfo 合并回去并重建 Column
```

合并时，布尔标志不会被 `False` 覆盖（`unique=False` 不会关掉继承来的 `unique=True`），`FieldInfoMetadata` 载体会被折叠成一个（SQLModel 只读第一个；否则类型别名里全空的载体会遮住 `= Field(primary_key=True)`——这正是 sqlmodel ≥ 0.0.32 上 `id: NonNegativeInt = Field(primary_key=True)` 丢失主键的原因）。

### 第 8–9 步：继承中的关系字段

```python
# 第 8 步：JTI 子类继承父类的 Relationship
# 第 9 步：禁止子类重新定义父类的 Relationship
for base in bases:
    for rel_name in getattr(base, '__sqlmodel_relationships__', {}):
        if rel_name in attrs:
            raise TypeError(f"Class {name} cannot redefine parent {base.__name__}'s Relationship field '{rel_name}'. ...")
```

### 第 10 步：继承字段描述

`use_attribute_docstrings` 从源码 AST 读 docstring。子类覆盖字段却没写 docstring，或者 `partial=True` 以编程方式生成了注解时，源码里没有 docstring，描述就会丢失。元类沿 MRO 从父类的 `model_fields` 补回描述——**描述只写一次**，派生 DTO 自动带上。

### 第 11 步：从 `model_fields` 中移除 Relationship 字段

Relationship 不是 Pydantic 字段；元类把它们从 `model_fields` / `__pydantic_fields__` 里删掉，必要时 `model_rebuild(force=True)`。

### 第 12 步：登记 partial 类

`partial=True` 创建的类按顺序追加到 `sqlmodel_ext.base.optional_dto_registry`，供契约测试等工具枚举。

### PEP 604 可空关系注解归一化

创建 Relationship 时，元类不直接调用 SQLModel 的 `get_relationship_to`，而是先经 `_resolve_relationship_target` 包装：把**扁平字符串 / ForwardRef 形式**的可空关系注解（如 `'Parent | None'`）用 `ast` 归一化为结构化的 `ForwardRef('Parent')`，再交给上游。

根因：`get_relationship_to` 只能从**已求值的** `typing.Union` 中剥离 `None`，无法解析「整体是字符串」的 PEP 604 注解——它会把整串 `'Parent | None'` 当类名丢给 SQLAlchemy。

```python
class Child(SQLModelBase, UUIDTableBaseMixin, table=True):
    parent_id: uuid.UUID | None = Field(default=None, foreign_key="parent.id")
    parent: 'Parent | None' = Relationship(back_populates="children")   # 无需 Optional['Parent']
```

覆盖全部形态：`Foo` / `pkg.Foo` / `Foo | None` / `None | Foo` / `Optional[Foo]` / `Union[Foo, None]` 以及内层再套引号（`Optional['Foo']`）。无法归一化为单一目标（如 `Foo | Bar`）时原样交还上游，抛它本来的错误。归一化只作用于传给 `get_relationship_to` 的临时值，不改写 `cls.__annotations__`。

未显式指定 `lazy` 的关系默认 `lazy='raise_on_sql'`：异步环境里意外的懒加载会立即报错，而不是变成 `MissingGreenlet`。

## `__DeclarativeMeta.__init__` — JTI 表创建

`__new__` 创建类后，`__init__` 做后续初始化。核心任务：**处理 JTI 子表的创建**。

```python
def __init__(cls, classname, bases, dict_, **kw):
    if not is_table_model_class(cls):
        ModelMetaclass.__init__(...)
        return

    base_is_table = any(is_table_model_class(base) for base in bases)
    if not base_is_table:
        cls._setup_relationships()
        DeclarativeMeta.__init__(...)
        return

    # 父类也是 table → 继承场景
    if is_joined_inheritance:
        # JTI：收集祖先表列名、找到子类自有字段、重建外键列、
        # 移除不属于子表的继承列、设置子类自有 Relationship
        DeclarativeMeta.__init__(...)
    else:
        # STI：子类共用父表
        ModelMetaclass.__init__(...)
        registry.map_imperatively(...)
```

::: info 为什么需要手动处理？
SQLModel 原本的逻辑是：如果父类已经是 table 模型，子类就**跳过** `DeclarativeMeta.__init__`。但 JTI 需要子类有自己的表！sqlmodel-ext 检测到 JTI 场景后手动调用来创建子表。对于 STI，使用 `registry.map_imperatively()` 把子类映射到父表。
:::

## `SQLModelBase` 本身

```python
class SQLModelBase(SQLModel, metaclass=__DeclarativeMeta):
    model_config = SQLModelExtConfig(
        use_attribute_docstrings=True,  # 属性 docstring 作为字段描述
        validate_by_name=True,          # 允许通过字段名验证
        extra='forbid',                 # 禁止传入未定义的字段
    )
```

除了配置，它还承载三态语义与几项构造期不变式：

| 成员 | 作用 |
|---|---|
| `annotation_is_omissible()` / `field_is_omissible()` | "这个字段能不能不传"——判断依据是注解里有没有 `Unset` |
| `_normalise_omitted_sentinel`（`mode='before'` 校验器） | 开启 `omitted_sentinel` 时，把入站 dict 中任意深度的 `'__omitted__'` 替换为 `Unset` |
| `model_json_schema()` | 开启 `omitted_sentinel` 时，给可省略字段（含嵌套模型与自引用模型）注入哨兵分支。必须在 `model_json_schema()` 出口处做：`__get_pydantic_json_schema__` 看到的是 Pydantic 之后还会重组的中间产物 |
| `__pydantic_init_subclass__` + `model_post_init` | 类创建时发现 `JSON100K` / `JSONList100K` 字段，构造时检查可编码性与 100K 上限——`table=True` 模型跳过 Pydantic 校验，这道检查依然生效。覆写 `model_post_init` 必须调用 `super()` |
| `__get_pydantic_json_schema__` | 修复 `$ref` 属性丢失 `description` |
| `submitted_fields_among()` | 显式提交的字段 ∩ 给定模型的字段 |
| `validate_list()` / `get_computed_field_names()` | 批量转换 / 列出 computed 字段 |

## `ExtraIgnoreModelBase` — 外部数据基类

```python
class ExtraIgnoreModelBase(SQLModelBase):
    model_config = SQLModelExtConfig(
        use_attribute_docstrings=True, validate_by_name=True, extra='ignore',
    )

    @model_validator(mode='before')
    @classmethod
    def _warn_unknown_fields(cls, data):
        ...  # 字段名、alias、validation_alias（含 AliasChoices 的每个字符串选项）算已知
        if unknown:
            logger.warning("External input contains unknown fields | model=%s ...", cls.__name__, ...)
        return data
```

与 `SQLModelBase`（`extra='forbid'`）不同，它静默忽略未知字段，但会**记录 WARNING 日志**帮助开发者发现第三方 API 变更。适用场景：第三方 API 响应、客户端 WebSocket 消息、外部 JSON 输入。

## `_compat.py` — Python 3.14 补丁

Python 3.14 引入 PEP 649（延迟求值注解），导致 SQLModel 内部函数出错。`_compat.py` 修复两处：

- **`get_sqlalchemy_type`**：原函数遇到 `ForwardRef`、`ClassVar`、`Literal[StrEnum.MEMBER]` 等类型时调用 `issubclass()` 导致 `TypeError`。补丁在调用前拦截这些特殊情况，并尊重用户显式设置的 `Field(sa_type=...)`。
- **`sqlmodel_table_construct`**：多态继承的 table 子类中，继承的 Relationship 字段默认值可能被替换为 `InstrumentedAttribute` 对象。补丁跳过这些"被污染"的默认值。

两个补丁只在 Python >= 3.14 时激活。

## 小结

| 元类步骤 | 消除的重复声明 / 解决的问题 |
|---------|-----------|
| 自动 `table=True` | "是不是表"只由继承 `TableBaseMixin` 表达 |
| 检测 JTI/STI | 继承方式由"有没有指向父表的外键"表达 |
| 合并 `__mapper_args__` | 多态配置用关键字参数，不写字典 |
| 乐观锁接线 | 混入 mixin 即可，不手写 `version_id_col` |
| 保留 `oplock_version` | 让误用在类创建时失败 |
| `partial=True` | 更新 DTO 从基类派生，不重写字段 |
| 提取 `sa_type` | 自定义类型自己声明列类型 |
| 恢复 SQLModel `FieldInfo` / 继承描述 | 约束与描述只写一次，继承链上不丢 |
| 修复继承关系字段 / JTI 子表创建 | 绕过 SQLModel/SQLAlchemy 在继承上的缺陷 |

**核心设计理念**：用户只声明式地写模型定义，元类在幕后把它派生成所有 SQLAlchemy 配置——每个事实只写一次。
