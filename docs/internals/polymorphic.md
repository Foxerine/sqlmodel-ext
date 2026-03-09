# 多态继承机制

::: tip 源码位置
`src/sqlmodel_ext/mixins/polymorphic.py` — `PolymorphicBaseMixin`、`AutoPolymorphicIdentityMixin`、`create_subclass_id_mixin`
:::

## `PolymorphicBaseMixin` — 自动配置父类

```python
class PolymorphicBaseMixin:
    _polymorphic_name: Mapped[str] = mapped_column(String, index=True)
```

`_polymorphic_name` 是**鉴别列**，数据库中存储 `"emailnotification"` 等字符串，SQLAlchemy 据此实例化对应子类。

单下划线前缀 `_` 的设计原因：存在数据库中（不像双下划线会被名称修饰）；不参与 API 序列化（Pydantic 默认跳过）；防止外部直接修改。

### `__init_subclass__` 自动配置

```python
def __init_subclass__(cls, polymorphic_on=None, polymorphic_abstract=None, **kwargs):
    super().__init_subclass__(**kwargs)

    if '__mapper_args__' not in cls.__dict__:
        cls.__mapper_args__ = {}

    # 自动设置鉴别列
    if 'polymorphic_on' not in cls.__mapper_args__:
        cls.__mapper_args__['polymorphic_on'] = polymorphic_on or '_polymorphic_name'

    # 自动检测是否为抽象类
    if polymorphic_abstract is None:
        has_abc = ABC in cls.__mro__
        has_abstract_methods = bool(getattr(cls, '__abstractmethods__', set()))
        polymorphic_abstract = has_abc and has_abstract_methods

    cls.__mapper_args__['polymorphic_abstract'] = polymorphic_abstract
```

`__init_subclass__` 在子类被定义时执行。效果：继承 `PolymorphicBaseMixin` 后不用手写 `__mapper_args__`；如果类继承了 `ABC` 且有抽象方法，自动标记为 `polymorphic_abstract=True`。

### 工具方法

```python
@classmethod
def _is_joined_table_inheritance(cls) -> bool:
    """子类表名与父类不同 → JTI"""

@classmethod
def get_concrete_subclasses(cls) -> list[type]:
    """递归获取所有非抽象子类"""

@classmethod
def get_identity_to_class_map(cls) -> dict[str, type]:
    """identity 字符串到类的映射"""
    # {'emailnotification': EmailNotification, ...}
```

## `create_subclass_id_mixin()` — JTI 外键

JTI 子类需要指向父表的外键。动态生成 Mixin：

```python
def create_subclass_id_mixin(parent_table_name: str) -> type:
    class SubclassIdMixin(SQLModelBase):
        id: UUID = Field(
            default_factory=uuid.uuid4,
            foreign_key=f'{parent_table_name}.id',
            primary_key=True,
        )
    SubclassIdMixin.__name__ = f'{ParentName}SubclassIdMixin'
    return SubclassIdMixin
```

动态生成而非手写：不同父表名导致外键目标不同，函数参数化解决。

**MRO 顺序至关重要**：Mixin 必须在继承列表**最前面**，其 `id` 才能覆盖 `UUIDTableBaseMixin` 的 `id`：

```python
class WebSearchTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
#                   ↑ 必须放在最前面
    ...  # ToolSubclassIdMixin 的 id（带外键）优先 // [!code highlight]
```

## `AutoPolymorphicIdentityMixin` — 自动 identity

```python
class AutoPolymorphicIdentityMixin:
    def __init_subclass__(cls, polymorphic_identity=None, **kwargs):
        super().__init_subclass__(**kwargs)

        if polymorphic_identity is not None:
            identity = polymorphic_identity        # 显式指定
        else:
            class_name = cls.__name__.lower()      # 类名小写

            parent_identity = None
            for base in cls.__mro__[1:]:
                if hasattr(base, '__mapper_args__'):
                    parent_identity = base.__mapper_args__.get('polymorphic_identity')
                    if parent_identity:
                        break

            if parent_identity:
                identity = f'{parent_identity}.{class_name}'
            else:
                identity = class_name

        cls.__mapper_args__['polymorphic_identity'] = identity
```

自动生成的 identity 格式为点分层级：

```python
class Function(Tool, ...)     # identity = 'function'
class CodeInterpreter(Function, ...)  # identity = 'function.codeinterpreter'
```

## STI 列注册（两阶段）

STI 子类字段需要作为 nullable 列添加到父表。这分两个阶段：

### Phase 1：`_register_sti_columns()`

在 `configure_mappers()` **之前**调用：

```python
@classmethod
def _register_sti_columns(cls):
    parent_table = None
    for base in cls.__mro__[1:]:
        if hasattr(base, '__table__'):
            parent_table = base.__table__
            break

    # JTI 检测——子类有自己的表就跳过
    if cls.__table__.name != parent_table.name:
        return

    for field_name, field_info in cls.model_fields.items():
        if field_name in parent_fields:   continue
        if field_name in existing_columns: continue

        column = get_column_from_field(field_info)
        column.nullable = True            # STI 子类字段必须 nullable // [!code warning]
        parent_table.append_column(column) # [!code focus]
```

### Phase 2：`_register_sti_column_properties()`

在 `configure_mappers()` **之后**调用：

```python
@classmethod
def _register_sti_column_properties(cls):
    child_mapper = inspect(cls).mapper
    parent_mapper = inspect(parent_class).mapper

    for field_name in cls.model_fields:
        if field_name in parent_fields: continue
        column = local_table.columns[field_name]

        child_mapper.add_property(field_name, ColumnProperty(column))
        parent_mapper.add_property(field_name, ColumnProperty(column))
```

### StrEnum 自动转换

STI 子类的 `StrEnum` 字段在数据库中存储为字符串。SQLAlchemy 加载时只返回 `str`，需要注册事件监听器自动转换：

```python
def _register_strenum_coercion_for_subclass(cls):
    strenum_fields = {}  # 找到所有非根类的 StrEnum 字段

    def _coerce(target):
        for field_name, enum_type in strenum_fields.items():
            raw = target.__dict__.get(field_name)
            if raw is not None and not isinstance(raw, enum_type):
                target.__dict__[field_name] = enum_type(str(raw))

    event.listens_for(cls, 'load')(_on_load)
    event.listens_for(cls, 'refresh')(_on_refresh)
```

## `_fix_polluted_model_fields()` — 修复默认值污染

SQLModel 继承时，SQLAlchemy 可能把字段默认值替换为 `InstrumentedAttribute` 或 `Column` 对象：

```python
def _fix_polluted_model_fields(cls):
    for field_name, current_field in cls.model_fields.items():
        if not isinstance(current_field.default, (InstrumentedAttribute, Column)):
            continue

        # 从 MRO 中找到原始 FieldInfo
        original = find_original_field_info(field_name)
        cls.model_fields[field_name] = FieldInfo(
            default=original.default,
            default_factory=original.default_factory,
            ...
        )
```

在多个地方被调用，确保 Pydantic 的 model_fields 始终包含正确的默认值。
