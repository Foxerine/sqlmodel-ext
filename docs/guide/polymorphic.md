# 多态继承

多态继承让不同类型的记录共享基础字段，同时各自拥有专属字段。查询时自动返回正确的子类实例。

## 什么时候需要多态？

当多种类型的对象共享相同的基础字段，但各自有额外的专属字段时。比如通知系统：

- 所有通知都有 `user_id` 和 `message`
- 邮件通知额外有 `email_to`
- 推送通知额外有 `device_token`

## 两种模式

### 联表继承（JTI）

每个子类一张表，通过外键关联父表。**子类字段差异大时选用。**

```
notification 表                emailnotification 表
┌────────────────────────┐    ┌───────────────────────┐
│ id (PK)                │←───│ id (PK, FK)           │
│ user_id                │    │ email_to              │
│ message                │    └───────────────────────┘
│ _polymorphic_name      │
└────────────────────────┘
```

### 单表继承（STI）

所有子类共用一张表。**子类额外字段少（1~2 个）时选用。**

```
notification 表
┌────────────────────────────────────┐
│ id (PK)                            │
│ user_id, message                   │
│ _polymorphic_name                  │  ← 鉴别列
│ email_to        (nullable)         │
│ device_token    (nullable)         │
└────────────────────────────────────┘
```

## JTI 用法

```python
from abc import ABC, abstractmethod
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
)

# 1. Base 类（纯字段，无表）
class ToolBase(SQLModelBase):
    name: str

# 2. 抽象父类（创建父表 tool）
class Tool(ToolBase, UUIDTableBaseMixin, PolymorphicBaseMixin, ABC):
    @abstractmethod
    async def execute(self) -> str: ...

# 3. 创建外键 Mixin（id → tool.id）
ToolSubclassIdMixin = create_subclass_id_mixin('tool')

# 4. 具体子类（各自创建子表）
class WebSearchTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    search_url: str
    async def execute(self) -> str:
        return f"Searching {self.search_url}"

class CalculatorTool(ToolSubclassIdMixin, Tool, AutoPolymorphicIdentityMixin, table=True):
    precision: int = 2
    async def execute(self) -> str:
        return "Calculating..."
```

::: warning MRO 顺序
`ToolSubclassIdMixin` 必须放在继承列表的**最前面**，这样它的 `id` 字段（带外键）才能覆盖 `UUIDTableBaseMixin` 的 `id`。
:::

查询时自动返回正确的子类：

```python
tools = await Tool.get(session, fetch_mode="all")
# tools[0] 是 WebSearchTool 实例
# tools[1] 是 CalculatorTool 实例
await tools[0].execute()  # 调用子类方法
```

## STI 用法

```python
class UserFile(SQLModelBase, UUIDTableBaseMixin, PolymorphicBaseMixin, table=True):
    filename: str

class PendingFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    upload_deadline: datetime | None = None   # nullable，加到 userfile 表

class CompletedFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    file_size: int | None = None              # nullable，加到 userfile 表

# 所有模型定义完成后（configure_mappers 前后）：
from sqlmodel_ext import (
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)
register_sti_columns_for_all_subclasses()       # Phase 1：加列
# configure_mappers() ...
register_sti_column_properties_for_all_subclasses()  # Phase 2：加属性
```

STI 子类的字段会自动以 nullable 列添加到父表中。

## 关键组件

| 组件 | 作用 |
|------|------|
| `PolymorphicBaseMixin` | 添加鉴别列 `_polymorphic_name`，自动配置 `__mapper_args__` |
| `AutoPolymorphicIdentityMixin` | 自动生成 `polymorphic_identity`（类名小写） |
| `create_subclass_id_mixin(table)` | 生成带外键的 ID Mixin（JTI 专用） |

## JTI vs STI 选择

| 考量 | 选 JTI | 选 STI |
|------|--------|--------|
| 子类额外字段多 | 是 | |
| 子类额外字段少（1~2个） | | 是 |
| 需要频繁查询所有类型 | | 是（不需要 JOIN） |
| 子类数据独立性要求高 | 是 | |
| 表结构简洁 | 是（各表紧凑） | 否（一张大宽表） |
