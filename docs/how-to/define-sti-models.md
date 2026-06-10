# 定义 STI（单表继承）模型

**目标**：把"用户文件"建模为一个父类 `UserFile` + 多个子类（`PendingFile`、`CompletedFile`），所有子类共享一张数据库表，子类专属字段以 nullable 列存在。

**何时选 STI**：

- 子类只多 1~2 个独占字段
- 你想避免 JOIN（单表查询更快）
- 你能接受表中有少量 NULL 列

如果子类字段差异大（5+ 个独占字段），选 [JTI](./define-jti-models)。

## 1. 定义父类

```python
from datetime import datetime
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    NonNegativeBigInt, Sha256Hex, Str256,
)

class UserFile(
    SQLModelBase,
    UUIDTableBaseMixin,
    PolymorphicBaseMixin,
    table=True,
):
    filename: Str256
    user_id: UUID = Field(foreign_key='user.id')
```

注意 STI 父类**不**继承 `ABC`——它是一张实际存在的表，子类**共享**它。

## 2. 定义子类（不要外键 Mixin）

```python
class PendingFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    upload_deadline: datetime | None = None  # 自动加为 nullable 列 // [!code highlight]


class CompletedFile(UserFile, AutoPolymorphicIdentityMixin, table=True):
    file_size: NonNegativeBigInt | None = None  # 自动加为 nullable 列 // [!code highlight]
    sha256: Sha256Hex | None = None
```

::: warning 子类字段必须是 nullable
STI 中所有子类共享一张表。`PendingFile.upload_deadline` 这一列对 `CompletedFile` 行没有意义，所以必须可空（`| None`）。sqlmodel-ext 会强制把列声明为 `nullable=True`。
:::

`AutoPolymorphicIdentityMixin` 自动设置 `_polymorphic_name = 'pendingfile'` / `'completedfile'`。

## 3. 调用两阶段注册函数

STI 子类的字段需要分两步注册到父表：**这两步必须在所有模型定义完成后调用**，通常放在应用启动代码或 `models/__init__.py` 末尾。

```python
from sqlmodel_ext import (
    register_sti_columns_for_all_subclasses,
    register_sti_column_properties_for_all_subclasses,
)
from sqlalchemy.orm import configure_mappers

# 所有 STI 模型 import 完成后：
register_sti_columns_for_all_subclasses()       # Phase 1：把列加到父表 // [!code warning]
configure_mappers()                              # SQLAlchemy 配置 mapper
register_sti_column_properties_for_all_subclasses()  # Phase 2：把列绑定到 mapper // [!code warning]
```

::: danger 调用顺序很重要
Phase 1 必须在 `configure_mappers()` **之前**，Phase 2 必须在**之后**。原因见 [多态继承机制讲解](/explanation/polymorphic-internals#sti-列注册（两阶段）)。
:::

## 4. 查询：自动按 `_polymorphic_name` 过滤

```python
# 查所有文件（无论子类）
all_files = await UserFile.get(session, fetch_mode='all')

# 只查 pending 文件
pending = await PendingFile.get(session, fetch_mode='all')
# 内部 SQL: SELECT * FROM userfile WHERE _polymorphic_name = 'pendingfile'
```

::: info STI 自动过滤
SQLAlchemy/SQLModel 默认**不会**为 STI 子类查询自动加 `WHERE _polymorphic_name IN (...)` 过滤。sqlmodel-ext 在 `get()` 中主动补上这个条件，使用 `mapper.self_and_descendants` 包含当前类及其所有子类。
:::

## 验证表结构

迁移后数据库只有一张表：

```sql
userfile (
    id UUID PRIMARY KEY,
    filename VARCHAR(256) NOT NULL,
    user_id UUID NOT NULL,
    _polymorphic_name VARCHAR NOT NULL,    -- 鉴别列（'pendingfile' / 'completedfile'）
    upload_deadline TIMESTAMP NULL,         -- PendingFile 的字段，对其他子类为 NULL
    file_size BIGINT NULL,                  -- CompletedFile 的字段
    sha256 VARCHAR(64) NULL,                -- CompletedFile 的字段
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
```

## STI 子类字段类型冲突

如果两个子类用相同字段名但不同类型（如 `Vendor1.duration: int` vs `Vendor2.duration: KlingDurationEnum`），sqlmodel-ext 会在 Phase 1 抛 `TypeError`。解决方法：**用 vendor 前缀重命名字段**（`vendor1_duration` / `kling_25_duration`）。

## 子类多层继承的 polymorphic identity

`AutoPolymorphicIdentityMixin` 生成的 identity 是点分层级：

```python
class Generator(SQLModelBase, ..., PolymorphicBaseMixin, table=True): ...
# identity = 'generator'

class FileGenerator(Generator, AutoPolymorphicIdentityMixin, table=True): ...
# identity = 'generator.filegenerator'

class ImageGenerator(FileGenerator, AutoPolymorphicIdentityMixin, table=True): ...
# identity = 'generator.filegenerator.imagegenerator'
```

数据迁移中按 `_polymorphic_name` 过滤时，要用 `LIKE '%xxx'` 匹配后缀。

## 相关参考

- [`register_sti_columns_for_all_subclasses` / `register_sti_column_properties_for_all_subclasses`](/reference/mixins#register-sti-columns-for-all-subclasses)
- [多态继承机制讲解（两阶段注册原理）](/explanation/polymorphic-internals#sti-列注册（两阶段）)
- [JTI 联表继承](./define-jti-models)（对比方案）
