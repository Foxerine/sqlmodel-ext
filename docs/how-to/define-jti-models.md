# 定义 JTI（联表继承）模型

**目标**：把"通知系统"建模为一个抽象基类 `Notification` + 多个具体子类（`EmailNotification`、`PushNotification`），每个子类有自己的字段和数据库表，通过外键关联父表。

**何时选 JTI**：

- 子类字段差异**大**（每个子类有 5+ 个独占字段）
- 你想要每张子表都紧凑、没有大量 NULL 列
- 你能接受 JOIN 的性能开销

如果子类只多 1~2 个字段，选 [STI](./define-sti-models) 更合适。

## 1. 定义抽象基类

```python
from abc import ABC, abstractmethod
from uuid import UUID
from sqlmodel import Field
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin,
    PolymorphicBaseMixin, AutoPolymorphicIdentityMixin,
    create_subclass_id_mixin,
    Str64,
)

# 1. Base 类：纯字段，无表
class NotificationBase(SQLModelBase):
    user_id: UUID = Field(foreign_key='user.id')
    message: Str64

# 2. 抽象父类：建表 + 抽象方法
class Notification(
    NotificationBase,
    UUIDTableBaseMixin,
    PolymorphicBaseMixin,
    ABC,
):
    @abstractmethod
    async def deliver(self) -> None: ...
```

`PolymorphicBaseMixin` 自动添加 `_polymorphic_name` 鉴别列。`ABC` + 抽象方法会让 `polymorphic_abstract=True` 自动启用——抽象类不能被实例化。

## 2. 创建子类外键 Mixin

```python
NotificationSubclassIdMixin = create_subclass_id_mixin('notification') # [!code highlight]
```

这个动态生成的 Mixin 提供 `id: UUID = Field(default_factory=uuid7, primary_key=True, foreign_key='notification.id')`——也就是子类的主键同时是父表的外键，组成 JTI 的核心。它的 `default_factory` 与 `UUIDTableBaseMixin.id` 相同（UUIDv7），父表与子表的主键版本一致。

## 3. 定义具体子类

```python
class EmailNotification(
    NotificationSubclassIdMixin,    # ← 必须放第一位 // [!code highlight]
    Notification,
    AutoPolymorphicIdentityMixin,
    table=True,
):
    email_to: Str64
    subject: Str64

    async def deliver(self) -> None:
        await send_email(self.email_to, self.subject, self.message)


class PushNotification(
    NotificationSubclassIdMixin,    # ← 必须放第一位 // [!code highlight]
    Notification,
    AutoPolymorphicIdentityMixin,
    table=True,
):
    device_token: Str64

    async def deliver(self) -> None:
        await send_push(self.device_token, self.message)
```

::: warning MRO 顺序
`NotificationSubclassIdMixin` **必须**放在继承列表第一位。原因：它的 `id` 字段（带外键到父表）需要覆盖 `UUIDTableBaseMixin` 的 `id`（普通主键）。MRO 顺序错了 → 子表没有外键 → JTI 失败。
:::

`AutoPolymorphicIdentityMixin` 自动把类名小写设为 identity，所以 `EmailNotification.__mapper_args__['polymorphic_identity'] == 'emailnotification'`。

## 4. 查询：自动返回正确的子类

```python
notifications = await Notification.get(session, fetch_mode='all')
# notifications[0] 可能是 EmailNotification 实例
# notifications[1] 可能是 PushNotification 实例

for n in notifications:
    await n.deliver()  # 多态调度，无需 isinstance 判断
```

`get()` 内部会自动用 `with_polymorphic(cls, '*')` JOIN 所有子表，避免 N+1 查询。

## 5. 按子类查询

```python
emails = await EmailNotification.get(session, fetch_mode='all')
# 只返回 emailnotification 表的记录
```

## 验证表结构

迁移后数据库会有 3 张表：

```sql
notification          -- 父表（id, user_id, message, _polymorphic_name, created_at, updated_at）
emailnotification     -- 子表（id PK FK→notification.id, email_to, subject）
pushnotification      -- 子表（id PK FK→notification.id, device_token）
```

每条 EmailNotification 在 `notification` 和 `emailnotification` 表中各占一行。

## 相关参考

- [`PolymorphicBaseMixin` / `create_subclass_id_mixin`](/reference/mixins#polymorphicbasemixin)
- [多态继承机制讲解](/explanation/polymorphic-internals)
- [STI 单表继承](./define-sti-models)（对比方案）
