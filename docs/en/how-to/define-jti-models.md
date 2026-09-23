# Define JTI (joined table inheritance) models

**Goal**: model a "notification system" as one abstract base class `Notification` plus several concrete subclasses (`EmailNotification`, `PushNotification`), each with its own fields and database table linked by a foreign key to the parent.

**When to choose JTI**:

- Subclasses have **many** distinct fields (5+ exclusive fields each)
- You want each sub-table to be compact, without lots of NULL columns
- You can accept the JOIN overhead

If subclasses only add 1–2 fields, [STI](./define-sti-models) is a better fit.

## 1. Define the abstract base class

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

# 1. Base class: pure fields, no table
class NotificationBase(SQLModelBase):
    user_id: UUID = Field(foreign_key='user.id')
    message: Str64

# 2. Abstract parent: table + abstract method
class Notification(
    NotificationBase,
    UUIDTableBaseMixin,
    PolymorphicBaseMixin,
    ABC,
):
    @abstractmethod
    async def deliver(self) -> None: ...
```

`PolymorphicBaseMixin` automatically adds the `_polymorphic_name` discriminator column. `ABC` + abstract methods automatically enable `polymorphic_abstract=True` — abstract classes cannot be instantiated.

## 2. Create the subclass FK mixin

```python
NotificationSubclassIdMixin = create_subclass_id_mixin('notification') # [!code highlight]
```

This dynamically generated mixin provides `id: UUID = Field(default_factory=uuid7, primary_key=True, foreign_key='notification.id')` — i.e. the subclass's primary key is also the parent's foreign key. That's the heart of JTI. Its `default_factory` is the same as `UUIDTableBaseMixin.id`'s (UUIDv7), so the parent and child table primary keys use the same version.

## 3. Define concrete subclasses

```python
class EmailNotification(
    NotificationSubclassIdMixin,    # ← must be first // [!code highlight]
    Notification,
    AutoPolymorphicIdentityMixin,
    table=True,
):
    email_to: Str64
    subject: Str64

    async def deliver(self) -> None:
        await send_email(self.email_to, self.subject, self.message)


class PushNotification(
    NotificationSubclassIdMixin,    # ← must be first // [!code highlight]
    Notification,
    AutoPolymorphicIdentityMixin,
    table=True,
):
    device_token: Str64

    async def deliver(self) -> None:
        await send_push(self.device_token, self.message)
```

::: warning MRO order
`NotificationSubclassIdMixin` **must** be first in the inheritance list. Reason: its `id` field (with FK to the parent table) needs to override `UUIDTableBaseMixin`'s `id` (plain primary key). Wrong MRO order → no foreign key in the sub-table → JTI broken.
:::

`AutoPolymorphicIdentityMixin` automatically uses the lowercased class name as the identity, so `EmailNotification.__mapper_args__['polymorphic_identity'] == 'emailnotification'`.

## 4. Querying: subclass instances returned automatically

```python
notifications = await Notification.get(session, fetch_mode='all')
# notifications[0] might be an EmailNotification instance
# notifications[1] might be a PushNotification instance

for n in notifications:
    await n.deliver()  # Polymorphic dispatch, no isinstance checks needed
```

`get()` automatically uses `with_polymorphic(cls, '*')` to JOIN every sub-table, avoiding N+1 queries.

## 5. Querying by subclass

```python
emails = await EmailNotification.get(session, fetch_mode='all')
# Only returns rows from the emailnotification table
```

## Verifying the schema

After migration the database has 3 tables:

```sql
notification          -- Parent (id, user_id, message, _polymorphic_name, created_at, updated_at)
emailnotification     -- Child (id PK FK→notification.id, email_to, subject)
pushnotification      -- Child (id PK FK→notification.id, device_token)
```

Each `EmailNotification` occupies one row in both `notification` and `emailnotification`.

## Related reference

- [`PolymorphicBaseMixin` / `create_subclass_id_mixin`](/en/reference/mixins#polymorphicbasemixin)
- [Polymorphic inheritance internals](/en/explanation/polymorphic-internals)
- [STI single table inheritance](./define-sti-models) (the alternative)
