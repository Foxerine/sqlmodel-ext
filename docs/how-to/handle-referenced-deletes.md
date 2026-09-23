# 处理"仍被引用"的删除

**目标**：删除一个仍被其它行通过外键（`RESTRICT` / `NO ACTION`）引用的记录时，给用户返回一条可操作的 409 消息，而不是 500 或者泄露约束名的数据库错误。

**前置条件**：

- PostgreSQL（翻译依赖 SQLSTATE `23503`）
- 外键的 `ondelete` 是 `RESTRICT` 或默认的 `NO ACTION`（级联删除 / 置空的配置见 [配置级联删除](./configure-cascade-delete)）

## 1. 发生了什么

```python
await Owner.delete(session, owner)
# → sqlmodel_ext.mixins.ResourceReferencedError: Cannot delete: this resource is still referenced by other resources
```

外键违反有**两个方向**，驱动抛出的异常完全相同（同一个 SQLSTATE、同一个约束名）：

| 方向 | 触发 | 语义 |
|------|------|------|
| 指向不存在的父行 | INSERT/UPDATE 子行，外键目标已不存在 | 404：引用的资源不存在 |
| 仍被引用 | DELETE 父行，子行还指着它 | **409**：资源存在，但现在不能删 |

所以方向只能由**调用点**决定：`delete()` 只在同时满足两个条件时把 `IntegrityError` 翻译成 `ResourceReferencedError`——错误在 `delete()` 内被捕获，**并且** `IntegrityError.statement` 是一条 `DELETE`。只满足第一条不够：commit 会 flush session 里所有待执行的操作，你之前排队的一条错误 `INSERT` 也会在 `delete()` 里冒出来——那条会原样重新抛出。

## 2. 在父模型旁边注册用户可见消息

```python
from sqlmodel_ext import TableBaseMixin

class Owner(SQLModelBase, UUIDTableBaseMixin, table=True):
    name: Str64

TableBaseMixin.register_fk_delete_restrict_message(
    'project_owner_id_fkey',
    "该负责人名下还有项目，请先删除或转移这些项目",
)
```

- 约定写在**被引用的父模型**旁边（与 `register_unique_violation_message` 同一约定：约束声明在哪里，消息就注册在哪里）。
- 消息应说明**下一步能做什么**，不能包含表名 / 列名。
- 未注册的约束回落到 `FK_DELETE_RESTRICT_FALLBACK_MESSAGE`（"Cannot delete: this resource is still referenced by other resources"）。

::: warning 约束名必须与数据库**完全一致**
拼错不会报错，只会静默回落到通用消息。从真实运行的数据库确认名字（例如 `delete()` 记录的 warning 日志里的 `constraint=` 字段），不要从 `create_all` 建的库里抄——迁移可能给约束起了不同的名字。PostgreSQL 的默认名是 `<table>_<column>_fkey`。
:::

这张注册表与 `register_foreign_key_violation_message`（"引用的资源不存在"方向）以同一个约束名为键，但服务**相反的方向**，互不干扰：`lookup_integrity_violation_message()` 只查后者，只有 `delete()` 查前者。

## 3. 在 FastAPI 中映射为 409

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlmodel_ext.mixins import ResourceReferencedError

@app.exception_handler(ResourceReferencedError)
async def resource_referenced_handler(request: Request, exc: ResourceReferencedError) -> JSONResponse:
    # exc.friendly_message 可以安全返回给客户端；exc.original_error 只用于诊断，不要暴露
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.friendly_message})
```

`ResourceReferencedError.status_code` 是 `409`（不是 404——行不但存在，而且正被使用）。它还带 `constraint_name`（驱动没给时为 `None`）与 `original_error`。

## 4. 需要"被什么引用、有几条"时：先显式检查

通用消息刻意不说明"被谁引用"。端点如果要告诉用户"还有 3 个项目"，应在删除**之前**自己查并抛出自己的错误；`ResourceReferencedError` 只是这条路径的兜底（例如检查与删除之间的 TOCTOU 竞态输了）：

```python
n = await Project.count(session, col(Project.owner_id) == owner.id)
if n:
    raise HTTPException(409, detail=f"该负责人名下还有 {n} 个项目")
await Owner.delete(session, owner)     # 并发插入仍可能让这里抛 ResourceReferencedError
```

## 5. 边界

- **`commit=False` 不覆盖**：实例模式下 `delete(..., commit=False)` 时真正的 `DELETE` 由你之后的 flush / commit 发出，发生在 `delete()` 之外，不会被翻译。
- **不要在别处抛 `ResourceReferencedError`**：它不校验自身用法。在通用的完整性错误处理器里抛它，会把"引用目标不存在"误报成"仍被引用"。
- **其它数据库**：没有 SQLSTATE，不会翻译，原始 `IntegrityError` 原样抛出。

## 相关参考

- [`delete()` 的异常](/reference/crud-methods#delete)
- [IntegrityError 友好消息注册表](/reference/crud-methods#integrityerror-友好消息注册表)
- [配置级联删除](./configure-cascade-delete)
