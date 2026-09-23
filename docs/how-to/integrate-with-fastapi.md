# 集成 FastAPI

**目标**：写一组完整的 CRUD 端点（GET 单个 / GET 列表 / POST / PATCH / DELETE），覆盖典型 RESTful 资源。

**前置条件**：

- 你已经有一个建表模型（继承 `UUIDTableBaseMixin` 或 `TableBaseMixin`）
- 你已经配置好 `AsyncSession` 依赖（通常叫 `SessionDep`），session 由 `SessionFactory(engine, class_=sqlmodel_ext.AsyncSession)` 构造
- 你已经有一个 `XxxBase` 数据模型 + `XxxResponse` DTO

## 1. 准备 DTO：每个事实只声明一次

```python
from uuid import UUID
from sqlmodel import Field
from sqlmodel_ext import (
    SQLModelBase, UUIDTableBaseMixin, UUIDIdDatetimeInfoMixin,
    NonEmptyStrippedStr64, Str64, Text10K,
)

class ArticleBase(SQLModelBase):
    title: NonEmptyStrippedStr64
    body: Text10K
    subtitle: Str64 | None = None

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id')

class ArticleCreateRequest(ArticleBase):
    """POST 请求体：字段与约束全部继承自 ArticleBase"""

class ArticleUpdateRequest(ArticleBase, partial=True):
    """PATCH 请求体：继承字段自动变成 ``Unset | T = Unset``，约束原样保留"""

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    """响应 DTO：必带 id 和时间戳"""
    author_id: UUID
```

字段约束（长度、非空、去空白）只写在 `ArticleBase` 里一次；表模型、创建请求、PATCH 请求、响应 DTO 全部从它派生。改一个约束，四处一起变，没有第二个地方会漂移。

`partial=True` 派生的 PATCH DTO 区分三种状态：

| 客户端发送 | 字段值 | `model_dump()` 中 |
|------|------|------|
| 不带 `subtitle` 键 | `Unset` | **不出现**——不会被写入 |
| `"subtitle": null` | `None` | `{'subtitle': None}`——写成 NULL（字段允许 `None` 时） |
| `"subtitle": "x"` | `'x'` | `{'subtitle': 'x'}` |

`title` 在 `ArticleBase` 里不允许 `None`，所以 PATCH 里 `"title": null` 仍然 422——"可以不传"与"可以传 null"是两件事。在代码里判断"客户端有没有传"用 `is not Unset`，而不是 `is not None`：

```python
from sqlmodel_ext import Unset

if patch.subtitle is not Unset:
    ...   # 客户端传了 subtitle（可能是 None，表示清空）
```

`UUIDIdDatetimeInfoMixin` 添加 `id: UUID`、`created_at: datetime`、`updated_at: datetime` 三个**必填**字段——反映"响应中这些字段一定有值"，区别于表模型中 INSERT 之前可能为空的状态。

## 2. 五种端点

```python
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlmodel_ext import ListResponse, TableViewRequest

router = APIRouter(prefix="/articles", tags=["articles"])
TableViewDep = Annotated[TableViewRequest, Depends()]

@router.post("", response_model=ArticleResponse)
async def create_article(
    session: SessionDep,
    current_user: CurrentUserDep,
    data: ArticleCreateRequest,
) -> Article:
    article = Article(**data.model_dump(), author_id=current_user.id)
    return await article.save(session)

@router.get("", response_model=ListResponse[ArticleResponse])
async def list_articles(
    session: SessionDep,
    table_view: TableViewDep,
) -> ListResponse[Article]:
    return await Article.get_with_count(session, table_view=table_view)

@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(
    session: SessionDep,
    article_id: UUID,
) -> Article:
    return await Article.get_exist_one(session, article_id)

@router.patch("/{article_id}", response_model=ArticleResponse)
async def update_article(
    session: SessionDep,
    article_id: UUID,
    data: ArticleUpdateRequest,
) -> Article:
    article = await Article.get_exist_one(session, article_id)
    return await article.update(session, data)

@router.delete("/{article_id}")
async def delete_article(
    session: SessionDep,
    article_id: UUID,
) -> dict[str, int]:
    article = await Article.get_exist_one(session, article_id, with_for_update=True)
    deleted = await Article.delete(session, article)
    return {"deleted": deleted}
```

`delete` 端点里的 `with_for_update=True` 关闭了"存在检查 → 删除"之间的 TOCTOU 窗口：并发的第二个请求会阻塞，等第一个提交后查不到行，得到与串行第二次删除相同的 404。

## 3. 应用级异常映射

```python
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlmodel_ext import OptimisticLockError
from sqlmodel_ext.mixins import KeysetCursorError, ResourceReferencedError

app = FastAPI()

@app.exception_handler(ValidationError)          # 查询参数 DTO 的跨字段校验（after_id + offset 等）
async def dto_validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={
        "detail": jsonable_encoder(exc.errors(include_url=False, include_context=False)),
    })

@app.exception_handler(KeysetCursorError)        # 游标失效 / 不支持，status_code = 422
async def keyset_error_handler(request: Request, exc: KeysetCursorError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

@app.exception_handler(ResourceReferencedError)  # 删除仍被外键引用的行，status_code = 409
async def referenced_handler(request: Request, exc: ResourceReferencedError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.friendly_message})

@app.exception_handler(OptimisticLockError)      # 乐观锁重试耗尽
async def oplock_handler(request: Request, exc: OptimisticLockError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "数据已被其他人修改，请刷新后重试"})
```

为什么需要第一个处理器：FastAPI 对 `Depends()` 类依赖逐个校验查询参数（单字段错误是 422），但跨字段的 `model_validator` 在 FastAPI 调用 `TableViewRequest(...)` 构造对象时才触发，不处理就是 500。

## 关键约定

| 约定 | 原因 |
|------|------|
| 所有 mutation 端点用 `await xxx.save(session)` 并**用返回值** | `commit()` 后对象过期，必须用刷新后的实例 |
| `get_exist_one()` 而不是 `get_one()` | 找不到自动抛 `HTTPException(404)`（FastAPI 已安装时）；`detail=` 可自定义文案 |
| 列表端点返回 `ListResponse[T]` 而不是 `list[T]` | `count` 字段让前端做分页 UI |
| PATCH 用 `partial=True` DTO + `update()` | `Unset` 字段不出现在 `model_dump()` 里，天然只写客户端传了的字段 |
| 权限判断用 `authoritative=True` 读取 | 授权依据必须是最新已提交的值，绕过 identity map 与 Redis |

## 关于权限和 scope

上面的代码假设 `CurrentUserDep` 已经做好认证。PATCH/DELETE 端点通常还要校验"当前用户是否有权操作这条记录"——这是业务逻辑，你应该在端点里自己检查 `article.author_id == current_user.id`。用来做授权判断的读取请传 `authoritative=True`：

```python
article = await Article.get_one(session, article_id, authoritative=True)
if article.author_id != current_user.id:
    raise HTTPException(403)
```

## 关于响应包含关系字段

如果 `ArticleResponse` 中包含关系字段（如 `author: UserResponse`），必须在查询时 `load=` 预加载，否则会触发 MissingGreenlet。具体见 [防止 MissingGreenlet 错误](./prevent-missing-greenlet)。

```python
from sqlmodel_ext import rel

# rel() 把 Relationship 字段 cast 为 QueryableAttribute——
# 类型检查器会把 Article.author 推断为 User 而非可加载属性
return await Article.get_exist_one(session, article_id, load=rel(Article.author))
```

## 相关参考

- [CRUD 方法完整签名](/reference/crud-methods)
- [信息响应 Mixin](/reference/pagination-types#信息响应-mixin-dto)
- [Keyset 游标分页](./keyset-pagination)
- [处理"仍被引用"的删除](./handle-referenced-deletes)
