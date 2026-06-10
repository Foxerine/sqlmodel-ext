# 集成 FastAPI

**目标**：写一组完整的 CRUD 端点（GET 单个 / GET 列表 / POST / PATCH / DELETE），覆盖典型 RESTful 资源。

**前置条件**：

- 你已经有一个建表模型（继承 `UUIDTableBaseMixin` 或 `TableBaseMixin`）
- 你已经配置好 `AsyncSession` 依赖（通常叫 `SessionDep`）
- 你已经有一个 `XxxBase` 数据模型 + `XxxResponse` DTO

## 1. 准备 DTO

```python
from sqlmodel_ext import SQLModelBase, UUIDIdDatetimeInfoMixin, NonEmptyStrippedStr64, Text10K

class ArticleBase(SQLModelBase):
    title: NonEmptyStrippedStr64
    body: Text10K

class Article(ArticleBase, UUIDTableBaseMixin, table=True):
    author_id: UUID = Field(foreign_key='user.id')

class ArticleCreateRequest(ArticleBase):
    """POST 请求体：所有字段都必填"""
    pass

class ArticleUpdateRequest(ArticleBase, all_fields_optional=True):
    """PATCH 请求体：继承字段自动转为 ``T | None = None``，约束原样保留（无需手写）"""
    pass

class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    """响应 DTO：必带 id 和时间戳"""
    author_id: UUID
```

`UUIDIdDatetimeInfoMixin` 自动添加 `id: UUID`、`created_at: datetime`、`updated_at: datetime` 三个**必填**字段——这反映了"响应中这些字段一定有值"的事实，区别于表模型中的 `id: UUID | None`。

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
    article = await Article.get_exist_one(session, article_id)
    deleted = await Article.delete(session, article)
    return {"deleted": deleted}
```

## 关键约定

| 约定 | 原因 |
|------|------|
| 所有 mutation 端点用 `await xxx.save(session)` 并**用返回值** | `commit()` 后对象过期，必须用刷新后的实例 |
| `get_exist_one()` 而不是 `get_one()` | 找不到自动抛 `HTTPException(404)`（FastAPI 已安装时） |
| 列表端点返回 `ListResponse[T]` 而不是 `list[T]` | `count` 字段让前端做分页 UI |
| `PATCH` 用 `update(other)` 而不是 `save()` | `update()` 默认 `exclude_unset=True`，即 PATCH 语义 |

## 关于权限和 scope

上面的代码假设 `CurrentUserDep` 已经做好认证。在真实项目中，PATCH/DELETE 端点通常还要校验"当前用户是否有权操作这条记录"——这是业务逻辑，sqlmodel-ext 不直接管，你应该在端点里自己检查 `article.author_id == current_user.id`。

## 关于响应包含关系字段

如果 `ArticleResponse` 中包含关系字段（如 `author: UserResponse`），你必须在查询时 `load=` 预加载，否则会触发 MissingGreenlet。具体见 [防止 MissingGreenlet 错误](./prevent-missing-greenlet)。

```python
from sqlmodel_ext import rel

# rel() 把 Relationship 字段 cast 为 QueryableAttribute——
# 类型检查器会把 Article.author 推断为 User 而非可加载属性
return await Article.get_exist_one(session, article_id, load=rel(Article.author))
```

## 相关参考

- [CRUD 方法完整签名](/reference/crud-methods)
- [信息响应 Mixin](/reference/pagination-types#信息响应-mixin-dto)
