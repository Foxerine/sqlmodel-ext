# 03 · 给博客加 Redis 缓存

教程 02 构建的博客 API 已经能跑了，但每次 `GET /articles/{id}` 都查数据库——一篇被疯转的文章会让数据库哭出来。这次我们给 `Article` 加 Redis 缓存层，让重复读零 SQL；顺便解决教程 02 末尾留下的"`MissingGreenlet`"伏笔。

预计 30 分钟。

## 你将做什么

1. 启动一个本地 Redis
2. 给 `Article` 模型加 `CachedTableBaseMixin`
3. 在应用启动时配置 Redis 客户端
4. 验证缓存命中（用 SQL 日志确认）
5. 给 `ArticleResponse` 加 `author: UserResponse`，并用 `load=` 解决 MissingGreenlet
6. 看缓存失效（修改文章后下一次读取会重新查数据库）

## 0. 启动 Redis

最快的方法是 Docker：

```bash
docker run --name blog-redis -p 6379:6379 -d redis:7
```

确认能连：

```bash
docker exec -it blog-redis redis-cli ping
# → PONG
```

## 1. 安装 Redis 异步客户端

延续教程 02 的目录：

```bash
pip install "redis[hiredis]>=5"
```

## 2. 给 `Article` 加缓存能力

打开 `models.py`，修改 `Article` 这一段：

```python
from sqlmodel_ext import (
    SQLModelBase,
    UUIDTableBaseMixin,
    UUIDIdDatetimeInfoMixin,
    CachedTableBaseMixin,    # ← 新增
    NonEmptyStrippedStr64,
    NonEmptyStrippedStr256,
    Text10K,
)

# ... User / UserBase / UserCreateRequest / UserResponse 不变 ...

class Article(
    CachedTableBaseMixin,    # ← 新增（必须放在最前）  // [!code highlight]
    ArticleBase,
    UUIDTableBaseMixin,
    table=True,
    cache_ttl=600,           # 10 分钟  // [!code highlight]
):
    author_id: UUID = Field(foreign_key="user.id", index=True)
    author: User = Relationship(back_populates="articles")
    comments: list["Comment"] = Relationship(back_populates="article")
```

::: warning MRO 顺序
`CachedTableBaseMixin` **必须**放在 `UUIDTableBaseMixin` 之前。它需要在 MRO 链中比基类先出现，才能让自己的 `get()` / `save()` / `update()` / `delete()` 重写生效。
:::

`cache_ttl=600` 是元类专门处理的关键字参数，会被翻译成 `__cache_ttl__: ClassVar[int] = 600`。默认 3600 秒。

## 3. 在 lifespan 中配置 Redis

修改 `db.py`：

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated

import redis.asyncio as redis                        # ← 新增
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel_ext import AsyncSession, CachedTableBaseMixin  # ← 新增（注意：增强版 AsyncSession）

engine = create_async_engine("sqlite+aiosqlite:///blog.db")
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

::: warning 必须用 sqlmodel_ext.AsyncSession
0.4.0 起缓存失效由增强版 `AsyncSession.commit()` 统一编排。session 工厂的 `class_=` 必须指向 `sqlmodel_ext.AsyncSession`（不是 `sqlmodel.ext.asyncio.session.AsyncSession`），否则失效退化为 fire-and-forget 补偿钩子，存在短暂 stale 窗口。
:::

```python


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 启动：建表 + 配置 Redis
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    redis_client = redis.from_url(                      # [!code ++]
        "redis://localhost:6379",                        # [!code ++]
        decode_responses=False,    # ← 关键，不能改  // [!code ++]
    )                                                    # [!code ++]
    CachedTableBaseMixin.configure_redis(redis_client)  # [!code ++]
    CachedTableBaseMixin.check_cache_config()           # [!code ++]

    yield

    # 关闭
    await redis_client.aclose()                          # [!code ++]
    await engine.dispose()


# ... get_session / SessionDep 不变 ...
```

::: danger `decode_responses=False`
缓存的 value 是 `bytes`（来自 `model_dump_json().encode()`）。设成 `True` 会让 redis-py 把 bytes 解成 str，破坏反序列化。
:::

`check_cache_config()` 会校验所有继承 `CachedTableBaseMixin` 的子类的 `__cache_ttl__` 是合法的正整数，并注册 SQLAlchemy 的 `after_commit` 事件钩子（用于 `commit=False` 场景的失效补偿）。

## 4. 验证缓存命中

重启服务：

```bash
fastapi dev main.py
```

回到教程 02 创建的文章 ID，连续两次 `curl`：

```bash
curl http://127.0.0.1:8000/articles/<article_id>
curl http://127.0.0.1:8000/articles/<article_id>
```

观察 `fastapi dev` 终端的 SQL 日志：

- **第一次** 会看到 `SELECT ... FROM article WHERE article.id = ?`
- **第二次** SQL 日志里**没有任何 SELECT**——直接命中 ID 缓存（key 形如 `id:Article:550e...`）

::: info 怎么看缓存里有什么
```bash
docker exec -it blog-redis redis-cli
> KEYS id:Article:*
1) "id:Article:550e8400-..."
> GET id:Article:550e8400-...
"{\"_t\":\"single\",\"_data\":{...},\"_c\":\"Article\"}"
> TTL id:Article:550e8400-...
(integer) 597
```

`_t` 是结果类型（single / list / none），`_c` 是实际的类名（多态安全），`_data` 是 `model_dump_json()` 的结果。
:::

## 5. 验证自动失效

```bash
curl -X PATCH http://127.0.0.1:8000/articles/<article_id> \
  -H "Content-Type: application/json" \
  -d '{"title":"新标题"}'

curl http://127.0.0.1:8000/articles/<article_id>
```

注意第二次 `curl` 又出现了 SQL 查询——为什么？因为 `update()` 内部调用了 `_invalidate_for_model()`，把 `id:Article:<id>` 失效了，同时把查询缓存的版本号 `+1`。下一次读取时缓存 miss → 查数据库 → 重新写入新缓存。

业务代码完全没感知。

## 6. 解决教程 02 末尾留下的 MissingGreenlet 隐患

现在我们要让 `ArticleResponse` 包含作者信息：

```python
# models.py
class ArticleResponse(ArticleBase, UUIDIdDatetimeInfoMixin):
    author_id: UUID
    author: UserResponse    # ← 新增
```

如果不改端点直接试一下：

```bash
curl http://127.0.0.1:8000/articles/<article_id>
```

服务会爆炸：

```
sqlalchemy.exc.InvalidRequestError: 'Article.author' is not available
due to lazy='raise_on_sql'
```

::: info 第三道防线
sqlmodel-ext 0.2.0 起把所有 `Relationship` 的默认 `lazy` 设为 `'raise_on_sql'`——访问未预加载的关系**立刻抛清晰错误**，而不是触发隐式同步查询导致 `MissingGreenlet`。这是把"难懂的 greenlet 错误"转成"可读的 InvalidRequestError"的安全网。
:::

修复很简单——告诉查询要预加载 `author`：

```python
# main.py
@articles.get("/{article_id}", response_model=ArticleResponse)
async def get_article(session: SessionDep, article_id: UUID) -> Article:
    return await Article.get_exist_one(
        session,
        article_id,
        load=Article.author,    # ← 新增
    )
```

`load=Article.author` 让 sqlmodel-ext 在底层用 `selectinload(Article.author)` 一次性把作者带回来。

试试看：

```bash
curl http://127.0.0.1:8000/articles/<article_id>
# → {"id":"...","title":"...","author":{"id":"...","name":"Alice","email":"..."}}
```

::: tip 嵌套关系也能预加载
如果你想同时拿到作者**和**作者的某个字段相关对象，写 `load=[Article.author, User.profile]`——sqlmodel-ext 会自动构建 `selectinload(author).selectinload(profile)` 链。
:::

## 7. 关于跳过缓存

某些场景你不想用缓存——例如 PATCH 后立刻读取需要拿到最新值。`get()` 接受 `no_cache=True`：

```python
fresh = await Article.get_one(session, article_id, no_cache=True)
```

不过通常你不需要——`save()` / `update()` 已经自动失效了缓存，下一次普通读取就能拿到新数据。

**自动跳过缓存的场景**（你不用手动指定）：

- `with_for_update=True`（行锁需要最新数据）
- `populate_existing=True`
- `options=` / `join=` 非空（无法稳定哈希）
- 当前事务内有待失效数据

## 8. 你刚才学到了什么

| 概念 | 操作 |
|------|------|
| `CachedTableBaseMixin` 必须放 MRO 第一位 | `class Article(CachedTableBaseMixin, ..., table=True)` |
| `cache_ttl` 是元类专属关键字 | `cache_ttl=600` |
| `configure_redis()` 启动时调用一次 | lifespan 中 |
| `check_cache_config()` 校验所有子类 | lifespan 中 |
| `decode_responses=False` 不能改 | redis 客户端配置 |
| 缓存失效完全自动 | `save()` / `update()` / `delete()` 内部处理 |
| `lazy='raise_on_sql'` 是 MissingGreenlet 的安全网 | 不用配置，默认开启 |
| `load=` 预加载关系 | `Article.get_exist_one(..., load=Article.author)` |

## 你已经会的

恭喜——三篇教程通关了。这套技能足够你写 80% 的真实项目：

- 定义模型（Base / Table / CreateRequest / UpdateRequest / Response 五件套）
- 全套 CRUD 端点（包括分页 + PATCH 语义）
- 关系 + 预加载（避免 MissingGreenlet）
- Redis 缓存（自动失效）

## 接下来去哪

- **遇到具体任务**？查 [操作指南](/how-to/) ——比如"怎么处理并发更新"、"怎么定义 STI 多态模型"。
- **想找某个 API 的精确签名**？查 [参考](/reference/)。
- **好奇内部实现**？看 [讲解](/explanation/) ——比如 [元类做了什么](/explanation/metaclass)、[Redis 缓存怎么实现自动失效](/explanation/cached-table)。
- **遇到 bug 或想提建议**？去 [GitHub Issues](https://github.com/Foxerine/sqlmodel-ext/issues)。
