# 静态分析器原理

::: tip 源码位置
`src/sqlmodel_ext/relation_load_checker.py` — `RelationLoadChecker`、`RelationLoadCheckMiddleware`、`run_model_checks`
:::

这是整个项目中**最复杂的模块**（约 2000 行），通过 AST 静态分析在应用启动时发现潜在的 `MissingGreenlet` 问题。

::: warning 0.3 起默认关闭
0.3.0 起这个模块被视为**实验性**，需要 `rlc.check_on_startup = True` 显式开启。本章解释它**怎么工作**，要看怎么在项目里启用它，请去 [防止 MissingGreenlet 错误](/how-to/prevent-missing-greenlet)。
:::

## 核心类

```python
class RelationLoadChecker:
    def __init__(self, model_base_class=None):
        self.model_base_class = model_base_class
        self.warnings: list[RelationLoadWarning] = []
```

使用 Python 的 `ast` 模块解析源码的**抽象语法树**，而不是执行代码：
- 不需要数据库连接
- 不需要运行任何业务逻辑
- 在导入阶段就能完成

## 分析流程

```mermaid
flowchart TD
    Start["应用启动"] --> A["run_model_checks(SQLModelBase)"]
    A --> B["扫描所有 SQLModelBase 子类"]
    B --> C["对每个类的方法做 AST 分析"]
    C --> D["生成 warnings"]

    Start --> E["RelationLoadCheckMiddleware"]
    E --> F["第一个请求到来时"]
    F --> G["扫描所有 FastAPI 路由函数"]
    G --> H["扫描已导入模块中的协程"]
    H --> I["生成 warnings → 记录到日志"]
```

## 检测规则详解

### RLC001：response_model 包含关系字段但端点未预加载

分析器：
1. 解析 `response_model=UserResponse`，发现它包含 `profile` 字段
2. 检查端点函数体中的查询调用，发现没有 `load=` 参数
3. 生成警告

```python
@router.get("/user/{id}", response_model=UserResponse)
async def get_user(session: SessionDep, id: UUID):
    return await User.get_exist_one(session, id) # [!code warning]
    # ⚠ RLC001: response_model 包含 profile，但查询没有 load=
```

### RLC002：save/update 后访问关系

追踪变量的"过期状态"——在 `save()` 或 `update()` 调用之后，对象的所有关系视为过期。

```python
user = await User.get_exist_one(session, id, load=rel(User.profile))
user = await user.update(session, data)   # 之后所有关系过期 // [!code warning]
return user.profile                        # RLC002 // [!code error]
```

### RLC003：访问未加载的关系（本地变量）

追踪本地变量绑定的对象类型和已加载的关系，发现对未加载关系的属性访问。

### RLC005：依赖函数未预加载 response_model 需要的关系

FastAPI 依赖（`Depends`）返回的 ORM 对象被端点直接 return 时，依赖内部的查询必须预加载 response_model 序列化所需的关系——RLC005 检查依赖函数体的 `load=` 是否覆盖。

### RLC007：commit 后访问过期对象的列属性

追踪 `session.commit()` 调用，之后访问相关对象的任何属性都视为危险。

### RLC008：commit 后调用过期对象的方法

类似 RLC007，但检测的是方法调用而非属性访问。

### RLC009：类型注解解析错误

检测混用已解析类型和字符串前向引用导致的类型解析问题。

### RLC010：把 commit 后的过期对象当参数传递

`commit()` 之后把已过期的 ORM 对象作为参数传给其他函数/方法——被调用方访问其列属性会触发同步懒加载 → `MissingGreenlet`。比 RLC007（本函数内访问）更隐蔽，因为崩溃点在调用链下游。

### RLC011：对未加载关系的对象做布尔判断

`if not obj:` / `if obj:` 会隐式调用 `__bool__()`/`__len__()`，可能触碰未加载的关系。建议改写为 `if obj is None:`。

### RLC012：STI 异构序列化字段缺失

端点 `response_model` 的字段不在 STI **所有**子类的 `model_fields` 中：端点返回 STI 基类时，查询可能产出缺少该字段的子类实例，序列化将失败。修法：在具体子类上调用 `get()`/`get_with_count()` 窄化返回类型。

### RLC013：async generator 在 yield 之后访问列属性

`yield` 把控制权交给持有同一 session 的消费方，消费方可能在 yield 期间 commit 使对象过期；恢复后访问列属性触发 `MissingGreenlet`。yield 前把需要的字段提取到局部变量。

## `RelationLoadWarning`

```python
class RelationLoadWarning:
    code: str          # "RLC001"
    message: str       # 人类可读的描述
    location: str      # "module.py:42 in function_name"
    severity: str      # "warning" 或 "error"
```

## `mark_app_check_completed()`

中间件的检查只执行一次。第一个请求到来时完成分析后，通过 `mark_app_check_completed()` 标记完成，后续请求不再重复。

## 为什么用 AST 而不用运行时检查？

| 方式 | 优点 | 缺点 |
|------|------|------|
| AST 静态分析 | 启动时发现、不执行代码、覆盖所有路径 | 可能有误报、无法分析动态代码 |
| 运行时检查 | 100% 准确 | 只有执行到的路径才会检查 |

静态分析器作为"第一道防线"配合运行时的 `@requires_relations` 和 `lazy='raise_on_sql'`，形成多层保护。

## 局限性

- **误报**：静态分析无法追踪运行时的动态行为（如 `getattr`、条件加载）
- **仅分析协程**：同步函数不在分析范围内（同步环境没有 MissingGreenlet 问题）
- **模块范围**：只分析已导入的模块，未导入的代码不会被扫描

::: warning 误报与项目结构假设
分析器的 AST 规则是针对特定项目结构（FastAPI 端点、STI 继承约定、`save`/`update`/`delete` 命名等）调优的，在不同项目上**可能产生误报或解析失败**。模块 API 不在 semver 稳定承诺范围内。
:::
