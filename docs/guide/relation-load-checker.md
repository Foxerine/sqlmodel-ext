# 静态分析器

静态分析器在**应用启动时**通过 AST 分析源码，提前发现可能导致 `MissingGreenlet` 错误的代码。

## 定位

这是 MissingGreenlet 问题的**第一道防线**——在任何请求到来之前，扫描你的代码，找出所有潜在问题。

```
1. 启动时 AST 静态分析（本模块）  ← 最早发现问题
2. @requires_relations 运行时预加载  ← 自动修复问题
3. lazy='raise_on_sql' 运行时拦截    ← 最后的安全网
```

## 检测规则

| 规则 | 说明 |
|------|------|
| **RLC001** | `response_model` 包含关系字段但端点未预加载 |
| **RLC002** | `save()`/`update()` 之后访问关系但没用 `load=` |
| **RLC003** | 访问关系但之前没有用 `load=` 加载（仅本地变量） |
| **RLC005** | 依赖函数未预加载 `response_model` 需要的关系 |
| **RLC007** | commit 后访问过期对象的列属性 |
| **RLC008** | commit 后在过期对象上调用方法 |
| **RLC009** | 类型注解解析错误（混用已解析类型和字符串前向引用） |

## 使用方式

### 自动检查（推荐）

```python
# 在 models/__init__.py 中，configure_mappers() 之后：
from sqlmodel_ext import run_model_checks, SQLModelBase
run_model_checks(SQLModelBase)

# 在 main.py 中：
from sqlmodel_ext import RelationLoadCheckMiddleware
app.add_middleware(RelationLoadCheckMiddleware)
```

`run_model_checks` 扫描所有模型类的方法。`RelationLoadCheckMiddleware` 在第一个请求到来时扫描所有 FastAPI 路由函数。

### 手动检查

```python
from sqlmodel_ext import RelationLoadChecker

checker = RelationLoadChecker(model_base_class=SQLModelBase)
checker.check_function(some_function)
checker.check_fastapi_app(app)

for warning in checker.warnings:
    print(f"[{warning.code}] {warning.message}")
    print(f"  位置: {warning.location}")
```

## 常见警告示例

### RLC001：response_model 未预加载

```python
class UserResponse(SQLModelBase):
    profile: ProfileResponse    # 关系字段

@router.get("/user/{id}", response_model=UserResponse)
async def get_user(session: SessionDep, id: UUID):
    return await User.get_exist_one(session, id)
    # RLC001: response_model 包含 profile，但查询没有 load=User.profile
```

### RLC002：save 后访问关系

```python
async def update_user(session, id, data):
    user = await User.get_exist_one(session, id, load=User.profile)
    user = await user.update(session, data)  # commit 后关系过期
    return user.profile                       # RLC002
```

### RLC007：commit 后访问列

```python
async def create_and_log(session, data):
    user = User(**data)
    session.add(user)
    await session.commit()     # user 过期
    print(user.name)           # RLC007
```

## `RelationLoadWarning`

每个警告包含：

| 属性 | 说明 |
|------|------|
| `code` | 规则编码（如 "RLC001"） |
| `message` | 人类可读的描述 |
| `location` | 位置（如 "module.py:42 in function_name"） |
| `severity` | "warning" 或 "error" |

## 局限性

- **误报**：无法追踪运行时的动态行为（如 `getattr`、条件加载）
- **仅分析协程**：同步函数不在分析范围内
- **模块范围**：只分析已导入的模块，未导入的代码不会被扫描
