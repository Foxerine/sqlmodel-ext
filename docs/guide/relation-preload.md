# 关系预加载

在异步 SQLAlchemy 中，访问未加载的关系会触发隐式的同步查询，导致 `MissingGreenlet` 错误。`@requires_relations` 装饰器解决了这个问题。

## 问题

```python
class MyFunction(SQLModelBase, UUIDTableBaseMixin, table=True):
    generator: Generator = Relationship()

    async def calculate_cost(self, session) -> int:
        config = self.generator.config    # MissingGreenlet!
        return config.price * 10
```

常规解决办法是调用时用 `load=` 预加载，但**调用者必须知道方法内部需要哪些关系**。

## `@requires_relations` 装饰器

把关系需求声明在方法本身：

```python
from sqlmodel_ext.mixins import RelationPreloadMixin, requires_relations

class MyFunction(SQLModelBase, UUIDTableBaseMixin, RelationPreloadMixin, table=True):
    generator: Generator = Relationship()

    @requires_relations('generator', Generator.config)
    async def calculate_cost(self, session) -> int:
        # generator 和 generator.config 在执行前自动加载
        return self.generator.config.price * 10
```

调用者不需要关心内部依赖：

```python
cost = await func.calculate_cost(session)  # 自动加载所需关系
```

### 参数格式

```python
@requires_relations(
    'generator',         # 字符串：本类的属性名
    Generator.config,    # RelationshipInfo：外部类的关系属性（嵌套）
)
```

- **字符串** — `self.generator` 这样的直接关系
- **RelationshipInfo** — `Generator.config` 表示嵌套关系

## 关键特性

| 特性 | 说明 |
|------|------|
| **声明式** | 关系需求声明在方法上，而非调用处 |
| **增量加载** | 已加载的关系不重复加载 |
| **导入时验证** | 关系名拼写错误在启动时立刻报错 |
| **session 自动发现** | 不强制 session 参数位置，自动从参数中找到 |
| **嵌套感知** | 自动处理多层关系链 |
| **支持异步生成器** | `async for` 也能用 |

## `@requires_for_update` 装饰器

声明方法必须在 `FOR UPDATE` 锁定的实例上调用：

```python
from sqlmodel_ext.mixins.relation_preload import requires_for_update

class Account(SQLModelBase, UUIDTableBaseMixin, RelationPreloadMixin, table=True):
    balance: int

    @requires_for_update
    async def adjust_balance(self, session: AsyncSession, *, amount: int) -> None:
        self.balance += amount
        await self.save(session)
```

调用者必须先获取锁：

```python
account = await Account.get(session, Account.id == uid, with_for_update=True)
await account.adjust_balance(session, amount=-100)  # OK

account = await Account.get_exist_one(session, uid)
await account.adjust_balance(session, amount=-100)  # RuntimeError!
```

运行时通过 `session.info` 检查锁定状态。静态分析器也能在启动时检测未锁定的调用。

## 默认 `lazy='raise_on_sql'`

0.2.0 起，所有 Relationship 字段默认设置 `lazy='raise_on_sql'`。这意味着在异步环境中访问未预加载的关系会**立刻抛出异常**，而非触发隐式查询。这是 MissingGreenlet 问题的最后一道安全网。

## 手动 API

通常不需要，装饰器自动处理一切。提供了手动接口用于特殊场景：

```python
# 获取方法声明的关系列表（用于构建查询）
rels = MyFunction.get_relations_for_method('calculate_cost')

# 获取多个方法的关系（去重）
rels = MyFunction.get_relations_for_methods('calculate_cost', 'validate')

# 手动预加载
await instance.preload_for(session, 'calculate_cost', 'validate')
```

## 三道防线

sqlmodel-ext 对 MissingGreenlet 问题提供了三层保护：

```
1. 启动时 AST 静态分析    ← 最早发现问题（见静态分析器）
2. @requires_relations    ← 运行时自动加载
3. lazy='raise_on_sql'    ← 最后的安全网
```
