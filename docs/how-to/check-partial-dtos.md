# 检查 partial DTO 的误用（check_derived）

::: warning 实验性功能
`sqlmodel_ext.check_derived` 与 `sqlmodel_ext.derived_decls` 是**实验性**的：命令行参数与模块 API 可能在任何版本变化。它有明确的退出计划——见文末 [何时移除](#何时移除)。
:::

**目标**：让 basedpyright 对 `partial=True` 派生的 PATCH DTO 说话——`if dto.x is not None: dto.x.upper()` 这类误用在提交前就被标红。

**前置条件**：

- 项目已安装 sqlmodel-ext 与 `basedpyright>=1.40.1`（见 [用 basedpyright 做类型检查](./type-check-with-basedpyright)）
- 了解 [Unset 三态](/explanation/unset-three-state)

## 1. 要解决的问题

`partial=True` 在**运行时**由元类把字段改写成 `Unset | T = Unset`，而 basedpyright 只读**源码**。于是在检查器眼里，派生类的字段仍是基类类型：

<!-- skip-run -->
```python
class ArticleBase(SQLModelBase):
    title: Str64
    subtitle: Str256 | None = None


class ArticleUpdate(ArticleBase, partial=True):
    pass                     # 运行时 subtitle: Unset | Str256 | None；静态仍是 Str256 | None


def preview(patch: ArticleUpdate) -> str:
    if patch.subtitle is not None:      # 没传时是 Unset，Unset is not None 为真
        return patch.subtitle[:20]      # 运行时 TypeError，basedpyright 却 0 error
    return ''
```

同样的盲区还出现在**继承下来的方法**里：基类里写得正确的 `@model_validator`（`if self.subtitle is not None`），被 `ArticleUpdate` 继承后在三态下就不够了——而 validator 在**每次构造**时都会运行，只传一个字段的 PATCH 请求体正是它的常态触发路径。

这不是配置问题，而是语言边界：Python 既没有编译期的派生宏（Rust `#[derive]`），类型系统也没有类型级映射（TypeScript `Partial<T>`）。

在没有这个工具时，办法只有一个：把需要静态收窄的字段在 partial 类体里**显式声明**成 `Unset | T = Unset`（类体里的声明优先于 `partial`）。`check_derived` 提供第二个办法：不改源码，让检查器直接看见运行时的三态。

## 2. 运行

```bash
python -m sqlmodel_ext.check_derived app            # 导入包 app（递归导入所有子模块），检查当前目录这个项目
python -m sqlmodel_ext.check_derived app --root .   # 显式指定项目根目录
python -m sqlmodel_ext.check_derived app --keep     # 保留临时副本，便于查看展开结果
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `MODULE ...`（位置参数，至少一个） | — | 要导入的模块或包（点分名）。包会被递归导入——元类在类创建时登记 partial 类，定义它们的模块必须被导入 |
| `--root ROOT` | 当前目录 | 要复制并检查的项目根目录 |
| `--import-path DIR` | 根目录；若存在 `src/` 再加上它 | 导入前放进 `sys.path` 的目录（相对 `--root`），可重复 |
| `--basedpyright PATH` | 当前解释器的 scripts 目录，其次 `PATH` | basedpyright 可执行文件 |
| `--python PATH` | 当前解释器（配置里写了 `venv` 时不传） | basedpyright 解析第三方包所用的解释器 |
| `--keep` | 关 | 保留临时副本并打印位置 |

退出码：`0` 没有新增的阻塞错误；`1` 有新增的阻塞错误；`2`（或 `[ABORT]` 开头的消息）工具无法完成工作——某个类无法展开、找不到 basedpyright、basedpyright 没有返回 JSON、或者**扫描了 0 个文件**（空扫描也会显示"0 errors"，不能当成通过）。

## 3. 放进 pre-commit（排在所有静态检查之前）

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-derived          # 必须是第一个静态检查
        name: check partial DTOs (sqlmodel-ext, experimental)
        entry: python -m sqlmodel_ext.check_derived app
        language: system
        pass_filenames: false
        types: [python]
  # ... 其后才是 basedpyright、ruff 等
```

**这个钩子必须排在其他所有静态检查之前**：它报告的是"类型检查器本来看不见"的错误，让它先跑，误用会以本来的面目出现（"派生 DTO 的某个字段没判 `Unset`"），而不是被后面钩子的输出淹没，或者在后面的钩子失败时根本没机会运行。`language: system` 让钩子使用项目自己的虚拟环境（它需要导入你的模块）。

在 CI 里同样直接运行这条命令即可。

## 4. 它做了什么

1. **导入**你给的模块；元类把每个 `partial=True` 类登记到 `optional_dto_registry`。受检范围是这些类**以及它们的全部子类**（三态注解会被继承，而子类自己不写 `partial=True`），且只限源文件位于 `--root` 之下的类。
2. **复制**项目里的 `.py` / `.pyi` 到系统临时目录（git 仓库：已跟踪文件加上未忽略的未跟踪文件；否则遍历目录，跳过隐藏目录与虚拟环境）。
3. **在副本里展开**，两件事缺一不可：
   - **字段声明**：元类确实改成三态的字段（由运行时 `model_fields` 决定，而不是重新推演元类规则），在派生类的 `if TYPE_CHECKING:` 块里写 `name: Unset | <基类源码里逐字的注解>`——保留 `Str64` 这样的别名，而不是展开成 `Annotated[...]`；
   - **按 MRO 展开方法**：派生类没有定义、但祖先定义了、且读取了三态字段（`self.<字段>`）的成员（方法、property、`model_validator`、`model_post_init`），复制进派生类。它们的 `self` 随之成为派生类类型，守卫不足由 basedpyright 原生报出。
4. 在副本上**跑 basedpyright**（沿用项目自己的配置：`pyrightconfig.json`，否则 `pyproject.toml` 的 `[tool.basedpyright]` / `[tool.pyright]`）。
5. 有错误时，再在**未展开**的副本上跑一次作为基线，只报告展开**新增**的错误。比对键是 `(文件, 规则, 消息)` 的计数，**不含行号**——展开会插入代码行，含行号会把每一条既有错误都算成"新增"。
6. 删除临时目录（`--keep` 时保留）。

**它不会修改你的工作树**：所有写入都在临时副本里；导入你的模块时关闭了字节码写入，连 `__pycache__` 都不会出现；写入函数拒绝把副本放在项目目录之内。仓库里不会有任何生成物。

**分级**：

| 新增错误落在哪里 | 级别 | 原因 |
|---|---|---|
| 展开进来的 `model_validator` / `model_post_init` | 阻塞 | 每次构造都会运行，运行时必然出错 |
| 生成代码之外（**消费侧**，别处读取了该字段） | 阻塞 | 真实调用点 |
| 展开进来的普通方法 / property | 潜在（打印，不失败） | 只有被调用才出错；PATCH DTO 通常不会调用基类的业务方法 |

## 5. 演示

仓库里的 [`examples/check_derived_demo`](https://github.com/Foxerine/sqlmodel-ext/tree/master/examples/check_derived_demo) 是一个最小项目：`shop/models.py` 有一个基类 validator 和一个派生的 `ArticleUpdate`，`shop/handlers.py` 有一处 `is not None` 误用。

普通的 basedpyright 看不出任何问题（在项目目录、激活虚拟环境后运行）：

```text
$ basedpyright
0 errors, 0 warnings, 0 notes
```

运行 `check_derived`（在仓库根目录，basedpyright 1.40.1，逐字输出）：

```text
$ python -m sqlmodel_ext.check_derived shop --root examples/check_derived_demo
[..] expanded 1 class(es) / 2 inherited member(s) (0.0s)
[..] 6 error(s) after expansion; running the unexpanded baseline

==============================================================================
Inherited methods are wrong for tri-state fields (fail only when called) (1)
==============================================================================
  ArticleUpdate <- headline
    shop/models.py  [reportAttributeAccessIssue]
      Cannot access attribute "upper" for class "MISSING"
  A PATCH DTO usually never calls these (they are behavior of the base class
  that the DTO inherited), so they do not fail the check. They are real:
  calling one on a PATCH DTO will fail.

==============================================================================
Code that runs on construction, or a consumer, is wrong for tri-state fields (2)
==============================================================================
  ArticleUpdate <- _normalize_subtitle
    shop/models.py  [reportAttributeAccessIssue]
      Cannot access attribute "strip" for class "MISSING"
  consumer
    shop/handlers.py  [reportIndexIssue]
      "__getitem__" method not defined on type "MISSING"
  model_validator / model_post_init run on every construction -- including a PATCH
  body that sets a single field -- so these fail at runtime, not in theory.
  Fix: test for omission with `x is Unset` (or `x is Unset or x is None`).
  Do not use `if x:` -- bool(Unset) is True, and a truthiness test also
  swallows legitimate 0 / '' / [].

[FAIL] blocking 2 / latent 1 (2.6s)
```

（`MISSING` 就是 `Unset`——两者是同一个对象。"6 error(s) after expansion" 里另外 3 条是展开本身必然产生的覆盖声明诊断，工具会识别并丢弃。）

阻塞项里的 validator 不是理论问题：`ArticleUpdate(title='x')` 在运行时确实抛出 `AttributeError: 'sentinel' object has no attribute 'strip'`。修法是把守卫写成三态的：

<!-- skip-run -->
```python
if self.subtitle is not Unset and self.subtitle is not None:
    self.subtitle = self.subtitle.strip()
```

::: tip 不要用 `if x:` 代替
`bool(Unset)` 为真，truthy 判断挡不住它；同时还会把合法的 `0` / `''` / `[]` 当成"没传"。
:::

## 6. 已知盲区

- `getattr(self, 'x', None)` 等**字符串形式**的属性访问——类型检查器无法从字符串字面量推断属性。
- 类型检查器本来就看不到的东西：不在 `include` 范围内的代码、`Any` 类型的值、`# pyright: ignore` 注释。
- 源文件不在 `--root` 之下的类，以及从 sqlmodel-ext / SQLModel / Pydantic 自身继承来的成员，不会被展开。
- 成员是否展开，取决于其源码里有没有出现 `self.<三态字段>`；通过别名访问（`other = self; other.x`）不会被识别。
- 副本取自**工作树**，而不是 git 暂存区。

## 何时移除

[PEP 827](https://peps.python.org/pep-0827/)（类型操作，草案）提出了 Python 缺失的那一块：把 PATCH DTO 写成基类的类型级变换，检查器直接看见三态。等它（或等价方案）被接受、且 basedpyright 支持之后，`check_derived` 与 `derived_decls` 会先标记弃用、再移除。因为它从不在仓库里留下生成物，移除时也没有任何需要清理的东西——删掉 pre-commit 里那一条即可。
