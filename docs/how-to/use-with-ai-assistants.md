# 配合 AI 编码助手使用

**目标**：让 Claude Code、Codex、Copilot 等 AI 编码助手在你的项目里**按 sqlmodel-ext 的正确方式**写模型和查询，而不是"看起来对"地乱用乱写；并用 basedpyright 把剩下的错误在运行前标红。

**前置条件**：

- 项目已经在用 sqlmodel-ext
- 能在项目里运行 `basedpyright`（`pip install basedpyright`，建议 >= 1.40.1——这是第一个能正确收窄 `x is Unset` 的版本）

## 为什么需要

sqlmodel-ext 的核心是**单点真相**：每个事实只在一个地方声明，其余一切由它派生。字段约束写在类型里一次（`Str64`），同时产出 Pydantic 校验、数据库列类型和 OpenAPI schema；更新 DTO 用 `partial=True` 从模型派生；"没传 / 传了 null / 传了值"由 `Unset` / `None` / 值三个状态区分。

AI 最常见的错误恰恰是**在第二个地方重复声明同一个事实**：

| AI 常写的 | 问题 | 正确写法 |
|---|---|---|
| 手写 `class XxxUpdate: title: str \| None = None` | 分不清"没传"和"清空"；约束与表模型慢慢漂移 | `class XxxUpdate(XxxBase, partial=True)` |
| `if patch.summary is not None:` | 把"没传"和"传了 null"混为一谈 | `if patch.summary is not Unset:` |
| `model_dump(exclude_unset=True)` | 多余——`Unset` 字段本来就不会出现在 dump 里 | 直接 `await obj.update(session, patch)` |
| `title[:64]` | 第二个 `64`，改了类型这里不会跟着变 | `title[:max_length_of(Str64)]` |
| `count = x or 0` | 合法的 `0` / `''` 也被兜底替换，问题被掩盖 | 显式 `if x is None:`，或让字段必填 |
| `obj.save(session)` 不接返回值 | 手上的对象已过期，再访问属性会触发 I/O | `obj = await obj.save(session)` |
| 访问未加载的关系 | 关系默认 `lazy='raise_on_sql'`，直接报错 | `load=rel(Model.rel)` / `@requires_relations` |

仓库附带的规则包把这些约定写成 AI 能直接遵守的指令。

## 1. 安装规则包

规则包在仓库的 [`ai-rules/`](https://github.com/Foxerine/sqlmodel-ext/tree/master/ai-rules) 目录，只有一份正文：

| 文件 | 谁读 | 内容 |
|---|---|---|
| `AGENTS.md` | Codex、Copilot、Cursor、Claude Code 等 | **规则正文**（唯一一份） |
| `CLAUDE.md` | Claude Code | 入口，用 Claude Code 的 `@AGENTS.md` 导入语法引入正文 |

**项目里还没有 `AGENTS.md` / `CLAUDE.md`**——两个文件都放到仓库根目录：

```bash
base=https://raw.githubusercontent.com/Foxerine/sqlmodel-ext/master/ai-rules
curl -fsSLO "$base/AGENTS.md" && curl -fsSLO "$base/CLAUDE.md"
```

**项目里已经有了**——规则单独存一个文件，只引用一次：

```bash
curl -fsSL -o sqlmodel-ext.rules.md \
  https://raw.githubusercontent.com/Foxerine/sqlmodel-ext/master/ai-rules/AGENTS.md
echo '@sqlmodel-ext.rules.md' >> CLAUDE.md
printf '\nWhen writing SQLModel models or queries, follow sqlmodel-ext.rules.md.\n' >> AGENTS.md
```

Claude Code 在会话开始时展开 `@` 导入；Codex 等工具没有导入语法，靠 `AGENTS.md` 里那句话指向规则文件。把 `master` 换成与你所用版本对应的分支或 tag。

## 2. 规则概览

`AGENTS.md` 按主题分为 11 节，每条都是"规则 → 原因 → 正例 / 反例"：

1. **模型布局**：共享字段放 `XxxBase`，表和所有 DTO 继承它；`Field` / `Relationship` 从 `sqlmodel` 导入；字段说明写 docstring
2. **约束放在类型别名里**：用 `Str64`、`NonNegativeInt`、`NonNegativeWriteDecimal38_18` 等，需要数字时用 `max_length_of()`
3. **PATCH = `partial=True` + `Unset`**：用 `is Unset` 判断是否提交，不用 `is not None`，不用 `exclude_unset`
4. **不静默兜底**：禁止 `or 0` / `or ''` / `.get(k, 0)`；错误要抛出，不返回 `None`
5. **写入**：`save()` / `update()` 必须接返回值；不用 `session.refresh()`
6. **读取**：参数化 `get(condition)`，不写 `find_by_xxx` 薄包装；`col()` / `cond()` / `rel()`
7. **关系一律预加载**：`load=`、`@requires_relations`、`ensure_relations_loaded_bulk`；可空关系写 `'Target | None'`；模型模块不写 `from __future__ import annotations`
8. **并发与事务**：`with_for_update` + `@requires_for_update`；`OptimisticLockMixin` 放在 MRO 最前；外部副作用用 `add_post_commit_callback`
9. **缓存**：原生 DML 之后 `invalidate_on_commit()`
10. **删除与完整性错误**：`ResourceReferencedError` → 409，在父模型旁注册文案
11. **边界类型**：客户端时间用 `AwareDatetime`；金额用 Decimal 别名，绝不用 `float`

## 3. 配合 basedpyright

规则告诉 AI 怎么写，basedpyright 负责检查它有没有照做——**每次修改后跑到 0 error** 是规则包里的强制步骤。sqlmodel-ext 的 API 类型设计让大部分误用直接成为类型错误：

| 误用 | basedpyright 报告 |
|---|---|
| 显式声明的 `Unset \| T` 字段没判断 `is not Unset` 就使用 | `"MISSING" is not assignable to "str"` |
| 对 `Unset \| T \| None` 只判断了 `is not None` | 同上——`None` 判断排除不了 `Unset` |
| 把 `get()` 默认返回的 `T \| None` 当 `T` 用 | `"name" is not a known attribute of "None"` |
| `await Model.delete(session)` 什么都没传 | `No overloads for "delete"` |
| `load=Model.relation` 没包 `rel()` | `"Target" is not assignable to "QueryableAttribute[Any]"` |
| `Model.field.in_(...)` 没包 `col()` | `Cannot access attribute "in_"` |
| `select()` 超过 9 列 | `No overloads for "select"` |
| 丢弃 `save()` 的返回值 | warning `reportUnusedCallResult` |

完整的真实输出见仓库中的 [`examples/11_type_errors_caught_by_basedpyright.py`](https://github.com/Foxerine/sqlmodel-ext/blob/master/examples/11_type_errors_caught_by_basedpyright.py) 与 [`examples/README.md`](https://github.com/Foxerine/sqlmodel-ext/blob/master/examples/README.md)。

::: warning partial=True 的静态检查盲区
`partial=True` 的三态注解在类创建时由元类生成，basedpyright 在派生类上看到的仍是基类注解（`title: str`），因此**不会强制**你写 `is Unset` 判断。运行时语义不受影响。需要静态强制时有两个办法：把字段在 partial 类体里显式声明为 `field: Unset | T = Unset`，或者用实验性的 `python -m sqlmodel_ext.check_derived`（见 [检查 partial DTO 的误用](./check-partial-dtos)）。
:::

建议把 basedpyright 放进 CI，并在给 AI 的指令里把"basedpyright 0 error"写成完成标准：

```bash
basedpyright           # 在项目根目录运行；0 errors 才算完成
```

## 常见陷阱

- **只装 `CLAUDE.md` 不装 `AGENTS.md`**：`CLAUDE.md` 只是一个导入入口，正文在 `AGENTS.md`。
- **把规则正文复制进多个文件**：复制品不会跟着更新，两份规则会慢慢漂移——这正是本库要消灭的问题。始终只保留一份，其他地方引用它。
- **让 AI 用 `# type: ignore` 消掉报错**：规则禁止这样做；`# pyright: ignore[规则名]  # 原因` 只允许用于第三方 stub 自身的缺陷。
