# Static analyzer internals

::: tip Source location
`src/sqlmodel_ext/relation_load_checker.py` — `RelationLoadChecker`, `RelationLoadCheckMiddleware`, `run_model_checks`
:::

This is the **most complex module** in the entire project (~2000 lines). It uses AST static analysis to surface potential `MissingGreenlet` problems at application startup.

::: warning Disabled by default since 0.3
Starting with 0.3.0 this module is **experimental** and requires `rlc.check_on_startup = True` to opt in. This chapter explains **how it works**; for steps to enable it in your project, see [Prevent MissingGreenlet errors](/en/how-to/prevent-missing-greenlet).
:::

## Core class

```python
class RelationLoadChecker:
    def __init__(self, model_base_class=None):
        self.model_base_class = model_base_class
        self.warnings: list[RelationLoadWarning] = []
```

Uses Python's `ast` module to parse source code's **abstract syntax tree**, rather than executing code:
- No database connection needed
- No business logic execution required
- Can be completed during the import phase

## Analysis flow

```mermaid
flowchart TD
    Start["Application Startup"] --> A["run_model_checks(SQLModelBase)"]
    A --> B["Scan all SQLModelBase subclasses"]
    B --> C["AST-analyze each class's methods"]
    C --> D["Generate warnings"]

    Start --> E["RelationLoadCheckMiddleware"]
    E --> F["First request arrives"]
    F --> G["Scan all FastAPI route functions"]
    G --> H["Scan coroutines in imported modules"]
    H --> I["Generate warnings → Log"]
```

## Detection rules

### RLC001: response_model contains relation fields but endpoint doesn't preload

The analyzer:
1. Parses `response_model=UserResponse`, finds it contains a `profile` field
2. Checks query calls in the endpoint function body, finds no `load=` parameter
3. Generates a warning

```python
@router.get("/user/{id}", response_model=UserResponse)
async def get_user(session: SessionDep, id: UUID):
    return await User.get_exist_one(session, id) # [!code warning]
    # ⚠ RLC001: response_model contains profile, but query has no load=
```

### RLC002: accessing relations after save/update

Tracks variable "expiration state" — after a `save()` or `update()` call, all relations on the object are considered expired.

```python
user = await User.get_exist_one(session, id, load=User.profile)
user = await user.update(session, data)   # All relations expire after this // [!code warning]
return user.profile                        # RLC002 // [!code error]
```

### RLC003: accessing unloaded relations (local variables)

Tracks the object types and loaded relations bound to local variables, detecting attribute access on unloaded relations.

### RLC005: dependency does not preload response_model relations

When an ORM object returned by a FastAPI dependency (`Depends`) is returned
directly by the endpoint, the dependency's query must preload every relation
the response_model serializes — RLC005 checks whether the dependency body's
`load=` covers them.

### RLC007: accessing expired object column attributes after commit

Tracks `session.commit()` calls — accessing any attribute on related objects afterward is considered dangerous.

### RLC008: calling methods on expired objects after commit

Similar to RLC007, but detects method calls rather than attribute access.

### RLC009: type annotation resolution errors

Detects issues caused by mixing resolved types with string forward references.

### RLC010: passing an expired post-commit object as an argument

Passing an already-expired ORM object to another function/method after
`commit()` — the callee's column access triggers a synchronous lazy load →
`MissingGreenlet`. Sneakier than RLC007 (in-function access) because the crash
site is downstream in the call chain.

### RLC011: boolean test on an object with unloaded relations

`if not obj:` / `if obj:` implicitly calls `__bool__()`/`__len__()`, which may
touch unloaded relations. Rewrite as `if obj is None:`.

### RLC012: STI heterogeneous serialization field missing

A `response_model` field is missing from the `model_fields` of **some** STI
subclasses: when the endpoint returns the STI base class, the query may yield
subclass instances lacking the field, and serialization fails. Fix: call
`get()`/`get_with_count()` on the concrete subclass to narrow the return type.

### RLC013: column access after yield in an async generator

`yield` hands control to a consumer holding the same session reference, which
may commit during the yield and expire the object; access after resuming
triggers `MissingGreenlet`. Extract needed fields into locals before yielding.

## `RelationLoadWarning`

```python
class RelationLoadWarning:
    code: str          # "RLC001"
    message: str       # Human-readable description
    location: str      # "module.py:42 in function_name"
    severity: str      # "warning" or "error"
```

## `mark_app_check_completed()`

The middleware check executes only once. After analysis is complete on the first request, `mark_app_check_completed()` marks it as done, and subsequent requests don't repeat the check.

## Why AST instead of runtime checks?

| Approach | Pros | Cons |
|----------|------|------|
| AST static analysis | Finds issues at startup, no code execution, covers all paths | Possible false positives, can't analyze dynamic code |
| Runtime checks | 100% accurate | Only checks executed paths |

The static analyzer serves as the "first line of defense" alongside runtime `@requires_relations` and `lazy='raise_on_sql'`, forming multi-layer protection.

## Limitations

- **False positives**: Static analysis cannot track runtime dynamic behavior (e.g., `getattr`, conditional loading)
- **Coroutines only**: Synchronous functions are not analyzed (no MissingGreenlet issue in sync environments)
- **Module scope**: Only analyzes imported modules; unimported code is not scanned

::: warning False positives & project assumptions
The analyzer's AST rules are tuned to a specific project layout (FastAPI endpoints, STI inheritance conventions, `save`/`update`/`delete` naming). On other projects it **may produce false positives or fail to parse**. The module API is not in the semver stability promise.
:::
