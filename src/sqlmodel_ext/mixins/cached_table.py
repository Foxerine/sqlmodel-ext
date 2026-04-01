"""
可缓存模型 Mixin — 充血模型

为 SQLModel 表模型添加 Redis 查询缓存能力。继承即启用，不需要就不继承。

设计原则：
- 充血模型：缓存读写、失效逻辑全部内聚在 Mixin 中
- Redis 客户端集中管理：由 configure_redis() at startup
- 快速失败：Redis not configured -> RuntimeError，运行时 Redis 异常 logger.error() + 降级到 DB
- 全量缓存：始终缓存全部字段，不支持字段子集
- 序列化：model_dump_json() → orjson.loads → model_validate（确保 _sa_instance_state）
- 显式失败：no_cache 参数仅存在于 CacheableModelMixin.get()，
  非缓存模型传 no_cache=True 会 TypeError（Python 自然报错）

缓存双层架构：
- ID 缓存（id:{ModelName}:{id_value}）：单行 ID 等值查询，行级失效
- 查询缓存（query:{ModelName}:v{version}:{hash}）：条件/列表查询，版本号失效
- 版本号（ver:{ModelName}）：查询缓存的命名空间版本，INCR 递增使旧 key 不可达

失效粒度：
- save/update：行级多 key DEL id:{cls}:{id} + O(1) pipeline INCR ver:{cls}
- delete(instances)：行级多 key DEL 每个实例 + O(1) pipeline INCR ver:{cls}
- delete(condition)：SCAN+DEL id:{cls}:*（稀有路径）+ O(1) INCR ver:{cls}
- STI 多态：子类变更时 pipeline INCR 自身和所有缓存祖先的版本号

缓存跳过条件：
- no_cache=True（调用方显式跳过）
- load 包含非 MANYTOONE 或不可缓存关系（无法使用多 ID 缓存优化）
- options is not None（ExecutableOption 改变加载行为）
- with_for_update（悲观锁必须读最新）
- populate_existing（明确要求刷新 identity map）
- join is not None（JOIN target 变更不触发主模型失效，有幻读风险）
  # TODO: 未来优化 — 支持 JOIN 查询缓存，需追踪 join target 的变更

依赖关系：
    redis.asyncio
    orjson
    base/sqlmodel_base.py ← SQLModelBase

用法::

    class Character(CacheableModelMixin, CharacterBase, UUIDTableBaseMixin, table=True):
        __cache_ttl__: ClassVar[int] = 1800  # 30 分钟
"""
import ast
import asyncio
import hashlib
import inspect
import textwrap
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self, cast, overload

import json
import logging
from collections.abc import Callable

try:
    import orjson as _json_lib

    def _json_dumps(obj: Any) -> bytes:
        return _json_lib.dumps(obj)

    def _json_loads(data: bytes | str) -> Any:
        return _json_lib.loads(data)
except ImportError:
    _json_lib = None  # type: ignore[assignment]

    def _json_dumps(obj: Any) -> bytes:
        return json.dumps(obj, separators=(",", ":"), default=str).encode("utf-8")

    def _json_loads(data: bytes | str) -> Any:
        return json.loads(data)

logger = logging.getLogger(__name__)

from pydantic import ValidationError
from sqlalchemy import ColumnElement, event, inspect as sa_inspect
from sqlalchemy.orm import InstanceState, QueryableAttribute, Session as _SyncSession, make_transient_to_detached
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.orm.relationships import MANYTOONE  # pyright: ignore[reportPrivateImportUsage]
from sqlalchemy.sql import operators
from sqlalchemy.sql.base import ExecutableOption
from sqlalchemy.sql.elements import BinaryExpression
from sqlmodel.ext.asyncio.session import AsyncSession

from sqlalchemy.sql._typing import _OnClauseArgument  # pyright: ignore[reportPrivateUsage]

from sqlmodel_ext.base import SQLModelBase
from sqlmodel_ext.mixins.polymorphic import PolymorphicBaseMixin
from sqlmodel_ext.mixins.table import TableBaseMixin
from sqlmodel_ext.pagination import TableViewRequest



class _CacheResultType(StrEnum):
    """缓存序列化包装类型 — 区分 None/单个/列表三种查询结果。"""
    NONE = 'none'
    LIST = 'list'
    SINGLE = 'single'


# 序列化包装 JSON 字段名
_WRAPPER_TYPE_KEY = '_t'
_WRAPPER_ITEMS_KEY = '_items'
_WRAPPER_DATA_KEY = '_data'
_WRAPPER_CLASS_KEY = '_c'  # 实际类名（多态场景下还原正确子类）

# session.info keys — 缓存失效状态跟踪
_SESSION_PENDING_CACHE_KEY = '_pending_cache_invalidation_types'
_SESSION_SYNCED_CACHE_KEY = '_synced_cache_invalidation_types'
_SESSION_CASCADE_DELETED_KEY = '_cascade_deleted_for_sync_invalidation'

# 哨兵值 — add() 场景：新增项无需失效 ID 缓存，只需失效查询缓存
_QUERY_ONLY_INVALIDATION = object()

# 哨兵值 — delete(condition) 场景：条件删除无法提取具体 ID，需模型级全量失效
_FULL_MODEL_INVALIDATION = object()

# 哨兵值 — _try_load_from_id_caches() 返回，表示缓存未命中（区分于 None 结果）
_LOAD_CACHE_MISS = object()

# 子类方法禁止直接调用的缓存失效方法名
# check_cache_config() 使用 AST 检查子类方法体，防止 commit 后访问过期属性（MissingGreenlet）
_FORBIDDEN_DIRECT_CALLS: frozenset[str] = frozenset({
    'invalidate_by_id',
    'invalidate_all',
    '_invalidate_for_model',
    '_invalidate_id_cache',
    '_invalidate_query_caches',
})


class CachedTableBaseMixin(TableBaseMixin):
    """
    继承此 Mixin 即启用 Redis 查询缓存。不需要缓存就不要继承。

    MRO: Model → CacheableModelMixin → Base → TableBaseMixin

    ClassVar 配置项：
        __cache_ttl__: 缓存 TTL（秒），子类按需覆盖
    """

    __cache_ttl__: ClassVar[int] = 3600
    """缓存 TTL（秒）。通过类定义参数 cache_ttl=N 覆盖（由元类设置）。"""

    _commit_hook_registered: ClassVar[bool] = False
    """after_commit 事件钩子注册状态标记"""

    # ---- 内部常量 ----
    _CACHE_KEY_PREFIX: ClassVar[str] = 'query'
    """查询缓存 key 前缀。格式: query:{ModelName}:v{version}:{hash}"""

    _ID_CACHE_KEY_PREFIX: ClassVar[str] = 'id'
    """ID 缓存 key 前缀。格式: id:{ModelName}:{id_value}"""

    _CACHE_KEY_HASH_LENGTH: ClassVar[int] = 16
    """缓存 key 中 MD5 哈希的截取长度。"""

    _VERSION_KEY_PREFIX: ClassVar[str] = 'ver'
    """查询缓存版本 key 前缀。格式: ver:{ModelName}"""

    _SCAN_BATCH_SIZE: ClassVar[int] = 100
    """Redis SCAN 命令每次迭代的 count 参数（仅用于稀有的模型级 ID 缓存全量清除）。"""

    _subclass_name_cache: ClassVar[dict[str, type]] = {}
    """class_name → type 缓存，避免 _resolve_subclass() 每次递归遍历。"""

    on_cache_hit: ClassVar[Callable[[str], None] | None] = None
    """Optional callback invoked on cache hit. Receives model class name. Set at startup for metrics."""

    on_cache_miss: ClassVar[Callable[[str], None] | None] = None
    """Optional callback invoked on cache miss. Receives model class name. Set at startup for metrics."""

    # ================================================================
    #  Redis 客户端访问（managed via configure_redis()）
    # ================================================================

    _redis_client: ClassVar[Any] = None
    """Redis client instance. Must be set via configure_redis() before use."""

    @classmethod
    def configure_redis(cls, client: Any) -> None:
        """Configure the Redis client for caching.

        Must be called once at application startup before any cache operations.

        :param client: A redis.asyncio.Redis instance (decode_responses=False)
        """
        cls._redis_client = client

    @classmethod
    def _get_client(cls) -> Any:
        """Get the Redis client for caching.

        Returns the client configured via configure_redis().
        Raises RuntimeError if not configured.

        Return type is Any because redis.asyncio.Redis generic parameter
        depends on runtime configuration (decode_responses).
        """
        if cls._redis_client is None:
            raise RuntimeError(
                f"{cls.__name__}: Redis client not configured. "
                f"Call CachedTableBaseMixin.configure_redis(redis_client) at startup."
            )
        return cls._redis_client

    # ================================================================
    #  缓存原语（运行时 Redis 异常 → l.error + 降级）
    # ================================================================

    @classmethod
    async def _cache_get(cls, key: str) -> bytes | None:
        """读缓存。Redis 异常时 logger.error() + 返回 None（降级到 DB）。"""
        try:
            return await cls._get_client().get(key)
        except RuntimeError:
            raise  # 未初始化：快速失败
        except Exception as e:
            logger.error(f"Redis 读取异常 key='{key}': {e}")
            return None

    @classmethod
    async def _cache_set(cls, key: str, value: bytes, ttl: int) -> None:
        """写缓存。Redis 异常时 logger.error() + 跳过（非关键路径）。"""
        try:
            await cls._get_client().set(key, value, ex=ttl)
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Redis 写入异常 key='{key}': {e}")

    @classmethod
    async def _cache_delete(cls, key: str) -> None:
        """删缓存。Redis 异常时向上抛出（调用方决定是否吞掉）。

        RuntimeError（未初始化）直接抛出；其他异常也抛出，
        以便 sync 路径可检测失败、不标记 synced，让补偿重试。
        """
        await cls._get_client().delete(key)

    @classmethod
    async def _cache_delete_pattern(cls, pattern: str) -> None:
        """SCAN + DEL 模式删除。避免 KEYS 阻塞。

        Redis 异常时向上抛出（同 _cache_delete）。
        """
        client = cls._get_client()
        cursor = 0
        while True:
            cursor, keys = await client.scan(
                cursor, match=pattern, count=cls._SCAN_BATCH_SIZE,
            )
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

    # ================================================================
    #  版本号管理（query cache 失效用 version bump 替代 SCAN+DEL）
    # ================================================================

    @classmethod
    def _build_version_key(cls) -> str:
        """构建版本 key。格式: ver:{ModelName}"""
        return f"{cls._VERSION_KEY_PREFIX}:{cls.__name__}"

    @classmethod
    async def _get_query_version(cls) -> int:
        """获取当前查询缓存版本号。key 不存在时返回 0（初始版本）。

        Redis 异常时返回 0（降级到无版本，query key 回退为 v0）。
        """
        try:
            raw = await cls._get_client().get(cls._build_version_key())
            return int(raw) if raw is not None else 0
        except RuntimeError:
            raise  # 未初始化：快速失败
        except Exception as e:
            logger.error(f"Redis 版本读取异常 ({cls.__name__}): {e}")
            return 0

    @classmethod
    async def _bump_query_version(cls) -> int:
        """递增查询缓存版本号。O(1) 替代 SCAN+DEL O(N)。

        INCR 对不存在的 key 自动从 0 开始递增（Redis 原子操作）。
        旧版本的 query key 自然通过 TTL 过期，无需主动清理。

        :return: 递增后的版本号（失败时返回 0）
        """
        try:
            new_ver: int = await cls._get_client().incr(cls._build_version_key())
            return new_ver
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Redis 版本递增异常 ({cls.__name__}): {e}")
            return 0

    # ================================================================
    #  静态检查
    # ================================================================

    @classmethod
    def check_cache_config(cls) -> None:
        """遍历所有 CacheableModelMixin 子类，检查配置错误，并注册 session 事件钩子。

        在 main.py 启动完成后调用一次（configure_redis() has been called）。

        检查项：
        1. Redis client is configured
        2. 子类（递归）不得重写 _get_client（破坏 Redis 访问机制）
        3. __cache_ttl__ 必须是正整数（None 也报错）
        4. 子类方法不得直接调用缓存失效方法（AST 检查）
        5. cascade_delete + passive_deletes 不得指向 CachedTableBaseMixin 子类
           （passive_deletes 会跳过 SA 级联加载，persistent_to_deleted 事件不触发）

        副作用：
        - 注册 SQLAlchemy Session after_commit/after_rollback/persistent_to_deleted 事件钩子
        """
        # 验证 Redis client is available
        _ = cls._get_client()

        violations: list[str] = []

        def _check_forbidden_calls(sub: type) -> None:
            """AST 检查子类方法体中是否直接调用了缓存失效方法。

            子类方法绕过 CRUD 后需要失效缓存时，应使用：
            - _register_pending_invalidation() 注册待失效 ID
            - _commit_and_invalidate() 或 _sync_invalidate_after_commit() 执行失效
            直接调用 invalidate_by_id() 等方法可能导致 commit 后访问过期属性（MissingGreenlet）。
            """
            for attr_name, attr in sub.__dict__.items():
                # 解包描述符，收集所有需要扫描的函数对象
                funcs: list[Any] = []
                if isinstance(attr, (classmethod, staticmethod)):
                    funcs.append(attr.__func__)
                elif isinstance(attr, property):
                    for accessor in (attr.fget, attr.fset, attr.fdel):
                        if accessor is not None:
                            funcs.append(accessor)
                elif inspect.isfunction(attr):
                    funcs.append(attr)
                seen: set[str] = set()
                for func_obj in funcs:
                    try:
                        source = textwrap.dedent(inspect.getsource(func_obj))
                        tree = ast.parse(source)
                    except (OSError, TypeError, SyntaxError):
                        continue
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call):
                            continue
                        func = node.func
                        call_name: str | None = None
                        if isinstance(func, ast.Attribute):
                            call_name = func.attr
                        elif isinstance(func, ast.Name):
                            call_name = func.id
                        if call_name and call_name in _FORBIDDEN_DIRECT_CALLS and call_name not in seen:
                            seen.add(call_name)
                            violations.append(f"  - {sub.__name__}.{attr_name}() -> {call_name}()")

        def _check_cascade_passive_deletes(sub: type) -> None:
            """检测 cascade_delete + passive_deletes + CachedTarget 的危险组合。

            passive_deletes 让 SA 跳过子记录加载，依赖 DB CASCADE 静默删除。
            此时 persistent_to_deleted 事件不触发，子模型缓存不会自动失效。
            """
            try:
                mapper = sa_inspect(sub)
            except Exception:
                return
            for rel in mapper.relationships:
                if not rel.cascade.delete or not rel.passive_deletes:
                    continue
                target = rel.mapper.class_
                if isinstance(target, type) and issubclass(target, CachedTableBaseMixin):
                    raise TypeError(
                        f"{sub.__name__}.{rel.key} → {target.__name__}: "
                        f"cascade_delete=True + passive_deletes={rel.passive_deletes!r} "
                        f"指向 CachedTableBaseMixin 子类。"
                        f"passive_deletes 会跳过 SA 级联加载，导致 persistent_to_deleted "
                        f"事件不触发，子模型缓存无法自动失效。"
                        f"移除 passive_deletes 或改用显式缓存失效。"
                    )

        def _check_subclasses(parent: type) -> None:
            for sub in parent.__subclasses__():
                if '_get_client' in sub.__dict__:
                    raise TypeError(f"{sub.__name__} 不得重写 _get_client")
                ttl = getattr(sub, '__cache_ttl__', None)
                if not isinstance(ttl, int) or ttl <= 0:
                    raise ValueError(
                        f"{sub.__name__}.__cache_ttl__ 必须是正整数，当前: {ttl!r}"
                    )
                _check_forbidden_calls(sub)
                _check_cascade_passive_deletes(sub)
                _check_subclasses(sub)

        _check_subclasses(cls)

        if violations:
            nl = '\n'
            raise TypeError(
                f"以下子类方法直接调用了缓存失效方法（可能导致 commit 后 MissingGreenlet）：\n"
                f"{nl.join(violations)}\n"
                f"应使用 _register_pending_invalidation() + _commit_and_invalidate()/_sync_invalidate_after_commit() 代替。"
            )

        # 注册 session 事件钩子（幂等）
        cls._register_session_commit_hook()

    @classmethod
    def _register_session_commit_hook(cls) -> None:
        """注册 SQLAlchemy Session after_commit/after_rollback 事件钩子。

        after_commit: 自动刷新 session.info 中累积的待失效缓存类型。
        覆盖所有 commit 路径：CRUD 方法 commit=True、直接 session.commit()。

        after_rollback: 清空累积的待失效类型（数据已回滚，无需失效）。

        幂等：多次调用只注册一次。

        局限性（fire-and-forget）：
        after_commit handler 是同步函数（SQLAlchemy event 限制），无法 await 异步失效。
        使用 loop.create_task() 调度失效，提交返回和失效完成之间有极短窗口（通常 < 1ms）。
        commit=True 的 CRUD 方法已做同步 await 失效（无此窗口），此路径仅覆盖
        commit=False → session.commit() 场景。TTL 提供最终一致性兜底。
        """
        if cls._commit_hook_registered:
            return

        def _after_commit_handler(session: _SyncSession) -> None:
            pending: dict[type, set[Any]] | None = session.info.pop(_SESSION_PENDING_CACHE_KEY, None)
            if not pending:
                return
            loop = asyncio.get_running_loop()

            # 创建本次 commit 周期的同步失效跟踪字典。
            # CRUD 方法在 super() 返回后（即本 handler 执行后）会往此字典写入已同步失效的
            # (类型, instance_ids) 对。_compensate task 按 instance_id 去重，只补偿未同步的 ID。
            synced: dict[type, set[Any]] = {}
            session.info[_SESSION_SYNCED_CACHE_KEY] = synced

            async def _compensate(
                    to_invalidate: dict[type, set[Any]],
                    already_synced: dict[type, set[Any]],
            ) -> None:
                sentinels = {_QUERY_ONLY_INVALIDATION, _FULL_MODEL_INVALIDATION}
                for model_type, pending_ids in to_invalidate.items():
                    if not issubclass(model_type, CachedTableBaseMixin):
                        continue

                    needs_full = _FULL_MODEL_INVALIDATION in pending_ids
                    synced_ids = already_synced.get(model_type)

                    if synced_ids is not None:
                        # 同步路径已对该类型执行过失效（query: 缓存已覆盖）。
                        if needs_full and _FULL_MODEL_INVALIDATION not in synced_ids:
                            # pending 含全量哨兵但同步路径未做全量 — 补偿模型级
                            try:
                                await model_type._invalidate_for_model()
                            except Exception as e:
                                logger.error(f"commit 后补偿模型级缓存失效失败 ({model_type.__name__}): {e}")
                            continue
                        # 仅补偿未被同步路径处理的 ID 缓存
                        remaining = pending_ids - synced_ids - sentinels
                        if remaining:
                            try:
                                for _id in remaining:
                                    await model_type._invalidate_id_cache(_id)
                            except Exception as e:
                                logger.error(f"commit 后补偿 ID 缓存失效失败 ({model_type.__name__}): {e}")
                        continue

                    # 该类型完全未被同步路径处理 — 完整补偿（未迁移路径）
                    logger.warning(
                        f"fallback 补偿触发: {model_type.__name__} 未走同步失效路径"
                        f"（pending_ids={len(pending_ids)}，请迁移到 cache_aware_commit）"
                    )
                    try:
                        if needs_full:
                            await model_type._invalidate_for_model()
                        else:
                            real_ids = pending_ids - sentinels
                            has_query_only = _QUERY_ONLY_INVALIDATION in pending_ids
                            if real_ids:
                                for _id in real_ids:
                                    await model_type._invalidate_id_cache(_id)
                                await model_type._invalidate_query_caches()
                            elif has_query_only:
                                await model_type._invalidate_query_caches()
                            else:
                                await model_type._invalidate_for_model()
                    except Exception as e:
                        logger.error(f"commit 后补偿缓存失效失败 ({model_type.__name__}): {e}")

            # fire-and-forget：同步路径已处理的 instance_id 被 synced 去重；
            # 仅对 commit=False 累积且未被同步失效的 ID 执行补偿。
            _ = loop.create_task(_compensate(pending, synced))

        def _after_rollback_handler(session: _SyncSession) -> None:
            session.info.pop(_SESSION_PENDING_CACHE_KEY, None)
            session.info.pop(_SESSION_SYNCED_CACHE_KEY, None)
            session.info.pop(_SESSION_CASCADE_DELETED_KEY, None)

        def _on_persistent_to_deleted(session: _SyncSession, instance: object) -> None:
            """级联删除缓存失效：监听 SA flush 时的 persistent→deleted 状态转换。

            SA async 模式下，passive_deletes=False 的 cascade_delete 关系在 flush 时
            会自动 SELECT + 逐条 DELETE 子记录。每条 DELETE 触发此事件。

            双写策略：
            1. _SESSION_PENDING_CACHE_KEY: after_commit 补偿路径（fire-and-forget）
            2. _SESSION_CASCADE_DELETED_KEY: delete() 同步路径（await 失效，无窗口）
            """
            if isinstance(instance, CachedTableBaseMixin):
                _id = getattr(instance, 'id', None)
                if _id is not None:
                    _cls = type(instance)
                    CachedTableBaseMixin._register_pending_invalidation(
                        session, _cls, _id,  # pyright: ignore[reportArgumentType]  # SA event 传 _SyncSession，.info 兼容
                    )
                    cascade_info: dict[type, set[Any]] = session.info.setdefault(
                        _SESSION_CASCADE_DELETED_KEY, {}
                    )
                    cascade_info.setdefault(_cls, set()).add(_id)

        event.listen(_SyncSession, "after_commit", _after_commit_handler)
        event.listen(_SyncSession, "after_rollback", _after_rollback_handler)
        event.listen(_SyncSession, "persistent_to_deleted", _on_persistent_to_deleted)

        cls._commit_hook_registered = True
        logger.debug("CacheableModelMixin: session commit/rollback/cascade 事件钩子已注册")

    # ================================================================
    #  ID 查询检测
    # ================================================================

    @classmethod
    def _extract_id_from_condition(cls, condition: Any) -> Any | None:
        """检测 condition 是否为 Model.id == value 形式的 ID 等值查询。

        :return: ID 值（如果是 ID 查询），否则 None
        """
        if not isinstance(condition, BinaryExpression):
            return None
        if condition.operator is not operators.eq:
            return None
        left = condition.left
        if hasattr(left, 'key') and left.key == 'id':
            right = condition.right
            if hasattr(right, 'value'):
                return right.value
        return None

    @classmethod
    def _build_id_cache_key(cls, id_value: Any) -> str:
        """构建 ID 缓存 key。格式: id:{ModelName}:{id_value}"""
        return f"{cls._ID_CACHE_KEY_PREFIX}:{cls.__name__}:{id_value}"

    # ================================================================
    #  load 关系缓存（多 ID 缓存联合查询）
    # ================================================================

    @classmethod
    def _has_pending_invalidation(cls, session: AsyncSession) -> bool:
        """检查 session 中是否有与 cls 相关的待提交缓存失效。"""
        pending: dict[type, set[Any]] | None = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if not pending:
            return False
        return any(
            issubclass(pending_type, cls) or issubclass(cls, pending_type)
            for pending_type in pending
        )

    @classmethod
    def _analyze_load_relations(
            cls,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]],
    ) -> 'list[tuple[str, type[CachedTableBaseMixin], str]] | None':
        """分析 load 参数是否全部为 MANYTOONE 且目标可缓存。

        仅处理直属于 cls（含继承）的 MANYTOONE 关系。
        链式加载（如 Character.tool_set → ToolSet.tools）不支持。

        :return: [(rel_name, target_cls, fk_attr_name), ...] 或 None（不满足条件时）
        """
        load_list = load if isinstance(load, list) else [load]
        mapper = sa_inspect(cls)
        relationships = mapper.relationships  # pyright: ignore[reportOptionalMemberAccess]
        result: list[tuple[str, type[CachedTableBaseMixin], str]] = []

        for attr in load_list:
            # 检查是否为 cls 的直属关系（含继承）
            attr_owner: type[Any] = attr.class_  # pyright: ignore[reportAssignmentType]
            if not issubclass(cls, attr_owner):
                return None  # 链式加载，无法从缓存处理

            rel_name: str = attr.key
            if rel_name not in relationships:
                return None  # 不是关系属性

            rel_prop = relationships[rel_name]
            if rel_prop.direction is not MANYTOONE:
                return None  # 只支持 MANYTOONE（FK 在主模型侧）

            target_cls = rel_prop.mapper.class_
            if not issubclass(target_cls, CachedTableBaseMixin):
                return None  # 目标模型不可缓存

            # 提取 FK 属性名（如 permission_id）
            local_col = rel_prop.local_remote_pairs[0][0]
            fk_attr_name: str = local_col.key
            result.append((rel_name, target_cls, fk_attr_name))

        return result

    @classmethod
    async def _try_load_from_id_caches(
            cls,
            session: AsyncSession,
            id_value: Any,
            rel_info: 'list[tuple[str, type[CachedTableBaseMixin], str]]',
    ) -> Any:
        """尝试从多个 ID 缓存联合加载主模型 + MANYTOONE 关系。

        流程：
        1. 查主模型 ID 缓存 → 反序列化
        2. 遍历 rel_info，按 FK 值查关系模型的 ID 缓存
        3. 全部命中 → merge 到 session + set_committed_value → 返回
        4. 任一未命中 → 返回 _LOAD_CACHE_MISS

        :return: 模型实例 | None（缓存的空结果）| _LOAD_CACHE_MISS（需回源 DB）
        """
        # 1. 查主模型 ID 缓存
        main_cache_key = cls._build_id_cache_key(id_value)
        main_raw = await cls._cache_get(main_cache_key)
        if main_raw is None:
            return _LOAD_CACHE_MISS

        try:
            main_obj = cls._deserialize_result(main_raw, 'first')
        except Exception:
            try:
                await cls._cache_delete(main_cache_key)
            except Exception:
                pass
            return _LOAD_CACHE_MISS

        if main_obj is None:
            return None  # 缓存的空结果

        # 2. 收集关系 FK → 构建缓存 key 列表，通过 pipeline mget 一次查完
        #    （N 个关系从 N+1 次 Redis RTT 降为 2 次：1 次 get 主模型 + 1 次 mget 全部关系）
        rel_entries: list[tuple[str, 'type[CachedTableBaseMixin]', str | None]] = []
        """(rel_name, target_cls, cache_key | None)"""
        pipeline_keys: list[str] = []

        for rel_name, target_cls, fk_attr_name in rel_info:
            fk_value = getattr(main_obj, fk_attr_name, None)
            if fk_value is None:
                rel_entries.append((rel_name, target_cls, None))
                continue
            if target_cls._has_pending_invalidation(session):
                return _LOAD_CACHE_MISS
            rel_cache_key = target_cls._build_id_cache_key(fk_value)
            rel_entries.append((rel_name, target_cls, rel_cache_key))
            pipeline_keys.append(rel_cache_key)

        # 批量读取所有关系缓存（1 次 RTT）
        pipeline_results: list[bytes | None] = []
        if pipeline_keys:
            try:
                pipeline_results = await cls._get_client().mget(pipeline_keys)
            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"Redis mget 异常: {e}")
                return _LOAD_CACHE_MISS

        # 3. 反序列化关系对象
        pipeline_idx = 0
        rel_objects: list[tuple[str, Any]] = []
        for rel_name, target_cls, cache_key in rel_entries:
            if cache_key is None:
                rel_objects.append((rel_name, None))
                continue

            rel_raw = pipeline_results[pipeline_idx]
            pipeline_idx += 1
            if rel_raw is None:
                return _LOAD_CACHE_MISS

            try:
                rel_obj = target_cls._deserialize_result(rel_raw, 'first')
            except Exception:
                try:
                    await target_cls._cache_delete(cache_key)
                except Exception:
                    pass
                return _LOAD_CACHE_MISS

            rel_objects.append((rel_name, rel_obj))

        # 3. 全部命中 → 合并到 session identity map
        # 先合并关系对象
        merged_rels: list[tuple[str, Any]] = []
        for rel_name, rel_obj in rel_objects:
            if rel_obj is not None:
                make_transient_to_detached(rel_obj)
                rel_obj = await session.merge(rel_obj, load=False)
            merged_rels.append((rel_name, rel_obj))

        # 合并主对象
        make_transient_to_detached(main_obj)
        main_obj = await session.merge(main_obj, load=False)

        # 设置关系属性（不触发 ORM 变更追踪）
        for rel_name, rel_obj in merged_rels:
            set_committed_value(main_obj, rel_name, rel_obj)

        return main_obj

    @classmethod
    async def _write_load_result_to_id_caches(
            cls,
            result: Any,
            rel_info: 'list[tuple[str, type[CachedTableBaseMixin], str]]',
    ) -> None:
        """DB 查询带 load 后，将主模型和 MANYTOONE 关系模型分别写入各自的 ID 缓存。

        主模型和关系模型独立缓存（各用各自的 ID cache key + TTL），
        失效也独立（各自 save/update/delete 时自然失效各自的缓存）。
        """
        if result is None:
            return

        items = [result] if not isinstance(result, list) else result

        for item in items:
            if item is None:
                continue

            # 写主模型到 cls 的 ID 缓存
            item_id = getattr(item, 'id', None)
            if item_id is not None:
                try:
                    cache_key = cls._build_id_cache_key(item_id)
                    serialized = cls._serialize_result(item)
                    await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
                except Exception as e:
                    logger.error(f"缓存主模型写入失败 ({cls.__name__}:{item_id}): {e}")

            # 写关系模型到 target_cls 的 ID 缓存
            for rel_name, target_cls, _fk_attr in rel_info:
                rel_obj = getattr(item, rel_name, None)
                if rel_obj is None:
                    continue
                rel_id = getattr(rel_obj, 'id', None)
                if rel_id is None:
                    continue
                actual_rel_cls = type(rel_obj)
                if not issubclass(actual_rel_cls, CachedTableBaseMixin):
                    continue
                try:
                    cache_key = target_cls._build_id_cache_key(rel_id)
                    serialized = target_cls._serialize_result(rel_obj)
                    await target_cls._cache_set(cache_key, serialized, target_cls.__cache_ttl__)
                except Exception as e:
                    logger.error(f"缓存关系模型写入失败 ({target_cls.__name__}:{rel_id}): {e}")

    # ================================================================
    #  缓存 Key 构建
    # ================================================================

    @classmethod
    def _build_cache_key(
            cls,
            condition: Any,
            fetch_mode: str,
            offset: int | None,
            limit: int | None,
            order_by: Any,
            load: Any,
            filter_expr: Any,
            table_view: Any,
            *time_args: Any,
            version: int = 0,
    ) -> str:
        """从查询参数构建确定性缓存 key（带版本号）。

        先将 table_view 合并到显式参数（镜像 table.py 的合并逻辑），
        确保语义相同的查询产生相同的 key，无论参数来源是 table_view 还是显式传参。

        time_args 顺序: created_before, created_after, updated_before, updated_after
        version: 查询缓存版本号，每次 _invalidate_query_caches() 递增

        格式: {_CACHE_KEY_PREFIX}:{ModelName}:v{version}:{md5_hash[:_CACHE_KEY_HASH_LENGTH]}
        """
        try:
            from sqlalchemy.dialects import postgresql
            _dialect = postgresql.dialect()
        except Exception:
            _dialect = None

        def _compile_sql(expr: Any) -> str:
            """Compile SQL expression to string, with dialect fallback."""
            if _dialect is not None:
                try:
                    return str(expr.compile(dialect=_dialect, compile_kwargs={"literal_binds": True}))
                except Exception:
                    pass
            try:
                return str(expr.compile(compile_kwargs={"literal_binds": True}))
            except Exception:
                return repr(expr)

        # ---- 归一化：将 table_view 合并到显式参数（镜像 table.py:890-911） ----
        # time_args: [created_before, created_after, updated_before, updated_after]
        merged_times = list(time_args)
        if table_view is not None:
            # 时间字段：显式参数优先，None 时取 table_view
            tv_times = [
                table_view.created_before_datetime,
                table_view.created_after_datetime,
                table_view.updated_before_datetime,
                table_view.updated_after_datetime,
            ]
            for i in range(min(len(merged_times), 4)):
                if merged_times[i] is None:
                    merged_times[i] = tv_times[i]
            # 分页：显式参数优先
            if offset is None:
                offset = table_view.offset
            if limit is None:
                limit = table_view.limit
            # 排序：显式 order_by 优先，否则用 table_view 描述
            if order_by is None:
                parts_order = f"ob={table_view.order},{'d' if table_view.desc else 'a'}"
            else:
                parts_order = None
        else:
            parts_order = None

        parts: list[str] = [fetch_mode]

        # condition → SQL string
        if condition is None:
            parts.append("none")
        elif isinstance(condition, bool):
            parts.append(str(condition))
        else:
            parts.append(_compile_sql(condition))

        # pagination (normalized)
        if offset is not None:
            parts.append(f"o={offset}")
        if limit is not None:
            parts.append(f"l={limit}")

        # ordering (normalized)
        if order_by:
            for ob in order_by:
                parts.append(_compile_sql(ob))
        elif parts_order is not None:
            parts.append(parts_order)

        # load args (affect returned data content)
        if load is not None:
            load_list = load if isinstance(load, list) else [load]
            parts.append("load=" + ",".join(str(item.key) for item in load_list))

        # filter (bool values also need inclusion: False = WHERE FALSE, True = no condition)
        if filter_expr is not None:
            if isinstance(filter_expr, bool):
                parts.append(f"f={filter_expr}")
            else:
                try:
                    parts.append("f=" + _compile_sql(filter_expr))
                except Exception:
                    parts.append("f=" + repr(filter_expr))

        # 时间过滤（已归一化，table_view 时间字段已合并）
        for i, ta in enumerate(merged_times):
            if ta is not None:
                parts.append(f"t{i}={ta.isoformat()}")

        key_hash = hashlib.md5("|".join(parts).encode()).hexdigest()[:cls._CACHE_KEY_HASH_LENGTH]
        return f"{cls._CACHE_KEY_PREFIX}:{cls.__name__}:v{version}:{key_hash}"

    # ================================================================
    #  序列化 / 反序列化
    # ================================================================

    @classmethod
    def _serialize_item(cls, item: Any) -> bytes:
        """将单个 SQLModelBase 实例序列化为 bytes，包含实际类名。

        多态场景下（STI），查询基类可能返回子类实例。
        记录实际类名以确保反序列化时还原正确子类。

        只序列化列字段（column attrs），排除：
        - 关系属性（lazy='raise_on_sql' 会报错，且 ID 缓存只存列数据）
        - computed_field（可能依赖关系属性，如 ToolSet.tool_count → self.tools）

        load 查询时，主模型和关系模型各自独立序列化到各自的 ID 缓存。
        """
        if isinstance(item, SQLModelBase):
            mapper = sa_inspect(type(item))
            column_fields: set[str] = {prop.key for prop in mapper.column_attrs}
            item_dict: dict[str, Any] = item.model_dump(mode='json', include=column_fields)
            item_dict[_WRAPPER_CLASS_KEY] = type(item).__name__
            return _json_dumps(item_dict)
        return _json_dumps(item)

    @classmethod
    def _serialize_result(cls, result: Any) -> bytes:
        """将 get() 查询结果序列化为 bytes（用于 Redis 存储）。

        使用 orjson 序列化 + 包装格式区分 None/single/list。
        每个 SQLModelBase 项包含 _c 字段记录实际类名（多态安全）。
        """
        if result is None:
            return _json_dumps({_WRAPPER_TYPE_KEY: _CacheResultType.NONE})
        elif isinstance(result, list):
            serialized_items = [cls._serialize_item(item).decode('utf-8') for item in result]
            items_json = "[" + ",".join(serialized_items) + "]"
            wrapper = (
                f'{{"{_WRAPPER_TYPE_KEY}":"{_CacheResultType.LIST}"'
                f',"{_WRAPPER_ITEMS_KEY}":{items_json}}}'
            )
            return wrapper.encode('utf-8')
        else:
            data_json = cls._serialize_item(result).decode('utf-8')
            wrapper = (
                f'{{"{_WRAPPER_TYPE_KEY}":"{_CacheResultType.SINGLE}"'
                f',"{_WRAPPER_DATA_KEY}":{data_json}}}'
            )
            return wrapper.encode('utf-8')

    @classmethod
    def _resolve_subclass(cls, class_name: str | None) -> type:
        """从类名解析出实际子类。用于多态反序列化。

        使用 _subclass_name_cache 缓存查找结果，首次递归遍历后 O(1) 命中。
        缓存 key 包含查询起点类名以区分不同继承树。
        """
        if class_name is None or cls.__name__ == class_name:
            return cls

        lookup_key = f"{cls.__name__}.{class_name}"
        cached = CachedTableBaseMixin._subclass_name_cache.get(lookup_key)
        if cached is not None:
            return cached

        def _walk(klass: type) -> type | None:
            for sub in klass.__subclasses__():
                if sub.__name__ == class_name:
                    return sub
                found = _walk(sub)
                if found is not None:
                    return found
            return None

        resolved = _walk(cls) or cls
        CachedTableBaseMixin._subclass_name_cache[lookup_key] = resolved
        return resolved

    @classmethod
    def _deserialize_item(cls, item_data: dict[str, Any]) -> Any:
        """从缓存 dict 重建单个模型实例。

        读取 _c 字段解析实际子类（多态安全），然后 pop 掉 _c 再 model_validate。
        """
        class_name = item_data.pop(_WRAPPER_CLASS_KEY, None)
        actual_cls = cls._resolve_subclass(class_name)
        return actual_cls.model_validate(item_data)

    @classmethod
    def _deserialize_result(cls, raw: bytes, _fetch_mode: str) -> Any:
        """从缓存 bytes 重建 get() 查询结果。

        使用 orjson.loads → model_validate（非 model_validate_json，
        因为 model_validate_json 对 table=True 模型的 UUID 字段返回 str）。

        :raises ValidationError: schema 不匹配时
        :raises orjson.JSONDecodeError: 无效 JSON
        """
        cached = _json_loads(raw)
        result_type = cached.get(_WRAPPER_TYPE_KEY)
        if result_type == _CacheResultType.NONE:
            return None
        elif result_type == _CacheResultType.LIST:
            return [cls._deserialize_item(item) for item in cached[_WRAPPER_ITEMS_KEY]]
        elif result_type == _CacheResultType.SINGLE:
            return cls._deserialize_item(cached[_WRAPPER_DATA_KEY])
        raise ValueError(
            f"未知的缓存结果类型: {result_type!r}，期望 {_CacheResultType.NONE!r}/{_CacheResultType.LIST!r}/{_CacheResultType.SINGLE!r}"
        )

    # ================================================================
    #  失效
    # ================================================================

    @classmethod
    def _cached_ancestors(cls) -> list[type['CachedTableBaseMixin']]:
        """收集 MRO 中除自身外的 CachedTableBaseMixin 缓存祖先（STI 继承链）。"""
        return [
            ancestor
            for ancestor in cls.__mro__
            if (
                ancestor is not cls
                and ancestor is not object
                and ancestor is not CachedTableBaseMixin
                and isinstance(ancestor, type)
                and issubclass(ancestor, CachedTableBaseMixin)
            )
        ]

    @classmethod
    async def _invalidate_id_cache(cls, instance_id: Any) -> None:
        """行级 DEL：单次多 key DELETE 删除自身 + 祖先的同 ID 缓存。

        Redis DELETE 原生支持多 key（单命令、单 RTT、原子执行）。
        """
        prefix = cls._ID_CACHE_KEY_PREFIX
        keys = [f"{prefix}:{cls.__name__}:{instance_id}"]
        for ancestor in cls._cached_ancestors():
            keys.append(f"{prefix}:{ancestor.__name__}:{instance_id}")
        await cls._get_client().delete(*keys)

    @classmethod
    async def _invalidate_query_caches(cls) -> None:
        """O(1) version bump：pipeline 批量 INCR 自身和所有缓存祖先的版本号。

        pipeline(transaction=False) 将多个 INCR 打包为一次 RTT 发送。
        旧版本的 query key 自然通过 TTL 过期，无需 SCAN+DEL。
        """
        ancestors = cls._cached_ancestors()
        if not ancestors:
            # 无祖先，直接单次 INCR（避免 pipeline 开销）
            await cls._bump_query_version()
            return
        client = cls._get_client()
        async with client.pipeline(transaction=False) as pipe:
            pipe.incr(cls._build_version_key())
            for ancestor in ancestors:
                pipe.incr(ancestor._build_version_key())
            await pipe.execute()

    @classmethod
    async def _invalidate_for_model(cls, _instance_id: Any | None = None) -> None:
        """失效缓存：ID 缓存 + 查询缓存版本号。

        save/update/delete 后调用。

        - _instance_id 提供时：行级多 key DEL 该 ID 的 id: 缓存
        - _instance_id 为 None 时：模型级 SCAN+DEL 所有 id: 缓存（稀有路径）
        - 查询缓存始终 O(1) pipeline INCR 版本号
        """
        if _instance_id is not None:
            await cls._invalidate_id_cache(_instance_id)
        else:
            # 无 instance_id → 模型级清除所有 ID 缓存（仅条件删除时触发，稀有路径）
            id_prefix = cls._ID_CACHE_KEY_PREFIX
            await cls._cache_delete_pattern(f"{id_prefix}:{cls.__name__}:*")
            for ancestor in cls._cached_ancestors():
                await cls._cache_delete_pattern(f"{id_prefix}:{ancestor.__name__}:*")
        # 查询缓存始终 O(1) 版本号递增
        await cls._invalidate_query_caches()

    @classmethod
    async def invalidate_by_id(cls, *_ids: Any) -> None:
        """公共 API：手动失效指定 ID 的缓存。

        仅供外部调用（管理脚本、测试、非模型代码）。
        模型内部的 raw SQL 方法应使用 _register_pending_invalidation() +
        _commit_and_invalidate()/_sync_invalidate_after_commit()，
        避免 commit 后访问过期属性导致 MissingGreenlet。

        每个 ID 行级多 key DEL id: 缓存，查询缓存 O(1) pipeline INCR 版本号。

        Redis 异常时记录日志并吞掉（fire-and-forget），
        避免 DB 已提交但接口返回 500 的不一致状态。
        """
        try:
            for _id in _ids:
                await cls._invalidate_id_cache(_id)
            await cls._invalidate_query_caches()
        except Exception as e:
            logger.error(f"invalidate_by_id() Redis 失效失败 ({cls.__name__}, ids={_ids}): {e}")

    @classmethod
    async def invalidate_all(cls) -> None:
        """公共 API：失效该模型的全部缓存（id: + query:）。

        Redis 异常时记录日志并吞掉（同 invalidate_by_id）。
        """
        try:
            await cls._invalidate_for_model()
        except Exception as e:
            logger.error(f"invalidate_all() Redis 失效失败 ({cls.__name__}): {e}")

    # ================================================================
    #  get() 重写 — 缓存读取路径
    # ================================================================

    @overload
    @classmethod
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["all"],
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> list[Self]: ...

    @overload
    @classmethod
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["one"],
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["first"] = ...,
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> Self | None: ...

    @classmethod  # @override — MRO 运行时覆盖 TableBaseMixin.get()，pyright 静态无法识别
    async def get(
            cls,
            session: AsyncSession,
            condition: ColumnElement[bool] | bool | None = None,
            *,
            offset: int | None = None,
            limit: int | None = None,
            fetch_mode: Literal["one", "first", "all"] = "first",
            join: type[TableBaseMixin] | tuple[type[TableBaseMixin], _OnClauseArgument] | None = None,
            options: list[ExecutableOption] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            order_by: list[ColumnElement[Any]] | None = None,
            filter: ColumnElement[bool] | bool | None = None,
            with_for_update: bool = False,
            table_view: TableViewRequest | None = None,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            populate_existing: bool = False,
            no_cache: bool = False,
            created_before_datetime: datetime | None = None,
            created_after_datetime: datetime | None = None,
            updated_before_datetime: datetime | None = None,
            updated_after_datetime: datetime | None = None,
    ) -> list[Self] | Self | None:
        """带缓存的 get() — 拦截 TableBaseMixin.get()，缓存命中时直接返回。

        - no_cache 仅存在于此 Mixin，非缓存模型传入会 TypeError（显式失败）
        - 事务内 commit=False 有未提交变更时自动跳过缓存（读写均跳过）
        - 缓存命中的对象通过 session.merge(load=False) 合并到 identity map
        - load 指定 MANYTOONE 可缓存关系时，尝试多 ID 缓存联合查询（零 SQL）
        - 缓存跳过条件详见模块文档
        """
        skip_cache = (
            no_cache
            or options is not None
            or with_for_update
            or populate_existing
            # TODO: 未来优化 — 支持 JOIN 查询缓存，需追踪 join target 的变更
            or join is not None
        )

        # 事务内有未提交变更 → 跳过缓存（读/写均跳过）
        if not skip_cache and cls._has_pending_invalidation(session):
            skip_cache = True

        # ---- load 查询：多 ID 缓存联合优化 ----
        # MANYTOONE 且目标可缓存时，分别查主模型和关系模型的 ID 缓存组装返回
        if load is not None:
            rel_info: list[tuple[str, type[CachedTableBaseMixin], str]] | None = None
            if not skip_cache and jti_subclasses is None:
                rel_info = cls._analyze_load_relations(load)
                if rel_info is not None:
                    # 检测 simple ID query（多 ID 缓存仅支持单行 ID 等值查询）
                    id_value = cls._extract_id_from_condition(condition)
                    is_simple = (
                        id_value is not None
                        and fetch_mode in ('first', 'one')
                        and offset is None and limit is None
                        and order_by is None and filter is None
                        and table_view is None
                        and created_before_datetime is None and created_after_datetime is None
                        and updated_before_datetime is None and updated_after_datetime is None
                    )
                    if is_simple:
                        cached = await cls._try_load_from_id_caches(session, id_value, rel_info)
                        if cached is not _LOAD_CACHE_MISS:
                            if cls.on_cache_hit is not None:
                                cls.on_cache_hit(cls.__name__)
                            return cached
                        # 实际查过 Redis 且未命中

            # 缓存未命中 / 非 simple query / 不可优化 → DB 查询
            result = await super().get(
                session, condition,
                offset=offset, limit=limit, fetch_mode=fetch_mode,
                join=join, options=options, load=load,
                order_by=order_by, filter=filter,
                with_for_update=with_for_update, table_view=table_view,
                jti_subclasses=jti_subclasses,
                populate_existing=populate_existing,
                created_before_datetime=created_before_datetime,
                created_after_datetime=created_after_datetime,
                updated_before_datetime=updated_before_datetime,
                updated_after_datetime=updated_after_datetime,
            )

            # 写入主模型 + 关系模型的 ID 缓存（仅可优化场景）
            if rel_info is not None and not skip_cache:
                await cls._write_load_result_to_id_caches(result, rel_info)

            return result

        # ---- 非 load 查询 ----
        cache_key: str | None = None
        if not skip_cache:
            # 检测是否为纯 ID 等值查询（无分页/排序/时间过滤等额外参数）
            id_value = cls._extract_id_from_condition(condition)
            is_simple_id_query = (
                id_value is not None
                and fetch_mode in ('first', 'one')
                and offset is None and limit is None
                and order_by is None and filter is None
                and table_view is None
                and created_before_datetime is None and created_after_datetime is None
                and updated_before_datetime is None and updated_after_datetime is None
            )

            if is_simple_id_query:
                cache_key = cls._build_id_cache_key(id_value)
            else:
                # 先获取当前版本号，嵌入 query key（版本递增后旧 key 自然不可达）
                _query_version = await cls._get_query_version()
                cache_key = cls._build_cache_key(
                    condition, fetch_mode, offset, limit,
                    order_by, load, filter, table_view,
                    created_before_datetime, created_after_datetime,
                    updated_before_datetime, updated_after_datetime,
                    version=_query_version,
                )

            raw = await cls._cache_get(cache_key)
            if raw is not None:
                # Phase 1: 反序列化 — 失败说明缓存数据已损坏（schema 变更等）
                try:
                    result = cls._deserialize_result(raw, fetch_mode)
                except Exception as e:
                    # 坏缓存：尝试删除并回源 DB（清理失败不影响正常查询）
                    logger.warning(
                        f"缓存反序列化失败，删除坏 key 并回源 DB: {cache_key} ({type(e).__name__}: {e})"
                    )
                    try:
                        await cls._cache_delete(cache_key)
                    except Exception as del_err:
                        logger.error(f"坏缓存清理失败 key='{cache_key}': {del_err}")
                    # fall through to DB query below
                else:
                    # Phase 2: 合并到 session identity map（保持与 DB 查询一致的语义）
                    if result is not None:
                        if isinstance(result, list):
                            merged_list: list[Self] = []
                            for item in result:
                                make_transient_to_detached(item)
                                merged_list.append(await session.merge(item, load=False))
                            return merged_list
                        else:
                            make_transient_to_detached(result)
                            result = await session.merge(result, load=False)
                    if cls.on_cache_hit is not None:
                        cls.on_cache_hit(cls.__name__)
                    return result

        # Cache miss callback (only when Redis was actually queried, not skip_cache path)
        if cache_key is not None and cls.on_cache_miss is not None:
            cls.on_cache_miss(cls.__name__)

        # DB 查询（通过 MRO 调用 TableBaseMixin.get()）
        # 注意：不转发 no_cache — TableBaseMixin.get() 不接受此参数
        result = await super().get(
            session, condition,
            offset=offset, limit=limit, fetch_mode=fetch_mode,
            join=join, options=options, load=load,
            order_by=order_by, filter=filter,
            with_for_update=with_for_update, table_view=table_view,
            jti_subclasses=jti_subclasses,
            populate_existing=populate_existing,
            created_before_datetime=created_before_datetime,
            created_after_datetime=created_after_datetime,
            updated_before_datetime=updated_before_datetime,
            updated_after_datetime=updated_after_datetime,
        )

        # 写入缓存（仅在非跳过条件下）
        if not skip_cache and cache_key is not None:
            try:
                serialized = cls._serialize_result(result)
                await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
            except Exception as e:
                logger.error(f"缓存序列化/写入失败: {type(e).__name__}: {e}")

        return result

    # ================================================================
    #  延迟提交补偿（session.info 跟踪 + after_commit 事件）
    # ================================================================

    @staticmethod
    def _register_pending_invalidation(
            session: AsyncSession,
            model_type: type,
            instance_id: Any | None = None,
    ) -> None:
        """将模型类型（和可选 instance_id）记录到 session.info，用于 commit 时补偿失效。

        pending 结构: dict[type, set[Any]]
        - key: 模型类型
        - value: 该类型的待失效 instance_id 集合

        哨兵语义：
        - _FULL_MODEL_INVALIDATION: 条件删除，需模型级全量失效（优先级最高，一旦存在永不降级）
        - _QUERY_ONLY_INVALIDATION: add() 场景，只需失效查询缓存
        - 普通 UUID: 行级 ID 失效

        携带 instance_id 使得 after_commit 补偿路径也能做行级失效，
        避免模型级 SCAN+DEL 误删其他行的 ID 缓存。
        """
        pending: dict[type, set[Any]] = session.info.setdefault(_SESSION_PENDING_CACHE_KEY, {})
        ids = pending.setdefault(model_type, set())
        if instance_id is not None:
            ids.add(instance_id)

    # ================================================================
    #  save() 重写 — 写后失效
    # ================================================================

    async def save(
            self,
            session: AsyncSession,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            refresh: bool = True,
            commit: bool = True,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            optimistic_retry_count: int = 0,
    ) -> Self:  # MRO override TableBaseMixin.save()
        """save() 后先失效缓存，再通过 get() 刷新（确保 get() 不会命中旧缓存）。

        流程（commit=True, refresh=True 时）：
        1. super().save(refresh=False) — 只做 commit，不刷新
        2. 同步缓存失效 — 确保旧数据从 Redis 移除
        3. get() 刷新 — 缓存已失效，get() 查 DB 并回填缓存

        新建对象（id 为 None）：注册 _QUERY_ONLY_INVALIDATION（不可能有旧 ID 缓存），
        save 后从 result 获取数据库生成的 id 用于同步路径标记。
        """
        model_type = type(self)
        # 新建对象 id 可能为 None（数据库生成），此时只需失效查询缓存
        instance_id = getattr(self, 'id', None)
        if instance_id is not None:
            self._register_pending_invalidation(session, model_type, instance_id)
        else:
            self._register_pending_invalidation(session, model_type, _QUERY_ONLY_INVALIDATION)

        # commit=True 时快照所有 pending 类型（commit handler 会 pop dict）。
        # 用于 commit 后同步失效 ALL 待失效模型（不仅是 model_type），
        # 消除其他模型（如 adjust_foxcoins 中的 User）被 _compensate 误判的竞态。
        # 常规场景仅 1 个 pending 类型，无额外开销。
        _captured_pending: dict[type, set[Any]] | None = None
        if commit:
            _raw = session.info.get(_SESSION_PENDING_CACHE_KEY)
            if _raw:
                _captured_pending = {k: set(v) for k, v in _raw.items()}

        # refresh=False：跳过 super() 内部的 get()，由本方法在失效后自行刷新
        result = await super().save(
            session,
            refresh=False,
            commit=commit,
            optimistic_retry_count=optimistic_retry_count,
        )

        # super().save(refresh=False) 后 commit 让对象过期，
        # 直接 getattr(result, 'id') 会触发同步懒加载 → MissingGreenlet。
        # 使用 sa_inspect 从 identity map 安全读取 id（不触发 DB 查询）。
        _insp = cast(InstanceState[Any], sa_inspect(result))
        if _insp.identity:
            instance_id = _insp.identity[0]

        # 同步缓存失效（确保后续 get() 不会命中旧数据）
        if _captured_pending:
            # 先同步标记 ALL 待失效类型为 synced（无 await），防止 _compensate 竞态：
            # 逐类型 await Redis 时事件循环可调度 _compensate，若后续类型尚未标记
            # 则被误判为"未走同步失效路径"触发 fallback WARNING。
            _synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
            if isinstance(_synced, dict):
                for _cls, _ids in _captured_pending.items():
                    if isinstance(_cls, type) and issubclass(_cls, CachedTableBaseMixin):
                        _synced.setdefault(_cls, set()).update(_ids)
            # commit 后同步失效 ALL 待失效模型类型（与 cache_aware_commit 逻辑一致）
            for _cls, _ids in _captured_pending.items():
                if isinstance(_cls, type) and issubclass(_cls, CachedTableBaseMixin):
                    await _cls._do_sync_invalidation(session, _ids)
        else:
            pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
            if not pending or model_type not in pending:
                synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
                if isinstance(synced, dict):
                    synced.setdefault(model_type, set()).add(
                        instance_id if instance_id is not None else _QUERY_ONLY_INVALIDATION
                    )
                try:
                    if instance_id is not None:
                        await model_type._invalidate_id_cache(instance_id)
                        await model_type._invalidate_query_caches()
                    else:
                        await model_type._invalidate_query_caches()
                except Exception as e:
                    logger.error(f"save() 同步缓存失效失败 ({model_type.__name__}): {e}")

        # 写穿（write-through）刷新：绕过缓存读 → 查 DB → 主动回填 ID 缓存。
        # 绕过读：避免读到部分失效场景下的旧缓存。
        # 主动回填：保持高缓存命中率，下一次外部 get() 直接命中新鲜数据。
        # 仅 commit=True 时回填：commit=False 的数据可能被 rollback，不能写入缓存。
        if refresh:
            assert instance_id is not None, f"{model_type.__name__} save 后 id 为 None"
            result = await model_type.get(
                session, model_type.id == instance_id,
                load=load, jti_subclasses=jti_subclasses,
                no_cache=True,
            )
            assert result is not None, f"{model_type.__name__} 记录不存在（id={instance_id}）"

            # 主动回填 ID 缓存（commit=True + 无 load 时）
            if commit and load is None:
                try:
                    cache_key = model_type._build_id_cache_key(instance_id)
                    serialized = model_type._serialize_result(result)
                    await model_type._cache_set(cache_key, serialized, model_type.__cache_ttl__)
                except Exception as e:
                    logger.error(f"save() 后缓存回填失败 ({model_type.__name__}): {e}")

        return result  # noqa: RLC007  refresh=False 时调用方显式接受过期对象

    # ================================================================
    #  update() 重写 — 写后失效
    # ================================================================

    async def update(
            self,
            session: AsyncSession,
            other: SQLModelBase,
            extra_data: dict[str, Any] | None = None,
            exclude_unset: bool = True,
            exclude: set[str] | None = None,
            load: QueryableAttribute[Any] | list[QueryableAttribute[Any]] | None = None,
            refresh: bool = True,
            commit: bool = True,
            jti_subclasses: list[type[PolymorphicBaseMixin]] | Literal['all'] | None = None,
            optimistic_retry_count: int = 0,
    ) -> Self:  # MRO override
        """update() 后先失效缓存，再通过 get() 刷新。逻辑同 save()。"""
        model_type = type(self)
        instance_id = getattr(self, 'id', None)
        self._register_pending_invalidation(session, model_type, instance_id)

        # refresh=False：跳过 super() 内部的 get()，由本方法在失效后自行刷新
        result = await super().update(
            session, other,
            extra_data=extra_data,
            exclude_unset=exclude_unset,
            exclude=exclude,
            refresh=False,
            commit=commit,
            optimistic_retry_count=optimistic_retry_count,
        )

        # 先失效缓存
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if not pending or model_type not in pending:
            # 先标记 synced 再执行 Redis 调用（理由同 _do_sync_invalidation docstring）
            synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
            if isinstance(synced, dict):
                synced.setdefault(model_type, set()).add(instance_id)
            try:
                await model_type._invalidate_for_model(instance_id)
            except Exception as e:
                logger.error(f"update() 同步缓存失效失败 ({model_type.__name__}): {e}")

        # 写穿刷新：绕过缓存读 → 查 DB → 主动回填 ID 缓存
        if refresh:
            assert instance_id is not None, f"{model_type.__name__} update 后 id 为 None"
            result = await model_type.get(
                session, model_type.id == instance_id,
                load=load, jti_subclasses=jti_subclasses,
                no_cache=True,
            )
            assert result is not None, f"{model_type.__name__} 记录不存在（id={instance_id}）"

            # 主动回填 ID 缓存（commit=True + 无 load 时）
            if commit and load is None:
                try:
                    cache_key = model_type._build_id_cache_key(instance_id)
                    serialized = model_type._serialize_result(result)
                    await model_type._cache_set(cache_key, serialized, model_type.__cache_ttl__)
                except Exception as e:
                    logger.error(f"update() 后缓存回填失败 ({model_type.__name__}): {e}")

        return result  # noqa: RLC007  refresh=False 时调用方显式接受过期对象

    # ================================================================
    #  delete() 重写 — 删后失效
    # ================================================================

    @classmethod  # MRO override TableBaseMixin.delete()
    async def delete(
            cls,
            session: AsyncSession,
            instances: Self | list[Self] | None = None,
            *,
            condition: ColumnElement[bool] | bool | None = None,
            commit: bool = True,
    ) -> int:
        """delete() 后失效缓存。

        - instances 提供时：行级 DEL 每个实例的 id: 缓存 + 模型级 query: 缓存
        - condition 或无参时：模型级 SCAN+DEL（id: + query:）
        - 级联删除：passive_deletes=False 的 cascade_delete 关系由 SA flush 自动处理，
          persistent_to_deleted 事件注册子模型缓存失效，本方法同步 await 失效
        """
        # 提取实例 ID（在 super().delete() 之前，因为 delete 后对象可能无法访问）
        instance_ids: list[Any] = []
        if instances is not None:
            _instances = instances if isinstance(instances, list) else [instances]
            for inst in _instances:
                _id = getattr(inst, 'id', None)
                if _id is not None:
                    instance_ids.append(_id)

        # 清除上一次残留的级联数据（防御性，正常流程不会残留）
        session.info.pop(_SESSION_CASCADE_DELETED_KEY, None)

        # 注册 pending 时携带 instance_ids，使补偿路径也能做行级失效
        for _id in instance_ids:
            cls._register_pending_invalidation(session, cls, _id)
        if not instance_ids:
            # 条件删除无法提取具体 ID，用哨兵标记需要模型级全量失效
            cls._register_pending_invalidation(session, cls, _FULL_MODEL_INVALIDATION)
        result = await super().delete(session, instances, condition=condition, commit=commit)
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if not pending or cls not in pending:
            # 先标记 synced 再执行 Redis 调用（理由同 _do_sync_invalidation docstring）
            synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
            if isinstance(synced, dict):
                if instance_ids:
                    synced.setdefault(cls, set()).update(instance_ids)
                else:
                    synced.setdefault(cls, set()).add(_FULL_MODEL_INVALIDATION)
            try:
                if instance_ids:
                    for _id in instance_ids:
                        await cls._invalidate_id_cache(_id)
                    await cls._invalidate_query_caches()
                else:
                    await cls._invalidate_for_model()
            except Exception as e:
                logger.error(f"delete() 同步缓存失效失败 ({cls.__name__}): {e}")

        # 级联缓存失效：同步路径
        # persistent_to_deleted 事件在 flush 时将 cascade 子模型写入此 key
        cascade_deleted: dict[type, set[Any]] = session.info.pop(_SESSION_CASCADE_DELETED_KEY, {})
        cascade_deleted.pop(cls, None)  # cls 已由上方同步路径处理
        for target_cls, child_ids in cascade_deleted.items():
            if not child_ids:
                continue
            # 先标记 synced 再执行 Redis 调用（理由同 _do_sync_invalidation docstring）
            synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
            if isinstance(synced, dict):
                synced.setdefault(target_cls, set()).update(child_ids)
            try:
                for child_id in child_ids:
                    await target_cls._invalidate_id_cache(child_id)
                await target_cls._invalidate_query_caches()
            except Exception as e:
                logger.error(f"delete() 级联缓存失效失败 ({target_cls.__name__}): {e}")

        return result

    # ================================================================
    #  add() 重写 — 写后失效
    # ================================================================

    @classmethod  # MRO override TableBaseMixin.add()
    async def add(
            cls,
            session: AsyncSession,
            instances: Self | list[Self],
            refresh: bool = True,
            commit: bool = True,
    ) -> Self | list[Self]:
        """add() 后先失效缓存，再通过 get() 刷新。

        始终失效查询缓存（列表查询可能需要包含新项）。
        显式 ID 的实例同时失效 id: 缓存（防止 ID 复用时留下陈旧缓存）。
        """
        # 收集显式 ID（调用方手工传入的 id，而非 default_factory 自动生成的）
        # model_fields_set 只包含构造时显式传入的字段，default_factory 生成的不在其中
        items = instances if isinstance(instances, list) else [instances]
        explicit_ids: list[Any] = [
            _id for item in items
            if isinstance(item, SQLModelBase) and 'id' in item.model_fields_set
            and (_id := getattr(item, 'id', None)) is not None
        ]

        cls._register_pending_invalidation(session, cls, _QUERY_ONLY_INVALIDATION)
        for _id in explicit_ids:
            cls._register_pending_invalidation(session, cls, _id)

        # refresh=False：跳过 super() 内部的 get()，由本方法在失效后自行刷新
        result = await super().add(session, instances, refresh=False, commit=commit)

        # 先失效缓存
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if not pending or cls not in pending:
            # 先标记 synced 再执行 Redis 调用（理由同 _do_sync_invalidation docstring）
            synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
            if isinstance(synced, dict):
                s = synced.setdefault(cls, set())
                s.add(_QUERY_ONLY_INVALIDATION)
                s.update(explicit_ids)
            try:
                for _id in explicit_ids:
                    await cls._invalidate_id_cache(_id)
                await cls._invalidate_query_caches()
            except Exception as e:
                logger.error(f"add() 同步缓存失效失败 ({cls.__name__}): {e}")

        # 写穿刷新：绕过缓存读 → 查 DB → 主动回填 ID 缓存
        # 仅 commit=True 时回填：commit=False 的数据可能被 rollback，不能写入缓存。
        if refresh:
            if isinstance(result, list):
                refreshed: list[Self] = []
                for inst in result:
                    # commit 后对象过期，用 sa_inspect 安全读取 id
                    _insp = cast(InstanceState[Any], sa_inspect(inst))
                    _inst_id = _insp.identity[0] if _insp.identity else None
                    assert _inst_id is not None, f"{cls.__name__} add 后 id 为 None"
                    r = await cls.get(session, cls.id == _inst_id, no_cache=True)
                    assert r is not None, f"{cls.__name__} 记录不存在（id={_inst_id}）"
                    if commit:
                        try:
                            cache_key = cls._build_id_cache_key(_inst_id)
                            serialized = cls._serialize_result(r)
                            await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
                        except Exception as e:
                            logger.error(f"add() 后缓存回填失败 ({cls.__name__}): {e}")
                    refreshed.append(r)
                return refreshed
            else:
                _insp = cast(InstanceState[Any], sa_inspect(result))
                _result_id = _insp.identity[0] if _insp.identity else None
                assert _result_id is not None, f"{cls.__name__} add 后 id 为 None"
                r = await cls.get(session, cls.id == _result_id, no_cache=True)
                assert r is not None, f"{cls.__name__} 记录不存在（id={_result_id}）"
                if commit:
                    try:
                        cache_key = cls._build_id_cache_key(_result_id)
                        serialized = cls._serialize_result(r)
                        await cls._cache_set(cache_key, serialized, cls.__cache_ttl__)
                    except Exception as e:
                        logger.error(f"add() 后缓存回填失败 ({cls.__name__}): {e}")
                return r

        return result

    # ================================================================
    #  Raw SQL 辅助 — commit + 同步失效
    # ================================================================

    async def _commit_and_invalidate(self, session: AsyncSession) -> None:
        """Raw SQL 方法专用：commit + 同步缓存失效。

        从 session.info pending 快照本类型的待失效 ID（after_commit 会 pop 整个 pending），
        commit 后使用快照 ID 执行同步失效，避免 commit 后访问 self 属性导致 MissingGreenlet。

        使用前必须先调用 _register_pending_invalidation() 注册待失效 ID。

        用法::

            self._register_pending_invalidation(session, type(self), self.id)
            if commit:
                await self._commit_and_invalidate(session)
        """
        model_type = type(self)
        # 快照待失效 ID（after_commit handler 会 pop 整个 pending dict）
        pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        captured_ids: set[Any] = set(pending.get(model_type, ())) if pending else set()

        await session.commit()

        await model_type._do_sync_invalidation(session, captured_ids)

    @classmethod
    async def _sync_invalidate_after_commit(
            cls,
            session: AsyncSession,
            instance_id: Any,
    ) -> None:
        """在其他 CRUD 方法触发 commit 后，同步失效本类型的缓存。

        适用于：本类型注册了 pending，但 commit 由另一个模型的 CRUD 触发的场景。
        例如 adjust_foxcoins() 中，User pending 注册后由 Transaction.save(commit=True) 触发 commit。

        如果 commit 未发生（pending 未被消费），本方法安全地 no-op。

        instance_id 必须在 commit 前提取（commit 后 self 属性过期）。

        用法::

            user_id = self.id  # commit 前提取
            self._register_pending_invalidation(session, type(self), user_id)
            transaction = await Transaction(...).save(session, commit=commit)
            await type(self)._sync_invalidate_after_commit(session, user_id)
        """
        await cls._do_sync_invalidation(session, {instance_id})

    @classmethod
    async def _do_sync_invalidation(
            cls,
            session: AsyncSession,
            captured_ids: set[Any],
    ) -> None:
        """commit 后同步失效的内部实现。

        检查 pending 是否已被 after_commit 消费，如是则执行同步失效并标记 synced。

        synced 标记在 Redis 调用之前写入，消除与 _compensate fire-and-forget task 的竞态：
        after_commit 通过 create_task 调度 _compensate，它可能在本方法的 Redis await 点
        被事件循环调度。若 synced 在 Redis 调用之后才写入，_compensate 会误判本类型
        未走同步路径并触发 fallback WARNING。提前标记消除了这个窗口。

        Redis 失败时的正确性：synced 已标记 → _compensate 不再 fallback。但 _compensate
        的 fallback 同样调用 Redis，在 Redis 不可用时也会失败。TTL 兜底保证最终一致性。

        :param session: 数据库会话
        :param captured_ids: 在 commit 前快照的待失效 ID 集合（可含哨兵值）
        """
        current_pending = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if current_pending and cls in current_pending:
            return  # pending 未被消费（commit 未发生），跳过

        sentinels = {_QUERY_ONLY_INVALIDATION, _FULL_MODEL_INVALIDATION}
        real_ids = captured_ids - sentinels
        needs_full = _FULL_MODEL_INVALIDATION in captured_ids

        # 先标记 synced 再执行 Redis 调用，防止 _compensate 在 await 点抢占后误判
        synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
        if isinstance(synced, dict):
            synced.setdefault(cls, set()).update(captured_ids)

        try:
            if needs_full:
                await cls._invalidate_for_model()
            elif real_ids:
                for _id in real_ids:
                    await cls._invalidate_id_cache(_id)
                await cls._invalidate_query_caches()
            elif _QUERY_ONLY_INVALIDATION in captured_ids:
                await cls._invalidate_query_caches()
        except Exception as e:
            logger.error(f"同步缓存失效失败 ({cls.__name__}): {e}")

    # ================================================================
    #  Session 级 cache-aware commit
    # ================================================================

    @staticmethod
    async def cache_aware_commit(session: AsyncSession) -> None:
        """Session 级 cache-aware commit：commit + 同步缓存失效。

        替代裸 ``session.commit()`` — 当 session 中有 CachedTableBaseMixin 模型的
        commit=False CRUD 操作时，commit 后同步执行版本号递增 + ID 缓存删除，
        消除 after_commit 补偿的 fire-and-forget 窗口。

        无 pending 失效时退化为普通 ``session.commit()``，零额外开销。

        与已有方法的区别：

        - ``_commit_and_invalidate()``：实例方法，只处理 type(self) 一种模型
        - ``_sync_invalidate_after_commit()``：类方法，只处理指定类的单个 ID
        - ``cache_aware_commit()``：session 级，遍历所有待失效模型类型一次性处理

        用法::

            # 多个 commit=False 操作
            await tool_set.save(session, commit=False)
            await ToolSetToolLink.delete(session, condition=..., commit=False)
            # 统一 commit + 同步失效
            await CachedTableBaseMixin.cache_aware_commit(session)
        """
        pending: dict[type, set[Any]] | None = session.info.get(_SESSION_PENDING_CACHE_KEY)
        if not pending:
            await session.commit()
            return

        # 快照 pending（after_commit handler 会 pop 整个 dict）
        captured: dict[type, set[Any]] = {k: set(v) for k, v in pending.items()}

        await session.commit()

        # 先同步标记 ALL 类型为 synced（无 await），防止 _compensate 竞态
        _synced = session.info.get(_SESSION_SYNCED_CACHE_KEY)
        if isinstance(_synced, dict):
            for model_type, ids in captured.items():
                if isinstance(model_type, type) and issubclass(model_type, CachedTableBaseMixin):
                    _synced.setdefault(model_type, set()).update(ids)
        # 同步失效所有已捕获的模型类型
        for model_type, ids in captured.items():
            if not (isinstance(model_type, type) and issubclass(model_type, CachedTableBaseMixin)):
                continue
            await model_type._do_sync_invalidation(session, ids)
