"""
Generic resource quota mixin.

Provides atomic quota-slot acquisition for "resource models whose count per
owner has an upper limit". It depends on nothing application-specific (no
user model, no HTTP framework) -- only on SQLAlchemy/SQLModel's
``AsyncSession``. Bind it to your owner model in a small subclass.

Subclass contract (three classmethods must be implemented):

- ``_lock_owner(session, owner_id, *, with_for_update=True) -> owner | None``:
  load the owner row (``SELECT ... FOR UPDATE`` when ``with_for_update``) and
  return it (used by ``_quota_max``); the owner id type is up to you.
- ``_quota_condition(owner_id) -> ColumnElement[bool]``: SQL filter selecting
  this owner's rows of this model.
- ``_quota_max(owner) -> int``: the maximum number of instances allowed.

The subclass must also inherit a table base providing
``count(session, condition) -> int`` (``TableBaseMixin`` /
``UUIDTableBaseMixin`` / ``CachedTableBaseMixin``), unless it overrides
``_count_owner_resources``.

Exceptions:

- :class:`QuotaExceededError`: the quota is full.
- :class:`QuotaOwnerNotFoundError`: ``_lock_owner`` returned ``None``.
- :class:`CallerDidNotCommitError`: the ``async with`` block exited normally
  but the caller did not commit (dangling-lock guard).

Map the first two to 4xx responses; the last one is a programming error.

Usage::

    async with Conversation.acquire_quota_lock(session, owner_id):
        instance = await instance.save(session, commit=False)
        # ... related inserts ...
        await instance.save(session)  # commits -> releases the owner lock
"""
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from sqlalchemy import ColumnElement, event as sa_event
from sqlmodel.ext.asyncio.session import AsyncSession

_QuotaIdempotentT = TypeVar('_QuotaIdempotentT')
"""Type of the existing resource returned by ``acquire_quota_lock``'s ``idempotent_check`` callback."""


class QuotaExceededError(ValueError):
    """
    The resource quota is exhausted.

    Attributes:
        model_name: Name of the model class the quota applies to
        max_allowed: Maximum number of instances allowed
        current_count: Current number of instances (>= max_allowed, or too
            close to it for the requested count)
    """
    status_code: int = 400

    def __init__(self, model_name: str, max_allowed: int, current_count: int):
        self.model_name: str = model_name
        self.max_allowed: int = max_allowed
        self.current_count: int = current_count
        super().__init__(
            f"{model_name} quota exceeded: {current_count}/{max_allowed}"
        )


class QuotaOwnerNotFoundError(LookupError):
    """
    The quota owner row could not be found (``_lock_owner`` returned ``None``).

    Attributes:
        owner_id: The owner identifier that was looked up
    """
    status_code: int = 404

    def __init__(self, owner_id: Any):
        self.owner_id: Any = owner_id
        super().__init__(f"quota owner not found: {owner_id}")


class CallerDidNotCommitError(RuntimeError):
    """
    The ``async with acquire_quota_lock()`` block exited normally but the caller did not commit.

    The owner's ``FOR UPDATE`` lock is released with the transaction; a still
    open transaction means the lock is still held. This guard turns "forgot
    to commit, lock held for a long time" from a latent bug into an immediate
    error. It is a programming error in the caller.
    """


class ResourceQuotaMixin:
    """
    Generic per-owner resource quota mixin.

    Subclasses declare the quota rules through ``_lock_owner`` /
    ``_quota_condition`` / ``_quota_max``; callers acquire slots with
    ``async with cls.acquire_quota_lock(session, owner_id):``. See the module
    docstring.
    """

    @classmethod
    async def _lock_owner(
            cls,
            session: AsyncSession,
            owner_id: Any,
            *,
            with_for_update: bool = True,
    ) -> Any:
        """
        Load the owner row and return the owner instance; ``None`` if it does not exist.

        ``with_for_update=True`` (default, ``acquire_quota_lock`` path) runs
        ``SELECT ... FOR UPDATE`` so "check quota + INSERT" is serialized per
        owner within one transaction -- concurrent requests cannot overshoot.

        ``with_for_update=False`` (``preflight_quota`` path) is a lock-free
        (cache-friendly) read for the cheap pre-check.

        Subclasses must override this and forward ``with_for_update`` to the
        underlying ``Model.get``.
        """
        del session, owner_id, with_for_update  # the signature is the override contract; this base only raises
        raise NotImplementedError(
            f"{cls.__name__} must implement _lock_owner (ResourceQuotaMixin contract)"
        )

    @classmethod
    def _quota_condition(cls, owner_id: Any) -> 'ColumnElement[bool]':
        """
        Return the SQL filter selecting this owner's rows of this model.

        Subclasses must override it, e.g. ``col(cls.owner_id) == owner_id``,
        or a subquery through a link table.
        """
        del owner_id  # the signature is the override contract; this base only raises
        raise NotImplementedError(
            f"{cls.__name__} must implement _quota_condition (ResourceQuotaMixin contract)"
        )

    @classmethod
    def _quota_max(cls, owner: Any) -> int:
        """
        Return the maximum number of instances the owner is allowed.

        Subclasses must override it; typically ``owner.max_<resource>``.
        """
        del owner  # the signature is the override contract; this base only raises
        raise NotImplementedError(
            f"{cls.__name__} must implement _quota_max (ResourceQuotaMixin contract)"
        )

    @classmethod
    async def _count_owner_resources(
            cls,
            session: AsyncSession,
            owner_id: Any,
            owner: Any,
    ) -> int:
        """
        Return how many resources the owner currently uses (for the quota check).

        Default: ``cls.count(session, _quota_condition(owner_id))`` -- a
        single-table count.

        Override for non-single-table semantics (e.g. an aggregate over
        several tables). An override may ignore ``_quota_condition``.

        :param session: Database session
        :param owner_id: The ``owner_id`` passed to ``acquire_quota_lock``
        :param owner: The instance returned by ``_lock_owner`` (unused by the default)
        """
        del owner  # unused by the default implementation, available to overrides
        condition = cls._quota_condition(owner_id)
        # cls.count() is provided by the table base the subclass must also inherit.
        return await cls.count(session, condition)  # pyright: ignore[reportAttributeAccessIssue]

    @classmethod
    async def preflight_quota(
            cls,
            session: AsyncSession,
            owner_id: Any,
            count: int = 1,
    ) -> None:
        """**Lock-free** quota pre-check (cheap, best-effort, race-prone soft gate).

        Call it **before** an expensive step (e.g. a paid external API call)
        so that owners already over quota are rejected before any cost is
        incurred. The atomic guarantee still comes from the subsequent
        ``async with cls.acquire_quota_lock(...)``: a slot may be taken by a
        concurrent request in the short window between both calls, in which
        case the lock phase raises.

        Preferring a pre-check over holding the ``FOR UPDATE`` lock across the
        expensive step avoids serializing all of an owner's concurrent
        requests for the duration of that step.

        Reuses ``_lock_owner(with_for_update=False)``, so one override serves
        both paths.

        :raises ValueError: ``count < 1``
        :raises QuotaOwnerNotFoundError: the owner does not exist
        :raises QuotaExceededError: ``current + count > max_allowed``
        """
        if count < 1:
            raise ValueError(f"preflight_quota count must be >= 1, got {count}")
        owner = await cls._lock_owner(session, owner_id, with_for_update=False)
        if owner is None:
            raise QuotaOwnerNotFoundError(owner_id)
        max_allowed = cls._quota_max(owner)
        current_count = await cls._count_owner_resources(session, owner_id, owner)
        if current_count + count > max_allowed:
            raise QuotaExceededError(cls.__name__, max_allowed, current_count)

    @classmethod
    @asynccontextmanager
    async def acquire_quota_lock(
            cls,
            session: AsyncSession,
            owner_id: Any,
            count: int = 1,
            defer_commit: bool = False,
            idempotent_check: 'Callable[[], Awaitable[_QuotaIdempotentT | None]] | None' = None,
    ) -> AsyncGenerator['_QuotaIdempotentT | None', None]:
        """
        Atomically acquire ``count`` quota slots (default 1) inside the caller's transaction.

        Use it as an ``async with`` block right before the INSERT: do all
        pre-work (validation, lookups, in-memory work) **before** entering, so
        the lock window only covers ``acquire -> INSERT -> commit``.

        Strong consistency: ``_lock_owner`` holds the owner lock in the
        caller's transaction, serializing concurrent requests of one owner;
        ``current + count <= max`` and the protected INSERT run atomically in
        the same transaction -- the quota cannot be exceeded.

        Lifecycle: the lock is released by the caller's commit/rollback. The
        block itself does not commit -- the caller must (typically the final
        ``model.save(session)`` with the default ``commit=True``).

        Fail-loud guard (``defer_commit=False``, default): if the block exits
        **normally** without a commit having happened,
        :class:`CallerDidNotCommitError` is raised. On exceptional exit the
        exception propagates and the guard does not run.

        :param count: number of slots to reserve (``>= 1``); creating several
            resources at once checks ``current + count <= max`` once.
        :param defer_commit: ``True`` disables the commit guard -- for blocks
            that ``save(commit=False)`` and let a larger caller transaction
            commit later. The lock is still released by that final
            commit/rollback.
        :param idempotent_check: optional "re-check under lock" callback, run
            after the owner lock is taken and **before** counting. A non-None
            result skips the quota check and the INSERT and is yielded to the
            caller for an idempotent return -- the race guard for concurrent
            get-or-create of one unique resource (the later request finds the
            earlier one's row instead of a false "quota exceeded" / unique
            violation). The callback must read the database truth (bypass
            caches, e.g. ``no_cache=True``). In this branch no commit guard is
            registered (nothing to commit), but the owner lock is held until
            the caller's transaction ends -- return promptly.

        :raises ValueError: ``count < 1``
        :raises QuotaOwnerNotFoundError: ``_lock_owner`` returned ``None``
        :raises QuotaExceededError: ``current + count > max_allowed``
        :raises CallerDidNotCommitError: ``defer_commit=False`` and the block
            exited normally without a commit
        """
        if count < 1:
            raise ValueError(f"acquire_quota_lock count must be >= 1, got {count}")
        owner = await cls._lock_owner(session, owner_id)
        if owner is None:
            raise QuotaOwnerNotFoundError(owner_id)

        if idempotent_check is not None:
            existing = await idempotent_check()
            if existing is not None:
                yield existing
                return

        max_allowed = cls._quota_max(owner)
        current_count = await cls._count_owner_resources(session, owner_id, owner)
        if current_count + count > max_allowed:
            raise QuotaExceededError(cls.__name__, max_allowed, current_count)

        if defer_commit:
            yield None
            return

        # Side-effect-free commit probe: a passive ``after_commit`` listener
        # flips a flag. ``session.in_transaction()`` cannot be used -- the
        # refresh after the caller's ``save()`` opens a new transaction.
        committed = [False]

        def _on_commit(_sync_session: object) -> None:
            committed[0] = True

        sync_session = session.sync_session
        sa_event.listen(sync_session, 'after_commit', _on_commit)
        try:
            yield None
        finally:
            # Always remove the listener so it cannot leak into the session's later life.
            sa_event.remove(sync_session, 'after_commit', _on_commit)

        # Only reached on the happy path (exceptions propagate from the yield).
        if not committed[0]:
            raise CallerDidNotCommitError(
                f"{cls.__name__}.acquire_quota_lock: caller did not commit before exiting"
                + " `async with` block. The owner lock is still held, defeating the purpose of"
                + " scoping the lock window. Ensure the protected work ends with a commit"
                + " (e.g., `await model.save(session)` with default commit=True), or call"
                + " `session.rollback()` if aborting."
            )
