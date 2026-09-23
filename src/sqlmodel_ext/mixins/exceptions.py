"""Domain exceptions raised by the table mixins.

This module imports nothing from ``sqlmodel_ext`` so that every mixin module
can depend on it without import cycles. The exceptions carry an HTTP-compatible
``status_code`` hint (same convention as ``RecordNotFoundError``); mapping them
to responses is left to the application.
"""
from typing import ClassVar

from sqlalchemy.exc import IntegrityError

FK_DELETE_RESTRICT_FALLBACK_MESSAGE = "Cannot delete: this resource is still referenced by other resources"
"""Generic message used when no message is registered for the violated FK constraint.

Deliberately contains no table / column / constraint names (internal
structure). Endpoints that need "referenced by what, and how many" should
check explicitly *before* deleting and raise their own error; this message is
the fallback for that path (e.g. a lost TOCTOU race)."""


class ResourceReferencedError(Exception):
    """Delete rejected: the row **exists** and is still referenced through a foreign key (``RESTRICT`` / ``NO ACTION``).

    Raised by :meth:`TableBaseMixin.delete` when the database reports a
    foreign-key violation *caused by the DELETE statement itself*. Suggested
    HTTP mapping: **409 Conflict** -- not 404: the row is not missing, it is
    the opposite.

    A foreign-key violation has two directions whose ``sqlstate`` /
    ``constraint_name`` / ``table_name`` are identical on the driver
    exception:

    - INSERT/UPDATE of a child pointing at a missing parent -> "the referenced
      resource does not exist" (404 semantics);
    - DELETE of a parent that is still referenced -> this exception (409).

    ``delete()`` therefore requires **both** conditions: the error was caught
    on the delete path **and** ``IntegrityError.statement`` is a ``DELETE``.
    The first alone is not enough: a commit flushes every pending operation of
    the session, so a bad ``INSERT`` queued earlier by the caller surfaces
    inside ``delete()`` too -- that one is re-raised untouched.

    This class does not validate its own usage; raising it elsewhere (e.g.
    from a generic integrity-error handler) would report "still referenced"
    for what is actually "reference target missing".

    :param friendly_message: user-facing message, from the registry
        (``TableBaseMixin.register_fk_delete_restrict_message``) or
        :data:`FK_DELETE_RESTRICT_FALLBACK_MESSAGE`.
    :param constraint_name: the violated FK constraint name; ``None`` when the
        driver did not provide one.
    :param original_error: the original ``IntegrityError`` for diagnostics;
        do not expose it to clients.
    """
    status_code: ClassVar[int] = 409

    def __init__(
            self,
            friendly_message: str,
            constraint_name: str | None = None,
            original_error: IntegrityError | None = None,
    ) -> None:
        super().__init__(friendly_message)
        self.friendly_message: str = friendly_message
        self.constraint_name: str | None = constraint_name
        self.original_error: IntegrityError | None = original_error


class KeysetCursorError(ValueError):
    """Base class for invalid use of the ``after_id`` keyset cursor (a caller error, not a library failure).

    One common parent lets the application catch every cursor error in one
    place (typically mapped to a 422-style parameter error). ``str(self)`` is
    a user-facing message that is safe to return to the client.
    """
    status_code: ClassVar[int] = 422


class KeysetCursorInvalidError(KeysetCursorError):
    """The ``after_id`` anchor no longer exists or is outside the query's visible range; restart from the first page."""


class KeysetCursorUnsupportedError(KeysetCursorError):
    """This query cannot use the ``after_id`` keyset cursor (e.g. it fixes its own ``order_by`` or uses ``join``); use offset pagination."""
