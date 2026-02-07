"""Exceptions for sqlmodel_ext."""


class RecordNotFoundError(Exception):
    """
    Raised when a database record is not found.

    Attributes:
        status_code: HTTP-compatible status code (404)
        detail: Human-readable error message
    """
    status_code: int = 404

    def __init__(self, detail: str = "Not found"):
        self.detail = detail
        super().__init__(detail)
