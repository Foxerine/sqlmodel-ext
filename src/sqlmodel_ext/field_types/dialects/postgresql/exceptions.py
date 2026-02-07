"""
PostgreSQL vector type exceptions.

Custom exception hierarchy for NumpyVector and vector operations.
"""


class VectorError(Exception):
    """Base exception for vector operations."""
    pass


class VectorDimensionError(VectorError):
    """
    Vector dimension mismatch.

    Raised when the provided numpy array dimensions do not match
    the expected dimensions.
    """
    pass


class VectorDecodeError(VectorError):
    """
    Vector decoding error.

    Raised when decoding vector data from base64 or database format fails.
    """
    pass


class VectorDTypeError(VectorError):
    """
    Vector data type error.

    Raised when the numpy array dtype cannot be converted to the target dtype.
    """
    pass
