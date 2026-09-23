"""UUIDv7 generation (RFC 9562) with a standard-library fast path.

``uuid.uuid7`` exists in the standard library from Python 3.14. On older
interpreters this module provides an equivalent implementation so that
``UUIDTableBaseMixin`` can generate time-ordered primary keys everywhere
without a third-party dependency.
"""
import os
import threading
import time
import uuid
from collections.abc import Callable

_UNIX_TS_MS_MASK = (1 << 48) - 1
_RAND_A_MASK = (1 << 12) - 1
_RAND_B_MASK = (1 << 62) - 1

_lock = threading.Lock()
_last_ts_ms = 0
_last_rand_a = 0


def _uuid7_fallback() -> uuid.UUID:
    """Build a UUIDv7 as specified by RFC 9562 section 5.7.

    Layout (most significant bit first)::

        unix_ts_ms (48) | ver = 0b0111 (4) | rand_a (12) | var = 0b10 (2) | rand_b (62)

    Monotonicity within one process (RFC 9562 section 6.2, method 1): when
    the clock has not advanced past the previous value, ``rand_a`` is used as
    a counter seeded from the previous UUID; if the 12-bit counter would
    overflow, the timestamp is advanced by one millisecond. Consecutive calls
    in the same process therefore always yield strictly increasing values.
    ``rand_b`` is always fresh randomness from ``os.urandom``.
    """
    global _last_ts_ms, _last_rand_a
    with _lock:
        ts_ms = time.time_ns() // 1_000_000
        if ts_ms > _last_ts_ms:
            rand_a = int.from_bytes(os.urandom(2), 'big') & _RAND_A_MASK
        else:
            ts_ms = _last_ts_ms
            rand_a = _last_rand_a + 1
            if rand_a > _RAND_A_MASK:
                ts_ms += 1
                rand_a = int.from_bytes(os.urandom(2), 'big') & _RAND_A_MASK
        _last_ts_ms = ts_ms
        _last_rand_a = rand_a
    rand_b = int.from_bytes(os.urandom(8), 'big') & _RAND_B_MASK
    value = (
        ((ts_ms & _UNIX_TS_MS_MASK) << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )
    return uuid.UUID(int=value)


_stdlib_uuid7: Callable[[], uuid.UUID] | None = getattr(uuid, 'uuid7', None)

uuid7: Callable[[], uuid.UUID] = _stdlib_uuid7 if _stdlib_uuid7 is not None else _uuid7_fallback
"""Return a new time-ordered UUIDv7 (``uuid.uuid7`` on Python 3.14+, the RFC 9562 fallback otherwise).

The first 48 bits are the Unix timestamp in milliseconds, so byte order is
creation-time order (within one process; across processes it is only as
ordered as their clocks). The ID therefore reveals its creation time with
millisecond precision."""
