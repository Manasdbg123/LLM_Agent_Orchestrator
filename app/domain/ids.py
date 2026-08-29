"""Time-ordered UUIDs.

UUIDv7 embeds a millisecond timestamp in its high bits, so primary keys sort by
creation time. That keeps B-tree inserts appending to the right-hand edge instead of
scattering across the index like UUIDv4 does, and it makes `ORDER BY id` a valid
cheap proxy for `ORDER BY created_at`.

`uuid.uuid7` exists from CPython 3.14; this module falls back to a spec-compliant
implementation on 3.12/3.13 so the deployed image and a local interpreter agree.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable

# getattr() because `uuid.uuid7` only exists from 3.14; the annotation restores the
# type that getattr erases, so callers still see a UUID rather than Any.
_native_uuid7: Callable[[], uuid.UUID] | None = getattr(uuid, "uuid7", None)


def _uuid7_fallback() -> uuid.UUID:
    # RFC 9562 layout: 48-bit big-endian unix_ts_ms | 4-bit version | 12 bits rand_a
    # | 2-bit variant | 62 bits rand_b.
    unix_ts_ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 62) & 0x0FFF
    rand_b = rand & ((1 << 62) - 1)
    value = (unix_ts_ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | rand_a << 64 | 0b10 << 62 | rand_b
    return uuid.UUID(int=value)


def uuid7() -> uuid.UUID:
    if _native_uuid7 is not None:
        return _native_uuid7()
    return _uuid7_fallback()


def new_id() -> uuid.UUID:
    return uuid7()


__all__ = ["new_id", "uuid7"]
