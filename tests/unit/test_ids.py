from __future__ import annotations

import time

from app.domain.ids import _uuid7_fallback, uuid7


def test_version_and_variant_bits() -> None:
    for value in (uuid7(), _uuid7_fallback()):
        assert value.version == 7
        assert (value.int >> 62) & 0b11 == 0b10  # RFC 9562 variant


def test_ids_sort_by_creation_time() -> None:
    """The property the schema relies on: UUID order approximates time order, which
    keeps index inserts at the right-hand edge instead of scattered."""
    first = uuid7()
    time.sleep(0.005)
    second = uuid7()
    assert str(first) < str(second)


def test_ids_are_unique_under_tight_looping() -> None:
    assert len({uuid7() for _ in range(10_000)}) == 10_000


def test_fallback_embeds_the_current_timestamp() -> None:
    before = time.time_ns() // 1_000_000
    value = _uuid7_fallback()
    after = time.time_ns() // 1_000_000
    embedded = value.int >> 80
    assert before <= embedded <= after
