"""Versioned model pricing.

Cost is computed from a *versioned* table and the version is stored on every
`llm_calls` row. When prices change we add a new version rather than editing the old
one, so a run's recorded cost stays reproducible instead of being silently re-priced
by history. Arithmetic is `Decimal` throughout — float error accumulates across
thousands of calls, and this number ends up in a budget comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.llm.base import Usage

#: Bump when any rate below changes. Never edit an existing version's numbers.
PRICE_VERSION = "2026-06-24"

_M = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens."""

    input: Decimal
    output: Decimal
    #: Cache reads are ~0.1x input; cache writes ~1.25x input.
    cache_read: Decimal
    cache_write: Decimal

    @classmethod
    def of(cls, input_rate: str, output_rate: str) -> ModelPrice:
        base = Decimal(input_rate)
        return cls(
            input=base,
            output=Decimal(output_rate),
            cache_read=base / Decimal(10),
            cache_write=base * Decimal("1.25"),
        )


PRICES: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice.of("10.00", "50.00"),
    "claude-mythos-5": ModelPrice.of("10.00", "50.00"),
    "claude-opus-5": ModelPrice.of("5.00", "25.00"),
    "claude-opus-4-8": ModelPrice.of("5.00", "25.00"),
    "claude-opus-4-7": ModelPrice.of("5.00", "25.00"),
    "claude-opus-4-6": ModelPrice.of("5.00", "25.00"),
    "claude-sonnet-5": ModelPrice.of("2.00", "10.00"),
    "claude-sonnet-4-6": ModelPrice.of("3.00", "15.00"),
    "claude-haiku-4-5": ModelPrice.of("1.00", "5.00"),
    #: The fake provider used by tests and the eval harness. Priced so that cost
    #: accounting is exercised end to end without spending anything.
    "fake-model": ModelPrice.of("1.00", "5.00"),
}

#: Charged when a model is not in the table. Deliberately not zero: an unpriced model
#: that costs nothing would slip past every budget guardrail unnoticed. Priced at the
#: most expensive known rate so the failure mode is "run stops early", not "run spends
#: without limit".
UNKNOWN_MODEL_PRICE = ModelPrice.of("10.00", "50.00")


def price_for(model: str) -> tuple[ModelPrice, bool]:
    """Returns (price, is_known)."""
    price = PRICES.get(model)
    return (price, True) if price is not None else (UNKNOWN_MODEL_PRICE, False)


def cost_of(model: str, usage: Usage) -> Decimal:
    """USD cost of one call, quantized to the schema's 6 decimal places."""
    price, _known = price_for(model)
    total = (
        Decimal(usage.input_tokens) * price.input
        + Decimal(usage.output_tokens) * price.output
        + Decimal(usage.cache_read_tokens) * price.cache_read
        + Decimal(usage.cache_write_tokens) * price.cache_write
    ) / _M
    return total.quantize(Decimal("0.000001"))


def estimate_max_cost(model: str, *, input_tokens: int, max_tokens: int) -> Decimal:
    """Worst-case cost of a call that has not happened yet.

    Used by the budget guardrail: we cannot know the true output length in advance,
    so the check assumes the model emits its full `max_tokens`. Pessimistic by
    design — a budget that can be exceeded by a call it approved is not a budget.
    """
    return cost_of(model, Usage(input_tokens=input_tokens, output_tokens=max_tokens))


__all__ = [
    "PRICES",
    "PRICE_VERSION",
    "ModelPrice",
    "cost_of",
    "estimate_max_cost",
    "price_for",
]
