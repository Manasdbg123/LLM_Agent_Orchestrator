from __future__ import annotations

from decimal import Decimal

import pytest

from app.llm.base import Usage
from app.llm.fake import FakeProvider, Turn, encode_script
from app.llm.pricing import PRICES, cost_of, estimate_max_cost, price_for


def test_cost_uses_published_rates() -> None:
    # Opus 5: $5/1M input, $25/1M output.
    cost = cost_of("claude-opus-5", Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == Decimal("30.000000")


def test_cost_is_decimal_not_float() -> None:
    # Float accumulation across thousands of calls drifts, and this number is
    # compared against a budget.
    cost = cost_of("claude-opus-5", Usage(input_tokens=3, output_tokens=7))
    assert isinstance(cost, Decimal)
    assert cost == Decimal("0.000190")


def test_cache_reads_are_cheaper_than_fresh_input() -> None:
    fresh = cost_of("claude-opus-5", Usage(input_tokens=100_000))
    cached = cost_of("claude-opus-5", Usage(cache_read_tokens=100_000))
    assert cached == fresh / 10


def test_cache_writes_cost_more_than_fresh_input() -> None:
    fresh = cost_of("claude-opus-5", Usage(input_tokens=100_000))
    written = cost_of("claude-opus-5", Usage(cache_write_tokens=100_000))
    assert written == fresh * Decimal("1.25")


def test_unknown_models_are_priced_high_not_free() -> None:
    """A model missing from the table must not be free.

    Zero-cost would let an unpriced model slip past every budget guardrail. Pricing
    it at the most expensive known rate makes the failure "the run stops early"
    rather than "the run spends without limit".
    """
    price, known = price_for("some-model-released-next-year")
    assert not known
    assert price.input > PRICES["claude-opus-5"].input
    assert cost_of("some-model-released-next-year", Usage(input_tokens=1000)) > 0


def test_budget_estimate_assumes_the_worst_case_output() -> None:
    estimate = estimate_max_cost("claude-opus-5", input_tokens=1000, max_tokens=8000)
    actual = cost_of("claude-opus-5", Usage(input_tokens=1000, output_tokens=8000))
    assert estimate == actual


async def test_fake_provider_is_a_pure_function_of_the_transcript() -> None:
    """The property that makes recovery tests meaningful.

    A worker that dies and is replaced rebuilds the transcript and calls the provider
    again. If the fake tracked its own call count, the replacement would get a
    *different* turn and the test would be exercising the fake, not the engine.
    """
    provider = FakeProvider()
    script = encode_script(
        [{"tools": [{"name": "calculator", "input": {"expression": "1+1"}}]}, {"text": "2"}]
    )
    messages = [{"role": "user", "content": f"task {script}"}]

    first = await provider.complete(
        model="fake-model", system="", messages=messages, tools=[], max_tokens=100
    )
    repeat = await provider.complete(
        model="fake-model", system="", messages=messages, tools=[], max_tokens=100
    )

    assert first.tool_uses[0].name == "calculator"
    # Identical inputs, identical output — including the tool_use id, which the
    # transcript needs in order to pair results with calls after a recovery.
    assert [t.id for t in first.tool_uses] == [t.id for t in repeat.tool_uses]
    assert first.content == repeat.content


async def test_fake_provider_advances_with_assistant_turns() -> None:
    provider = FakeProvider()
    script = encode_script([{"tools": [{"name": "calculator", "input": {}}]}, {"text": "done"}])
    messages: list[dict] = [{"role": "user", "content": f"task {script}"}]

    first = await provider.complete(
        model="fake-model", system="", messages=messages, tools=[], max_tokens=100
    )
    assert first.stop_reason == "tool_use"

    messages.append({"role": "assistant", "content": first.content})
    messages.append({"role": "user", "content": [{"type": "tool_result", "content": "2"}]})
    second = await provider.complete(
        model="fake-model", system="", messages=messages, tools=[], max_tokens=100
    )
    assert second.stop_reason == "end_turn"
    assert second.text == "done"


async def test_fake_provider_can_inject_classified_failures() -> None:
    from app.domain.errors import RetryableError

    provider = FakeProvider(default_script=[Turn(raise_retryable=True)])
    with pytest.raises(RetryableError):
        await provider.complete(
            model="fake-model", system="", messages=[], tools=[], max_tokens=100
        )
