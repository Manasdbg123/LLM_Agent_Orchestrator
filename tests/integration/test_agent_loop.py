"""The ReAct loop end to end, driven by real workers against a real database.

The model is the scripted provider, so what these tests measure is the *engine*:
that a turn becomes a durable step, that tool results reach the next turn, that the
transcript is rebuilt correctly, and that malformed model output is survivable.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.domain.states import RunStatus, StepKind
from tests.helpers import (
    Cluster,
    get_llm_calls,
    get_run,
    get_steps,
    get_tool_calls,
    get_transitions,
    make_agent_run,
    wait_for_run_terminal,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture
def queue(pg_queue):
    return pg_queue


async def test_single_tool_task_runs_plan_act_observe_answer(queue) -> None:
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "calculator", "input": {"expression": "(12.5 * 3) + 2 ** 8"}}]},
            {"text": "The answer is 293.5"},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    assert [str(s.kind) for s in steps] == ["agent_turn", "tool_call", "agent_turn", "finalize"]
    assert [str(s.status) for s in steps] == ["succeeded"] * 4

    # The tool actually ran and its result reached the model.
    tool_calls = await get_tool_calls(run_id)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "calculator"
    assert tool_calls[0].result["content"] == "293.5"
    assert tool_calls[0].effect_status == "committed"

    run = await get_run(run_id)
    assert run.output["answer"] == "The answer is 293.5"
    assert run.output["model_turns"] == 2
    assert run.output["tool_calls"] == 1


async def test_multi_tool_task_chains_calls_sequentially(queue) -> None:
    """Two tool_use blocks in one turn become two sequential steps.

    The run-serialization invariant permits one active step, so siblings are chained
    rather than fanned out — and both results must still reach the model together.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {"name": "calculator", "input": {"expression": "2+2"}},
                    {"name": "calculator", "input": {"expression": "10*10"}},
                ]
            },
            {"text": "4 and 100"},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    assert [str(s.kind) for s in steps] == [
        "agent_turn",
        "tool_call",
        "tool_call",
        "agent_turn",
        "finalize",
    ]
    # Both tool_call steps hang off the same assistant turn.
    assert steps[1].parent_step_id == steps[0].id
    assert steps[2].parent_step_id == steps[0].id

    results = [tc.result["content"] for tc in await get_tool_calls(run_id)]
    assert results == ["4", "100"]


async def test_malformed_tool_arguments_are_corrected_by_the_model(queue) -> None:
    """Invalid arguments must not kill the run.

    The engine hands the validation error back as an errored tool_result; the model
    reads it and calls the tool properly on its next turn. Failing the run here would
    discard everything over a mistake the model fixes in one turn.
    """
    run_id, _ = await make_agent_run(
        [
            # `expression` is required and `expr` is forbidden by the schema.
            {"tools": [{"name": "calculator", "input": {"expr": "2+2"}}]},
            {"tools": [{"name": "calculator", "input": {"expression": "2+2"}}]},
            {"text": "4"},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    tool_calls = await get_tool_calls(run_id)
    assert len(tool_calls) == 2
    assert tool_calls[0].is_error is True
    assert "Invalid arguments" in tool_calls[0].result["content"]
    assert tool_calls[1].is_error is False
    assert tool_calls[1].result["content"] == "4"

    run = await get_run(run_id)
    assert run.output["answer"] == "4"


async def test_an_unknown_tool_is_reported_not_fatal(queue) -> None:
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "teleport", "input": {"destination": "mars"}}]},
            {"text": "no such tool available"},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    tool_step = next(s for s in steps if s.kind == StepKind.TOOL_CALL)
    assert tool_step.output["is_error"] is True
    assert "unknown tool" in tool_step.output["content"]


async def test_a_tool_error_is_visible_to_the_model_as_an_error_result(queue) -> None:
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "calculator", "input": {"expression": "1 / 0"}}]},
            {"text": "that expression is undefined"},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    tool_calls = await get_tool_calls(run_id)
    assert tool_calls[0].is_error is True
    assert "division by zero" in tool_calls[0].result["content"]


async def test_search_then_calculate_multi_step_task(queue) -> None:
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "web_search", "input": {"query": "speed of light vacuum"}}]},
            {"tools": [{"name": "calculator", "input": {"expression": "299792458 * 60"}}]},
            {"text": "17987547480 metres per minute"},
        ],
        queue=queue,
        tools=["web_search", "calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    tool_calls = await get_tool_calls(run_id)
    assert [tc.tool_name for tc in tool_calls] == ["web_search", "calculator"]
    assert "299792458" in tool_calls[0].result["content"]
    assert tool_calls[1].result["content"] == "17987547480"


async def test_cost_rollup_equals_the_sum_of_its_ledger(queue) -> None:
    """The denormalised rollup must match the ledger it summarises.

    `agent_runs.cost_usd` is incremented in the same transaction as each `llm_calls`
    insert precisely so this holds; the assertion is what keeps that honest.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [{"name": "calculator", "input": {"expression": "1+1"}}],
                "input_tokens": 1500,
                "output_tokens": 300,
            },
            {"text": "2", "input_tokens": 2000, "output_tokens": 100},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    run = await get_run(run_id)
    calls = await get_llm_calls(run_id)

    assert len(calls) == 2
    assert run.cost_usd == sum((c.cost_usd for c in calls), Decimal(0))
    assert run.input_tokens == sum(c.input_tokens for c in calls) == 3500
    assert run.output_tokens == sum(c.output_tokens for c in calls) == 400
    assert run.cost_usd > 0
    # Cost is reproducible after a price change because the version is recorded.
    assert all(c.price_version for c in calls)


async def test_thinking_blocks_are_persisted_for_verbatim_replay(queue) -> None:
    """Thinking blocks must survive the round trip through the database.

    They have to be echoed back to the model unchanged on the next turn; storing only
    the text would silently drop them and degrade multi-step reasoning.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "thinking": "I should compute this exactly rather than estimate.",
                "tools": [{"name": "calculator", "input": {"expression": "7*6"}}],
            },
            {"text": "42"},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    first_turn = steps[0]
    block_types = [b["type"] for b in first_turn.output["content"]]
    assert block_types == ["thinking", "tool_use"]
    assert first_turn.output["content"][0]["thinking"].startswith("I should compute")


async def test_a_persistent_model_error_exhausts_retries_then_fails_cleanly(queue) -> None:
    """A model call that always fails must retry, then stop and say why.

    Note the interaction with the fake provider: because it is a pure function of the
    transcript, a *failed* turn adds no assistant message, so every retry replays the
    same failing entry. That is the correct model of a genuinely broken upstream.
    """
    run_id, _ = await make_agent_run([{"raise_retryable": True}], queue=queue, tools=["calculator"])
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=90) is RunStatus.FAILED

    steps = await get_steps(run_id)
    assert steps[0].attempt == steps[0].max_attempts
    assert steps[0].error["code"] == "attempts_exhausted"

    reasons = [t.reason for t in await get_transitions(run_id) if t.step_id == steps[0].id]
    assert reasons.count("retryable_error") == steps[0].max_attempts - 1


async def test_a_transient_tool_failure_is_retried_then_succeeds(queue) -> None:
    """The step is retried in place and the second attempt completes the run.

    This also walks the ambiguous-claim path: the first attempt left an `in_flight`
    ledger row, and the retry resolves it via the tool's SAFE_TO_REPLAY policy rather
    than refusing to continue.
    """
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "calculator", "input": {"expression": "6*7"}}]},
            {"text": "42"},
        ],
        queue=queue,
        tools=["calculator"],
        tool_faults={"calculator": {"kind": "error", "phase": "before_effect", "times": 1}},
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=90) is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    tool_step = next(s for s in steps if s.kind == StepKind.TOOL_CALL)
    assert tool_step.attempt == 2, "expected one failure then a success"

    tool_calls = await get_tool_calls(run_id)
    # One ledger row, not two: the retry reused the same effect key.
    assert len(tool_calls) == 1
    assert tool_calls[0].effect_status == "committed"
    assert tool_calls[0].result["content"] == "42"
    assert (await get_run(run_id)).output["answer"] == "42"


async def test_a_model_refusal_fails_the_run_with_a_clear_reason(queue) -> None:
    run_id, _ = await make_agent_run([{"refuse": True}], queue=queue, tools=["calculator"])
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.FAILED

    run = await get_run(run_id)
    assert run.error["code"] == "model_output_invalid"
    # A refusal is terminal, not retried: the same request would be refused again.
    steps = await get_steps(run_id)
    assert steps[0].attempt == 1
