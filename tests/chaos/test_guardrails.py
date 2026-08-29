"""Tests written to break the guardrails.

Every one of these drives the engine into a limit deliberately and asserts that it
stops *at* the limit with a specific reason — not that it stops eventually, and not
that it merely logs something. A guardrail that fails open is worse than no guardrail,
because it is trusted.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
import sqlalchemy as sa

from app.db import session_scope
from app.domain.models import AgentRun, LLMCall, Step
from app.domain.states import RunStatus, StepKind
from tests.helpers import (
    Cluster,
    get_llm_calls,
    get_run,
    get_steps,
    get_tool_calls,
    make_agent_run,
    wait_for,
    wait_for_run_terminal,
)

pytestmark = [pytest.mark.chaos, pytest.mark.integration]


@pytest.fixture
def queue(pg_queue):
    return pg_queue


def _loop_forever(n: int = 40) -> list[dict]:
    """A model that never stops calling tools."""
    return [{"tools": [{"name": "calculator", "input": {"expression": "1+1"}}]}] * n


async def test_max_steps_stops_the_run_exactly_at_its_budget(queue) -> None:
    run_id, _ = await make_agent_run(
        _loop_forever(), queue=queue, tools=["calculator"], max_steps=6
    )
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.FAILED
    run = await get_run(run_id)
    assert run.error["code"] == "max_steps_exceeded"
    # Exactly at the cap, not one over: the check happens before a step is created.
    assert run.steps_used == 6
    assert len(await get_steps(run_id)) == 6


async def test_the_last_turn_before_the_cap_is_forced_to_answer(queue) -> None:
    """One step from the cap the model is called with no tools.

    Otherwise a run ends on a tool call it has no budget left to execute, which is
    a failure the user sees as "it just stopped" rather than an answer.
    """
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "calculator", "input": {"expression": "2+2"}}]},
            {"tools": [{"name": "calculator", "input": {"expression": "3+3"}}]},
            {"text": "I ran out of room but the answer is 4 and 6"},
        ],
        queue=queue,
        tools=["calculator"],
        max_steps=6,
    )
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.SUCCEEDED
    run = await get_run(run_id)
    assert "answer" in run.output


async def test_the_budget_stops_a_run_that_would_overspend(queue) -> None:
    """A cheap cap plus expensive turns must halt the run.

    The check runs before each model call against the recorded spend, so the run
    stops as soon as the ledger says the budget is gone.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [{"name": "calculator", "input": {"expression": "1+1"}}],
                "input_tokens": 200_000,
                "output_tokens": 50_000,
            }
        ]
        * 20,
        queue=queue,
        tools=["calculator"],
    )
    # fake-model is $1/1M in, $5/1M out -> ~$0.45 per turn. A $1 cap allows a couple.
    async with session_scope() as session:
        await session.execute(
            sa.update(AgentRun).where(AgentRun.id == run_id).values(max_cost_usd=Decimal("1.0"))
        )

    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=120)

    assert status is RunStatus.FAILED
    run = await get_run(run_id)
    assert run.error["code"] == "budget_exceeded"
    assert run.cost_usd >= run.max_cost_usd

    # The documented bound: overshoot is at most one call's worth, because the check
    # is before the call rather than during it.
    calls = await get_llm_calls(run_id)
    assert run.cost_usd - max(c.cost_usd for c in calls) < run.max_cost_usd


async def test_the_budget_check_reads_the_ledger_not_a_cached_value(queue) -> None:
    """Spend recorded by a previous step must be visible to the next one.

    The run object a handler holds was loaded when its step was claimed, so a budget
    check reading that copy would always see a stale, lower number.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [{"name": "calculator", "input": {"expression": "1+1"}}],
                "input_tokens": 900_000,
                "output_tokens": 100_000,
            },
            {"text": "should never be reached", "input_tokens": 10, "output_tokens": 10},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with session_scope() as session:
        await session.execute(
            sa.update(AgentRun).where(AgentRun.id == run_id).values(max_cost_usd=Decimal("0.5"))
        )

    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.FAILED
    assert (await get_run(run_id)).error["code"] == "budget_exceeded"
    # The first call was allowed (budget was intact); the second was refused.
    assert len(await get_llm_calls(run_id)) == 1


async def test_a_model_looping_on_invalid_tool_calls_is_stopped(queue) -> None:
    """Self-correction that never converges is just an expensive path to max_steps."""
    run_id, _ = await make_agent_run(
        # Every call omits the required `expression` field.
        [{"tools": [{"name": "calculator", "input": {"wrong": "x"}}]}] * 20,
        queue=queue,
        tools=["calculator"],
        max_steps=40,
    )
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=120)

    assert status is RunStatus.FAILED
    run = await get_run(run_id)
    assert run.error["code"] == "model_output_invalid"

    # Stopped by the invalid-call cap, well before max_steps.
    assert run.steps_used < 40
    tool_calls = await get_tool_calls(run_id)
    assert len(tool_calls) == 3
    assert all(tc.is_error for tc in tool_calls)


async def test_recovering_from_invalid_calls_resets_the_counter(queue) -> None:
    """The cap must count *consecutive* failures, not lifetime ones.

    A model that stumbles twice, recovers, then stumbles twice again is working as
    intended; failing it would punish exactly the self-correction the design wants.
    """
    bad = {"tools": [{"name": "calculator", "input": {"wrong": "x"}}]}
    good = {"tools": [{"name": "calculator", "input": {"expression": "1+1"}}]}
    run_id, _ = await make_agent_run(
        [bad, bad, good, bad, bad, good, {"text": "finally"}],
        queue=queue,
        tools=["calculator"],
        max_steps=30,
    )
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=120)

    assert status is RunStatus.SUCCEEDED
    assert (await get_run(run_id)).output["answer"] == "finally"


async def test_a_deadline_fails_a_run_that_is_taking_too_long(queue) -> None:
    run_id, _ = await make_agent_run(
        _loop_forever(), queue=queue, tools=["calculator"], timeout_seconds=3, max_steps=100
    )
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.FAILED
    assert (await get_run(run_id)).error["code"] == "deadline_exceeded"


async def test_a_per_tool_retry_budget_overrides_the_default(queue) -> None:
    """`send_email` declares fewer attempts than the engine default.

    Retrying a send harder makes the blast radius bigger rather than smaller, so the
    tool gets to say so — and the step created to run it must inherit that number
    rather than the engine-wide default.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "send_email",
                        "input": {"to": "x@example.test", "subject": "S", "body": "B"},
                    }
                ]
            },
            {"text": "unreachable"},
        ],
        queue=queue,
        tools=["send_email"],
    )

    async with Cluster(queue, workers=1):
        # The tool step only exists once the model has asked for it, so wait for it
        # rather than for the run, which parks on the approval gate.
        assert await wait_for(lambda: _tool_step_exists(run_id), timeout=45)

    steps = await get_steps(run_id)
    turn = next(s for s in steps if s.kind == StepKind.AGENT_TURN)
    tool_step = next(s for s in steps if s.kind == StepKind.TOOL_CALL)

    assert turn.max_attempts == 3, "a model turn keeps the engine default"
    assert tool_step.max_attempts == 2, "the tool's declared budget must reach its step"


async def _tool_step_exists(run_id) -> bool:
    async with session_scope() as session:
        return (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(Step)
                .where(Step.run_id == run_id, Step.kind == StepKind.TOOL_CALL)
            )
        ).scalar_one() > 0


async def test_guardrail_failures_are_all_distinguishable(queue) -> None:
    """Each limit reports its own code.

    "The run failed" is not actionable; "it ran out of budget" versus "it looped"
    versus "it ran out of time" are three different bugs to go and fix.
    """
    codes = set()

    steps_run, _ = await make_agent_run(
        _loop_forever(), queue=queue, tools=["calculator"], max_steps=4
    )
    async with Cluster(queue, workers=1):
        await wait_for_run_terminal(steps_run, timeout=90)
    codes.add((await get_run(steps_run)).error["code"])

    invalid_run, _ = await make_agent_run(
        [{"tools": [{"name": "calculator", "input": {"wrong": "x"}}]}] * 20,
        queue=queue,
        tools=["calculator"],
        max_steps=40,
    )
    async with Cluster(queue, workers=1):
        await wait_for_run_terminal(invalid_run, timeout=120)
    codes.add((await get_run(invalid_run)).error["code"])

    assert codes == {"max_steps_exceeded", "model_output_invalid"}


async def test_cost_accounting_survives_a_guardrail_failure(queue) -> None:
    """A run that fails on a guardrail must still report what it spent."""
    run_id, _ = await make_agent_run(
        _loop_forever(), queue=queue, tools=["calculator"], max_steps=5
    )
    async with Cluster(queue, workers=1):
        await wait_for_run_terminal(run_id, timeout=90)

    run = await get_run(run_id)
    async with session_scope() as session:
        ledger_total = (
            await session.execute(
                sa.select(sa.func.coalesce(sa.func.sum(LLMCall.cost_usd), 0)).where(
                    LLMCall.run_id == run_id
                )
            )
        ).scalar_one()
    assert run.cost_usd == ledger_total
    assert run.cost_usd > 0
