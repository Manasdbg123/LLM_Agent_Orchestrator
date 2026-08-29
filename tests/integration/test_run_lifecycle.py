"""End-to-end runs through real workers, on both queue backends."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from app.db import session_scope
from app.domain.models import Step
from app.domain.states import RunStatus, StepKind, StepStatus
from tests.helpers import (
    Cluster,
    get_effects,
    get_run,
    get_steps,
    get_transitions,
    make_run,
    wait_for,
    wait_for_run_terminal,
)

pytestmark = [pytest.mark.integration]

PLAN = [{"label": "one"}, {"label": "two"}, {"label": "three"}]


@pytest.fixture(params=["postgres", "redis"])
def queue(request):
    """Run each test on both backends.

    The backend is resolved lazily via `getfixturevalue`: requesting both as direct
    parameters would make an unavailable Redis skip the Postgres runs too, silently
    reducing coverage to nothing. Kept synchronous because `getfixturevalue` on an
    async fixture cannot run from inside an already-running event loop.
    """
    name = "pg_queue" if request.param == "postgres" else "redis_queue"
    return request.getfixturevalue(name)


async def test_a_run_executes_every_step_in_order_then_finalizes(queue) -> None:
    run_id, _ = await make_run(PLAN, queue=queue)
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=45)

    assert status is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    assert [str(s.kind) for s in steps] == ["dummy", "dummy", "dummy", "finalize"]
    assert [str(s.status) for s in steps] == ["succeeded"] * 4
    assert [s.seq for s in steps] == [1, 2, 3, 4]

    # Each step executed exactly once: no accidental replay on the happy path.
    effects = await get_effects(run_id)
    assert {label: e.executions for label, e in effects.items()} == {
        "one": 1,
        "two": 1,
        "three": 1,
    }

    run = await get_run(run_id)
    # steps_used counts every step including finalize; work_steps_completed counts
    # only the steps that produced results, so the two legitimately differ by one.
    assert run.steps_used == 4
    assert run.output["work_steps_completed"] == 3
    assert run.started_at is not None and run.ended_at is not None


async def test_the_audit_log_explains_the_whole_run(queue) -> None:
    run_id, _ = await make_run([{"label": "solo"}], queue=queue)
    async with Cluster(queue, workers=1):
        await wait_for_run_terminal(run_id, timeout=45)

    transitions = await get_transitions(run_id)
    run_level = [(t.from_status, t.to_status) for t in transitions if t.entity == "run"]
    assert run_level == [
        (None, "pending"),
        ("pending", "running"),
        ("running", "succeeded"),
    ]
    # Every transition names who did it and why.
    assert all(t.actor and t.reason for t in transitions)


async def test_multiple_workers_never_execute_a_step_twice(queue) -> None:
    """Horizontal scaling must not duplicate work.

    Four workers all subscribe to the same queue; duplicate delivery is expected. The
    lease is what makes the outcome correct, and the effect counters are the proof.
    """
    runs = [await make_run(PLAN, queue=queue) for _ in range(6)]
    async with Cluster(queue, workers=4):
        for run_id, _ in runs:
            assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    for run_id, _ in runs:
        effects = await get_effects(run_id)
        assert len(effects) == 3
        for label, effect in effects.items():
            assert effect.executions == 1, f"{label} ran {effect.executions} times"

    # And every step attempt is 1: nothing was reclaimed or retried on a clean run.
    async with session_scope() as s:
        attempts = (
            (await s.execute(sa.select(Step.attempt).where(Step.run_id.in_([r for r, _ in runs]))))
            .scalars()
            .all()
        )
    assert set(attempts) == {1}


async def test_a_retryable_failure_is_retried_and_then_succeeds(queue) -> None:
    """The step fails twice with a retryable error, then succeeds on attempt 3."""
    plan = [
        {"label": "flaky", "fault": {"kind": "error", "phase": "before_effect", "times": 2}},
        {"label": "after"},
    ]
    run_id, _ = await make_run(plan, queue=queue)
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=60)

    assert status is RunStatus.SUCCEEDED
    steps = await get_steps(run_id)
    assert steps[0].attempt == 3, "expected two failures then a success"

    # The fault fires before the effect, so the effect happened exactly once even
    # though the step ran three times.
    effects = await get_effects(run_id)
    assert effects["flaky"].executions == 1

    reasons = [t.reason for t in await get_transitions(run_id) if t.step_id == steps[0].id]
    assert reasons.count("retryable_error") == 2


async def test_a_terminal_failure_fails_the_run_without_retrying(queue) -> None:
    plan = [
        {
            "label": "doomed",
            "fault": {
                "kind": "error",
                "phase": "before_effect",
                "times": 99,
                "retryable": False,
            },
        }
    ]
    run_id, _ = await make_run(plan, queue=queue)
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=45)

    assert status is RunStatus.FAILED
    steps = await get_steps(run_id)
    assert steps[0].attempt == 1, "a terminal error must not consume retries"
    assert StepStatus(steps[0].status) is StepStatus.FAILED
    run = await get_run(run_id)
    assert run.error["code"] == "injected_fault"


async def test_retries_are_exhausted_and_reported_as_such(queue) -> None:
    plan = [{"label": "always", "fault": {"kind": "error", "phase": "before_effect", "times": 99}}]
    run_id, _ = await make_run(plan, queue=queue)
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.FAILED
    steps = await get_steps(run_id)
    assert steps[0].attempt == steps[0].max_attempts
    # The recorded cause must say "we ran out of attempts", not "unretryable bug".
    assert steps[0].error["code"] == "attempts_exhausted"
    assert (await get_run(run_id)).error["code"] == "attempts_exhausted"


async def test_max_steps_guardrail_stops_an_over_long_run(queue) -> None:
    run_id, _ = await make_run([{"label": f"s{i}"} for i in range(10)], queue=queue, max_steps=4)
    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=60)

    assert status is RunStatus.FAILED
    run = await get_run(run_id)
    assert run.error["code"] == "max_steps_exceeded"
    assert run.steps_used == 4, "the guardrail must stop it exactly at the budget"


async def test_cancel_stops_a_run_and_leaves_a_clean_state(queue) -> None:
    from app.core.runs import request_cancel

    # A long first step, so cancellation lands while the run is genuinely in flight.
    plan = [{"label": "slow", "sleep_ms": 4000}, {"label": "never"}]
    run_id, _ = await make_run(plan, queue=queue)

    async with Cluster(queue, workers=1):
        await wait_for(lambda: _step_is_running(run_id), timeout=20)
        async with session_scope() as s:
            assert await request_cancel(s, run_id=run_id, actor="test")
        status = await wait_for_run_terminal(run_id, timeout=45)

    assert status is RunStatus.CANCELLED
    steps = await get_steps(run_id)
    # The second step must never have been created.
    assert [str(s.kind) for s in steps] == ["dummy"]
    assert all(s.lease_owner is None for s in steps)


async def _step_is_running(run_id) -> bool:
    async with session_scope() as s:
        count = (
            await s.execute(
                sa.select(sa.func.count())
                .select_from(Step)
                .where(Step.run_id == run_id, Step.status == StepStatus.RUNNING)
            )
        ).scalar_one()
    return bool(count)


async def test_finalize_is_a_real_step_in_the_timeline(queue) -> None:
    run_id, _ = await make_run([{"label": "x"}], queue=queue)
    async with Cluster(queue, workers=1):
        await wait_for_run_terminal(run_id, timeout=45)
    steps = await get_steps(run_id)
    assert StepKind(steps[-1].kind) is StepKind.FINALIZE
    assert steps[-1].started_at is not None and steps[-1].ended_at is not None
