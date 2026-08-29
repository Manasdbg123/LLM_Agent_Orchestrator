"""Observability against a real run: metrics move, traces join up, pages render."""

from __future__ import annotations

import uuid

import httpx
import pytest
import sqlalchemy as sa

from app.api.main import app
from app.db import session_scope
from app.domain.models import StateTransition
from app.domain.states import RunStatus
from app.obs import metrics
from tests.helpers import (
    Cluster,
    get_run,
    get_steps,
    get_transitions,
    make_agent_run,
    wait_for_run_terminal,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture
def queue(pg_queue):
    return pg_queue


def _counter_value(counter, **labels) -> float:
    """Read a counter's current value so a delta can be asserted."""
    metric = counter.labels(**labels) if labels else counter
    return metric._value.get()


async def test_a_run_moves_the_metrics_that_describe_it(queue) -> None:
    before_steps = _counter_value(metrics.STEPS_COMPLETED, kind="agent_turn", status="succeeded")
    before_tools = _counter_value(metrics.TOOL_CALLS, tool="calculator", outcome="ok")
    before_cost = _counter_value(metrics.LLM_COST, model="fake-model")

    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "calculator", "input": {"expression": "2+2"}}]},
            {"text": "4"},
        ],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    assert (
        _counter_value(metrics.STEPS_COMPLETED, kind="agent_turn", status="succeeded")
        == before_steps + 2
    )
    assert _counter_value(metrics.TOOL_CALLS, tool="calculator", outcome="ok") == before_tools + 1
    assert _counter_value(metrics.LLM_COST, model="fake-model") > before_cost


async def test_state_gauges_reflect_the_database(queue) -> None:
    """Gauges are read from Postgres, so they must match what Postgres says."""
    run_id, _ = await make_agent_run(
        [{"tools": [{"name": "calculator", "input": {"expression": "1+1"}}]}, {"text": "2"}],
        queue=queue,
        tools=["calculator"],
    )

    await metrics.refresh_gauges()
    assert metrics.RUNS_IN_PROGRESS.labels(status="pending")._value.get() >= 1

    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    await metrics.refresh_gauges()
    # Everything finished, so nothing should be reported as in flight.
    assert metrics.RUNS_IN_PROGRESS.labels(status="running")._value.get() == 0
    assert metrics.STEPS_RUNNING._value.get() == 0


async def test_every_transition_records_the_trace_that_caused_it(queue) -> None:
    """The join between the audit log and the traces.

    Given a failed run you can go from its transition rows straight to the trace,
    which is the difference between "something went wrong" and "here is where".
    """
    run_id, _ = await make_agent_run(
        [{"tools": [{"name": "calculator", "input": {"expression": "1+1"}}]}, {"text": "2"}],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    transitions = await get_transitions(run_id)
    executed = [t for t in transitions if t.actor.startswith("w")]
    assert executed, "no worker-driven transitions recorded"
    assert all(t.trace_id for t in executed), "a transition was recorded with no trace"
    assert all(len(t.trace_id) == 32 for t in executed)


async def test_a_run_and_its_steps_share_one_trace(queue) -> None:
    run_id, _ = await make_agent_run(
        [{"tools": [{"name": "calculator", "input": {"expression": "1+1"}}]}, {"text": "2"}],
        queue=queue,
        tools=["calculator"],
    )
    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    run = await get_run(run_id)
    assert run.traceparent, "the run has no persisted trace context"
    run_trace_id = run.traceparent.split("-")[1]

    async with session_scope() as session:
        trace_ids = (
            (
                await session.execute(
                    sa.select(StateTransition.trace_id).where(
                        StateTransition.run_id == run_id,
                        StateTransition.trace_id.isnot(None),
                    )
                )
            )
            .scalars()
            .all()
        )

    assert set(trace_ids) == {run_trace_id}, (
        "steps did not join the run's trace; a crash and its recovery would appear "
        "as unrelated traces"
    )


@pytest.mark.chaos
async def test_a_crash_and_its_recovery_appear_in_the_same_trace(queue) -> None:
    """The picture that makes a recovery legible.

    The step is executed by two different workers in two different spans. Both must
    carry the run's trace id, or an operator sees two unexplained failures instead of
    one recovery.
    """
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "calculator", "input": {"expression": "1+1"}}]},
            {"text": "2"},
        ],
        queue=queue,
        tools=["calculator"],
        tool_faults={
            "calculator": {
                "kind": "hang",
                "phase": "after_effect",
                "seconds": 7,
                "suppress_heartbeat": True,
                "times": 1,
            }
        },
    )
    async with Cluster(queue, workers=2):
        assert await wait_for_run_terminal(run_id, timeout=120) is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    tool_step = next(s for s in steps if str(s.kind) == "tool_call")
    assert tool_step.attempt == 2, "the step was not actually re-executed"

    run = await get_run(run_id)
    run_trace_id = run.traceparent.split("-")[1]

    claims = [
        t
        for t in await get_transitions(run_id)
        if t.step_id == tool_step.id and t.to_status == "running"
    ]
    assert len(claims) == 2, "expected the step to be claimed twice"
    # Both executions, in two processes, under one trace.
    assert {c.trace_id for c in claims} == {run_trace_id}


# --- the dashboard -------------------------------------------------------------


@pytest.fixture
async def client(clean_db: None, pg_queue):
    app.state.queue = pg_queue
    from app.api.deps import queue_holder

    queue_holder.set(pg_queue)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_dashboard_pages_render_with_real_data(client, pg_queue) -> None:
    run_id, _ = await make_agent_run(
        [
            {"tools": [{"name": "calculator", "input": {"expression": "6*7"}}]},
            {"text": "42"},
        ],
        queue=pg_queue,
        tools=["calculator"],
    )
    async with Cluster(pg_queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    listing = await client.get("/ui")
    assert listing.status_code == 200
    assert "Recent runs" in listing.text
    assert str(run_id)[:8] in listing.text or "detail" in listing.text

    detail = await client.get(f"/ui/runs/{run_id}")
    assert detail.status_code == 200
    # The four things the run detail page exists to show.
    assert "Step timeline" in detail.text
    assert "Audit log" in detail.text
    assert "agent_turn" in detail.text and "tool_call" in detail.text
    assert "42" in detail.text

    approvals_page = await client.get("/ui/approvals")
    assert approvals_page.status_code == 200
    assert "Nothing is waiting" in approvals_page.text


async def test_dashboard_shows_and_decides_an_approval(client, pg_queue) -> None:
    """The approve button goes through the same service call as the API."""
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "send_email",
                        "input": {"to": "a@example.test", "subject": "S", "body": "B"},
                    }
                ]
            },
            {"text": "sent"},
        ],
        queue=pg_queue,
        tools=["send_email"],
    )

    async with Cluster(pg_queue, workers=1):
        from app.domain.models import ApprovalRequest
        from tests.helpers import wait_for

        async def gated() -> bool:
            async with session_scope() as session:
                return (
                    await session.execute(
                        sa.select(sa.func.count())
                        .select_from(ApprovalRequest)
                        .where(ApprovalRequest.run_id == run_id)
                    )
                ).scalar_one() > 0

        assert await wait_for(gated, timeout=45)

        page = await client.get("/ui/approvals")
        assert "send_email" in page.text
        assert "Approve" in page.text

        async with session_scope() as session:
            approval_id = (
                await session.execute(
                    sa.select(ApprovalRequest.id).where(ApprovalRequest.run_id == run_id)
                )
            ).scalar_one()

        posted = await client.post(
            f"/ui/approvals/{approval_id}/decide",
            data={"decision": "approve", "decided_by": "tester", "reason": "fine"},
        )
        assert posted.status_code in (200, 303)

        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED


async def test_a_missing_run_page_is_a_404(client) -> None:
    assert (await client.get(f"/ui/runs/{uuid.uuid4()}")).status_code == 404
