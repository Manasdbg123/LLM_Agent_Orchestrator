"""Human-in-the-loop gates.

The properties that matter: a gated step must never be executable before a decision,
a decision must be irreversible, and the run must resume (or stop) correctly
afterwards — including after a worker restart, since a run can sit awaiting approval
for far longer than any worker's lifetime.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa

from app.core import approvals
from app.db import session_scope
from app.domain.models import ApprovalRequest
from app.domain.states import RunStatus, StepStatus
from tests.helpers import (
    Cluster,
    count_emails,
    get_run,
    get_steps,
    get_transitions,
    make_agent_run,
    wait_for,
    wait_for_run_terminal,
)

pytestmark = [pytest.mark.integration]

SEND_EMAIL = {
    "tools": [
        {
            "name": "send_email",
            "input": {"to": "a@example.test", "subject": "S", "body": "B"},
        }
    ]
}


@pytest.fixture
def queue(pg_queue):
    return pg_queue


async def _pending_approval(run_id: uuid.UUID) -> ApprovalRequest | None:
    async with session_scope() as session:
        return (
            await session.execute(
                sa.select(ApprovalRequest).where(
                    ApprovalRequest.run_id == run_id,
                    ApprovalRequest.decision == approvals.Decision.PENDING,
                )
            )
        ).scalar_one_or_none()


async def _has_pending(run_id: uuid.UUID) -> bool:
    return await _pending_approval(run_id) is not None


async def _wait_for_gate(run_id: uuid.UUID, *, timeout: float = 45) -> ApprovalRequest:
    assert await wait_for(lambda: _has_pending(run_id), timeout=timeout), (
        "no approval gate appeared"
    )
    approval = await _pending_approval(run_id)
    assert approval is not None
    return approval


async def test_a_risky_tool_parks_the_run_instead_of_running(queue) -> None:
    """`send_email` declares that it needs a human, so nothing is sent yet."""
    run_id, _ = await make_agent_run(
        [SEND_EMAIL, {"text": "sent"}], queue=queue, tools=["send_email"]
    )
    async with Cluster(queue, workers=2):
        approval = await _wait_for_gate(run_id)

        # Give the workers a clear opportunity to (incorrectly) pick the step up.
        await asyncio.sleep(3)

        run = await get_run(run_id)
        assert RunStatus(run.status) is RunStatus.AWAITING_APPROVAL
        steps = await get_steps(run_id)
        gated = next(s for s in steps if s.id == approval.step_id)
        assert StepStatus(gated.status) is StepStatus.AWAITING_APPROVAL
        # Never claimed: a gated step is never published, so no lease was ever taken.
        assert gated.attempt == 0
        assert gated.lease_owner is None

    assert await count_emails() == 0, "the email was sent without approval"
    assert approval.tool_name == "send_email"
    assert "requiring approval" in (approval.reason or "")


async def test_approving_resumes_the_run_and_the_effect_happens(queue) -> None:
    run_id, _ = await make_agent_run(
        [SEND_EMAIL, {"text": "sent"}], queue=queue, tools=["send_email"]
    )
    async with Cluster(queue, workers=1):
        approval = await _wait_for_gate(run_id)

        async with session_scope() as session:
            result = await approvals.decide(
                session,
                approval_id=approval.id,
                decision=approvals.Decision.APPROVED,
                decided_by="operator@example.test",
                decision_reason="looks fine",
            )
        assert result.applied
        assert result.resume_step_id == approval.step_id
        await queue.publish(step_id=result.resume_step_id, run_id=run_id)

        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    assert await count_emails() == 1
    reasons = [t.reason for t in await get_transitions(run_id)]
    assert "approval_required" in reasons
    assert "approved" in reasons


async def test_rejecting_feeds_the_refusal_back_to_the_model(queue) -> None:
    """The default policy steers rather than kills.

    The rejected call becomes an errored tool_result the model reads, and the run
    continues so it can choose a different course.
    """
    run_id, _ = await make_agent_run(
        [SEND_EMAIL, {"text": "understood, I will not send it"}],
        queue=queue,
        tools=["send_email"],
    )
    async with Cluster(queue, workers=1):
        approval = await _wait_for_gate(run_id)
        async with session_scope() as session:
            result = await approvals.decide(
                session,
                approval_id=approval.id,
                decision=approvals.Decision.REJECTED,
                decided_by="operator@example.test",
                decision_reason="wrong recipient",
            )
        assert result.applied
        assert result.detail == "fed_back_to_model"
        await queue.publish(step_id=result.resume_step_id, run_id=run_id)

        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    assert await count_emails() == 0, "a rejected call still performed its effect"

    steps = await get_steps(run_id)
    gated = next(s for s in steps if s.id == approval.step_id)
    assert StepStatus(gated.status) is StepStatus.FAILED
    # The failed step still carries a tool_result payload for the transcript.
    assert gated.output["is_error"] is True
    assert "wrong recipient" in gated.output["content"]

    run = await get_run(run_id)
    assert run.output["answer"] == "understood, I will not send it"


async def test_rejecting_can_be_configured_to_fail_the_run(queue) -> None:
    run_id, _ = await make_agent_run(
        [SEND_EMAIL, {"text": "unreachable"}], queue=queue, tools=["send_email"]
    )
    async with session_scope() as session:
        from app.domain.models import AgentDefinition, AgentRun

        await session.execute(
            sa.update(AgentDefinition)
            .where(
                AgentDefinition.id
                == sa.select(AgentRun.agent_definition_id)
                .where(AgentRun.id == run_id)
                .scalar_subquery()
            )
            .values(on_approval_rejected="fail_run")
        )

    async with Cluster(queue, workers=1):
        approval = await _wait_for_gate(run_id)
        async with session_scope() as session:
            result = await approvals.decide(
                session,
                approval_id=approval.id,
                decision=approvals.Decision.REJECTED,
                decided_by="operator",
                decision_reason="not authorised",
            )
        assert result.detail == "run_failed"
        assert await wait_for_run_terminal(run_id, timeout=45) is RunStatus.FAILED

    run = await get_run(run_id)
    assert run.error["code"] == "approval_rejected"
    assert await count_emails() == 0


async def test_a_decision_cannot_be_reversed(queue) -> None:
    """Idempotent by design: a retried request must not flip an existing decision."""
    run_id, _ = await make_agent_run(
        [SEND_EMAIL, {"text": "sent"}], queue=queue, tools=["send_email"]
    )
    async with Cluster(queue, workers=1):
        approval = await _wait_for_gate(run_id)

        async with session_scope() as session:
            first = await approvals.decide(
                session,
                approval_id=approval.id,
                decision=approvals.Decision.APPROVED,
                decided_by="operator",
            )
        assert first.applied

        async with session_scope() as session:
            second = await approvals.decide(
                session,
                approval_id=approval.id,
                decision=approvals.Decision.REJECTED,
                decided_by="someone-else",
            )
        assert not second.applied
        assert second.detail == "already_decided"
        assert second.decision is approvals.Decision.APPROVED

        await queue.publish(step_id=first.resume_step_id, run_id=run_id)
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED


async def test_an_unanswered_approval_expires_and_fails_the_run(queue) -> None:
    """A forgotten approval must not pin a run open forever."""
    run_id, _ = await make_agent_run(
        [SEND_EMAIL, {"text": "sent"}], queue=queue, tools=["send_email"]
    )
    async with Cluster(queue, workers=1, reaper=False):
        approval = await _wait_for_gate(run_id)

    async with session_scope() as session:
        await session.execute(
            sa.update(ApprovalRequest)
            .where(ApprovalRequest.id == approval.id)
            .values(expires_at=sa.func.now() - sa.text("interval '1 second'"))
        )

    async with session_scope() as session:
        expired = await approvals.expire_stale(session)
    assert expired == [approval.id]

    run = await get_run(run_id)
    assert RunStatus(run.status) is RunStatus.FAILED
    assert run.error["code"] == "approval_expired"
    assert await count_emails() == 0


async def test_approval_is_data_dependent_not_merely_per_tool(queue) -> None:
    """`database_write` gates on the target namespace, not on being a write."""
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "database_write",
                        "input": {"namespace": "notes", "key": "k", "value": "v"},
                    }
                ]
            },
            {
                "tools": [
                    {
                        "name": "database_write",
                        "input": {"namespace": "customers", "key": "c1", "value": "v"},
                    }
                ]
            },
            {"text": "done"},
        ],
        queue=queue,
        tools=["database_write"],
    )
    async with Cluster(queue, workers=1):
        # The first write is routine and runs without a gate; the second is sensitive.
        approval = await _wait_for_gate(run_id, timeout=60)
        assert approval.arguments["namespace"] == "customers"
        assert "sensitive namespace" in (approval.reason or "")

        async with session_scope() as session:
            result = await approvals.decide(
                session,
                approval_id=approval.id,
                decision=approvals.Decision.APPROVED,
                decided_by="operator",
            )
        await queue.publish(step_id=result.resume_step_id, run_id=run_id)
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    # Exactly one gate: the routine write was never stopped.
    async with session_scope() as session:
        all_gates = (
            (
                await session.execute(
                    sa.select(ApprovalRequest).where(ApprovalRequest.run_id == run_id)
                )
            )
            .scalars()
            .all()
        )
    assert len(all_gates) == 1


async def test_a_run_survives_a_full_worker_restart_while_gated(queue) -> None:
    """Approval state lives in Postgres, not in a worker.

    A run can wait days for a human — far longer than any worker process lives — so
    the workers are stopped entirely and a fresh set started after the decision.
    """
    run_id, _ = await make_agent_run(
        [SEND_EMAIL, {"text": "sent"}], queue=queue, tools=["send_email"]
    )
    async with Cluster(queue, workers=1):
        approval = await _wait_for_gate(run_id)

    # Every worker is gone at this point.
    async with session_scope() as session:
        result = await approvals.decide(
            session,
            approval_id=approval.id,
            decision=approvals.Decision.APPROVED,
            decided_by="operator",
        )
    assert result.applied

    async with Cluster(queue, workers=1):
        assert await wait_for_run_terminal(run_id, timeout=60) is RunStatus.SUCCEEDED

    assert await count_emails() == 1
