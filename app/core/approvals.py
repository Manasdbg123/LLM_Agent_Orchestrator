"""Human-in-the-loop approval gates.

A gated step is created directly in `awaiting_approval` and is **never enqueued**, so
there is no window in which a worker could pick it up before a human has seen it. The
run moves to `awaiting_approval` too, which makes the approval queue a single indexed
query rather than a join across steps.

The gate is *data-dependent*: each tool decides from the actual arguments whether a
particular call needs a human (`Tool.approval_reason`). `send_email` always does;
`database_write` only when the target namespace is sensitive. A flat per-tool flag
would force a choice between gating every write and gating none.

Rejection feeds back to the model rather than killing the run
--------------------------------------------------------------
`on_approval_rejected` defaults to `feed_back_to_model`: the rejected call becomes a
`tool_result` with `is_error: true` carrying the approver's reason, and the run
continues so the model can choose a different course. That makes the gate a steering
mechanism rather than a kill switch, and nothing is lost — a run can still be ended
outright with `POST /v1/runs/{id}/cancel`. Setting `fail_run` restores the blunt
behaviour for agents where any rejection should be terminal.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.transitions import transition_run, transition_step
from app.domain.errors import ErrorCode
from app.domain.models import AgentDefinition, AgentRun, ApprovalRequest, Step
from app.domain.states import RunStatus, StepKind, StepStatus, TransitionReason
from app.engine.messages import TOOL_CONTENT, TOOL_IS_ERROR, TOOL_USE_ID
from app.obs import metrics
from app.obs.logging import get_logger

log = get_logger("approvals")


class Decision(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RejectionPolicy(StrEnum):
    FEED_BACK_TO_MODEL = "feed_back_to_model"
    FAIL_RUN = "fail_run"


@dataclass(slots=True)
class DecisionResult:
    applied: bool
    decision: Decision
    #: Set when the run should continue: the step to enqueue next.
    resume_step_id: uuid.UUID | None = None
    detail: str | None = None


async def create_gate(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    step_id: uuid.UUID,
    tool_name: str,
    arguments: dict[str, Any],
    reason: str,
    actor: str,
    ttl_seconds: int | None = None,
) -> ApprovalRequest:
    """Record the gate and park the run.

    Called in the same transaction that creates the step, so a gated step can never
    exist without its approval request.
    """
    approval = ApprovalRequest(
        run_id=run_id,
        step_id=step_id,
        tool_name=tool_name,
        arguments=arguments,
        reason=reason,
        decision=Decision.PENDING,
        expires_at=sa.func.now()
        + dt.timedelta(seconds=ttl_seconds or settings.approval_ttl_seconds),
    )
    session.add(approval)
    await session.flush()

    await transition_run(
        session,
        run_id=run_id,
        to=RunStatus.AWAITING_APPROVAL,
        reason=str(TransitionReason.APPROVAL_REQUIRED),
        actor=actor,
        expect=RunStatus.RUNNING,
        details={"tool": tool_name, "approval_id": str(approval.id)},
    )
    log.info("approval_required", tool=tool_name, reason=reason, approval_id=str(approval.id))
    return approval


async def decide(
    session: AsyncSession,
    *,
    approval_id: uuid.UUID,
    decision: Decision,
    decided_by: str,
    decision_reason: str | None = None,
) -> DecisionResult:
    """Approve or reject. Idempotent: deciding twice returns the original decision."""
    approval = (
        await session.execute(
            sa.select(ApprovalRequest).where(ApprovalRequest.id == approval_id).with_for_update()
        )
    ).scalar_one_or_none()
    if approval is None:
        return DecisionResult(False, Decision.PENDING, detail="not_found")

    if approval.decision != Decision.PENDING:
        # Already decided. Not an error — a retried request must not flip a decision.
        return DecisionResult(False, Decision(approval.decision), detail="already_decided")

    await session.execute(
        sa.update(ApprovalRequest)
        .where(ApprovalRequest.id == approval_id)
        .values(
            decision=decision,
            decided_by=decided_by,
            decision_reason=decision_reason,
            decided_at=sa.func.now(),
        )
    )

    metrics.APPROVAL_DECISIONS.labels(decision=str(decision)).inc()
    if approval.requested_at is not None:
        waited = (dt.datetime.now(dt.UTC) - approval.requested_at).total_seconds()
        metrics.APPROVAL_WAIT.observe(max(waited, 0.0))

    if decision is Decision.APPROVED:
        return await _resume(session, approval, decided_by=decided_by)
    return await _refuse(
        session, approval, decision=decision, decided_by=decided_by, reason=decision_reason
    )


async def _resume(
    session: AsyncSession, approval: ApprovalRequest, *, decided_by: str
) -> DecisionResult:
    applied = await transition_step(
        session,
        step_id=approval.step_id,
        to=StepStatus.PENDING,
        reason=str(TransitionReason.APPROVED),
        actor=decided_by,
        expect=StepStatus.AWAITING_APPROVAL,
        values={"available_at": sa.func.now()},
        details={"approval_id": str(approval.id)},
    )
    if not applied:
        return DecisionResult(False, Decision.APPROVED, detail=applied.blocked_by)

    await transition_run(
        session,
        run_id=approval.run_id,
        to=RunStatus.RUNNING,
        reason=str(TransitionReason.APPROVED),
        actor=decided_by,
        expect=RunStatus.AWAITING_APPROVAL,
    )
    log.info("approval_granted", approval_id=str(approval.id), by=decided_by)
    return DecisionResult(True, Decision.APPROVED, resume_step_id=approval.step_id)


async def _refuse(
    session: AsyncSession,
    approval: ApprovalRequest,
    *,
    decision: Decision,
    decided_by: str,
    reason: str | None,
) -> DecisionResult:
    """Reject or expire: fail the step, then either steer or stop."""
    policy = await _rejection_policy(session, approval.run_id)
    verb = "rejected" if decision is Decision.REJECTED else "expired"
    message = f"Tool call {verb} by {decided_by}" + (f": {reason}" if reason else ".")

    step = (await session.execute(sa.select(Step).where(Step.id == approval.step_id))).scalar_one()

    # The failed step still carries a tool_result payload, so the transcript builder
    # renders it as an errored result the model can read.
    applied = await transition_step(
        session,
        step_id=approval.step_id,
        to=StepStatus.FAILED,
        reason=str(
            TransitionReason.REJECTED
            if decision is Decision.REJECTED
            else TransitionReason.APPROVAL_EXPIRED
        ),
        actor=decided_by,
        expect=StepStatus.AWAITING_APPROVAL,
        values={
            "ended_at": sa.func.now(),
            "output": {
                TOOL_USE_ID: (step.input or {}).get("tool_use_id", ""),
                TOOL_CONTENT: message,
                TOOL_IS_ERROR: True,
                "data": {"approval_id": str(approval.id), "decision": str(decision)},
            },
            "error": {
                "code": str(
                    ErrorCode.APPROVAL_REJECTED
                    if decision is Decision.REJECTED
                    else ErrorCode.APPROVAL_EXPIRED
                ),
                "class": "terminal",
                "message": message,
            },
        },
    )
    if not applied:
        return DecisionResult(False, decision, detail=applied.blocked_by)

    if policy is RejectionPolicy.FAIL_RUN or decision is Decision.EXPIRED:
        # An expiry always ends the run: nobody is coming, and continuing would mean
        # acting as though a human had answered.
        from app.core.runs import fail_run

        await transition_run(
            session,
            run_id=approval.run_id,
            to=RunStatus.RUNNING,
            reason=str(TransitionReason.REJECTED),
            actor=decided_by,
            expect=RunStatus.AWAITING_APPROVAL,
        )
        await fail_run(
            session,
            run_id=approval.run_id,
            code=(
                ErrorCode.APPROVAL_REJECTED
                if decision is Decision.REJECTED
                else ErrorCode.APPROVAL_EXPIRED
            ),
            message=message,
            reason=str(TransitionReason.REJECTED),
            actor=decided_by,
        )
        log.info(
            "approval_refused_run_failed",
            approval_id=str(approval.id),
            decision=str(decision),
        )
        return DecisionResult(True, decision, detail="run_failed")

    # Steer: hand the refusal back to the model as a tool error and keep going.
    await transition_run(
        session,
        run_id=approval.run_id,
        to=RunStatus.RUNNING,
        reason=str(TransitionReason.REJECTED),
        actor=decided_by,
        expect=RunStatus.AWAITING_APPROVAL,
    )

    from app.core.runs import create_next_step

    run = (
        await session.execute(sa.select(AgentRun).where(AgentRun.id == approval.run_id))
    ).scalar_one()
    next_step = await create_next_step(
        session, run=run, kind=StepKind.AGENT_TURN, input={}, actor=decided_by
    )
    log.info("approval_rejected_fed_back", approval_id=str(approval.id))
    return DecisionResult(
        True,
        decision,
        resume_step_id=next_step.id if next_step else None,
        detail="fed_back_to_model",
    )


async def _rejection_policy(session: AsyncSession, run_id: uuid.UUID) -> RejectionPolicy:
    raw = (
        await session.execute(
            sa.select(AgentDefinition.on_approval_rejected)
            .join(AgentRun, AgentRun.agent_definition_id == AgentDefinition.id)
            .where(AgentRun.id == run_id)
        )
    ).scalar_one_or_none()
    try:
        return RejectionPolicy(raw or RejectionPolicy.FEED_BACK_TO_MODEL)
    except ValueError:
        return RejectionPolicy.FEED_BACK_TO_MODEL


async def expire_stale(session: AsyncSession, *, limit: int = 100) -> list[uuid.UUID]:
    """Fail approvals nobody answered in time.

    Without this a forgotten approval pins a run open indefinitely, holding its
    budget and its slot in the dashboard's queue.
    """
    stale = (
        (
            await session.execute(
                sa.select(ApprovalRequest.id)
                .where(
                    ApprovalRequest.decision == Decision.PENDING,
                    ApprovalRequest.expires_at < sa.func.now(),
                )
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
        .scalars()
        .all()
    )

    expired: list[uuid.UUID] = []
    for approval_id in stale:
        result = await decide(
            session,
            approval_id=approval_id,
            decision=Decision.EXPIRED,
            decided_by="reaper",
            decision_reason="no decision before the approval deadline",
        )
        if result.applied:
            expired.append(approval_id)
            log.warning("approval_expired", approval_id=str(approval_id))
    return expired


__all__ = [
    "Decision",
    "DecisionResult",
    "RejectionPolicy",
    "create_gate",
    "decide",
    "expire_stale",
]
