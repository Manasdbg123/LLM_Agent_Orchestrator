"""The only sanctioned way to write a status column.

Two rules the rest of the codebase depends on:

1. A status write is legal only if the transition table in `app.domain.states` has
   that edge. An illegal transition raises `IllegalTransitionError` — it is a
   programming error, so it is loud rather than logged and swallowed.

2. Every status write appends a row to `state_transitions` **in the same
   transaction** and emits exactly one log line. The audit log therefore cannot drift
   from the state it describes, because a single function writes both.

The one sanctioned exception is `app.core.leases.claim_step`, which must be a single
atomic UPDATE to be race-free and so cannot use the read-then-write shape here. It
compensates by calling `record_transition` itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.errors import IllegalTransitionError
from app.domain.models import AgentRun, StateTransition, Step
from app.domain.states import (
    RunStatus,
    StepStatus,
    TransitionReason,
    run_transition_legal,
    step_transition_legal,
)
from app.obs.logging import get_logger
from app.obs.tracing import current_trace_id

log = get_logger("transitions")


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Outcome of an attempted status write.

    `applied=False` is a normal, expected outcome — it means someone else got there
    first (a competing worker, the reaper, a cancel). Callers must handle it; they
    must not treat it as an error.
    """

    applied: bool
    from_status: str | None
    to_status: str
    #: Why the write did not apply, when it did not.
    blocked_by: str | None = None

    def __bool__(self) -> bool:
        return self.applied


#: Columns a caller may set alongside a step status change. Anything not listed is
#: rejected, so `transition_step` cannot become a general-purpose UPDATE.
_STEP_WRITABLE = frozenset(
    {
        "output",
        "error",
        "available_at",
        "started_at",
        "ended_at",
        "lease_owner",
        "lease_expires_at",
        "attempt",
        "recoveries",
        "traceparent",
    }
)

_RUN_WRITABLE = frozenset({"output", "error", "started_at", "ended_at", "cancel_requested"})


def _check(values: dict[str, Any], allowed: frozenset[str], what: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{what}: not writable via transition(): {sorted(unknown)}")


async def record_transition(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    step_id: uuid.UUID | None,
    entity: str,
    from_status: str | None,
    to_status: str,
    reason: str,
    actor: str,
    attempt: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Append to the audit log and emit the matching log line.

    Public because `leases.claim_step` needs it; not to be called for a status write
    that did not actually happen.
    """
    session.add(
        StateTransition(
            run_id=run_id,
            step_id=step_id,
            entity=entity,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            actor=actor,
            attempt=attempt,
            details=details or {},
            # Joins the audit log to the traces: from a failed run's transition rows
            # you can go straight to the trace that produced them.
            trace_id=current_trace_id(),
        )
    )
    # `details` is nested rather than splatted: it is caller-supplied and may carry
    # any key, including ones this call already binds (`attempt` did exactly that),
    # which would raise TypeError at the log site and take down the transition.
    log.info(
        "state_transition",
        entity=entity,
        run_id=str(run_id),
        step_id=str(step_id) if step_id else None,
        **{"from": from_status, "to": to_status},
        reason=reason,
        actor=actor,
        attempt=attempt,
        details=details or {},
    )


async def transition_step(
    session: AsyncSession,
    *,
    step_id: uuid.UUID,
    to: StepStatus,
    reason: str,
    actor: str,
    expect: StepStatus | set[StepStatus] | None = None,
    require_owner: str | None = None,
    require_epoch: int | None = None,
    values: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> TransitionResult:
    """Move a step to `to`, guarded.

    `require_owner` / `require_epoch` are the fencing guard: a worker whose lease was
    reclaimed while it was stalled will match zero rows here and get
    `applied=False, blocked_by="fenced"`, which is how it learns to throw its result
    away instead of clobbering the worker that took over.
    """
    values = dict(values or {})
    _check(values, _STEP_WRITABLE, "step")

    # SELECT ... FOR UPDATE serializes this against any competing transition on the
    # same step, so the legality check below cannot be invalidated between read and
    # write.
    row = (
        await session.execute(
            sa.select(Step.status, Step.attempt, Step.lease_owner, Step.lease_epoch, Step.run_id)
            .where(Step.id == step_id)
            .with_for_update()
        )
    ).one_or_none()
    if row is None:
        return TransitionResult(False, None, str(to), blocked_by="missing")

    current = StepStatus(row.status)

    if require_owner is not None and row.lease_owner != require_owner:
        return TransitionResult(False, str(current), str(to), blocked_by="fenced")
    if require_epoch is not None and row.lease_epoch != require_epoch:
        return TransitionResult(False, str(current), str(to), blocked_by="fenced")

    if expect is not None:
        allowed = {expect} if isinstance(expect, StepStatus) else expect
        if current not in allowed:
            return TransitionResult(False, str(current), str(to), blocked_by="unexpected_status")

    if current == to and to in {StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.CANCELLED}:
        # Idempotent re-application of a terminal state: not an error, not a write.
        return TransitionResult(False, str(current), str(to), blocked_by="already_terminal")

    if not step_transition_legal(current, to):
        raise IllegalTransitionError(
            f"step {step_id}: {current} -> {to} is not a legal transition",
            details={"step_id": str(step_id), "from": str(current), "to": str(to)},
        )

    await session.execute(
        sa.update(Step)
        .where(Step.id == step_id)
        .values(status=to, updated_at=sa.func.now(), **values)
    )
    await record_transition(
        session,
        run_id=row.run_id,
        step_id=step_id,
        entity="step",
        from_status=str(current),
        to_status=str(to),
        reason=reason,
        actor=actor,
        attempt=values.get("attempt", row.attempt),
        details=details,
    )
    _count_step_transition(to, reason)
    return TransitionResult(True, str(current), str(to))


async def transition_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    to: RunStatus,
    reason: str,
    actor: str,
    expect: RunStatus | set[RunStatus] | None = None,
    values: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> TransitionResult:
    values = dict(values or {})
    _check(values, _RUN_WRITABLE, "run")

    row = (
        await session.execute(
            sa.select(AgentRun.status).where(AgentRun.id == run_id).with_for_update()
        )
    ).one_or_none()
    if row is None:
        return TransitionResult(False, None, str(to), blocked_by="missing")

    current = RunStatus(row.status)

    if expect is not None:
        allowed = {expect} if isinstance(expect, RunStatus) else expect
        if current not in allowed:
            return TransitionResult(False, str(current), str(to), blocked_by="unexpected_status")

    if current == to:
        return TransitionResult(False, str(current), str(to), blocked_by="noop")

    if not run_transition_legal(current, to):
        # A run that is already terminal is the common case here (a late step result
        # arriving after cancellation). That is expected, not a bug, so it is a
        # blocked result rather than a raise.
        from app.domain.states import TERMINAL_RUN_STATUSES

        if current in TERMINAL_RUN_STATUSES:
            return TransitionResult(False, str(current), str(to), blocked_by="already_terminal")
        raise IllegalTransitionError(
            f"run {run_id}: {current} -> {to} is not a legal transition",
            details={"run_id": str(run_id), "from": str(current), "to": str(to)},
        )

    if to in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
        values.setdefault("ended_at", sa.func.now())
    if to is RunStatus.RUNNING and current is RunStatus.PENDING:
        values.setdefault("started_at", sa.func.now())

    await session.execute(
        sa.update(AgentRun)
        .where(AgentRun.id == run_id)
        .values(status=to, updated_at=sa.func.now(), **values)
    )
    await record_transition(
        session,
        run_id=run_id,
        step_id=None,
        entity="run",
        from_status=str(current),
        to_status=str(to),
        reason=reason,
        actor=actor,
        details=details,
    )
    _count_run_transition(to, values)
    return TransitionResult(True, str(current), str(to))


def _count_step_transition(to: StepStatus, reason: str) -> None:
    """Metrics for a step status change.

    Deliberately colocated with the write: a counter incremented somewhere else can
    disagree with the state it claims to describe, and then nobody trusts either.
    """
    from app.obs import metrics

    if to is StepStatus.RETRYING:
        metrics.STEP_RETRIES.labels(kind="step", error_code=reason).inc()
    elif to is StepStatus.PENDING and reason == str(TransitionReason.LEASE_EXPIRED):
        metrics.LEASE_EXPIRATIONS.inc()
    elif to is StepStatus.FAILED and reason == str(TransitionReason.TOO_MANY_RECOVERIES):
        metrics.STEPS_ABANDONED.inc()


def _count_run_transition(to: RunStatus, values: dict[str, Any]) -> None:
    from app.obs import metrics

    if to is RunStatus.RUNNING:
        metrics.RUNS_STARTED.inc()
    elif to in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
        error = values.get("error") or {}
        code = error.get("code", "") if isinstance(error, dict) else ""
        metrics.RUNS_COMPLETED.labels(status=str(to), error_code=code).inc()


__all__ = ["TransitionResult", "record_transition", "transition_run", "transition_step"]
