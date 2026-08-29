"""Run-level operations: creation, step chaining, guardrails, cancellation."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.transitions import transition_run, transition_step
from app.domain.errors import ErrorCode
from app.domain.ids import new_id
from app.domain.models import AgentDefinition, AgentRun, Step
from app.domain.states import RunStatus, StepKind, StepStatus, TransitionReason
from app.obs.logging import get_logger
from app.obs.tracing import new_run_context

log = get_logger("runs")


async def get_or_create_definition(
    session: AsyncSession,
    *,
    name: str,
    model: str = "none",
    system_prompt: str = "",
    tools: list[str] | None = None,
    max_steps: int | None = None,
    max_cost_usd: float | None = None,
    timeout_seconds: int | None = None,
) -> AgentDefinition:
    """Fetch the latest version of a definition, creating v1 if absent.

    Definitions are immutable, so this never mutates an existing row; callers wanting
    different settings create a new version explicitly.

    Creation is `ON CONFLICT DO NOTHING` followed by a re-read rather than a bare
    INSERT. A plain read-then-insert loses under concurrency: several requests naming
    the same new agent all see nothing, all insert, and every one but the winner takes
    a unique violation. That is not hypothetical -- the load test produced it at a
    concurrency of 20, as a 500 on 13% of submissions. Under READ COMMITTED the
    conflicting INSERT blocks until the winner commits and then affects no rows, so
    the SELECT that follows takes a fresh snapshot and sees the winner's row.
    """
    existing = (
        await session.execute(
            sa.select(AgentDefinition)
            .where(AgentDefinition.name == name)
            .order_by(AgentDefinition.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    values = {
        "id": new_id(),
        "name": name,
        "version": 1,
        "model": model,
        "system_prompt": system_prompt,
        "tools": tools or [],
        "max_steps": max_steps or settings.default_max_steps,
        "max_cost_usd": Decimal(str(max_cost_usd or settings.default_max_cost_usd)),
        "timeout_seconds": timeout_seconds or settings.default_timeout_seconds,
    }
    inserted_id = (
        await session.execute(
            pg_insert(AgentDefinition)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_agent_definitions_name_version")
            .returning(AgentDefinition.id)
        )
    ).scalar_one_or_none()

    if inserted_id is None:
        # A concurrent caller created it first. Re-read rather than raising: both
        # callers asked for the same immutable definition and both should get it.
        winner = (
            await session.execute(
                sa.select(AgentDefinition)
                .where(AgentDefinition.name == name)
                .order_by(AgentDefinition.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if winner is None:  # pragma: no cover - would mean the constraint is gone
            raise RuntimeError(f"agent definition {name!r} neither inserted nor found")
        return winner

    return (
        await session.execute(
            sa.select(AgentDefinition).where(AgentDefinition.id == inserted_id)
        )
    ).scalar_one()


async def create_run(
    session: AsyncSession,
    *,
    definition: AgentDefinition,
    task: str,
    input: dict[str, Any] | None = None,
    first_step_kind: StepKind = StepKind.DUMMY,
    first_step_input: dict[str, Any] | None = None,
    client_idempotency_key: str | None = None,
    max_steps: int | None = None,
    max_cost_usd: float | None = None,
    timeout_seconds: int | None = None,
    actor: str = "api",
) -> tuple[AgentRun, Step]:
    """Create a run and its first step in one transaction.

    One transaction matters: a run with no first step would sit `pending` forever with
    nothing to pick it up, and a step with no run would violate its own foreign key.
    """
    if client_idempotency_key:
        existing = (
            await session.execute(
                sa.select(AgentRun).where(AgentRun.client_idempotency_key == client_idempotency_key)
            )
        ).scalar_one_or_none()
        if existing is not None:
            first = (
                await session.execute(
                    sa.select(Step).where(Step.run_id == existing.id).order_by(Step.seq).limit(1)
                )
            ).scalar_one()
            log.info("run_create_deduped", run_id=str(existing.id))
            return existing, first

    timeout = timeout_seconds or definition.timeout_seconds
    run_id = new_id()
    run = AgentRun(
        id=run_id,
        # Minted here so every step of the run, in whatever process, can join the
        # same trace by restoring this context.
        traceparent=new_run_context(run_id, task),
        agent_definition_id=definition.id,
        status=RunStatus.PENDING,
        task=task,
        input=input or {},
        client_idempotency_key=client_idempotency_key,
        max_steps=max_steps or definition.max_steps,
        max_cost_usd=Decimal(str(max_cost_usd)) if max_cost_usd else definition.max_cost_usd,
        deadline_at=sa.func.now() + dt.timedelta(seconds=timeout),
    )
    session.add(run)
    await session.flush()

    step = Step(
        run_id=run.id,
        seq=1,
        kind=first_step_kind,
        status=StepStatus.PENDING,
        input=first_step_input or {},
        max_attempts=settings.default_max_attempts,
        max_recoveries=settings.default_max_recoveries,
    )
    session.add(step)
    await session.flush()

    from app.core.transitions import record_transition

    await record_transition(
        session,
        run_id=run.id,
        step_id=None,
        entity="run",
        from_status=None,
        to_status=str(RunStatus.PENDING),
        reason=str(TransitionReason.CREATED),
        actor=actor,
        details={"task": task[:200]},
    )
    await record_transition(
        session,
        run_id=run.id,
        step_id=step.id,
        entity="step",
        from_status=None,
        to_status=str(StepStatus.PENDING),
        reason=str(TransitionReason.CREATED),
        actor=actor,
        details={"kind": str(first_step_kind), "seq": 1},
    )
    return run, step


async def fail_run(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    code: ErrorCode,
    message: str,
    reason: str,
    actor: str,
    details: dict[str, Any] | None = None,
) -> None:
    await transition_run(
        session,
        run_id=run_id,
        to=RunStatus.FAILED,
        reason=reason,
        actor=actor,
        values={
            "error": {
                "code": str(code),
                "class": "terminal",
                "message": message,
                **({"details": details} if details else {}),
            }
        },
    )


async def _guardrails_ok(session: AsyncSession, run: AgentRun, *, actor: str) -> bool:
    """Enforce max_steps and the wall-clock deadline before extending a run.

    Checked at step-creation time rather than step-start time, so a run cannot be
    extended past its budget even by one step.
    """
    # Evaluated server-side: comparing a deadline against a worker's clock would make
    # guardrail enforcement depend on which machine happened to pick the step up.
    expired = (
        await session.execute(
            sa.select(AgentRun.deadline_at < sa.func.now()).where(AgentRun.id == run.id)
        )
    ).scalar_one()
    if expired:
        await fail_run(
            session,
            run_id=run.id,
            code=ErrorCode.DEADLINE_EXCEEDED,
            message="run exceeded its wall-clock deadline",
            reason=str(TransitionReason.DEADLINE_EXCEEDED),
            actor=actor,
            details={"deadline_at": run.deadline_at.isoformat()},
        )
        return False

    used = (
        await session.execute(sa.select(AgentRun.steps_used).where(AgentRun.id == run.id))
    ).scalar_one()
    if used >= run.max_steps:
        await fail_run(
            session,
            run_id=run.id,
            code=ErrorCode.MAX_STEPS_EXCEEDED,
            message=f"run reached max_steps={run.max_steps}",
            reason=str(TransitionReason.MAX_STEPS_EXCEEDED),
            actor=actor,
            details={"steps_used": used, "max_steps": run.max_steps},
        )
        return False
    return True


async def create_next_step(
    session: AsyncSession,
    *,
    run: AgentRun,
    kind: StepKind,
    input: dict[str, Any] | None = None,
    parent_step_id: uuid.UUID | None = None,
    delay_seconds: float = 0.0,
    actor: str,
    max_attempts: int | None = None,
    approval_reason: str | None = None,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> Step | None:
    """Append the next step to a run, or fail the run if a guardrail says stop.

    Returns None when no step was created; the run is already terminal in that case.
    """
    if not await _guardrails_ok(session, run, actor=actor):
        return None

    next_seq = (
        await session.execute(
            sa.select(sa.func.coalesce(sa.func.max(Step.seq), 0) + 1).where(Step.run_id == run.id)
        )
    ).scalar_one()

    # A gated step is born in `awaiting_approval` and is never published, so there is
    # no window in which a worker could claim it before a human has seen it.
    gated = approval_reason is not None
    step = Step(
        run_id=run.id,
        seq=next_seq,
        kind=kind,
        status=StepStatus.AWAITING_APPROVAL if gated else StepStatus.PENDING,
        input=input or {},
        parent_step_id=parent_step_id,
        max_attempts=max_attempts or settings.default_max_attempts,
        max_recoveries=settings.default_max_recoveries,
        available_at=(
            sa.func.now() + dt.timedelta(seconds=delay_seconds) if delay_seconds else sa.func.now()
        ),
    )
    session.add(step)
    await session.flush()

    from app.core.transitions import record_transition

    await record_transition(
        session,
        run_id=run.id,
        step_id=step.id,
        entity="step",
        from_status=None,
        to_status=str(step.status),
        reason=str(TransitionReason.CREATED),
        actor=actor,
        details={"kind": str(kind), "seq": next_seq},
    )

    if gated:
        from app.core.approvals import create_gate

        await create_gate(
            session,
            run_id=run.id,
            step_id=step.id,
            tool_name=tool_name or "",
            arguments=arguments or {},
            reason=approval_reason or "",
            actor=actor,
        )
    return step


async def request_cancel(session: AsyncSession, *, run_id: uuid.UUID, actor: str) -> bool:
    """Cooperative cancel.

    Sets the flag and cancels any step that is not currently executing. A running
    step is left alone: it observes the flag at its next checkpoint. We do not
    interrupt work mid-side-effect, because doing so manufactures exactly the
    ambiguity the rest of the system exists to avoid.
    """
    run = (
        await session.execute(sa.select(AgentRun).where(AgentRun.id == run_id).with_for_update())
    ).scalar_one_or_none()
    if run is None:
        return False
    if RunStatus(run.status) in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
        return False

    await session.execute(
        sa.update(AgentRun)
        .where(AgentRun.id == run_id)
        .values(cancel_requested=True, updated_at=sa.func.now())
    )

    idle = (
        (
            await session.execute(
                sa.select(Step.id).where(
                    Step.run_id == run_id,
                    Step.status.in_(
                        [StepStatus.PENDING, StepStatus.RETRYING, StepStatus.AWAITING_APPROVAL]
                    ),
                )
            )
        )
        .scalars()
        .all()
    )
    for step_id in idle:
        await transition_step(
            session,
            step_id=step_id,
            to=StepStatus.CANCELLED,
            reason=str(TransitionReason.CANCELLED),
            actor=actor,
            values={"ended_at": sa.func.now(), "lease_owner": None, "lease_expires_at": None},
        )

    running = (
        await session.execute(
            sa.select(sa.func.count())
            .select_from(Step)
            .where(Step.run_id == run_id, Step.status == StepStatus.RUNNING)
        )
    ).scalar_one()
    if running == 0:
        await transition_run(
            session,
            run_id=run_id,
            to=RunStatus.CANCELLED,
            reason=str(TransitionReason.CANCELLED),
            actor=actor,
        )
    return True


__all__ = [
    "create_next_step",
    "create_run",
    "fail_run",
    "get_or_create_definition",
    "request_cancel",
]
