"""FastAPI surface.

Phase 2 covers run creation, inspection and cancellation. Approvals, cost breakdown
and the SSE tail arrive with the phases that give them something to report.

The API is a thin transactional shell: it writes state and hints the queue. It never
executes a step, which is why it can be scaled, restarted or lost without affecting a
run in flight.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import sqlalchemy as sa
from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import SessionDep, queue_holder
from app.config import settings
from app.core import approvals
from app.core.runs import create_run, get_or_create_definition, request_cancel
from app.dashboard.views import router as dashboard_router
from app.db import dispose_engine
from app.domain.models import AgentRun, ApprovalRequest, LLMCall, StateTransition, Step
from app.domain.states import RunStatus, StepKind
from app.obs import metrics
from app.obs.logging import configure_logging, get_logger
from app.obs.tracing import configure_tracing
from app.queue.factory import build_queue
from app.tools.registry import default_registry

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    configure_tracing()
    app.state.queue = build_queue()
    await app.state.queue.setup()
    queue_holder.set(app.state.queue)

    # State gauges are refreshed here and only here. Workers each reporting their own
    # view of "runs in progress" would give Prometheus several partial answers to sum
    # into a wrong one; Postgres is the only component that knows the real number.
    gauges = (
        asyncio.create_task(metrics.gauge_refresh_loop(settings.gauge_refresh_seconds))
        if settings.metrics_enabled
        else None
    )
    log.info("api_started", queue_backend=settings.queue_backend)
    try:
        yield
    finally:
        if gauges is not None:
            gauges.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await gauges
        await app.state.queue.close()
        await dispose_engine()


app = FastAPI(
    title="Agent Orchestration Engine",
    version="0.5.0",
    summary="Durable execution for LLM agents",
    lifespan=lifespan,
)
app.include_router(dashboard_router)


# --- schemas --------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    agent: str = Field(default="dummy", description="Agent definition name")
    task: str = Field(..., min_length=1, max_length=10_000)
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Run payload. For the dummy agent: {'plan': [ {step spec}, ... ]}",
    )
    first_step_kind: Literal["dummy", "agent_turn"] = "agent_turn"
    tools: list[str] | None = Field(
        default=None, description="Tool names the agent may use (default: all registered)"
    )
    model: str | None = Field(default=None, description="Model id; defaults to configured")
    system_prompt: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=200)
    max_cost_usd: float | None = Field(default=None, gt=0)
    timeout_seconds: int | None = Field(default=None, ge=1)


class RunSummary(BaseModel):
    id: uuid.UUID
    status: RunStatus
    task: str
    steps_used: int
    max_steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    max_cost_usd: float
    created_at: dt.datetime
    started_at: dt.datetime | None
    ended_at: dt.datetime | None
    deadline_at: dt.datetime
    cancel_requested: bool
    error: dict[str, Any] | None = None
    output: dict[str, Any] | None = None

    @classmethod
    def of(cls, run: AgentRun) -> RunSummary:
        return cls(
            id=run.id,
            status=RunStatus(run.status),
            task=run.task,
            steps_used=run.steps_used,
            max_steps=run.max_steps,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            cost_usd=float(run.cost_usd),
            max_cost_usd=float(run.max_cost_usd),
            created_at=run.created_at,
            started_at=run.started_at,
            ended_at=run.ended_at,
            deadline_at=run.deadline_at,
            cancel_requested=run.cancel_requested,
            error=run.error,
            output=run.output,
        )


class StepView(BaseModel):
    id: uuid.UUID
    seq: int
    kind: str
    status: str
    attempt: int
    max_attempts: int
    recoveries: int
    lease_owner: str | None
    lease_expires_at: dt.datetime | None
    available_at: dt.datetime
    started_at: dt.datetime | None
    ended_at: dt.datetime | None
    duration_ms: int | None
    output: dict[str, Any] | None
    error: dict[str, Any] | None


class TransitionView(BaseModel):
    id: int
    entity: str
    step_id: uuid.UUID | None
    from_status: str | None
    to_status: str
    reason: str
    actor: str
    attempt: int | None
    details: dict[str, Any]
    created_at: dt.datetime


# --- endpoints ------------------------------------------------------------------


@app.post("/v1/runs", response_model=RunSummary, status_code=201, tags=["runs"])
async def create_run_endpoint(
    body: CreateRunRequest,
    session: SessionDep,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> RunSummary:
    """Create a run and enqueue its first step.

    Supplying `Idempotency-Key` makes creation safe to retry: a repeated request
    returns the original run rather than starting a second one.
    """
    definition = await get_or_create_definition(
        session,
        name=body.agent,
        model=body.model or settings.llm_model,
        system_prompt=body.system_prompt or "",
        tools=body.tools if body.tools is not None else default_registry().names,
    )
    run, step = await create_run(
        session,
        definition=definition,
        task=body.task,
        input=body.input,
        first_step_kind=StepKind(body.first_step_kind),
        first_step_input=(body.input.get("plan") or [{}])[0]
        if body.first_step_kind == "dummy"
        else {},
        client_idempotency_key=idempotency_key,
        max_steps=body.max_steps,
        max_cost_usd=body.max_cost_usd,
        timeout_seconds=body.timeout_seconds,
    )
    await session.commit()
    await session.refresh(run)

    # After the commit, never before: an enqueued id whose transaction rolled back
    # would send a worker chasing a step that does not exist.
    try:
        await app.state.queue.publish(step_id=step.id, run_id=run.id)
    except Exception as exc:
        log.warning("enqueue_failed", run_id=str(run.id), error=str(exc))
        response.headers["X-Enqueue-Deferred"] = "reaper"

    return RunSummary.of(run)


@app.get("/v1/runs", response_model=list[RunSummary], tags=["runs"])
async def list_runs(
    session: SessionDep,
    status: RunStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[RunSummary]:
    stmt = sa.select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit)
    if status is not None:
        stmt = stmt.where(AgentRun.status == status)
    rows = (await session.execute(stmt)).scalars().all()
    return [RunSummary.of(r) for r in rows]


async def _load_run(session: AsyncSession, run_id: uuid.UUID) -> AgentRun:
    run = (
        await session.execute(sa.select(AgentRun).where(AgentRun.id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return run


@app.get("/v1/runs/{run_id}", response_model=RunSummary, tags=["runs"])
async def get_run(run_id: uuid.UUID, session: SessionDep) -> RunSummary:
    return RunSummary.of(await _load_run(session, run_id))


@app.get("/v1/runs/{run_id}/steps", response_model=list[StepView], tags=["runs"])
async def get_steps(run_id: uuid.UUID, session: SessionDep) -> list[StepView]:
    await _load_run(session, run_id)
    rows = (
        (await session.execute(sa.select(Step).where(Step.run_id == run_id).order_by(Step.seq)))
        .scalars()
        .all()
    )
    return [
        StepView(
            id=s.id,
            seq=s.seq,
            kind=str(s.kind),
            status=str(s.status),
            attempt=s.attempt,
            max_attempts=s.max_attempts,
            recoveries=s.recoveries,
            lease_owner=s.lease_owner,
            lease_expires_at=s.lease_expires_at,
            available_at=s.available_at,
            started_at=s.started_at,
            ended_at=s.ended_at,
            duration_ms=(
                int((s.ended_at - s.started_at).total_seconds() * 1000)
                if s.started_at and s.ended_at
                else None
            ),
            output=s.output,
            error=s.error,
        )
        for s in rows
    ]


@app.get("/v1/runs/{run_id}/transitions", response_model=list[TransitionView], tags=["runs"])
async def get_transitions(run_id: uuid.UUID, session: SessionDep) -> list[TransitionView]:
    """The audit log: every state change, who caused it, and why."""
    await _load_run(session, run_id)
    rows = (
        (
            await session.execute(
                sa.select(StateTransition)
                .where(StateTransition.run_id == run_id)
                .order_by(StateTransition.id)
            )
        )
        .scalars()
        .all()
    )
    return [
        TransitionView(
            id=t.id,
            entity=t.entity,
            step_id=t.step_id,
            from_status=t.from_status,
            to_status=t.to_status,
            reason=t.reason,
            actor=t.actor,
            attempt=t.attempt,
            details=t.details,
            created_at=t.created_at,
        )
        for t in rows
    ]


@app.post("/v1/runs/{run_id}/cancel", response_model=RunSummary, tags=["runs"])
async def cancel_run(run_id: uuid.UUID, session: SessionDep) -> RunSummary:
    """Cooperative cancellation.

    Returns immediately. A step already executing finishes or aborts at its next
    checkpoint; we do not interrupt work that may be mid-side-effect.
    """
    await _load_run(session, run_id)
    changed = await request_cancel(session, run_id=run_id, actor="api")
    if not changed:
        raise HTTPException(status_code=409, detail="run is already in a terminal state")
    await session.commit()
    return RunSummary.of(await _load_run(session, run_id))


class CostBreakdown(BaseModel):
    run_id: uuid.UUID
    cost_usd: float
    max_cost_usd: float
    budget_remaining_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    llm_call_count: int
    calls: list[dict[str, Any]]


@app.get("/v1/runs/{run_id}/cost", response_model=CostBreakdown, tags=["runs"])
async def get_cost(run_id: uuid.UUID, session: SessionDep) -> CostBreakdown:
    """Per-call cost breakdown, plus the rollup and what is left of the budget."""
    run = await _load_run(session, run_id)
    rows = (
        (
            await session.execute(
                sa.select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.created_at)
            )
        )
        .scalars()
        .all()
    )
    return CostBreakdown(
        run_id=run_id,
        cost_usd=float(run.cost_usd),
        max_cost_usd=float(run.max_cost_usd),
        budget_remaining_usd=float(run.max_cost_usd - run.cost_usd),
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cache_read_tokens=run.cache_read_tokens,
        cache_write_tokens=run.cache_write_tokens,
        llm_call_count=len(rows),
        calls=[
            {
                "step_id": str(c.step_id),
                "model": c.model,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "cost_usd": float(c.cost_usd),
                "price_version": c.price_version,
                "latency_ms": c.latency_ms,
                "stop_reason": c.stop_reason,
            }
            for c in rows
        ],
    )


class ApprovalView(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    step_id: uuid.UUID
    tool_name: str
    arguments: dict[str, Any]
    reason: str | None
    decision: str
    decided_by: str | None
    decision_reason: str | None
    requested_at: dt.datetime
    expires_at: dt.datetime
    decided_at: dt.datetime | None

    @classmethod
    def of(cls, a: ApprovalRequest) -> ApprovalView:
        return cls(
            id=a.id,
            run_id=a.run_id,
            step_id=a.step_id,
            tool_name=a.tool_name,
            arguments=a.arguments,
            reason=a.reason,
            decision=a.decision,
            decided_by=a.decided_by,
            decision_reason=a.decision_reason,
            requested_at=a.requested_at,
            expires_at=a.expires_at,
            decided_at=a.decided_at,
        )


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    decided_by: str = Field(..., min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=2000)


@app.get("/v1/approvals", response_model=list[ApprovalView], tags=["approvals"])
async def list_approvals(
    session: SessionDep,
    status: Literal["pending", "approved", "rejected", "expired"] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ApprovalView]:
    """The approval queue. A single indexed read, because a gated run parks itself in
    `awaiting_approval` rather than leaving the state implicit in its steps."""
    rows = (
        (
            await session.execute(
                sa.select(ApprovalRequest)
                .where(ApprovalRequest.decision == status)
                .order_by(ApprovalRequest.requested_at)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [ApprovalView.of(a) for a in rows]


@app.get("/v1/approvals/{approval_id}", response_model=ApprovalView, tags=["approvals"])
async def get_approval(approval_id: uuid.UUID, session: SessionDep) -> ApprovalView:
    approval = (
        await session.execute(sa.select(ApprovalRequest).where(ApprovalRequest.id == approval_id))
    ).scalar_one_or_none()
    if approval is None:
        raise HTTPException(status_code=404, detail=f"approval {approval_id} not found")
    return ApprovalView.of(approval)


@app.post("/v1/approvals/{approval_id}/decision", response_model=ApprovalView, tags=["approvals"])
async def decide_approval(
    approval_id: uuid.UUID, body: DecisionRequest, session: SessionDep
) -> ApprovalView:
    """Approve or reject a gated tool call.

    Idempotent: a repeated decision returns 409 with the decision that already
    stands, rather than flipping it. Approving resumes the run; rejecting feeds the
    refusal back to the model as an errored tool result unless the agent definition
    says `fail_run`.
    """
    result = await approvals.decide(
        session,
        approval_id=approval_id,
        decision=(
            approvals.Decision.APPROVED
            if body.decision == "approve"
            else approvals.Decision.REJECTED
        ),
        decided_by=body.decided_by,
        decision_reason=body.reason,
    )
    if not result.applied and result.detail == "not_found":
        raise HTTPException(status_code=404, detail=f"approval {approval_id} not found")
    if not result.applied and result.detail == "already_decided":
        raise HTTPException(
            status_code=409,
            detail=f"approval already {result.decision}; decisions are not reversible",
        )
    if not result.applied:
        raise HTTPException(status_code=409, detail=result.detail or "could not apply decision")

    await session.commit()

    # Published after the commit, never before: an enqueued id whose transaction
    # rolled back would send a worker after a step that does not exist.
    if result.resume_step_id is not None:
        approval = (
            await session.execute(
                sa.select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
            )
        ).scalar_one()
        try:
            await app.state.queue.publish(step_id=result.resume_step_id, run_id=approval.run_id)
        except Exception as exc:
            log.warning("resume_enqueue_failed", approval_id=str(approval_id), error=str(exc))

    return await get_approval(approval_id, session)


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
async def readyz(session: SessionDep) -> dict[str, str]:
    """Ready means the database answers. Without it the API can do nothing useful."""
    await session.execute(sa.text("SELECT 1"))
    return {"status": "ready", "queue_backend": settings.queue_backend}


__all__ = ["app"]
