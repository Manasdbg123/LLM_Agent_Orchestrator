"""A small server-rendered operator dashboard.

Server-rendered rather than a SPA: this is a read-mostly view of server state plus an
approve/reject button. HTML from Jinja with a meta-refresh is a few hundred lines with
no build step, no second dependency tree and no CORS surface, and it renders the same
data the API already exposes. A React app here would add a toolchain whose only
benefit is looking like a frontend project.

The pages answer the four questions an operator actually has: what is running, what
happened inside this run, what is it costing, and what needs my decision.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.deps import SessionDep, queue_holder
from app.core import approvals
from app.domain.models import AgentRun, ApprovalRequest, LLMCall, StateTransition, Step, ToolCall

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

router = APIRouter(tags=["dashboard"])


@router.get("/ui", response_class=HTMLResponse)
async def runs_page(request: Request, session: SessionDep) -> HTMLResponse:
    runs = (
        (await session.execute(sa.select(AgentRun).order_by(AgentRun.created_at.desc()).limit(50)))
        .scalars()
        .all()
    )

    counts = {
        status: count
        for status, count in (
            await session.execute(
                sa.select(AgentRun.status, sa.func.count()).group_by(AgentRun.status)
            )
        ).all()
    }
    pending_approvals = (
        await session.execute(
            sa.select(sa.func.count())
            .select_from(ApprovalRequest)
            .where(ApprovalRequest.decision == approvals.Decision.PENDING)
        )
    ).scalar_one()
    total_cost = (
        await session.execute(sa.select(sa.func.coalesce(sa.func.sum(AgentRun.cost_usd), 0)))
    ).scalar_one()

    return TEMPLATES.TemplateResponse(
        request=request,
        name="runs.html",
        context={
            "runs": runs,
            "counts": counts,
            "pending_approvals": pending_approvals,
            "total_cost": total_cost,
        },
    )


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: uuid.UUID, session: SessionDep) -> HTMLResponse:
    run = (
        await session.execute(sa.select(AgentRun).where(AgentRun.id == run_id))
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    steps = (
        (await session.execute(sa.select(Step).where(Step.run_id == run_id).order_by(Step.seq)))
        .scalars()
        .all()
    )
    transitions = (
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
    llm_calls = (
        (await session.execute(sa.select(LLMCall).where(LLMCall.run_id == run_id))).scalars().all()
    )
    tool_calls = {
        tc.step_id: tc
        for tc in (await session.execute(sa.select(ToolCall).where(ToolCall.run_id == run_id)))
        .scalars()
        .all()
    }
    cost_by_step = {c.step_id: c for c in llm_calls}

    # The longest step sets the bar width, so the timeline shows relative cost of
    # time at a glance rather than a row of identical bars.
    longest = max(
        ((s.ended_at - s.started_at).total_seconds() for s in steps if s.started_at and s.ended_at),
        default=1.0,
    )

    return TEMPLATES.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "run": run,
            "steps": steps,
            "transitions": transitions,
            "cost_by_step": cost_by_step,
            "tool_calls": tool_calls,
            "longest": max(longest, 0.001),
        },
    )


@router.get("/ui/approvals", response_class=HTMLResponse)
async def approvals_page(request: Request, session: SessionDep) -> HTMLResponse:
    pending = (
        (
            await session.execute(
                sa.select(ApprovalRequest)
                .where(ApprovalRequest.decision == approvals.Decision.PENDING)
                .order_by(ApprovalRequest.requested_at)
            )
        )
        .scalars()
        .all()
    )
    recent = (
        (
            await session.execute(
                sa.select(ApprovalRequest)
                .where(ApprovalRequest.decision != approvals.Decision.PENDING)
                .order_by(ApprovalRequest.decided_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    return TEMPLATES.TemplateResponse(
        request=request,
        name="approvals.html",
        context={"pending": pending, "recent": recent},
    )


@router.post("/ui/approvals/{approval_id}/decide")
async def decide_from_ui(
    approval_id: uuid.UUID,
    session: SessionDep,
    decision: Annotated[str, Form()],
    decided_by: Annotated[str, Form()] = "dashboard",
    reason: Annotated[str, Form()] = "",
) -> RedirectResponse:
    """Goes through exactly the same service call as the API endpoint.

    The dashboard is a client of the engine, not a second way into it — a UI with its
    own approval logic is a UI that will eventually disagree with the API.
    """
    result = await approvals.decide(
        session,
        approval_id=approval_id,
        decision=(
            approvals.Decision.APPROVED if decision == "approve" else approvals.Decision.REJECTED
        ),
        decided_by=decided_by or "dashboard",
        decision_reason=reason or None,
    )
    await session.commit()

    if result.applied and result.resume_step_id is not None:
        approval = (
            await session.execute(
                sa.select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
            )
        ).scalar_one()
        queue = queue_holder.get()
        if queue is not None:
            try:
                await queue.publish(step_id=result.resume_step_id, run_id=approval.run_id)
            except Exception:
                # Non-fatal: the reaper dispatches it within the enqueue grace window.
                pass

    return RedirectResponse(url="/ui/approvals", status_code=303)


__all__ = ["router"]
