"""A small server-rendered operator dashboard.

Server-rendered rather than a SPA: this is a read-mostly view of server state plus an
approve/reject button. HTML from Jinja with a meta-refresh is a few hundred lines with
no build step, no second dependency tree and no CORS surface, and it renders the same
data the API already exposes. A React app here would add a toolchain whose only
benefit is looking like a frontend project.

The pages answer the four questions an operator actually has: what is running, what
happened inside this run, what is it costing, and what needs my decision. They also
let the operator act: start a run, cancel one, and decide approvals - each through
the same core service calls the API uses, never a second code path.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.deps import SessionDep, queue_holder
from app.config import settings
from app.core import approvals
from app.core.runs import create_run, get_or_create_definition, request_cancel
from app.domain.models import AgentRun, ApprovalRequest, LLMCall, StateTransition, Step, ToolCall
from app.domain.states import TERMINAL_RUN_STATUSES, RunStatus, StepKind
from app.tools.registry import default_registry

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

STATUS_FILTERS = ["running", "awaiting_approval", "pending", "succeeded", "failed", "cancelled"]


def _ago(value: dt.datetime | None) -> str:
    """'4s ago', '12m ago' - an operator scanning a list reads age, not timestamps."""
    if value is None:
        return ""
    seconds = int((dt.datetime.now(dt.UTC) - value).total_seconds())
    if seconds < 60:
        return f"{max(seconds, 0)}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _duration(start: dt.datetime | None, end: dt.datetime | None) -> str:
    if start is None:
        return ""
    seconds = ((end or dt.datetime.now(dt.UTC)) - start).total_seconds()
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _task_text(task: str) -> str:
    """The task as a person wrote it: runs driven by the fake provider carry their
    script inside the task text, which is noise on an operator's screen."""
    from app.llm.fake import SCRIPT_MARKER

    return task.split(SCRIPT_MARKER, 1)[0].strip() or task


def _pct(part: float, whole: float) -> float:
    return 0.0 if not whole else round(min(100.0, 100.0 * float(part) / float(whole)), 1)


TEMPLATES.env.filters["ago"] = _ago
TEMPLATES.env.filters["pretty"] = _pretty
TEMPLATES.env.filters["task_text"] = _task_text
TEMPLATES.env.globals["duration"] = _duration
TEMPLATES.env.globals["pct"] = _pct

router = APIRouter(tags=["dashboard"])


@router.get("/ui", response_class=HTMLResponse)
async def runs_page(
    request: Request,
    session: SessionDep,
    status: str | None = None,
    q: str | None = None,
) -> HTMLResponse:
    stmt = sa.select(AgentRun).order_by(AgentRun.created_at.desc()).limit(100)
    if status in STATUS_FILTERS:
        stmt = stmt.where(AgentRun.status == RunStatus(status))
    else:
        status = None
    query = (q or "").strip()[:200]
    if query:
        stmt = stmt.where(AgentRun.task.ilike(f"%{query}%"))
    runs = (await session.execute(stmt)).scalars().all()

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
            "total_runs": sum(counts.values()),
            "pending_approvals": pending_approvals,
            "total_cost": total_cost,
            "status_filter": status,
            "status_filters": STATUS_FILTERS,
            "q": query,
            "tools": default_registry().names,
            "default_model": settings.llm_model,
            "nav": "runs",
            "live": True,
        },
    )


@router.post("/ui/runs")
async def create_run_from_ui(
    session: SessionDep,
    task: Annotated[str, Form()],
    max_steps: Annotated[int | None, Form()] = None,
    max_cost_usd: Annotated[float | None, Form()] = None,
) -> RedirectResponse:
    """Start an agent run. Same service calls as `POST /v1/runs`, same guardrails."""
    task = task.strip()
    if not task:
        raise HTTPException(status_code=422, detail="task must not be empty")
    definition = await get_or_create_definition(
        session,
        name="dashboard",
        model=settings.llm_model,
        system_prompt="",
        tools=default_registry().names,
    )
    run, step = await create_run(
        session,
        definition=definition,
        task=task[:10_000],
        first_step_kind=StepKind.AGENT_TURN,
        first_step_input={},
        max_steps=max(1, min(200, max_steps)) if max_steps else None,
        max_cost_usd=max_cost_usd if max_cost_usd and max_cost_usd > 0 else None,
        actor="dashboard",
    )
    await session.commit()

    # After the commit, never before - the same ordering rule as the API.
    queue = queue_holder.get()
    if queue is not None:
        try:
            await queue.publish(step_id=step.id, run_id=run.id)
        except Exception:
            # Non-fatal: the reaper dispatches it within the enqueue grace window.
            pass
    return RedirectResponse(url=f"/ui/runs/{run.id}", status_code=303)


@router.post("/ui/runs/{run_id}/cancel")
async def cancel_from_ui(run_id: uuid.UUID, session: SessionDep) -> RedirectResponse:
    """Cooperative cancel, via the same call as `POST /v1/runs/{id}/cancel`."""
    await request_cancel(session, run_id=run_id, actor="dashboard")
    await session.commit()
    return RedirectResponse(url=f"/ui/runs/{run_id}", status_code=303)


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
            "terminal": RunStatus(run.status) in TERMINAL_RUN_STATUSES,
            "nav": "runs",
            "live": RunStatus(run.status) not in TERMINAL_RUN_STATUSES,
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
        context={"pending": pending, "recent": recent, "nav": "approvals", "live": True},
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
