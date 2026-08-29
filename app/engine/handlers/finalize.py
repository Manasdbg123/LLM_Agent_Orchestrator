"""Closes a run.

A dedicated step rather than a special case inside the executor: the run's completion
is then just another durable, retryable, auditable unit of work, and there is no code
path that ends a run without appearing in the step timeline.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa

from app.db import session_scope
from app.domain.models import Step
from app.domain.states import StepKind, StepStatus
from app.engine.types import HandlerResult, StepContext, register


@register(StepKind.FINALIZE)
async def handle_finalize(ctx: StepContext) -> HandlerResult:
    async with session_scope() as session:
        rows = (
            await session.execute(
                sa.select(Step.seq, Step.kind, Step.output, Step.input)
                .where(
                    Step.run_id == ctx.run_id,
                    Step.status == StepStatus.SUCCEEDED,
                    # Exclude finalize itself: it is still RUNNING at this point, so
                    # counting "succeeded steps" would silently mean "everything
                    # except me" and read as an off-by-one to anyone checking it.
                    Step.kind != StepKind.FINALIZE,
                )
                .order_by(Step.seq)
            )
        ).all()

    summary: dict[str, Any] = {
        "work_steps_completed": len(rows),
        "model_turns": sum(1 for r in rows if r.kind == StepKind.AGENT_TURN),
        "tool_calls": sum(1 for r in rows if r.kind == StepKind.TOOL_CALL),
    }

    # An agent run carries its answer in from the final model turn; a dummy run has
    # no answer and reports its trace instead. One closer serves both run styles.
    answer = ctx.input.get("answer")
    if answer is not None:
        summary["answer"] = answer
        # From the step's *input*, which is where the executor recorded which tool
        # this step invokes. The result payload is the tool's own `data` dict and
        # carries no tool name, so reading it here yielded raw tool_use ids.
        summary["tool_sequence"] = [
            (r.input or {}).get("tool_name", "") for r in rows if r.kind == StepKind.TOOL_CALL
        ]
        return HandlerResult(
            output={"summary": {"work_steps_completed": len(rows)}}, run_output=summary
        )

    summary["results"] = [{"seq": r.seq, "kind": str(r.kind), "output": r.output} for r in rows]
    declared = ctx.run.input.get("expected_output")
    if declared:
        summary["result"] = declared
    return HandlerResult(
        output={"summary": {"work_steps_completed": len(rows)}}, run_output=summary
    )


__all__ = ["handle_finalize"]
