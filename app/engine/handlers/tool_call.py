"""Execute exactly one tool call, at most once.

The three-phase protocol from `app.core.idempotency` is what this handler is built
around:

    CLAIM (committed)  ->  EXECUTE (the external effect)  ->  COMMIT (committed)

The window between EXECUTE and COMMIT is unavoidable — an external effect and a local
transaction cannot be made atomic — so instead it is made *recoverable*. A worker
killed inside that window leaves an `in_flight` ledger row; its replacement finds
that row and resolves it by the tool's declared effect policy rather than by
guessing.

Note what is deliberately *not* a step failure here. Invalid arguments, an unknown
tool, and a tool that reports an error all produce a `tool_result` with
`is_error: true`, which the model reads on its next turn and corrects. Failing the
run on the model's first malformed argument would throw away a run over something
the model can usually fix by itself in one turn.
"""

from __future__ import annotations

import asyncio
from typing import Any

import sqlalchemy as sa
from pydantic import ValidationError

from app.core.idempotency import (
    ClaimOutcome,
    claim_effect,
    commit_effect,
    resolve_ambiguous,
)
from app.db import session_scope
from app.domain.errors import ErrorCode, RetryableError, TerminalError
from app.domain.models import Step
from app.domain.states import StepKind, StepStatus
from app.engine import faults
from app.engine.messages import TOOL_CONTENT, TOOL_IS_ERROR, TOOL_USE_ID, unresolved_tool_uses
from app.engine.types import HandlerResult, NextStep, StepContext, register
from app.obs import metrics
from app.obs.logging import get_logger
from app.obs.tracing import child_span
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import default_registry

log = get_logger("handler.tool_call")


@register(StepKind.TOOL_CALL)
async def handle_tool_call(ctx: StepContext) -> HandlerResult:
    spec = ctx.input
    tool_use_id = str(spec.get("tool_use_id") or "")
    tool_name = str(spec.get("tool_name") or "")
    arguments: dict[str, Any] = spec.get("arguments") or {}
    fault = faults.parse(spec.get("fault"))
    lease_key = faults.make_lease_key(ctx.step_id, ctx.lease.epoch)

    registry = default_registry()
    try:
        tool = registry.get(tool_name)
    except TerminalError as exc:
        # The model asked for a tool that does not exist. Tell it so; do not kill
        # the run over a typo it can correct next turn.
        return await _finish(
            ctx, tool_use_id, ToolResult(content=f"Error: {exc.message}", is_error=True)
        )

    # Phase 1: CLAIM. Committed before anything external happens, so a crash between
    # here and the effect still leaves a row proving an attempt was in flight.
    async with session_scope() as session:
        claim = await claim_effect(
            session,
            run_id=ctx.run_id,
            step_id=ctx.step_id,
            tool_name=tool_name,
            arguments=arguments,
            effect_policy=tool.effect_policy,
            attempt=ctx.lease.attempt,
            provider_tool_use_id=tool_use_id or None,
        )

    if claim.outcome is ClaimOutcome.ALREADY_COMMITTED:
        metrics.EFFECT_DEDUPES.labels(tool=tool_name, reason="already_committed").inc()
        # This exact effect already happened. Return the stored result and do not
        # execute again — this is the line that makes the run idempotent.
        stored = claim.result or {}
        log.info("tool_call_deduplicated", tool=tool_name, attempt=ctx.lease.attempt)
        return await _finish(
            ctx,
            tool_use_id,
            ToolResult(
                content=str(stored.get("content", "")),
                is_error=claim.is_error,
                data=stored.get("data") or {},
            ),
            replayed=True,
        )

    if claim.outcome is ClaimOutcome.AMBIGUOUS:
        metrics.EFFECT_DEDUPES.labels(tool=tool_name, reason="ambiguous_resumed").inc()
        # Raises for UNSAFE_TO_REPLAY; returns True for policies that can be resumed.
        resolve_ambiguous(tool.effect_policy, tool_name=tool_name)
        log.warning(
            "tool_call_resuming_ambiguous",
            tool=tool_name,
            policy=str(tool.effect_policy),
            attempt=ctx.lease.attempt,
        )

    try:
        args = tool.args_model.model_validate(arguments)
    except ValidationError as exc:
        # Structured-output validation failure. Handed back to the model as an error
        # result with the specific problem, which it can usually fix in one turn.
        message = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        result = ToolResult(
            content=f"Invalid arguments for {tool_name}: {message}",
            is_error=True,
            data={"validation_errors": message, "received": arguments},
        )
        async with session_scope() as session:
            await commit_effect(
                session,
                tool_call_id=claim.tool_call_id,
                result={"content": result.content, "data": result.data},
                is_error=True,
            )
        log.info("tool_args_invalid", tool=tool_name, errors=message)
        return await _finish(ctx, tool_use_id, result)

    tool_ctx = ToolContext(
        run_id=ctx.run_id,
        step_id=ctx.step_id,
        idempotency_key=claim.idempotency_key,
        attempt=ctx.lease.attempt,
        worker_id=ctx.worker_id,
    )

    await faults.apply(fault, phase="before_effect", attempt=ctx.lease.attempt, lease_key=lease_key)

    # Phase 2: EXECUTE.
    try:
        with child_span(
            f"tool.{tool_name}",
            {
                "agentorc.tool": tool_name,
                "agentorc.effect_policy": str(tool.effect_policy),
                "agentorc.idempotency_key": claim.idempotency_key[:16],
                "agentorc.attempt": ctx.lease.attempt,
            },
        ):
            async with asyncio.timeout(tool.timeout_seconds):
                result = await tool.execute(tool_ctx, args)
    except TimeoutError as exc:
        metrics.TOOL_CALLS.labels(tool=tool_name, outcome="timeout").inc()
        # A timeout is exactly the ambiguous case: the effect may or may not have
        # landed. The ledger row stays `in_flight`, so a retry resolves it by policy
        # instead of blindly re-sending.
        raise RetryableError(
            f"tool {tool_name!r} timed out after {tool.timeout_seconds}s",
            code=ErrorCode.TIMEOUT,
        ) from exc

    # The crash window: the effect has happened, the result has not been recorded.
    await faults.apply(fault, phase="after_effect", attempt=ctx.lease.attempt, lease_key=lease_key)

    # Phase 3: COMMIT.
    async with session_scope() as session:
        await commit_effect(
            session,
            tool_call_id=claim.tool_call_id,
            result={"content": result.content, "data": result.data},
            is_error=result.is_error,
        )

    metrics.TOOL_CALLS.labels(tool=tool_name, outcome="error" if result.is_error else "ok").inc()
    log.info("tool_call_done", tool=tool_name, is_error=result.is_error)
    return await _finish(ctx, tool_use_id, result)


async def _finish(
    ctx: StepContext,
    tool_use_id: str,
    result: ToolResult,
    *,
    replayed: bool = False,
) -> HandlerResult:
    """Record the result and pick the next step.

    Siblings are chained rather than fanned out: the next unresolved tool_use from
    the same assistant turn becomes the next step, and only when none remain does the
    run go back to the model.
    """
    output = {
        TOOL_USE_ID: tool_use_id,
        TOOL_CONTENT: result.content,
        TOOL_IS_ERROR: result.is_error,
        "data": result.data,
        "replayed": replayed,
    }

    next_sibling = await _next_sibling_tool_use(ctx)
    if next_sibling is not None:
        # The same builder the first call of the turn went through, so a gate cannot
        # be enforced on one sibling and skipped on the next.
        from app.engine.handlers.agent_turn import build_tool_step

        return HandlerResult(output=output, next_step=build_tool_step(ctx, next_sibling))
    return HandlerResult(output=output, next_step=NextStep(kind=StepKind.AGENT_TURN, input={}))


async def _next_sibling_tool_use(ctx: StepContext) -> dict[str, Any] | None:
    """The next tool_use block from this step's parent turn that has no result yet."""
    parent_id = ctx.lease.parent_step_id
    if parent_id is None:
        return None

    async with session_scope() as session:
        parent_output = (
            await session.execute(sa.select(Step.output).where(Step.id == parent_id))
        ).scalar_one_or_none()
        sibling_outputs = (
            (
                await session.execute(
                    sa.select(Step.output).where(
                        Step.parent_step_id == parent_id,
                        Step.kind == StepKind.TOOL_CALL,
                        Step.status.in_(
                            [StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.CANCELLED]
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )

    resolved = {(o or {}).get(TOOL_USE_ID, "") for o in sibling_outputs} - {""}
    # The step being committed right now is not yet SUCCEEDED, so include it.
    resolved.add(str(ctx.input.get("tool_use_id") or ""))

    pending = unresolved_tool_uses((parent_output or {}).get("content") or [], resolved)
    return pending[0] if pending else None


__all__ = ["handle_tool_call"]
