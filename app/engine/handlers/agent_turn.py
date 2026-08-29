"""One model turn: think, then either call a tool or answer.

This handler is the "plan/observe" half of the ReAct loop. It is deliberately *one*
model call and nothing else — no tool execution, no looping. That granularity is what
makes crash recovery worth having: if the worker dies after the model has answered
but before the tool ran, the expensive part is already durable and recovery resumes
at the tool.

The next step is chosen here and created by the executor after this step commits:

    tool_use blocks  ->  a tool_call step for the FIRST unresolved block
    end_turn         ->  a finalize step

Only one tool_call step is spawned at a time. The run-serialization invariant allows
a single non-terminal step per run, so sibling tool calls are chained by the
tool_call handler rather than fanned out. Sequential execution costs latency and buys
the absence of join logic, partial-fan-out recovery, and intra-run races.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import sqlalchemy as sa

from app.config import settings
from app.db import session_scope
from app.domain.errors import ErrorCode, TerminalError
from app.domain.models import AgentDefinition, AgentRun, LLMCall, Step
from app.domain.states import StepKind, StepStatus
from app.engine import faults
from app.engine.messages import TOOL_USE_ID, build_messages, unresolved_tool_uses
from app.engine.types import HandlerResult, NextStep, StepContext, register
from app.llm.base import LLMResponse
from app.llm.factory import default_provider
from app.llm.pricing import PRICE_VERSION, cost_of, estimate_max_cost
from app.obs import metrics
from app.obs.logging import get_logger
from app.obs.tracing import child_span
from app.tools.registry import default_registry

log = get_logger("handler.agent_turn")

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful task-executing agent. Work step by step and use the tools "
    "provided rather than guessing. When you have enough information to answer, "
    "give the final answer directly and stop calling tools. If a tool returns an "
    "error, read it and adjust rather than repeating the same call."
)


@register(StepKind.AGENT_TURN)
async def handle_agent_turn(ctx: StepContext) -> HandlerResult:
    async with session_scope() as session:
        definition = (
            await session.execute(
                sa.select(AgentDefinition).where(AgentDefinition.id == ctx.run.agent_definition_id)
            )
        ).scalar_one()
        # Bounded to steps before this one, so the turn never sees its own result.
        messages = await build_messages(session, run=ctx.run, up_to_seq=ctx.lease.seq)

    registry = default_registry()
    system = definition.system_prompt or DEFAULT_SYSTEM_PROMPT
    model = definition.model if definition.model != "none" else settings.llm_model

    await _check_budget(ctx, model=model)
    await _check_consecutive_invalid_tool_calls(ctx)

    # One step from the cap, the model is called with no tools at all. It cannot then
    # request work the run has no budget left to execute, so it answers with what it
    # has instead of ending the run on an unusable tool call.
    tools = registry.schemas(list(definition.tools) or None)
    if settings.reserve_final_answer_step and ctx.run.steps_used >= ctx.run.max_steps - 2:
        log.info(
            "final_answer_forced",
            steps_used=ctx.run.steps_used,
            max_steps=ctx.run.max_steps,
        )
        tools = []
        system = (
            system + "\n\nYou have no tool calls remaining. Answer now using what you "
            "already know, and say plainly if the answer is incomplete."
        )

    provider = default_provider()
    with child_span(
        "llm.complete",
        {
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": model,
            "gen_ai.request.max_tokens": settings.llm_max_tokens,
            "agentorc.tool_count": len(tools),
            "agentorc.message_count": len(messages),
        },
    ) as span:
        response = await provider.complete(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=settings.llm_max_tokens,
            effort=settings.llm_effort,
        )
        span.set_attribute("gen_ai.response.finish_reason", response.stop_reason)
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)

    cost = await _record_llm_call(ctx, response, model=model)
    metrics.observe_llm_call(response.model or model, response.usage, cost, response.latency_ms)

    if response.refused:
        # A policy refusal is not a bug and not retryable: the same request will be
        # refused again. Fail with the category so the run's error explains itself.
        raise TerminalError(
            f"model refused to continue: {(response.stop_details or {}).get('category')}",
            code=ErrorCode.MODEL_OUTPUT_INVALID,
            details={"stop_details": response.stop_details},
        )

    if response.stop_reason == "max_tokens":
        # The turn was cut mid-generation, so any tool_use block in it may be
        # truncated. Acting on half a tool call is worse than stopping.
        raise TerminalError(
            "model response hit max_tokens and was truncated; raise llm_max_tokens",
            code=ErrorCode.MODEL_OUTPUT_INVALID,
            details={"max_tokens": settings.llm_max_tokens},
        )

    output: dict[str, Any] = {
        # Raw blocks, replayed verbatim next turn — thinking blocks included.
        "content": response.content,
        "text": response.text,
        "stop_reason": response.stop_reason,
        "model": response.model,
        "tool_uses": [{"id": t.id, "name": t.name, "input": t.input} for t in response.tool_uses],
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_tokens": response.usage.cache_read_tokens,
            "cache_write_tokens": response.usage.cache_write_tokens,
        },
        "cost_usd": str(cost),
        "latency_ms": response.latency_ms,
    }

    if not response.tool_uses:
        log.info("agent_turn_final", stop_reason=response.stop_reason, chars=len(response.text))
        return HandlerResult(
            output=output,
            next_step=NextStep(kind=StepKind.FINALIZE, input={"answer": response.text}),
        )

    # On a normal turn nothing is resolved yet; after a recovery some sibling tool
    # calls may already have committed, and re-spawning them would redo their effects.
    resolved = await _resolved_tool_use_ids(ctx)
    pending = unresolved_tool_uses(response.content, resolved)
    if not pending:
        return HandlerResult(output=output, next_step=NextStep(kind=StepKind.AGENT_TURN, input={}))

    first = pending[0]
    log.info(
        "agent_turn_tools",
        requested=len(response.tool_uses),
        pending=len(pending),
        next_tool=first.get("name"),
    )
    return HandlerResult(output=output, next_step=build_tool_step(ctx, first))


def build_tool_step(ctx: StepContext, tool_use: dict[str, Any]) -> NextStep:
    """Turn a tool_use block into the next step, gated when the call warrants it.

    Shared with the tool_call handler so a gate cannot be enforced on the first call
    of a turn and silently skipped on its siblings.
    """
    name = str(tool_use.get("name") or "")
    arguments = tool_use.get("input") or {}

    approval_reason: str | None = None
    max_attempts: int | None = None
    registry = default_registry()
    if name in registry:
        tool = registry.get(name)
        # Data-dependent: the tool inspects the actual arguments, so `database_write`
        # can gate on the target namespace rather than on being a write.
        approval_reason = tool.approval_reason(arguments)
        max_attempts = tool.max_attempts

    # A tool_call step chains siblings from the same assistant turn, so the parent is
    # this step's parent rather than this step.
    parent = ctx.lease.parent_step_id if ctx.lease.kind is StepKind.TOOL_CALL else ctx.step_id

    return NextStep(
        kind=StepKind.TOOL_CALL,
        input={
            "tool_use_id": tool_use.get("id"),
            "tool_name": name,
            "arguments": arguments,
            "fault": faults.fault_for_tool(ctx.run.input, name),
        },
        parent_step_id=parent,
        max_attempts=max_attempts,
        approval_reason=approval_reason,
        tool_name=name,
        arguments=arguments,
    )


async def _check_budget(ctx: StepContext, *, model: str) -> None:
    """Stop before spending past the run's cap.

    Checked before the call rather than enforced during it, so the final call can
    overshoot by at most `max_tokens` of output. Hard enforcement would require
    streaming with a mid-stream abort; the overshoot is bounded and cheap, and the
    alternative is a budget that does nothing until after the money is spent.
    """
    async with session_scope() as session:
        row = (
            await session.execute(
                sa.select(AgentRun.cost_usd, AgentRun.max_cost_usd).where(AgentRun.id == ctx.run_id)
            )
        ).one()

    if row.cost_usd >= row.max_cost_usd:
        raise TerminalError(
            f"run exhausted its budget of ${row.max_cost_usd} (spent ${row.cost_usd})",
            code=ErrorCode.BUDGET_EXCEEDED,
            details={
                "cost_usd": str(row.cost_usd),
                "max_cost_usd": str(row.max_cost_usd),
                "worst_case_next_call_usd": str(
                    estimate_max_cost(model, input_tokens=0, max_tokens=settings.llm_max_tokens)
                ),
            },
        )


async def _check_consecutive_invalid_tool_calls(ctx: StepContext) -> None:
    """A model that keeps emitting unusable tool calls is looping, not recovering.

    Errored tool results are fed back deliberately so the model can self-correct, but
    self-correction that never converges is only an expensive route to `max_steps`.
    """
    cap = settings.max_consecutive_invalid_tool_calls
    async with session_scope() as session:
        recent = (
            (
                await session.execute(
                    sa.select(Step.output)
                    .where(
                        Step.run_id == ctx.run_id,
                        Step.kind == StepKind.TOOL_CALL,
                        Step.status.in_([StepStatus.SUCCEEDED, StepStatus.FAILED]),
                    )
                    .order_by(Step.seq.desc())
                    .limit(cap)
                )
            )
            .scalars()
            .all()
        )

    if len(recent) >= cap and all((row or {}).get("is_error") for row in recent):
        raise TerminalError(
            f"model produced {cap} unusable tool calls in a row without recovering",
            code=ErrorCode.MODEL_OUTPUT_INVALID,
            details={"consecutive_invalid": cap},
        )


async def _resolved_tool_use_ids(ctx: StepContext) -> set[str]:
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    sa.select(Step.output).where(
                        Step.run_id == ctx.run_id,
                        Step.parent_step_id == ctx.step_id,
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
    return {(row or {}).get(TOOL_USE_ID, "") for row in rows} - {""}


async def _record_llm_call(ctx: StepContext, response: LLMResponse, *, model: str) -> Decimal:
    """Persist the token/cost ledger row and increment the run's rollups.

    Both happen in one transaction so the denormalised rollup on `agent_runs` cannot
    drift from the sum of its ledger. The eval harness asserts that invariant.
    """
    cost = cost_of(response.model or model, response.usage)
    async with session_scope() as session:
        session.add(
            LLMCall(
                step_id=ctx.step_id,
                run_id=ctx.run_id,
                model=response.model or model,
                request=response.raw_request,
                response=response.raw_response,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_tokens,
                cache_write_tokens=response.usage.cache_write_tokens,
                cost_usd=cost,
                price_version=PRICE_VERSION,
                latency_ms=response.latency_ms,
            )
        )
        await session.execute(
            sa.update(AgentRun)
            .where(AgentRun.id == ctx.run_id)
            .values(
                input_tokens=AgentRun.input_tokens + response.usage.input_tokens,
                output_tokens=AgentRun.output_tokens + response.usage.output_tokens,
                cache_read_tokens=AgentRun.cache_read_tokens + response.usage.cache_read_tokens,
                cache_write_tokens=AgentRun.cache_write_tokens + response.usage.cache_write_tokens,
                cost_usd=AgentRun.cost_usd + cost,
                updated_at=sa.func.now(),
            )
        )
    return cost


__all__ = ["DEFAULT_SYSTEM_PROMPT", "handle_agent_turn"]
