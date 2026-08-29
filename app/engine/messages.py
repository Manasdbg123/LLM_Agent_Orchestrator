"""Rebuild the model transcript from persisted steps.

There is no conversation table. The `messages` array is derived from the step
timeline every turn, which means it cannot drift from the state machine and a worker
that picks up a recovered run reconstructs exactly the context its predecessor had.

Two rules here are protocol requirements rather than style choices:

1. **Assistant content is replayed verbatim.** The stored blocks — including thinking
   blocks — go back unchanged. Reducing an assistant turn to its text would drop the
   thinking blocks, and on the current models those must be echoed back intact for
   the model to continue its own reasoning.

2. **All tool results for one assistant turn go in a single user message.** Splitting
   them across several messages is accepted by the API but teaches the model to stop
   emitting parallel tool calls, so it degrades behaviour silently rather than
   failing loudly.
"""

from __future__ import annotations

import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import AgentRun, Step
from app.domain.states import StepKind, StepStatus

#: Keys of a stored tool_call step output.
TOOL_USE_ID = "tool_use_id"
TOOL_CONTENT = "content"
TOOL_IS_ERROR = "is_error"


def tool_result_block(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": output.get(TOOL_USE_ID, ""),
        "content": output.get(TOOL_CONTENT, ""),
        "is_error": bool(output.get(TOOL_IS_ERROR, False)),
    }


async def build_messages(
    session: AsyncSession, *, run: AgentRun, up_to_seq: int | None = None
) -> list[dict[str, Any]]:
    """Derive the transcript for the next model call.

    `up_to_seq` bounds the rebuild to steps strictly before the step being executed,
    so a step never sees its own (not yet existent) result.
    """
    stmt = (
        sa.select(Step)
        .where(Step.run_id == run.id, Step.kind.in_([StepKind.AGENT_TURN, StepKind.TOOL_CALL]))
        .order_by(Step.seq)
    )
    if up_to_seq is not None:
        stmt = stmt.where(Step.seq < up_to_seq)
    steps = list((await session.execute(stmt)).scalars().all())

    messages: list[dict[str, Any]] = [{"role": "user", "content": run.task}]

    # Index tool_call steps by the agent_turn that requested them, so results can be
    # emitted as one grouped user message per assistant turn.
    results_by_parent: dict[uuid.UUID, list[Step]] = {}
    for step in steps:
        if step.kind == StepKind.TOOL_CALL and step.parent_step_id is not None:
            results_by_parent.setdefault(step.parent_step_id, []).append(step)

    for step in steps:
        if step.kind != StepKind.AGENT_TURN or step.status != StepStatus.SUCCEEDED:
            continue

        content = (step.output or {}).get("content") or []
        if not content:
            continue
        messages.append({"role": "assistant", "content": content})

        children = sorted(results_by_parent.get(step.id, []), key=lambda s: s.seq)
        blocks = [
            tool_result_block(child.output or {})
            for child in children
            if child.status in (StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.CANCELLED)
            and (child.output or {}).get(TOOL_USE_ID)
        ]
        if blocks:
            messages.append({"role": "user", "content": blocks})

    return messages


def unresolved_tool_uses(
    assistant_content: list[dict[str, Any]], resolved_ids: set[str]
) -> list[dict[str, Any]]:
    """tool_use blocks from an assistant turn that still have no result.

    Used to decide what work an agent_turn must spawn, and — on recovery — to avoid
    re-spawning tool calls that already completed.
    """
    return [
        block
        for block in assistant_content
        if block.get("type") == "tool_use" and block.get("id") not in resolved_ids
    ]


__all__ = [
    "TOOL_CONTENT",
    "TOOL_IS_ERROR",
    "TOOL_USE_ID",
    "build_messages",
    "tool_result_block",
    "unresolved_tool_uses",
]
