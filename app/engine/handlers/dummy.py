"""The Phase 2 step: deterministic, instrumented, no LLM.

Durability is proven here, before any agent code exists. If crash recovery is not
demonstrable on a step this simple, it will not become more demonstrable once there
is a model in the loop — it will just become harder to test.

The "effect" this step performs is a row in `dummy_effects` with an execution
counter. That turns the central claim into an assertion on an integer:

    completed steps were not re-executed  ->  executions == 1

The effect is committed in its *own* transaction, before the step result is
committed. That gap is deliberate: it is exactly the window in which a crash leaves
an effect that happened and a result that did not, which is the situation the Phase
3/4 idempotency ledger exists to resolve. In Phase 2 the dummy effect is knowingly
non-idempotent, so the chaos test can observe the duplicate and show what the ledger
will have to prevent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import session_scope
from app.domain.errors import ErrorCode, TerminalError
from app.domain.models import DummyEffect
from app.domain.states import StepKind
from app.engine import faults
from app.engine.types import HandlerResult, NextStep, StepContext, register
from app.obs.logging import get_logger

log = get_logger("handler.dummy")


async def _record_effect(ctx: StepContext, label: str) -> int:
    """Perform the (deliberately non-idempotent) side effect. Returns its count."""
    async with session_scope() as session:
        stmt = (
            pg_insert(DummyEffect)
            .values(
                step_id=ctx.step_id,
                run_id=ctx.run_id,
                label=label,
                executions=1,
                last_worker=ctx.worker_id,
            )
            .on_conflict_do_update(
                index_elements=[DummyEffect.step_id],
                set_={
                    "executions": DummyEffect.executions + 1,
                    "last_worker": ctx.worker_id,
                    "last_started_at": sa.func.now(),
                },
            )
            .returning(DummyEffect.executions)
        )
        return int((await session.execute(stmt)).scalar_one())


def _plan(ctx: StepContext) -> list[dict[str, Any]]:
    plan = ctx.run.input.get("plan") or []
    if not isinstance(plan, list):
        raise TerminalError("run input 'plan' must be a list", code=ErrorCode.INVALID_INPUT)
    return plan


@register(StepKind.DUMMY)
async def handle_dummy(ctx: StepContext) -> HandlerResult:
    spec = ctx.input
    label = str(spec.get("label", f"step-{ctx.lease.seq}"))
    fault = faults.parse(spec.get("fault"))
    lease_key = faults.make_lease_key(ctx.step_id, ctx.lease.epoch)

    await faults.apply(fault, phase="before_effect", attempt=ctx.lease.attempt, lease_key=lease_key)

    executions = await _record_effect(ctx, label)
    log.info("dummy_effect_recorded", label=label, executions=executions, seq=ctx.lease.seq)

    # Committed effect, uncommitted result: the crash window.
    await faults.apply(fault, phase="after_effect", attempt=ctx.lease.attempt, lease_key=lease_key)

    if sleep_ms := int(spec.get("sleep_ms", 0)):
        await asyncio.sleep(sleep_ms / 1000)

    await faults.apply(fault, phase="before_commit", attempt=ctx.lease.attempt, lease_key=lease_key)

    output = {
        "label": label,
        "executions": executions,
        "attempt": ctx.lease.attempt,
        "worker_id": ctx.worker_id,
        "echo": spec.get("echo"),
    }

    # The plan is positional: plan[i] is the step at seq i+1. `lease.seq` is therefore
    # the index of the *next* entry.
    plan = _plan(ctx)
    next_index = ctx.lease.seq
    if next_index < len(plan):
        return HandlerResult(
            output=output,
            next_step=NextStep(kind=StepKind.DUMMY, input=plan[next_index]),
        )
    return HandlerResult(output=output, next_step=NextStep(kind=StepKind.FINALIZE, input={}))


__all__ = ["handle_dummy"]
