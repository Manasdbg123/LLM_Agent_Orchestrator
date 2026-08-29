"""Claim -> execute -> commit.

The transaction boundaries here are the whole design, so they are worth stating
plainly:

  txn A   claim the lease, load the run, start the run if this is its first step
  (none)  run the handler, with a heartbeat task extending the lease beside it
  txn B   commit the outcome: step status, run rollups, the next step

The handler runs *outside* a transaction. A handler executing inside txn B would hold
a write transaction open for the entire duration of an LLM call or an HTTP request,
which at any real concurrency turns the database into the bottleneck and makes long
steps indistinguishable from lock contention.

The gap between the handler's own side effect and txn B is the crash window. It
cannot be closed — an external effect and a local transaction cannot be made atomic
— so instead it is made *recoverable*: the effect ledger (Phase 3) records the effect
before it happens, and the lease guarantees only one executor is ever in that window
for a given step.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import sqlalchemy as sa

from app.config import settings
from app.core import leases as lease_ops
from app.core.retry import RetryAction, RetryPolicy, decide
from app.core.runs import create_next_step, fail_run
from app.core.transitions import transition_run, transition_step
from app.db import session_scope
from app.domain.errors import ErrorCode, LeaseLostError
from app.domain.models import AgentRun
from app.domain.states import RunStatus, StepStatus, TransitionReason
from app.engine import faults

# Importing the handler modules is what populates the registry.
from app.engine.handlers import agent_turn as _agent_turn  # noqa: F401
from app.engine.handlers import dummy as _dummy  # noqa: F401
from app.engine.handlers import finalize as _finalize  # noqa: F401
from app.engine.handlers import tool_call as _tool_call  # noqa: F401
from app.engine.types import HandlerResult, StepContext, get_handler
from app.obs import logging as obslog
from app.obs import metrics, tracing
from app.queue.base import StepMessage, StepQueue

log = obslog.get_logger("executor")


class Outcome(StrEnum):
    NOT_CLAIMED = "not_claimed"
    SUCCEEDED = "succeeded"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: This worker lost its lease mid-step; another worker owns the step now.
    FENCED = "fenced"


@dataclass(slots=True)
class ExecutionReport:
    outcome: Outcome
    step_id: uuid.UUID
    run_id: uuid.UUID
    attempt: int = 0
    #: Set when a follow-on step was created and should be published.
    next_step_id: uuid.UUID | None = None
    detail: str | None = None


class StepExecutor:
    def __init__(self, queue: StepQueue, worker_id: str) -> None:
        self.queue = queue
        self.worker_id = worker_id

    async def execute(self, message: StepMessage) -> ExecutionReport:
        # The run's trace context is read *before* the claim so the span is already
        # open when the claim writes its audit row. Otherwise the `claimed`
        # transition — the one that records which worker took the step, and the row
        # that makes a crash and its recovery legible as one trace — is the only one
        # with no trace id on it.
        traceparent = await self._run_traceparent(message.run_id)
        started = time.perf_counter()

        with tracing.step_span(
            name="step",
            parent_traceparent=traceparent,
            attributes={
                "agentorc.run_id": str(message.run_id),
                "agentorc.step_id": str(message.step_id),
                "agentorc.worker_id": self.worker_id,
            },
        ) as span:
            try:
                claimed = await self._claim(message)
            except Exception:
                log.exception("claim_failed", step_id=str(message.step_id))
                raise

            if claimed is None:
                # Normal under duplicate delivery: someone else owns it.
                span.set_attribute("agentorc.claimed", False)
                return ExecutionReport(
                    Outcome.NOT_CLAIMED,
                    message.step_id,
                    message.run_id,
                    detail="not_claimable",
                )
            lease, run, precondition = claimed

            span.set_attribute("agentorc.claimed", True)
            span.set_attribute("agentorc.step_kind", str(lease.kind))
            span.set_attribute("agentorc.step_seq", lease.seq)
            span.set_attribute("agentorc.attempt", lease.attempt)
            span.update_name(f"step.{lease.kind}")

            obslog.bind(
                run_id=str(lease.run_id),
                step_id=str(lease.step_id),
                attempt=lease.attempt,
                worker_id=self.worker_id,
                step_kind=str(lease.kind),
            )
            try:
                if precondition is not None:
                    return await self._abort(lease, precondition)

                ctx = StepContext(run=run, lease=lease, worker_id=self.worker_id)
                try:
                    result = await self._run_handler(ctx)
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        tracing.record_exception(span, exc)
                    report = await self._commit_failure(lease, run, exc)
                    self._observe(lease, report, started)
                    return report
                report = await self._commit_success(lease, run, result)
                self._observe(lease, report, started)
                return report
            finally:
                span.set_attribute(
                    "agentorc.duration_ms", int((time.perf_counter() - started) * 1000)
                )
                obslog.clear()

    async def _run_traceparent(self, run_id: uuid.UUID) -> str | None:
        """The run's persisted trace context.

        One extra cheap read per delivered message, which buys complete trace
        coverage: without it the claim is outside the span and the handoff between a
        dead worker and its replacement is invisible.
        """
        async with session_scope() as session:
            return (
                await session.execute(sa.select(AgentRun.traceparent).where(AgentRun.id == run_id))
            ).scalar_one_or_none()

    def _observe(self, lease: lease_ops.Lease, report: ExecutionReport, started: float) -> None:
        duration = time.perf_counter() - started
        metrics.observe_step_finished(str(lease.kind), str(report.outcome), duration)
        if report.outcome is Outcome.FENCED:
            metrics.FENCED_WRITES.inc()

    # --- txn A ------------------------------------------------------------------

    async def _claim(
        self, message: StepMessage
    ) -> tuple[lease_ops.Lease, AgentRun, str | None] | None:
        async with session_scope() as session:
            lease = await lease_ops.claim_step(
                session,
                step_id=message.step_id,
                worker_id=self.worker_id,
                ttl_seconds=settings.lease_ttl_seconds,
            )
            if lease is None:
                return None

            run = (
                await session.execute(sa.select(AgentRun).where(AgentRun.id == lease.run_id))
            ).scalar_one()

            # Checked after the claim rather than before: holding the lease means no
            # one else can act on this step while we decide what to do with it.
            precondition: str | None = None
            if run.cancel_requested:
                precondition = "cancelled"
            elif RunStatus(run.status) in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                precondition = "run_terminal"

            if precondition is None and RunStatus(run.status) is RunStatus.PENDING:
                await transition_run(
                    session,
                    run_id=run.id,
                    to=RunStatus.RUNNING,
                    reason=str(TransitionReason.RUN_STARTED),
                    actor=self.worker_id,
                    expect=RunStatus.PENDING,
                )
            return lease, run, precondition

    # --- handler + heartbeat ----------------------------------------------------

    async def _run_handler(self, ctx: StepContext) -> HandlerResult:
        handler = get_handler(ctx.lease.kind)
        lease_lost = asyncio.Event()

        handler_task: asyncio.Task[HandlerResult] = asyncio.create_task(
            handler(ctx), name=f"step-{ctx.step_id}"
        )
        hb_task = asyncio.create_task(
            self._heartbeat_loop(ctx.lease, handler_task, lease_lost),
            name=f"hb-{ctx.step_id}",
        )
        try:
            return await handler_task
        except asyncio.CancelledError:
            # Distinguish "we cancelled it because the lease died" from a genuine
            # outside cancellation (process shutdown). Getting this wrong would mark
            # a step failed when the worker was merely being shut down.
            if lease_lost.is_set():
                raise LeaseLostError(
                    "lease lost while executing; another worker now owns this step",
                    code=ErrorCode.LEASE_LOST,
                ) from None
            raise
        finally:
            hb_task.cancel()
            with_suppressed = asyncio.gather(hb_task, return_exceptions=True)
            await with_suppressed

    async def _heartbeat_loop(
        self,
        lease: lease_ops.Lease,
        handler_task: asyncio.Task[Any],
        lease_lost: asyncio.Event,
    ) -> None:
        interval = settings.heartbeat_interval_seconds
        while True:
            await asyncio.sleep(interval)

            if faults.heartbeat_suppressed(faults.make_lease_key(lease.step_id, lease.epoch)):
                # Test affordance: simulate a stalled worker whose lease expires
                # under it while the process itself stays alive.
                continue

            try:
                async with session_scope() as session:
                    alive = await lease_ops.heartbeat(
                        session,
                        step_id=lease.step_id,
                        worker_id=self.worker_id,
                        epoch=lease.epoch,
                        ttl_seconds=settings.lease_ttl_seconds,
                    )
                    cancelled = (
                        await session.execute(
                            sa.select(AgentRun.cancel_requested).where(AgentRun.id == lease.run_id)
                        )
                    ).scalar_one_or_none()
            except Exception as exc:  # database unreachable
                # We can no longer prove we own the lease. Continuing to execute
                # unowned work is exactly how double execution happens, so stop.
                log.warning("heartbeat_error", error=str(exc))
                lease_lost.set()
                handler_task.cancel()
                return

            if not alive:
                log.warning("lease_lost", step_id=str(lease.step_id), epoch=lease.epoch)
                lease_lost.set()
                handler_task.cancel()
                return

            if cancelled:
                log.info("cancel_observed", step_id=str(lease.step_id))
                handler_task.cancel()
                return

    # --- txn B ------------------------------------------------------------------

    async def _commit_success(
        self, lease: lease_ops.Lease, run: AgentRun, result: HandlerResult
    ) -> ExecutionReport:
        async with session_scope() as session:
            applied = await transition_step(
                session,
                step_id=lease.step_id,
                to=StepStatus.SUCCEEDED,
                reason=str(TransitionReason.COMPLETED),
                actor=self.worker_id,
                expect=StepStatus.RUNNING,
                require_owner=self.worker_id,
                require_epoch=lease.epoch,
                values={
                    "output": result.output,
                    "ended_at": sa.func.now(),
                    "lease_owner": None,
                    "lease_expires_at": None,
                },
            )
            if not applied:
                log.warning(
                    "commit_rejected",
                    blocked_by=applied.blocked_by,
                    step_id=str(lease.step_id),
                )
                return ExecutionReport(
                    Outcome.FENCED, lease.step_id, lease.run_id, detail=applied.blocked_by
                )

            await session.execute(
                sa.update(AgentRun)
                .where(AgentRun.id == lease.run_id)
                .values(steps_used=AgentRun.steps_used + 1, updated_at=sa.func.now())
            )

            next_step_id: uuid.UUID | None = None
            if result.run_output is not None:
                await transition_run(
                    session,
                    run_id=lease.run_id,
                    to=RunStatus.SUCCEEDED,
                    reason=str(TransitionReason.COMPLETED),
                    actor=self.worker_id,
                    values={"output": result.run_output},
                )
            elif result.next_step is not None:
                # Re-read the run so guardrail checks see the rollup we just wrote.
                fresh = (
                    await session.execute(sa.select(AgentRun).where(AgentRun.id == lease.run_id))
                ).scalar_one()
                nxt = await create_next_step(
                    session,
                    run=fresh,
                    kind=result.next_step.kind,
                    input=result.next_step.input,
                    parent_step_id=result.next_step.parent_step_id,
                    delay_seconds=result.next_step.delay_seconds,
                    actor=self.worker_id,
                    max_attempts=result.next_step.max_attempts,
                    approval_reason=result.next_step.approval_reason,
                    tool_name=result.next_step.tool_name,
                    arguments=result.next_step.arguments,
                )
                # A gated step must not be published: it waits for a human, and an
                # enqueued id would invite a worker to claim it first.
                if nxt is not None and nxt.status == StepStatus.PENDING:
                    next_step_id = nxt.id

        # Published only after the commit: publishing inside the transaction could
        # hand a worker an id that then rolls back.
        if next_step_id is not None:
            await self._publish(next_step_id, lease.run_id)

        return ExecutionReport(
            Outcome.SUCCEEDED,
            lease.step_id,
            lease.run_id,
            attempt=lease.attempt,
            next_step_id=next_step_id,
        )

    async def _commit_failure(
        self, lease: lease_ops.Lease, run: AgentRun, exc: BaseException
    ) -> ExecutionReport:
        if isinstance(exc, LeaseLostError):
            # Another worker owns the step. Discard the result; do not write.
            log.warning("fenced", step_id=str(lease.step_id), epoch=lease.epoch)
            return ExecutionReport(Outcome.FENCED, lease.step_id, lease.run_id, detail="lease_lost")

        if isinstance(exc, asyncio.CancelledError):
            # Re-read the flag rather than trusting `run`, which was loaded when the
            # step was claimed and therefore predates any cancellation. Using the
            # stale copy made every cancel look like a process shutdown, so the step
            # was abandoned to lease expiry instead of being cancelled promptly.
            async with session_scope() as session:
                cancel_requested = (
                    await session.execute(
                        sa.select(AgentRun.cancel_requested).where(AgentRun.id == lease.run_id)
                    )
                ).scalar_one_or_none()
            if cancel_requested:
                return await self._abort(lease, "cancelled")
            # Process shutdown: leave the step RUNNING and let its lease expire. The
            # reaper reassigns it. Marking it failed here would turn a routine deploy
            # into a run failure.
            log.info("step_abandoned_on_shutdown", step_id=str(lease.step_id))
            raise exc

        # `max_attempts` lives on the step, set from the tool's declaration when the
        # step was created, so a tool that talks to a flaky network can be given a
        # different budget from one that cannot fail transiently.
        policy = dataclasses.replace(RetryPolicy.default(), max_attempts=lease.max_attempts)
        decision = decide(exc, attempt=lease.attempt, policy=policy)

        if decision.action is RetryAction.RETRY:
            async with session_scope() as session:
                applied = await transition_step(
                    session,
                    step_id=lease.step_id,
                    to=StepStatus.RETRYING,
                    reason=str(TransitionReason.RETRYABLE_ERROR),
                    actor=self.worker_id,
                    expect=StepStatus.RUNNING,
                    require_owner=self.worker_id,
                    require_epoch=lease.epoch,
                    values={
                        "error": decision.error,
                        "lease_owner": None,
                        "lease_expires_at": None,
                        # Backoff is a timestamp, not a sleep: it survives a worker
                        # restart and does not occupy a worker slot while it elapses.
                        "available_at": sa.func.now()
                        + dt.timedelta(seconds=decision.delay_seconds),
                    },
                    details={"delay_seconds": round(decision.delay_seconds, 3)},
                )
            if not applied:
                return ExecutionReport(
                    Outcome.FENCED, lease.step_id, lease.run_id, detail=applied.blocked_by
                )
            log.info(
                "step_retry_scheduled",
                delay_seconds=round(decision.delay_seconds, 3),
                attempt=lease.attempt,
                error_code=str(decision.code),
            )
            return ExecutionReport(
                Outcome.RETRY_SCHEDULED,
                lease.step_id,
                lease.run_id,
                attempt=lease.attempt,
                detail=str(decision.code),
            )

        reason = (
            TransitionReason.ATTEMPTS_EXHAUSTED
            if decision.code is ErrorCode.ATTEMPTS_EXHAUSTED
            else TransitionReason.TERMINAL_ERROR
        )
        async with session_scope() as session:
            applied = await transition_step(
                session,
                step_id=lease.step_id,
                to=StepStatus.FAILED,
                reason=str(reason),
                actor=self.worker_id,
                expect=StepStatus.RUNNING,
                require_owner=self.worker_id,
                require_epoch=lease.epoch,
                values={
                    "error": decision.error,
                    "ended_at": sa.func.now(),
                    "lease_owner": None,
                    "lease_expires_at": None,
                },
            )
            if not applied:
                return ExecutionReport(
                    Outcome.FENCED, lease.step_id, lease.run_id, detail=applied.blocked_by
                )
            await fail_run(
                session,
                run_id=lease.run_id,
                code=decision.code,
                message=str(decision.error.get("message", "step failed")),
                reason=str(TransitionReason.STEP_FAILED),
                actor=self.worker_id,
                details={"step_id": str(lease.step_id), "seq": lease.seq},
            )
        log.warning("step_failed", error_code=str(decision.code), attempt=lease.attempt)
        return ExecutionReport(
            Outcome.FAILED,
            lease.step_id,
            lease.run_id,
            attempt=lease.attempt,
            detail=str(decision.code),
        )

    async def _abort(self, lease: lease_ops.Lease, precondition: str) -> ExecutionReport:
        """Release a claimed step whose run should no longer proceed."""
        target = StepStatus.CANCELLED if precondition == "cancelled" else StepStatus.FAILED
        async with session_scope() as session:
            await transition_step(
                session,
                step_id=lease.step_id,
                to=target,
                reason=str(TransitionReason.CANCELLED),
                actor=self.worker_id,
                expect=StepStatus.RUNNING,
                require_owner=self.worker_id,
                require_epoch=lease.epoch,
                values={
                    "ended_at": sa.func.now(),
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "error": {
                        "code": str(ErrorCode.CANCELLED),
                        "class": "terminal",
                        "message": f"step aborted: {precondition}",
                    },
                },
            )
            if precondition == "cancelled":
                await transition_run(
                    session,
                    run_id=lease.run_id,
                    to=RunStatus.CANCELLED,
                    reason=str(TransitionReason.CANCELLED),
                    actor=self.worker_id,
                )
        return ExecutionReport(Outcome.CANCELLED, lease.step_id, lease.run_id, detail=precondition)

    async def _publish(self, step_id: uuid.UUID, run_id: uuid.UUID) -> None:
        try:
            await self.queue.publish(step_id=step_id, run_id=run_id)
        except Exception as exc:
            # Non-fatal by design: the step row is committed, so the reaper will
            # dispatch it within `enqueue_grace`. A queue outage costs latency here,
            # never work.
            log.warning("publish_failed", step_id=str(step_id), error=str(exc))


__all__ = ["ExecutionReport", "Outcome", "StepExecutor"]
