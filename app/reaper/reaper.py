"""The reaper: the component that makes crashes survivable.

Three sweeps on a fixed interval, all driven by the same source of truth:

  1. expired leases   -> hand orphaned steps back to the pool (crash recovery)
  2. runnable steps   -> re-publish anything the queue lost or never delivered,
                         and dispatch retries whose backoff has elapsed
  3. run deadlines    -> fail runs that ran out of wall clock
  4. stale approvals  -> fail runs nobody answered in time

Sweep 2 is the one that makes Redis optional. It is the same predicate a worker's
claim uses, so any step that *should* be running but is not gets picked up here
regardless of what the queue does or does not remember.

Safe to run several replicas: every action it takes is a guarded transition, so two
reapers racing produce one effect and one no-op.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from dataclasses import dataclass, field

import sqlalchemy as sa

from app.config import settings
from app.core import leases as lease_ops
from app.core.approvals import expire_stale
from app.core.runs import fail_run
from app.core.transitions import transition_step
from app.db import dispose_engine, session_scope
from app.domain.errors import ErrorCode
from app.domain.models import AgentRun, Step
from app.domain.states import RunStatus, StepStatus, TransitionReason
from app.obs.logging import configure_logging, get_logger
from app.queue.base import StepQueue
from app.queue.factory import build_queue

log = get_logger("reaper")

ACTOR = "reaper"


@dataclass(slots=True)
class SweepStats:
    reclaimed: int = 0
    abandoned: int = 0
    republished: int = 0
    deadlined: int = 0
    approvals_expired: int = 0
    by_status: dict[str, int] = field(default_factory=dict)

    def merge(self, other: SweepStats) -> None:
        self.reclaimed += other.reclaimed
        self.abandoned += other.abandoned
        self.republished += other.republished
        self.deadlined += other.deadlined
        self.approvals_expired += other.approvals_expired


class Reaper:
    def __init__(self, queue: StepQueue | None = None) -> None:
        self.queue = queue or build_queue()
        self._stop = asyncio.Event()
        self.totals = SweepStats()

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        configure_logging()
        await self.queue.setup()
        log.info("reaper_started", interval=settings.reaper_interval_seconds)
        try:
            while not self._stop.is_set():
                try:
                    stats = await self.sweep()
                    self.totals.merge(stats)
                except Exception as exc:
                    # The reaper is the recovery mechanism; it must never be the thing
                    # that dies. Log and keep sweeping.
                    log.exception("sweep_failed", error=str(exc))
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=settings.reaper_interval_seconds
                    )
        finally:
            await self.queue.close()
            log.info("reaper_stopped", totals=self.totals)

    async def sweep(self) -> SweepStats:
        stats = SweepStats()
        await self._reclaim_expired_leases(stats)
        await self._enforce_deadlines(stats)
        await self._expire_approvals(stats)
        await self._republish_runnable(stats)
        if stats.reclaimed or stats.abandoned or stats.deadlined or stats.approvals_expired:
            log.info(
                "sweep",
                reclaimed=stats.reclaimed,
                abandoned=stats.abandoned,
                republished=stats.republished,
                deadlined=stats.deadlined,
                approvals_expired=stats.approvals_expired,
            )
        return stats

    async def _reclaim_expired_leases(self, stats: SweepStats) -> None:
        async with session_scope() as session:
            rows = await lease_ops.find_expired_leases(session, limit=settings.reaper_batch_size)
            for row in rows:
                status = await lease_ops.reclaim_expired_lease(session, row=row, actor=ACTOR)
                if status == str(StepStatus.PENDING):
                    stats.reclaimed += 1
                    log.warning(
                        "lease_reclaimed",
                        step_id=str(row.id),
                        run_id=str(row.run_id),
                        orphaned_owner=row.lease_owner,
                        recoveries=row.recoveries + 1,
                    )
                elif status == str(StepStatus.FAILED):
                    stats.abandoned += 1
                    log.error(
                        "step_abandoned_poison",
                        step_id=str(row.id),
                        run_id=str(row.run_id),
                        recoveries=row.recoveries + 1,
                    )
                    await fail_run(
                        session,
                        run_id=row.run_id,
                        code=ErrorCode.TOO_MANY_RECOVERIES,
                        message="step repeatedly killed its worker",
                        reason=str(TransitionReason.STEP_FAILED),
                        actor=ACTOR,
                        details={"step_id": str(row.id)},
                    )

    async def _enforce_deadlines(self, stats: SweepStats) -> None:
        """Fail runs past their wall clock.

        A step that is currently RUNNING is left alone rather than force-failed: its
        worker may be mid-side-effect. It discovers the run is terminal when it tries
        to extend the run, and its own result is committed harmlessly.
        """
        async with session_scope() as session:
            expired = (
                (
                    await session.execute(
                        sa.select(AgentRun.id)
                        .where(
                            AgentRun.status.in_(
                                [RunStatus.PENDING, RunStatus.RUNNING, RunStatus.AWAITING_APPROVAL]
                            ),
                            AgentRun.deadline_at < sa.func.now(),
                        )
                        .limit(settings.reaper_batch_size)
                        .with_for_update(skip_locked=True)
                    )
                )
                .scalars()
                .all()
            )

            for run_id in expired:
                idle_steps = (
                    (
                        await session.execute(
                            sa.select(Step.id).where(
                                Step.run_id == run_id,
                                Step.status.in_(
                                    [
                                        StepStatus.PENDING,
                                        StepStatus.RETRYING,
                                        StepStatus.AWAITING_APPROVAL,
                                    ]
                                ),
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                for step_id in idle_steps:
                    await transition_step(
                        session,
                        step_id=step_id,
                        to=StepStatus.FAILED,
                        reason=str(TransitionReason.DEADLINE_EXCEEDED),
                        actor=ACTOR,
                        values={
                            "ended_at": sa.func.now(),
                            "error": {
                                "code": str(ErrorCode.DEADLINE_EXCEEDED),
                                "class": "terminal",
                                "message": "run deadline exceeded before this step ran",
                            },
                        },
                    )
                await fail_run(
                    session,
                    run_id=run_id,
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    message="run exceeded its wall-clock deadline",
                    reason=str(TransitionReason.DEADLINE_EXCEEDED),
                    actor=ACTOR,
                )
                stats.deadlined += 1
                log.warning("run_deadline_exceeded", run_id=str(run_id))

    async def _expire_approvals(self, stats: SweepStats) -> None:
        """Fail runs whose approval nobody answered.

        Without an expiry a forgotten approval pins a run open forever, holding its
        budget and its place in the operator's queue. An expiry always ends the run
        rather than steering the model: nobody is coming, and continuing would mean
        pretending a human had answered.
        """
        async with session_scope() as session:
            expired = await expire_stale(session, limit=settings.reaper_batch_size)
        stats.approvals_expired += len(expired)

    async def _republish_runnable(self, stats: SweepStats) -> None:
        async with session_scope() as session:
            rows = await lease_ops.find_runnable_steps(
                session,
                limit=settings.reaper_batch_size,
                grace_seconds=settings.enqueue_grace_seconds,
            )
        for row in rows:
            try:
                await self.queue.publish(step_id=row.id, run_id=row.run_id)
                stats.republished += 1
            except Exception as exc:
                log.warning("republish_failed", step_id=str(row.id), error=str(exc))
                return
        if stats.republished:
            log.info("republished_runnable_steps", count=stats.republished)


async def main() -> None:
    reaper = Reaper()
    loop = asyncio.get_running_loop()

    def _handler() -> None:
        reaper.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _handler())

    try:
        await reaper.run()
    finally:
        await dispose_engine()


__all__ = ["ACTOR", "Reaper", "SweepStats", "main"]
