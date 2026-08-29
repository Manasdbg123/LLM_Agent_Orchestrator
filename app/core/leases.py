"""Leases: the mechanism that makes at-most-one-executor-per-step true.

Every time comparison in this module is evaluated by `now()` *inside Postgres*.
Comparing a lease against a worker's local clock is a real source of double
execution once workers' clocks drift, and nothing here ever does it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.transitions import record_transition, transition_step
from app.domain.errors import ErrorCode
from app.domain.states import StepKind, StepStatus, TransitionReason
from app.obs.logging import get_logger

log = get_logger("leases")


@dataclass(frozen=True, slots=True)
class Lease:
    """Proof that this worker owns this step, for this epoch."""

    step_id: uuid.UUID
    run_id: uuid.UUID
    seq: int
    #: The agent_turn that requested this step, for tool_call steps.
    parent_step_id: uuid.UUID | None
    kind: StepKind
    input: dict[str, Any]
    #: Incremented by the claim. Every later write by this worker is conditioned on
    #: it, so a stalled worker's late write cannot land after reassignment.
    epoch: int
    attempt: int
    max_attempts: int
    recoveries: int
    max_recoveries: int
    worker_id: str


# A single atomic statement. This is the entire duplicate-execution defense: Postgres
# row locks serialize contenders, the loser matches zero rows and simply moves on.
#
# The `prev` CTE exists only so the audit log can record the true previous status;
# CTEs see the statement-start snapshot, so it reports the pre-UPDATE value.
_CLAIM_SQL = sa.text(
    """
WITH prev AS (
    SELECT id, status FROM steps WHERE id = :step_id
), upd AS (
    UPDATE steps
       SET status           = 'running',
           attempt          = attempt + 1,
           lease_owner      = :worker_id,
           lease_epoch      = lease_epoch + 1,
           lease_expires_at = now() + make_interval(secs => :ttl_seconds),
           started_at       = COALESCE(started_at, now()),
           updated_at       = now()
     WHERE id = :step_id
       AND status IN ('pending', 'retrying')
       AND available_at <= now()
       AND (lease_expires_at IS NULL OR lease_expires_at < now())
    RETURNING id, run_id, seq, parent_step_id, kind, input, attempt, lease_epoch,
              max_attempts, recoveries, max_recoveries
)
SELECT upd.id, upd.run_id, upd.seq, upd.parent_step_id, upd.kind, upd.input,
       upd.attempt, upd.lease_epoch, upd.max_attempts, upd.recoveries,
       upd.max_recoveries, prev.status AS from_status
  FROM upd JOIN prev ON prev.id = upd.id
"""
)


async def claim_step(
    session: AsyncSession,
    *,
    step_id: uuid.UUID,
    worker_id: str,
    ttl_seconds: int,
) -> Lease | None:
    """Try to take ownership of a step. Returns None if someone else has it.

    None is the normal outcome under duplicate queue delivery and is not an error.
    """
    row = (
        await session.execute(
            _CLAIM_SQL,
            {"step_id": step_id, "worker_id": worker_id, "ttl_seconds": ttl_seconds},
        )
    ).one_or_none()
    if row is None:
        return None

    await record_transition(
        session,
        run_id=row.run_id,
        step_id=row.id,
        entity="step",
        from_status=row.from_status,
        to_status=str(StepStatus.RUNNING),
        reason=str(TransitionReason.CLAIMED),
        actor=worker_id,
        attempt=row.attempt,
        details={"lease_epoch": row.lease_epoch},
    )
    return Lease(
        step_id=row.id,
        run_id=row.run_id,
        seq=row.seq,
        parent_step_id=row.parent_step_id,
        kind=StepKind(row.kind),
        input=row.input or {},
        epoch=row.lease_epoch,
        attempt=row.attempt,
        max_attempts=row.max_attempts,
        recoveries=row.recoveries,
        max_recoveries=row.max_recoveries,
        worker_id=worker_id,
    )


_HEARTBEAT_SQL = sa.text(
    """
UPDATE steps
   SET lease_expires_at = now() + make_interval(secs => :ttl_seconds),
       updated_at = now()
 WHERE id = :step_id
   AND status = 'running'
   AND lease_owner = :worker_id
   AND lease_epoch = :epoch
RETURNING id
"""
)


async def heartbeat(
    session: AsyncSession,
    *,
    step_id: uuid.UUID,
    worker_id: str,
    epoch: int,
    ttl_seconds: int,
) -> bool:
    """Extend the lease. False means this worker has been fenced and must stop.

    Note the `lease_epoch` guard: a worker that was reaped and whose step was already
    re-claimed by someone else will not resurrect its own lease here.
    """
    row = (
        await session.execute(
            _HEARTBEAT_SQL,
            {
                "step_id": step_id,
                "worker_id": worker_id,
                "epoch": epoch,
                "ttl_seconds": ttl_seconds,
            },
        )
    ).one_or_none()
    return row is not None


async def release_lease(
    session: AsyncSession, *, step_id: uuid.UUID, worker_id: str, epoch: int
) -> None:
    """Clear lease fields on a step this worker still owns.

    Called after a terminal transition so the row does not keep a stale owner. Safe
    to call when fenced: the guard simply matches nothing.
    """
    await session.execute(
        sa.text(
            """
            UPDATE steps SET lease_owner = NULL, lease_expires_at = NULL, updated_at = now()
             WHERE id = :step_id AND lease_owner = :worker_id AND lease_epoch = :epoch
            """
        ),
        {"step_id": step_id, "worker_id": worker_id, "epoch": epoch},
    )


# --- reaper queries -------------------------------------------------------------

_EXPIRED_LEASES_SQL = sa.text(
    """
SELECT id, run_id, lease_owner, recoveries, max_recoveries, attempt
  FROM steps
 WHERE status = 'running'
   AND lease_expires_at IS NOT NULL
   AND lease_expires_at < now()
 ORDER BY lease_expires_at
 LIMIT :limit
 FOR UPDATE SKIP LOCKED
"""
)


async def find_expired_leases(session: AsyncSession, *, limit: int) -> list[Any]:
    return list((await session.execute(_EXPIRED_LEASES_SQL, {"limit": limit})).all())


async def reclaim_expired_lease(session: AsyncSession, *, row: Any, actor: str = "reaper") -> str:
    """Hand one orphaned step back to the pool, or give up on it.

    Returns the resulting status, for metrics and for the reaper's log line.

    A worker death is *not* a step failure: it moves running -> pending and consumes
    the separate `recoveries` budget rather than the retry budget. Without that
    separation, a step that reliably kills whatever worker picks it up would quietly
    exhaust the user's retries and present as an ordinary application error.
    """
    if row.recoveries + 1 > row.max_recoveries:
        result = await transition_step(
            session,
            step_id=row.id,
            to=StepStatus.FAILED,
            reason=str(TransitionReason.TOO_MANY_RECOVERIES),
            actor=actor,
            expect=StepStatus.RUNNING,
            values={
                "recoveries": row.recoveries + 1,
                "lease_owner": None,
                "lease_expires_at": None,
                "ended_at": sa.func.now(),
                "error": {
                    "code": str(ErrorCode.TOO_MANY_RECOVERIES),
                    "class": "terminal",
                    "message": (
                        f"step exceeded max_recoveries={row.max_recoveries}; "
                        "it repeatedly killed the worker executing it"
                    ),
                },
            },
            details={"orphaned_owner": row.lease_owner},
        )
        return str(StepStatus.FAILED) if result.applied else "unchanged"

    result = await transition_step(
        session,
        step_id=row.id,
        to=StepStatus.PENDING,
        reason=str(TransitionReason.LEASE_EXPIRED),
        actor=actor,
        expect=StepStatus.RUNNING,
        values={
            "recoveries": row.recoveries + 1,
            "lease_owner": None,
            "lease_expires_at": None,
            "available_at": sa.func.now(),
        },
        details={"orphaned_owner": row.lease_owner, "attempt": row.attempt},
    )
    return str(StepStatus.PENDING) if result.applied else "unchanged"


# Steps that are runnable but have gone unclaimed. This is simultaneously:
#   - delayed-retry dispatch (available_at in the future until the backoff elapses),
#   - lost-queue-message recovery,
#   - post-crash re-dispatch of steps the reaper just reset to pending.
# One predicate, three failure modes. The `grace` term keeps the reaper from racing
# the inline enqueue that just happened for a freshly created step.
_RUNNABLE_SQL = sa.text(
    """
SELECT s.id, s.run_id
  FROM steps s
  JOIN agent_runs r ON r.id = s.run_id
 WHERE s.status IN ('pending', 'retrying')
   AND s.available_at <= now() - make_interval(secs => :grace_seconds)
   AND (s.lease_expires_at IS NULL OR s.lease_expires_at < now())
   AND r.status IN ('pending', 'running')
   AND NOT r.cancel_requested
 ORDER BY s.available_at
 LIMIT :limit
"""
)


async def find_runnable_steps(
    session: AsyncSession, *, limit: int, grace_seconds: float
) -> list[Any]:
    return list(
        (
            await session.execute(_RUNNABLE_SQL, {"limit": limit, "grace_seconds": grace_seconds})
        ).all()
    )


__all__ = [
    "Lease",
    "claim_step",
    "find_expired_leases",
    "find_runnable_steps",
    "heartbeat",
    "reclaim_expired_lease",
    "release_lease",
]
