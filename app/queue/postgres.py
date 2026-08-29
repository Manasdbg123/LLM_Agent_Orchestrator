"""Postgres-only queue backend: no Redis at all.

This exists for two reasons.

1. It is a real deployment mode for small installs — one dependency instead of two.
2. It is a live proof of the central claim in DESIGN.md: Redis is a latency
   optimization, not a source of truth. The runnable predicate below is the *same*
   predicate the reaper uses, so if the system works with this backend, the Redis
   backend cannot be load-bearing for correctness.

`publish` is a no-op because the step row already is the work item. `ack` is a no-op
for the same reason: the step's status *is* the acknowledgement.
"""

from __future__ import annotations

import asyncio
import uuid

import sqlalchemy as sa

from app.db import session_scope
from app.queue.base import StepMessage

# Identical to `leases._RUNNABLE_SQL` minus the grace term: with no separate
# publisher there is no inline enqueue to race with, so a step is dispatchable the
# instant it becomes runnable.
_POLL_SQL = sa.text(
    """
SELECT s.id, s.run_id
  FROM steps s
  JOIN agent_runs r ON r.id = s.run_id
 WHERE s.status IN ('pending', 'retrying')
   AND s.available_at <= now()
   AND (s.lease_expires_at IS NULL OR s.lease_expires_at < now())
   AND r.status IN ('pending', 'running')
   AND NOT r.cancel_requested
 ORDER BY s.available_at
 LIMIT :limit
"""
)


class PostgresQueue:
    """Polling dispatcher over the runnable predicate."""

    def __init__(self, poll_interval_seconds: float = 0.25) -> None:
        self.poll_interval_seconds = poll_interval_seconds

    async def setup(self) -> None:
        return None

    async def publish(self, *, step_id: uuid.UUID, run_id: uuid.UUID) -> None:
        return None

    async def consume(
        self, *, consumer: str, max_messages: int, block_ms: int
    ) -> list[StepMessage]:
        deadline = asyncio.get_running_loop().time() + block_ms / 1000
        while True:
            async with session_scope() as session:
                rows = (await session.execute(_POLL_SQL, {"limit": max_messages})).all()
            if rows:
                # Several workers will see the same rows; the atomic claim decides.
                # At this scale that is cheaper than partitioning the poll.
                return [StepMessage(step_id=r.id, run_id=r.run_id) for r in rows]
            if asyncio.get_running_loop().time() >= deadline:
                return []
            await asyncio.sleep(self.poll_interval_seconds)

    async def ack(self, message: StepMessage) -> None:
        return None

    async def reclaim_stalled(
        self, *, consumer: str, min_idle_ms: int, count: int
    ) -> list[StepMessage]:
        # Nothing to reclaim: an orphaned step is found by the same poll, once its
        # lease expires.
        return []

    async def depth(self) -> int:
        async with session_scope() as session:
            return int(
                (
                    await session.execute(
                        sa.text(
                            """
                            SELECT count(*) FROM steps
                             WHERE status IN ('pending','retrying') AND available_at <= now()
                            """
                        )
                    )
                ).scalar_one()
            )

    async def close(self) -> None:
        return None


__all__ = ["PostgresQueue"]
