"""The queue port.

Deliberately narrow, and deliberately *not* trusted. The queue's only job is to tell
a worker quickly that a step might be runnable; Postgres decides whether it actually
is. That is what makes swapping Redis Streams for Kafka an adapter change rather than
a correctness argument, and what makes losing the queue entirely a latency incident
rather than a data-loss incident.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StepMessage:
    step_id: uuid.UUID
    run_id: uuid.UUID
    #: Backend-specific ack handle (a Redis stream entry id). None for backends where
    #: the work item is not a separate durable object.
    receipt: str | None = None
    #: How many times this message has been delivered, when the backend tracks it.
    delivery_count: int = 1


@runtime_checkable
class StepQueue(Protocol):
    async def setup(self) -> None:
        """Idempotently create whatever the backend needs (stream, group)."""

    async def publish(self, *, step_id: uuid.UUID, run_id: uuid.UUID) -> None:
        """Hint that a step is runnable.

        Failure here must never fail the caller's transaction: the step row is
        already committed, so the reaper will pick it up within `enqueue_grace`.
        """

    async def consume(
        self, *, consumer: str, max_messages: int, block_ms: int
    ) -> list[StepMessage]: ...

    async def ack(self, message: StepMessage) -> None: ...

    async def reclaim_stalled(
        self, *, consumer: str, min_idle_ms: int, count: int
    ) -> list[StepMessage]:
        """Take over messages another consumer accepted but never acked."""

    async def depth(self) -> int:
        """Approximate backlog, for metrics."""

    async def close(self) -> None: ...


__all__ = ["StepMessage", "StepQueue"]
