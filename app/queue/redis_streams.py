"""Redis Streams adapter.

Shape mapping, for the "why not Kafka" argument in DESIGN.md:

    XADD                -> produce
    consumer group      -> consumer group
    XREADGROUP          -> poll with at-least-once delivery
    XACK                -> commit offset
    XAUTOCLAIM          -> partition reassignment after a consumer dies

Duplicate delivery is expected and harmless: `leases.claim_step` is atomic, so a
second delivery loses the race and is acked without executing anything.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from redis import asyncio as aioredis
from redis.exceptions import ResponseError

from app.config import settings
from app.obs.logging import get_logger
from app.queue.base import StepMessage

log = get_logger("queue.redis")


#: What XREADGROUP / XAUTOCLAIM hand back: [(stream, [(entry_id, {field: value})])].
#: redis-py declares a far wider union covering `decode_responses=False`, where every
#: string is bytes. This client sets `decode_responses=True` (see `redis` below), so
#: the str shape is the real one and the call sites cast to it -- narrowing at the
#: boundary once, rather than a blanket `type: ignore` on the loop that reads it.
StreamResponse = list[tuple[Any, list[tuple[str, dict[str, str]]]]] | None


class RedisStreamQueue:
    def __init__(
        self,
        url: str | None = None,
        stream: str | None = None,
        group: str | None = None,
    ) -> None:
        self._url = url or settings.redis_url
        self.stream = stream or settings.stream_name
        self.group = group or settings.consumer_group
        self._redis: aioredis.Redis | None = None

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            # Bounded timeouts: an unreachable Redis must surface in seconds. The
            # reaper covers the gap, so failing fast costs latency, never work.
            self._redis = aioredis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=30,
                health_check_interval=30,
            )
        return self._redis

    async def setup(self) -> None:
        try:
            # mkstream so the group can be created before the first XADD.
            await self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            log.info("consumer_group_created", stream=self.stream, group=self.group)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish(self, *, step_id: uuid.UUID, run_id: uuid.UUID) -> None:
        await self.redis.xadd(self.stream, {"step_id": str(step_id), "run_id": str(run_id)})

    async def consume(
        self, *, consumer: str, max_messages: int, block_ms: int
    ) -> list[StepMessage]:
        # ">" = entries never delivered to this group. Redelivery of this consumer's
        # own un-acked entries is handled by reclaim_stalled, not here, so a crashed
        # worker's backlog is not silently pinned to its dead consumer name.
        response = await self.redis.xreadgroup(
            groupname=self.group,
            consumername=consumer,
            streams={self.stream: ">"},
            count=max_messages,
            block=block_ms,
        )
        return await self._to_messages(cast("StreamResponse", response))

    async def ack(self, message: StepMessage) -> None:
        if message.receipt is not None:
            await self.redis.xack(self.stream, self.group, message.receipt)

    async def reclaim_stalled(
        self, *, consumer: str, min_idle_ms: int, count: int
    ) -> list[StepMessage]:
        """Adopt entries a dead consumer never acked.

        `min_idle_ms` must exceed the lease TTL (enforced in Settings), otherwise this
        would steal messages from workers that are alive and legitimately busy. Even
        then it would not cause double execution — the lease would still hold — but it
        would generate pointless claim failures.
        """
        try:
            _cursor, entries, _deleted = await self.redis.xautoclaim(
                name=self.stream,
                groupname=self.group,
                consumername=consumer,
                min_idle_time=min_idle_ms,
                start_id="0-0",
                count=count,
            )
        except ResponseError as exc:  # group vanished (FLUSHALL, failover)
            log.warning("xautoclaim_failed", error=str(exc))
            await self.setup()
            return []
        return await self._to_messages([(self.stream, entries)])

    async def depth(self) -> int:
        try:
            info = await self.redis.xinfo_groups(self.stream)
        except ResponseError:
            return 0
        for group in info:
            if group.get("name") == self.group:
                return int(group.get("lag") or group.get("pending") or 0)
        return 0

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def _to_messages(self, response: StreamResponse) -> list[StepMessage]:
        messages: list[StepMessage] = []
        if not response:
            return messages
        for _stream, entries in response:
            for entry_id, fields in entries or []:
                try:
                    messages.append(
                        StepMessage(
                            step_id=uuid.UUID(fields["step_id"]),
                            run_id=uuid.UUID(fields["run_id"]),
                            receipt=entry_id,
                        )
                    )
                except (KeyError, ValueError):
                    # A malformed entry can never become valid. It must be acked as
                    # well as dropped: an unacked entry stays in the pending list and
                    # would be reclaimed by XAUTOCLAIM on every sweep, forever.
                    log.warning("dropping_malformed_entry", entry_id=entry_id, fields=fields)
                    await self.redis.xack(self.stream, self.group, entry_id)
        return messages


__all__ = ["RedisStreamQueue"]
