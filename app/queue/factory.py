from __future__ import annotations

from app.config import settings
from app.queue.base import StepQueue


def build_queue() -> StepQueue:
    if settings.queue_backend == "postgres":
        from app.queue.postgres import PostgresQueue

        return PostgresQueue()
    from app.queue.redis_streams import RedisStreamQueue

    return RedisStreamQueue()


__all__ = ["build_queue"]
