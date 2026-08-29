"""Worker process: pull step ids, claim them, execute them.

Stateless and disposable by construction. Everything a worker knows is either in its
lease (which expires) or in Postgres (which does not), so killing a worker at any
instant is a recoverable event rather than a data-loss event.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import uuid
from typing import Any

from app.config import settings
from app.db import dispose_engine
from app.engine.executor import Outcome, StepExecutor
from app.obs.logging import configure_logging, get_logger
from app.queue.base import StepMessage, StepQueue
from app.queue.factory import build_queue

log = get_logger("worker")


def generate_worker_id() -> str:
    """Unique per process, and traceable back to a host and pid when debugging.

    The random suffix matters: a container that restarts with the same hostname and
    pid would otherwise reuse the identity of the worker it replaced, and a lease
    guard keyed on `lease_owner` would accept writes from its predecessor.
    """
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class Worker:
    def __init__(
        self,
        queue: StepQueue | None = None,
        *,
        worker_id: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        self.queue = queue or build_queue()
        self.worker_id = worker_id or generate_worker_id()
        self.concurrency = concurrency or settings.worker_concurrency
        self.executor = StepExecutor(self.queue, self.worker_id)
        self._stop = asyncio.Event()
        self._inflight: set[asyncio.Task[Any]] = set()
        self._sweeps = 0
        self.stats: dict[str, int] = {}

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        configure_logging()
        await self.queue.setup()
        log.info(
            "worker_started",
            worker_id=self.worker_id,
            concurrency=self.concurrency,
            queue=settings.queue_backend,
        )
        try:
            while not self._stop.is_set():
                await self._tick()
        finally:
            await self._drain()
            await self.queue.close()
            log.info("worker_stopped", worker_id=self.worker_id, stats=self.stats)

    async def _tick(self) -> None:
        free = self.concurrency - len(self._inflight)
        if free <= 0:
            await self._await_any()
            return

        messages = await self._fetch(free)
        for message in messages:
            task = asyncio.create_task(self._handle(message))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _fetch(self, limit: int) -> list[StepMessage]:
        # Periodically adopt entries that a dead worker accepted and never acked.
        # Note this is a queue-level concern only: the step's *lease* is what
        # actually protects it, so adopting a message never risks double execution.
        self._sweeps += 1
        if self._sweeps % 10 == 1:
            try:
                stalled = await self.queue.reclaim_stalled(
                    consumer=self.worker_id,
                    min_idle_ms=settings.stalled_message_idle_ms,
                    count=limit,
                )
            except Exception as exc:
                log.warning("reclaim_failed", error=str(exc))
                stalled = []
            if stalled:
                log.info("reclaimed_stalled_messages", count=len(stalled))
                return stalled
        try:
            return await self.queue.consume(
                consumer=self.worker_id,
                max_messages=limit,
                block_ms=settings.worker_poll_block_ms,
            )
        except Exception as exc:
            # A queue outage must not spin the CPU or kill the worker: the reaper
            # keeps dispatching from Postgres, and this worker retries shortly.
            log.warning("consume_failed", error=str(exc))
            await asyncio.sleep(1.0)
            return []

    async def _handle(self, message: StepMessage) -> None:
        try:
            report = await self.executor.execute(message)
            self.stats[str(report.outcome)] = self.stats.get(str(report.outcome), 0) + 1
        except asyncio.CancelledError:
            # Shutdown mid-step: leave the step RUNNING and let the lease expire.
            # The reaper reassigns it, which is strictly safer than guessing here.
            raise
        except Exception as exc:
            self.stats["error"] = self.stats.get("error", 0) + 1
            log.exception("step_execution_error", step_id=str(message.step_id), error=str(exc))
            return
        else:
            if report.outcome is not Outcome.NOT_CLAIMED:
                log.info(
                    "step_done",
                    outcome=str(report.outcome),
                    step_id=str(report.step_id),
                    detail=report.detail,
                )
        finally:
            # Ack regardless of outcome: the message has been fully processed by this
            # worker. The step's fate lives in Postgres, so an unacked message would
            # only produce a duplicate delivery that loses the claim race anyway.
            with contextlib.suppress(Exception):
                await self.queue.ack(message)

    async def _await_any(self) -> None:
        if not self._inflight:
            await asyncio.sleep(0.05)
            return
        await asyncio.wait(self._inflight, return_when=asyncio.FIRST_COMPLETED)

    async def _drain(self) -> None:
        if not self._inflight:
            return
        log.info("draining", inflight=len(self._inflight))
        _done, pending = await asyncio.wait(self._inflight, timeout=settings.shutdown_grace_seconds)
        for task in pending:
            task.cancel()
        if pending:
            log.warning("drain_timeout", abandoned=len(pending))
            await asyncio.gather(*pending, return_exceptions=True)


async def main() -> None:
    worker = Worker()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        log.info("shutdown_signal_received", worker_id=worker.worker_id)
        worker.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows: add_signal_handler is unavailable on the proactor loop.
            signal.signal(sig, lambda *_: _signal_handler())

    try:
        await worker.run()
    finally:
        await dispose_engine()


__all__ = ["Worker", "generate_worker_id", "main"]
