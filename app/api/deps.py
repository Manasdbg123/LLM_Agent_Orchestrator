"""Shared API dependencies.

Extracted so the dashboard does not have to import the FastAPI app to get a session
or the queue. That import was circular — `main` imported the dashboard router at the
bottom while the dashboard imported `main` at the top — and FastAPI resolved it by
silently registering nothing, which is the worst possible failure mode: no error, no
routes, a 404 that looks like a routing typo.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_sessionmaker
from app.queue.base import StepQueue


async def get_session() -> AsyncIterator[AsyncSession]:
    """One transaction per request, committed on a clean response."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_session)]


class _QueueHolder:
    """The process-wide queue handle, set once by the API lifespan.

    A holder rather than a module global so tests can swap the backend, and so the
    dashboard can publish a resumed step without reaching into the app object.
    """

    def __init__(self) -> None:
        self._queue: StepQueue | None = None

    def set(self, queue: StepQueue) -> None:
        self._queue = queue

    def get(self) -> StepQueue | None:
        return self._queue


queue_holder = _QueueHolder()


__all__ = ["SessionDep", "get_session", "queue_holder"]
