"""Shared test scaffolding."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import sqlalchemy as sa

from app.core.runs import create_run, get_or_create_definition
from app.db import session_scope
from app.domain.models import AgentRun, DummyEffect, StateTransition, Step
from app.domain.states import RunStatus, StepKind
from app.queue.base import StepQueue
from app.reaper.reaper import Reaper
from app.worker.worker import Worker


async def make_run(
    plan: list[dict[str, Any]],
    *,
    queue: StepQueue | None = None,
    agent: str = "chaos",
    task: str = "test task",
    timeout_seconds: int = 120,
    max_steps: int | None = None,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Create a dummy run from a positional plan. Returns (run_id, first_step_id)."""
    async with session_scope() as session:
        definition = await get_or_create_definition(session, name=agent)
        run, step = await create_run(
            session,
            definition=definition,
            task=task,
            input={"plan": plan},
            first_step_kind=StepKind.DUMMY,
            first_step_input=plan[0] if plan else {},
            timeout_seconds=timeout_seconds,
            max_steps=max_steps,
        )
        run_id, step_id = run.id, step.id
    if queue is not None:
        await queue.publish(step_id=step_id, run_id=run_id)
    return run_id, step_id


async def get_run(run_id: uuid.UUID) -> AgentRun:
    async with session_scope() as s:
        return (await s.execute(sa.select(AgentRun).where(AgentRun.id == run_id))).scalar_one()


async def get_steps(run_id: uuid.UUID) -> list[Step]:
    async with session_scope() as s:
        return list(
            (await s.execute(sa.select(Step).where(Step.run_id == run_id).order_by(Step.seq)))
            .scalars()
            .all()
        )


async def get_effects(run_id: uuid.UUID) -> dict[str, DummyEffect]:
    """Side-effect ledger for a run, keyed by label."""
    async with session_scope() as s:
        rows = (
            (await s.execute(sa.select(DummyEffect).where(DummyEffect.run_id == run_id)))
            .scalars()
            .all()
        )
    return {r.label: r for r in rows}


async def get_transitions(run_id: uuid.UUID) -> list[StateTransition]:
    async with session_scope() as s:
        return list(
            (
                await s.execute(
                    sa.select(StateTransition)
                    .where(StateTransition.run_id == run_id)
                    .order_by(StateTransition.id)
                )
            )
            .scalars()
            .all()
        )


async def wait_for(
    predicate: Callable[[], Awaitable[bool]], *, timeout: float = 30.0, interval: float = 0.1
) -> bool:
    """Poll until true or timeout. Returns whether it became true."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def wait_for_run_terminal(run_id: uuid.UUID, *, timeout: float = 60.0) -> RunStatus:
    async def done() -> bool:
        run = await get_run(run_id)
        return RunStatus(run.status) in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }

    await wait_for(done, timeout=timeout)
    return RunStatus((await get_run(run_id)).status)


class Cluster:
    """N in-process workers plus a reaper, started and stopped together."""

    def __init__(self, queue: StepQueue, *, workers: int = 1, reaper: bool = True) -> None:
        self.queue = queue
        self.workers = [
            Worker(queue, worker_id=f"w{i}-{uuid.uuid4().hex[:4]}", concurrency=2)
            for i in range(workers)
        ]
        self.reaper = Reaper(queue) if reaper else None
        self._tasks: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> Cluster:
        for w in self.workers:
            self._tasks.append(asyncio.create_task(w.run(), name=f"worker-{w.worker_id}"))
        if self.reaper is not None:
            self._tasks.append(asyncio.create_task(self.reaper.run(), name="reaper"))
        return self

    async def __aexit__(self, *exc: object) -> None:
        for w in self.workers:
            w.request_stop()
        if self.reaper is not None:
            self.reaper.request_stop()
        done, pending = await asyncio.wait(self._tasks, timeout=15)
        for t in pending:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # Surface a worker that died of a real bug rather than letting the test see
        # only "nothing happened".
        for t in done:
            if not t.cancelled() and t.exception() is not None:
                raise AssertionError(f"{t.get_name()} crashed") from t.exception()


async def make_agent_run(
    script: list[dict[str, Any]],
    *,
    queue: StepQueue | None = None,
    tools: list[str] | None = None,
    task: str = "complete the task",
    tool_faults: dict[str, dict[str, Any]] | None = None,
    max_steps: int | None = None,
    timeout_seconds: int = 120,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Create an LLM-driven run whose model turns follow `script`.

    The script rides along in the task text, which is how the fake provider finds it
    without any extra plumbing through the engine.
    """
    from app.core.runs import create_run, get_or_create_definition
    from app.llm.fake import encode_script

    run_input: dict[str, Any] = {}
    if tool_faults:
        run_input["tool_faults"] = tool_faults

    async with session_scope() as session:
        definition = await get_or_create_definition(
            session,
            # Unique per run: definitions are immutable and looked up by name, so a
            # shared name would silently reuse another test's tool list.
            name=f"agent-{uuid.uuid4().hex[:8]}",
            model="fake-model",
            tools=tools if tools is not None else ["calculator", "web_search"],
        )
        run, step = await create_run(
            session,
            definition=definition,
            task=f"{task} {encode_script(script)}",
            input=run_input,
            first_step_kind=StepKind.AGENT_TURN,
            first_step_input={},
            timeout_seconds=timeout_seconds,
            max_steps=max_steps,
        )
        run_id, step_id = run.id, step.id
    if queue is not None:
        await queue.publish(step_id=step_id, run_id=run_id)
    return run_id, step_id


async def get_tool_calls(run_id: uuid.UUID) -> list[Any]:
    from app.domain.models import ToolCall

    async with session_scope() as s:
        return list(
            (
                await s.execute(
                    sa.select(ToolCall)
                    .where(ToolCall.run_id == run_id)
                    .order_by(ToolCall.started_at)
                )
            )
            .scalars()
            .all()
        )


async def get_llm_calls(run_id: uuid.UUID) -> list[Any]:
    from app.domain.models import LLMCall

    async with session_scope() as s:
        return list(
            (
                await s.execute(
                    sa.select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.created_at)
                )
            )
            .scalars()
            .all()
        )


async def count_emails() -> int:
    from app.domain.models import EmailOutbox

    async with session_scope() as s:
        return int(
            (await s.execute(sa.select(sa.func.count()).select_from(EmailOutbox))).scalar_one()
        )


class AutoApprover:
    """Stands in for an operator watching the approval queue.

    Phase 4 gates `send_email` behind a human, so any test exercising the *effect*
    path has to supply that human or the run parks forever. Approving in a background
    task rather than inline keeps the tests honest: the run genuinely stops, is
    genuinely decided by a separate actor, and genuinely resumes.
    """

    def __init__(self, queue: StepQueue, run_id: uuid.UUID, *, poll: float = 0.2) -> None:
        self.queue = queue
        self.run_id = run_id
        self.poll = poll
        self.approved: list[uuid.UUID] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> AutoApprover:
        self._task = asyncio.create_task(self._loop(), name="auto-approver")
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        from app.core import approvals
        from app.domain.models import ApprovalRequest

        while not self._stop.is_set():
            async with session_scope() as session:
                pending = (
                    (
                        await session.execute(
                            sa.select(ApprovalRequest.id).where(
                                ApprovalRequest.run_id == self.run_id,
                                ApprovalRequest.decision == approvals.Decision.PENDING,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )

            for approval_id in pending:
                async with session_scope() as session:
                    result = await approvals.decide(
                        session,
                        approval_id=approval_id,
                        decision=approvals.Decision.APPROVED,
                        decided_by="auto-approver",
                        decision_reason="approved by test operator",
                    )
                if result.applied and result.resume_step_id is not None:
                    self.approved.append(approval_id)
                    await self.queue.publish(step_id=result.resume_step_id, run_id=self.run_id)
            await asyncio.sleep(self.poll)
