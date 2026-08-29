"""The eval runner.

Submits every task in the catalog to a real cluster -- real Postgres, real workers,
a real reaper -- waits for each run to reach a terminal state, and grades it against
its expectations.

**Why this runs the engine in-process rather than over HTTP.** The API is a thin
layer over `app.core.runs` and is already covered by `tests/integration/test_api.py`;
driving it here would add a port, a server lifecycle and a class of "is the API up
yet" flake without testing anything new. What the eval is measuring is the
*execution* path -- planning, tool calls, retries, gates, recovery -- so the harness
starts workers and a reaper in this process and talks to the same core functions the
API handlers call. The load test (`scripts/load_test.py`) is the one that goes over
HTTP, because throughput under concurrent submission is exactly where the transport
matters.

**Grading reads ground truth.** "Sent one email" is a `SELECT count(*)` against the
provider's outbox; "wrote the record" is a row in `sandbox_records`. The engine's own
summary of what it did is reported but never used to decide pass or fail.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa

from app.core import approvals
from app.core.runs import create_run, get_or_create_definition
from app.db import session_scope
from app.domain.models import (
    AgentRun,
    ApprovalRequest,
    EmailOutbox,
    SandboxRecord,
    Step,
)
from app.domain.states import RunStatus, StepKind
from app.llm.fake import encode_script
from app.obs.logging import get_logger
from app.queue.base import StepQueue
from app.reaper.reaper import Reaper
from app.worker.worker import Worker
from eval.tasks import EvalTask

log = get_logger("eval")

TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


@dataclass(slots=True)
class TaskResult:
    """One graded run. Everything the report needs, and nothing derived."""

    task_id: str
    tier: str
    intent: str
    run_id: uuid.UUID | None
    passed: bool
    status: str
    error_code: str | None = None
    answer: str | None = None
    steps_used: int = 0
    model_turns: int = 0
    tool_calls: int = 0
    tool_sequence: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    #: Engine-observed duration: run creation to terminal.
    latency_s: float = 0.0
    #: Harness-observed duration, including the poll interval. Always >= latency_s.
    wall_s: float = 0.0
    retries: int = 0
    recoveries: int = 0
    approvals: int = 0
    emails: int = 0
    #: Human-readable reasons this task failed. Empty iff `passed`.
    failures: list[str] = field(default_factory=list)


class Cluster:
    """Workers plus a reaper, started and stopped together.

    The reaper is not optional. Half the catalog's reliability tier depends on
    something noticing an expired lease, and an eval that quietly ran without one
    would report those tasks as hangs rather than as the recoveries they are.
    """

    def __init__(self, queue: StepQueue, *, workers: int = 2) -> None:
        self.queue = queue
        self.workers = [
            Worker(queue, worker_id=f"eval-w{i}-{uuid.uuid4().hex[:4]}", concurrency=2)
            for i in range(workers)
        ]
        self.reaper = Reaper(queue)
        self._tasks: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> Cluster:
        for w in self.workers:
            self._tasks.append(asyncio.create_task(w.run(), name=f"worker-{w.worker_id}"))
        self._tasks.append(asyncio.create_task(self.reaper.run(), name="reaper"))
        return self

    async def __aexit__(self, *exc: object) -> None:
        for w in self.workers:
            w.request_stop()
        self.reaper.request_stop()
        done, pending = await asyncio.wait(self._tasks, timeout=20)
        for t in pending:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for t in done:
            if not t.cancelled() and t.exception() is not None:
                raise RuntimeError(f"{t.get_name()} crashed") from t.exception()


async def _submit(task: EvalTask, queue: StepQueue, *, live: bool) -> uuid.UUID:
    """Create the run and enqueue its first step. Returns the run id.

    Against the scripted provider the script rides along in the task text, which is
    how the fake finds it without extra plumbing through the engine. Against a live
    model only the instruction is sent.
    """
    text = task.task if live else task.task + encode_script(task.script)
    run_input: dict[str, Any] = {}
    if task.tool_faults:
        run_input["tool_faults"] = task.tool_faults

    async with session_scope() as session:
        definition = await get_or_create_definition(
            session,
            # Unique per submission: definitions are immutable and looked up by name,
            # so a shared name would silently reuse an earlier task's tool list.
            name=f"eval-{task.id}-{uuid.uuid4().hex[:8]}",
            model="fake-model",
            tools=task.tools,
        )
        run, step = await create_run(
            session,
            definition=definition,
            task=text,
            input=run_input,
            first_step_kind=StepKind.AGENT_TURN,
            first_step_input={},
            max_steps=task.max_steps,
            max_cost_usd=task.max_cost_usd,
            timeout_seconds=task.timeout_seconds,
            actor="eval",
        )
        run_id, step_id = run.id, step.id

    await queue.publish(step_id=step_id, run_id=run_id)
    return run_id


async def _drive_approvals(run_id: uuid.UUID, task: EvalTask, queue: StepQueue) -> None:
    """Stand in for an operator watching the approval queue.

    Deliberately goes through `approvals.decide` -- the same function the API endpoint
    calls -- rather than updating the row directly, so the eval exercises the real
    resume path including the enqueue of the released step.
    """
    if not (task.approve or task.reject):
        return

    async with session_scope() as session:
        pending = (
            (
                await session.execute(
                    sa.select(ApprovalRequest.id).where(
                        ApprovalRequest.run_id == run_id,
                        ApprovalRequest.decision == approvals.Decision.PENDING,
                    )
                )
            )
            .scalars()
            .all()
        )

    decision = approvals.Decision.APPROVED if task.approve else approvals.Decision.REJECTED
    for approval_id in pending:
        async with session_scope() as session:
            result = await approvals.decide(
                session,
                approval_id=approval_id,
                decision=decision,
                decided_by="eval-harness",
                decision_reason="auto-decided by the eval harness",
            )
        if result.resume_step_id is not None:
            await queue.publish(step_id=result.resume_step_id, run_id=run_id)


async def _await_terminal(
    run_id: uuid.UUID, task: EvalTask, queue: StepQueue, *, poll: float = 0.25
) -> RunStatus:
    """Poll until the run settles or the task's own timeout elapses.

    The timeout is the harness's, not the engine's: the run also carries a deadline
    and will fail itself. Giving the harness the longer leash means a run that blows
    its deadline is reported as a graded failure with a real error code, rather than
    as a harness timeout that says nothing about why.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + task.timeout_seconds + 30
    while loop.time() < deadline:
        await _drive_approvals(run_id, task, queue)
        async with session_scope() as session:
            status = (
                await session.execute(sa.select(AgentRun.status).where(AgentRun.id == run_id))
            ).scalar_one()
        if RunStatus(status) in TERMINAL:
            return RunStatus(status)
        await asyncio.sleep(poll)
    return RunStatus.RUNNING


async def _collect(run_id: uuid.UUID) -> dict[str, Any]:
    """Everything observable about a finished run, from the store."""
    async with session_scope() as session:
        run = (
            await session.execute(sa.select(AgentRun).where(AgentRun.id == run_id))
        ).scalar_one()
        steps = list(
            (await session.execute(sa.select(Step).where(Step.run_id == run_id).order_by(Step.seq)))
            .scalars()
            .all()
        )
        approval_count = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(ApprovalRequest)
                .where(ApprovalRequest.run_id == run_id)
            )
        ).scalar_one()
        emails = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(EmailOutbox)
                .where(EmailOutbox.run_id == run_id)
            )
        ).scalar_one()
        records = {
            (r.namespace, r.key): r.value
            for r in (
                await session.execute(
                    sa.select(SandboxRecord).where(SandboxRecord.run_id == run_id)
                )
            )
            .scalars()
            .all()
        }

    output = run.output or {}
    latency = (
        (run.ended_at - run.created_at).total_seconds()
        if run.ended_at is not None and run.created_at is not None
        else 0.0
    )
    return {
        "run": run,
        "status": RunStatus(run.status),
        "error_code": (run.error or {}).get("code"),
        "answer": output.get("answer"),
        "tool_sequence": [t for t in output.get("tool_sequence", []) if t],
        "model_turns": int(output.get("model_turns", 0)),
        "tool_calls_reported": int(output.get("tool_calls", 0)),
        "tool_call_steps": sum(1 for s in steps if s.kind == StepKind.TOOL_CALL),
        "steps_used": run.steps_used,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cost_usd": float(run.cost_usd),
        "latency_s": latency,
        # attempt is 1 on a first execution, so anything above that is a retry.
        "retries": sum(max(s.attempt - 1, 0) for s in steps),
        "recoveries": sum(s.recoveries for s in steps),
        "approvals": int(approval_count),
        "emails": int(emails),
        "records": records,
    }


def _grade(task: EvalTask, obs: dict[str, Any]) -> list[str]:
    """Every expectation that did not hold. Empty means the task passed.

    All checks run rather than short-circuiting on the first failure: when a task
    goes wrong you want the whole picture, not the first symptom.
    """
    expect = task.expect
    failures: list[str] = []

    if obs["status"] is not expect.status:
        failures.append(f"status {obs['status']} != expected {expect.status}")
    if expect.error_code is not None and obs["error_code"] != expect.error_code:
        failures.append(f"error code {obs['error_code']!r} != expected {expect.error_code!r}")

    answer = (obs["answer"] or "").lower()
    for needle in expect.answer_contains:
        if needle.lower() not in answer:
            failures.append(f"answer missing {needle!r}")

    if expect.tool_sequence is not None:
        actual = tuple(obs["tool_sequence"])
        if actual != expect.tool_sequence:
            failures.append(
                f"tool sequence {list(actual)} != expected {list(expect.tool_sequence)}"
            )
    if obs["tool_call_steps"] < expect.min_tool_calls:
        failures.append(
            f"{obs['tool_call_steps']} tool call steps < expected {expect.min_tool_calls}"
        )

    if expect.emails_sent is not None and obs["emails"] != expect.emails_sent:
        # The check the whole idempotency story rests on, so it names itself clearly.
        failures.append(
            f"{obs['emails']} email(s) in the outbox != expected {expect.emails_sent}"
        )

    for namespace, key, value in expect.records:
        actual_value = obs["records"].get((namespace, key))
        if actual_value is None:
            failures.append(f"no record {namespace}/{key} written by this run")
        elif actual_value != value:
            failures.append(f"record {namespace}/{key} = {actual_value!r} != expected {value!r}")

    if obs["approvals"] != expect.approvals:
        failures.append(f"{obs['approvals']} approval gate(s) != expected {expect.approvals}")
    if obs["retries"] < expect.min_retries:
        failures.append(f"{obs['retries']} retries < expected {expect.min_retries}")
    if obs["recoveries"] < expect.min_recoveries:
        failures.append(f"{obs['recoveries']} recoveries < expected {expect.min_recoveries}")

    return failures


async def run_task(task: EvalTask, queue: StepQueue, *, live: bool = False) -> TaskResult:
    """Submit, wait, grade."""
    started = time.perf_counter()
    try:
        run_id = await _submit(task, queue, live=live)
    except Exception as exc:  # pragma: no cover - a submission failure is a hard stop
        return TaskResult(
            task_id=task.id,
            tier=str(task.tier),
            intent=task.intent,
            run_id=None,
            passed=False,
            status="submit_failed",
            failures=[f"submission raised {type(exc).__name__}: {exc}"],
            wall_s=time.perf_counter() - started,
        )

    final = await _await_terminal(run_id, task, queue)
    wall = time.perf_counter() - started
    obs = await _collect(run_id)

    failures = _grade(task, obs)
    if final not in TERMINAL:
        failures.insert(0, "run did not reach a terminal state before the harness timeout")

    result = TaskResult(
        task_id=task.id,
        tier=str(task.tier),
        intent=task.intent,
        run_id=run_id,
        passed=not failures,
        status=str(obs["status"]),
        error_code=obs["error_code"],
        answer=obs["answer"],
        steps_used=obs["steps_used"],
        model_turns=obs["model_turns"],
        tool_calls=obs["tool_call_steps"],
        tool_sequence=list(obs["tool_sequence"]),
        input_tokens=obs["input_tokens"],
        output_tokens=obs["output_tokens"],
        cost_usd=obs["cost_usd"],
        latency_s=obs["latency_s"],
        wall_s=wall,
        retries=obs["retries"],
        recoveries=obs["recoveries"],
        approvals=obs["approvals"],
        emails=obs["emails"],
        failures=failures,
    )
    log.info(
        "eval_task_done",
        task=task.id,
        passed=result.passed,
        status=result.status,
        steps=result.steps_used,
        cost_usd=round(result.cost_usd, 6),
    )
    return result


async def run_suite(
    tasks: list[EvalTask],
    queue: StepQueue,
    *,
    live: bool = False,
    concurrency: int = 1,
    on_result: Any = None,
) -> list[TaskResult]:
    """Run the catalog.

    Concurrency defaults to 1 so the reported latencies mean "how long this task
    takes", not "how long it takes while sharing two workers with three other runs".
    Raise it to exercise the engine under load; the pass/fail column stays valid
    either way, the timing column does not.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[TaskResult | None] = [None] * len(tasks)

    async def one(index: int, task: EvalTask) -> None:
        async with semaphore:
            result = await run_task(task, queue, live=live)
        results[index] = result
        if on_result is not None:
            on_result(result)

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return [r for r in results if r is not None]


__all__ = ["Cluster", "TaskResult", "run_suite", "run_task"]
