"""Crash recovery with real worker processes that really die.

`test_crash_recovery.py` simulates a stalled worker in-process. This suite goes
further: it starts actual OS processes and one of them terminates without warning,
mid-step, holding a lease. Nothing runs on the way out — no atexit hook, no `finally`
block, no lease release. Recovery has to come from lease expiry or not at all.

Uses the Postgres queue backend so the subprocesses need no coordination about
stream names, which has the side benefit of proving recovery on the Redis-free path.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import psutil
import pytest
import sqlalchemy as sa

from app.db import session_scope
from app.domain.models import DummyEffect, Step
from app.domain.states import RunStatus, StepStatus
from app.queue.postgres import PostgresQueue
from app.reaper.reaper import Reaper
from tests.helpers import get_effects, get_run, get_steps, get_transitions, make_run, wait_for

pytestmark = [pytest.mark.chaos, pytest.mark.integration]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env["AGENTORC_QUEUE_BACKEND"] = "postgres"
    env["AGENTORC_ENABLE_FAULT_INJECTION"] = "true"
    env["PYTHONPATH"] = REPO_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _spawn_worker() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker"],
        cwd=REPO_ROOT,
        env=_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


class _ReaperTask:
    """The reaper, in-process, so the test can control its lifetime."""

    def __init__(self) -> None:
        self.reaper = Reaper(PostgresQueue())

    async def __aenter__(self) -> _ReaperTask:
        self._task = asyncio.create_task(self.reaper.run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.reaper.request_stop()
        await asyncio.wait([self._task], timeout=10)
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


async def test_a_worker_that_dies_mid_step_does_not_lose_the_run(clean_db: None) -> None:
    """Self-inflicted `os._exit` at the exact moment after the side effect.

    Deterministic where an external kill would be a race: the fault fires inside the
    handler, in the window between the effect landing and the result being committed.
    """
    plan = [
        {"label": "before"},
        {
            "label": "victim",
            "fault": {"kind": "crash", "phase": "after_effect", "times": 1},
        },
        {"label": "after"},
    ]
    run_id, _ = await make_run(plan)

    procs = [_spawn_worker() for _ in range(2)]
    try:
        async with _ReaperTask():
            done = await wait_for(lambda: _run_is_terminal(run_id), timeout=120, interval=0.5)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    assert done, "run never reached a terminal state"
    assert RunStatus((await get_run(run_id)).status) is RunStatus.SUCCEEDED

    # Exactly one worker died of the injected crash; the other survived to finish.
    exit_codes = [p.returncode for p in procs]
    assert 137 in exit_codes, f"expected a worker to die with code 137, got {exit_codes}"

    steps = await get_steps(run_id)
    assert [str(s.status) for s in steps] == ["succeeded"] * 4

    effects = await get_effects(run_id)
    # The completed step before the crash was NOT redone.
    assert effects["before"].executions == 1
    assert effects["after"].executions == 1
    # The interrupted step was redone by the surviving worker, and the two executions
    # were performed by two different workers.
    assert effects["victim"].executions == 2

    victim = steps[1]
    assert victim.recoveries == 1
    assert victim.attempt == 2
    reasons = [t.reason for t in await get_transitions(run_id) if t.step_id == victim.id]
    assert "lease_expired" in reasons


async def test_killing_a_worker_from_outside_is_survivable(clean_db: None) -> None:
    """No fault injection: the test finds a worker holding a lease and kills it.

    This is the version with nothing staged inside the process — the worker has no
    idea it is about to die.
    """
    plan = [{"label": "before"}, {"label": "victim", "sleep_ms": 20_000}, {"label": "after"}]
    run_id, _ = await make_run(plan)

    procs = [_spawn_worker() for _ in range(2)]
    try:
        async with _ReaperTask():
            owner = await _wait_for_lease_owner(run_id, label="victim", timeout=60)
            assert owner is not None, "no worker ever picked up the victim step"

            killed_pid = _kill_lease_owner(procs, owner)
            assert killed_pid is not None, f"lease owner {owner} is not one of our processes"

            done = await wait_for(lambda: _run_is_terminal(run_id), timeout=180, interval=0.5)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    assert done, "run never recovered after its worker was killed"
    assert RunStatus((await get_run(run_id)).status) is RunStatus.SUCCEEDED

    effects = await get_effects(run_id)
    assert effects["before"].executions == 1, "a completed step was re-executed"
    assert effects["victim"].executions == 2
    assert effects["after"].executions == 1

    steps = await get_steps(run_id)
    assert steps[1].recoveries >= 1
    assert steps[1].lease_owner is None


def _worker_pid(worker_id: str) -> int | None:
    """Worker ids are "<host>-<pid>-<rand>"; the pid is the second-to-last field."""
    try:
        return int(worker_id.split("-")[-2])
    except (ValueError, IndexError):
        return None


def _kill_lease_owner(procs: list[subprocess.Popen[bytes]], worker_id: str) -> int | None:
    """Kill the process actually running the worker that holds the lease.

    `Popen.pid` cannot be matched against the worker id directly: on Windows a venv's
    `python.exe` is a launcher that runs the real interpreter as a *child*, so the
    process that reports `os.getpid()` in its worker id is one level down. We walk
    each spawned process's descendants to find it, which also keeps the kill safely
    scoped to processes this test started.
    """
    target_pid = _worker_pid(worker_id)
    if target_pid is None:
        return None
    for proc in procs:
        try:
            parent = psutil.Process(proc.pid)
            candidates = [parent, *parent.children(recursive=True)]
        except psutil.NoSuchProcess:
            continue
        for candidate in candidates:
            if candidate.pid == target_pid:
                # SIGKILL / TerminateProcess: unmaskable, no cleanup, no lease release.
                candidate.kill()
                return target_pid
    return None


async def _run_is_terminal(run_id) -> bool:
    run = await get_run(run_id)
    return RunStatus(run.status) in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


async def _wait_for_lease_owner(run_id, *, label: str, timeout: float) -> str | None:
    """Wait until the named step is RUNNING, its effect has landed, and we know who
    holds the lease. Waiting for the effect (not just the status) puts the kill
    squarely in the effect-happened-but-result-not-committed window."""
    holder: list[str] = []

    async def ready() -> bool:
        async with session_scope() as s:
            row = (
                await s.execute(
                    sa.select(Step.lease_owner)
                    .join(DummyEffect, DummyEffect.step_id == Step.id)
                    .where(
                        Step.run_id == run_id,
                        Step.status == StepStatus.RUNNING,
                        DummyEffect.label == label,
                        DummyEffect.executions >= 1,
                    )
                )
            ).scalar_one_or_none()
        if row:
            holder.append(row)
            return True
        return False

    await wait_for(ready, timeout=timeout, interval=0.2)
    return holder[0] if holder else None
