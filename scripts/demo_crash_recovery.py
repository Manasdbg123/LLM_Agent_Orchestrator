"""Crash-recovery demo you can watch happen.

    python scripts/demo_crash_recovery.py

Starts two real worker processes, submits a three-step run, waits until one worker is
provably mid-step (it has performed its side effect but has not committed its
result), and kills that process outright. No cleanup runs. The lease it held is never
released; it simply goes stale.

Then it prints what the system does about that, straight from the database: the
reaper noticing the expired lease, the step going back into the pool, another worker
picking it up, and the run finishing. It ends with the assertions that matter —
completed steps were not re-executed, and the run reached `succeeded`.

Requires Postgres (Redis is not used: the demo runs on the Postgres queue backend, so
what you are watching cannot be explained by queue redelivery).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("AGENTORC_QUEUE_BACKEND", "postgres")
os.environ.setdefault("AGENTORC_ENABLE_FAULT_INJECTION", "true")
os.environ.setdefault("AGENTORC_LEASE_TTL_SECONDS", "6")
os.environ.setdefault("AGENTORC_REAPER_INTERVAL_SECONDS", "1")
os.environ.setdefault("AGENTORC_ENQUEUE_GRACE_SECONDS", "2")
os.environ.setdefault("AGENTORC_STALLED_MESSAGE_IDLE_MS", "20000")
os.environ.setdefault("AGENTORC_LOG_LEVEL", "WARNING")

import sqlalchemy as sa  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.runs import create_run, get_or_create_definition  # noqa: E402
from app.db import dispose_engine, session_scope  # noqa: E402
from app.domain.models import AgentRun, DummyEffect, StateTransition, Step  # noqa: E402
from app.domain.states import RunStatus, StepKind, StepStatus  # noqa: E402
from app.queue.postgres import PostgresQueue  # noqa: E402
from app.reaper.reaper import Reaper  # noqa: E402
from app.runtime import run as run_async  # noqa: E402

PLAN = [
    {"label": "step-1-completes-normally"},
    {"label": "step-2-worker-gets-killed-here", "sleep_ms": 30_000},
    {"label": "step-3-runs-after-recovery"},
]

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)


def say(msg: str = "", *, color: str = "") -> None:
    print(f"{color}{msg}{RESET}" if color else msg, flush=True)


def rule(title: str) -> None:
    say()
    say(f"{BOLD}{'=' * 78}{RESET}")
    say(f"{BOLD}  {title}{RESET}")
    say(f"{BOLD}{'=' * 78}{RESET}")


def spawn_worker() -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def kill_lease_owner(workers: list[subprocess.Popen[bytes]], worker_id: str) -> int | None:
    """Kill the process actually running the worker that holds the lease.

    The pid in a worker id is `os.getpid()` from inside the worker, which is not
    necessarily `Popen.pid`: on Windows a venv's `python.exe` is a launcher that runs
    the real interpreter as a child. Walking each spawned process's descendants finds
    the right one and keeps the kill scoped to processes this script started.
    """
    try:
        import psutil
    except ImportError:
        say('psutil is required for the demo:  pip install -e ".[dev]"', color=RED)
        raise

    try:
        target_pid = int(worker_id.split("-")[-2])
    except (ValueError, IndexError):
        return None

    for proc in workers:
        try:
            parent = psutil.Process(proc.pid)
            for candidate in [parent, *parent.children(recursive=True)]:
                if candidate.pid == target_pid:
                    candidate.kill()
                    return target_pid
        except psutil.NoSuchProcess:
            continue
    return None


async def create_demo_run() -> str:
    async with session_scope() as session:
        definition = await get_or_create_definition(session, name="crash-demo")
        run, _step = await create_run(
            session,
            definition=definition,
            task="demonstrate crash recovery",
            input={"plan": PLAN},
            first_step_kind=StepKind.DUMMY,
            first_step_input=PLAN[0],
            timeout_seconds=600,
        )
        return str(run.id)


async def wait_until(predicate, *, timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def find_victim(run_id: str) -> tuple[str, str] | None:
    """(lease_owner, step_id) for a step that is running AND has already performed
    its side effect. Killing before the effect would prove nothing interesting."""
    async with session_scope() as s:
        row = (
            await s.execute(
                sa.select(Step.lease_owner, Step.id)
                .join(DummyEffect, DummyEffect.step_id == Step.id)
                .where(
                    Step.run_id == sa.cast(run_id, sa.Uuid),
                    Step.status == StepStatus.RUNNING,
                    DummyEffect.executions >= 1,
                )
            )
        ).one_or_none()
    return (row.lease_owner, str(row.id)) if row else None


async def run_status(run_id: str) -> RunStatus:
    async with session_scope() as s:
        return RunStatus(
            (
                await s.execute(
                    sa.select(AgentRun.status).where(AgentRun.id == sa.cast(run_id, sa.Uuid))
                )
            ).scalar_one()
        )


async def print_timeline(run_id: str) -> list[Step]:
    async with session_scope() as s:
        steps = list(
            (
                await s.execute(
                    sa.select(Step)
                    .where(Step.run_id == sa.cast(run_id, sa.Uuid))
                    .order_by(Step.seq)
                )
            )
            .scalars()
            .all()
        )
        effects = {
            e.step_id: e
            for e in (
                await s.execute(
                    sa.select(DummyEffect).where(DummyEffect.run_id == sa.cast(run_id, sa.Uuid))
                )
            )
            .scalars()
            .all()
        }

    say()
    say(f"{BOLD}{'seq':<4}{'kind':<10}{'status':<11}{'att':<5}{'rec':<5}{'execs':<7}label{RESET}")
    say("-" * 78)
    for step in steps:
        effect = effects.get(step.id)
        execs = effect.executions if effect else 0
        label = effect.label if effect else "-"
        color = GREEN if str(step.status) == "succeeded" else RED
        flag = f"  {YELLOW}<- executed twice{RESET}" if execs > 1 else ""
        say(
            f"{step.seq:<4}{step.kind!s:<10}{color}{step.status!s:<11}{RESET}"
            f"{step.attempt:<5}{step.recoveries:<5}{execs:<7}{label}{flag}"
        )
    return steps


async def print_audit_trail(run_id: str) -> list[StateTransition]:
    async with session_scope() as s:
        rows = list(
            (
                await s.execute(
                    sa.select(StateTransition)
                    .where(StateTransition.run_id == sa.cast(run_id, sa.Uuid))
                    .order_by(StateTransition.id)
                )
            )
            .scalars()
            .all()
        )
    say()
    say(f"{BOLD}{'entity':<7}{'transition':<28}{'reason':<22}actor{RESET}")
    say("-" * 78)
    for t in rows:
        arrow = f"{t.from_status or '-'} -> {t.to_status}"
        highlight = t.reason in {"lease_expired", "claimed"}
        color = YELLOW if t.reason == "lease_expired" else (DIM if not highlight else "")
        say(f"{color}{t.entity:<7}{arrow:<28}{t.reason:<22}{t.actor}{RESET}")
    return rows


async def main() -> int:
    rule("Crash recovery demo")
    say(f"database      : {settings.database_url.split('@')[-1]}")
    say(f"queue backend : {settings.queue_backend}  (Redis is not involved)")
    say(f"lease TTL     : {settings.lease_ttl_seconds}s")
    say(f"reaper poll   : {settings.reaper_interval_seconds}s")

    run_id = await create_demo_run()
    say(f"\nrun           : {run_id}")
    say("plan          : 3 steps; step 2 sleeps for 30s once its side effect has landed")

    workers = [spawn_worker() for _ in range(2)]
    say(f"workers       : started pids {[w.pid for w in workers]}")

    reaper = Reaper(PostgresQueue())
    reaper_task = asyncio.create_task(reaper.run())

    try:
        rule("1. Wait until a worker is provably mid-step")
        found: list[tuple[str, str]] = []

        async def victim_ready() -> bool:
            v = await find_victim(run_id)
            if v and "step-2" in str(await _label_of(v[1])):
                found.append(v)
                return True
            return False

        if not await wait_until(victim_ready, timeout=60):
            say("could not catch a worker mid-step", color=RED)
            return 1

        owner, step_id = found[-1]
        say(f"step {step_id} is RUNNING")
        say(f"lease held by {owner}")
        say("its side effect is already committed; its result is not.", color=YELLOW)

        pid = kill_lease_owner(workers, owner)
        if pid is None:
            say(f"lease owner {owner} is not one of our workers", color=RED)
            return 1

        rule(f"2. Kill worker pid {pid} with no warning and no cleanup")
        say(f"worker {pid} is gone (SIGKILL / TerminateProcess)", color=RED)
        say("it never released its lease; nothing ran on the way out.")

        rule("3. Watch the lease expire and the step come back")
        say(f"waiting up to {settings.lease_ttl_seconds + 5}s for the lease to go stale...")

        async def reclaimed() -> bool:
            # Look for the reclaim *event*, not the current status. The step is
            # handed back to the pool and re-claimed by the surviving worker within
            # milliseconds, so polling for "not running" usually misses the window
            # and reports a failure that did not happen.
            async with session_scope() as s:
                found = (
                    await s.execute(
                        sa.select(StateTransition.id).where(
                            StateTransition.step_id == sa.cast(step_id, sa.Uuid),
                            StateTransition.reason == "lease_expired",
                        )
                    )
                ).first()
            return found is not None

        if await wait_until(reclaimed, timeout=settings.lease_ttl_seconds + 20):
            say("the reaper reclaimed the orphaned step and returned it to the pool", color=GREEN)
        else:
            say("the step was never reclaimed", color=RED)

        rule("4. Wait for the surviving worker to finish the run")
        await wait_until(lambda: _is_terminal(run_id), timeout=180, interval=0.5)
        status = await run_status(run_id)
        say(f"run status: {status}", color=GREEN if status is RunStatus.SUCCEEDED else RED)

        rule("5. What actually happened")
        steps = await print_timeline(run_id)
        await print_audit_trail(run_id)

        rule("6. Assertions")
        async with session_scope() as s:
            effects = {
                e.label: e.executions
                for e in (
                    await s.execute(
                        sa.select(DummyEffect).where(DummyEffect.run_id == sa.cast(run_id, sa.Uuid))
                    )
                )
                .scalars()
                .all()
            }

        checks = [
            ("run reached succeeded", status is RunStatus.SUCCEEDED),
            ("every step is succeeded", all(str(s.status) == "succeeded" for s in steps)),
            (
                "step 1 was NOT re-executed after the crash",
                effects.get("step-1-completes-normally") == 1,
            ),
            (
                "step 3 ran exactly once",
                effects.get("step-3-runs-after-recovery") == 1,
            ),
            (
                "the interrupted step was reassigned (recoveries >= 1)",
                any(s.recoveries >= 1 for s in steps),
            ),
            (
                "the interrupted step ran under two leases (attempt == 2)",
                any(s.attempt == 2 for s in steps),
            ),
            ("no step still holds a lease", all(s.lease_owner is None for s in steps)),
        ]
        ok = True
        for name, passed in checks:
            ok &= passed
            say(f"  {GREEN}PASS{RESET}  {name}" if passed else f"  {RED}FAIL{RESET}  {name}")

        say()
        victim_execs = effects.get("step-2-worker-gets-killed-here")
        say(
            f"{YELLOW}Note:{RESET} the interrupted step's side effect ran {victim_execs} times.\n"
            "That is not a recovery bug — the effect landed, then the process died before\n"
            "the result could be committed, so the replacement worker had no way to know.\n"
            "Eliminating that duplicate is exactly what the Phase 3/4 idempotency ledger\n"
            "does; Phase 2 leaves it visible rather than hiding it."
        )

        say()
        say(f"{BOLD}{GREEN}DEMO PASSED{RESET}" if ok else f"{BOLD}{RED}DEMO FAILED{RESET}")
        return 0 if ok else 1
    finally:
        reaper.request_stop()
        await asyncio.wait([reaper_task], timeout=10)
        reaper_task.cancel()
        await asyncio.gather(reaper_task, return_exceptions=True)
        for w in workers:
            if w.poll() is None:
                w.terminate()
        for w in workers:
            try:
                w.wait(timeout=10)
            except subprocess.TimeoutExpired:
                w.kill()
        await dispose_engine()


async def _label_of(step_id: str) -> str | None:
    async with session_scope() as s:
        return (
            await s.execute(
                sa.select(DummyEffect.label).where(DummyEffect.step_id == sa.cast(step_id, sa.Uuid))
            )
        ).scalar_one_or_none()


async def _is_terminal(run_id: str) -> bool:
    return await run_status(run_id) in {
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }


if __name__ == "__main__":
    raise SystemExit(run_async(main()))
