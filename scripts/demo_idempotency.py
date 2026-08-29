"""Prove that a side effect happens once, even when the worker dies after doing it.

    python scripts/demo_idempotency.py

Phase 2's crash demo ended on an honest caveat: the interrupted step's side effect
ran twice. The worker performed the effect, died before recording it, and the
replacement had no way to know. This demo runs the same scenario with a *real* side
effect — sending an email through a provider that deduplicates on an idempotency key
— and shows the duplicate is gone.

What you are watching:

  1. An agent run whose one tool call sends an email. `send_email` is gated, so the
     run parks and waits for a human before anything leaves the building.
  2. An operator approves it and the run resumes.
  3. The worker sends the email, then stalls without heartbeating. Its lease expires.
  4. The reaper hands the step back; a second worker re-runs the tool call.
  5. The provider recognises the idempotency key and returns the original message
     instead of sending again.
  6. The outbox is queried directly: exactly one email, one ledger row.

The assertion is made against the provider's own outbox table, not against anything
the engine says about itself. The claim is "the customer received one email", and
only the outbox can settle that.

Requires Postgres. Redis is not used.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("AGENTORC_QUEUE_BACKEND", "postgres")
os.environ.setdefault("AGENTORC_ENABLE_FAULT_INJECTION", "true")
os.environ.setdefault("AGENTORC_LLM_PROVIDER", "fake")
os.environ.setdefault("AGENTORC_LEASE_TTL_SECONDS", "5")
os.environ.setdefault("AGENTORC_REAPER_INTERVAL_SECONDS", "1")
os.environ.setdefault("AGENTORC_ENQUEUE_GRACE_SECONDS", "2")
os.environ.setdefault("AGENTORC_STALLED_MESSAGE_IDLE_MS", "20000")
os.environ.setdefault("AGENTORC_LOG_LEVEL", "WARNING")

import sqlalchemy as sa  # noqa: E402

from app.config import settings  # noqa: E402
from app.core import approvals  # noqa: E402
from app.core.runs import create_run, get_or_create_definition  # noqa: E402
from app.db import dispose_engine, session_scope  # noqa: E402
from app.domain.models import (  # noqa: E402
    AgentRun,
    ApprovalRequest,
    EmailOutbox,
    Step,
    ToolCall,
)
from app.domain.states import RunStatus, StepKind  # noqa: E402
from app.llm.fake import encode_script  # noqa: E402
from app.queue.postgres import PostgresQueue  # noqa: E402
from app.reaper.reaper import Reaper  # noqa: E402
from app.runtime import run as run_async  # noqa: E402
from app.worker.worker import Worker  # noqa: E402

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)

RECIPIENT = "customer@example.test"

SCRIPT = [
    {
        "tools": [
            {
                "name": "send_email",
                "input": {
                    "to": RECIPIENT,
                    "subject": "Your order has shipped",
                    "body": "Tracking number 1Z999AA10123456784.",
                },
            }
        ]
    },
    {"text": "I sent the shipping confirmation."},
]

#: Stall for longer than the lease TTL, without heartbeating, immediately after the
#: email has been sent but before the result is recorded. Deterministic: it fires at
#: an exact point in the handler rather than racing a sleep.
STALL = {
    "kind": "hang",
    "phase": "after_effect",
    "seconds": 9,
    "suppress_heartbeat": True,
    "times": 1,
}


def say(msg: str = "", *, color: str = "") -> None:
    print(f"{color}{msg}{RESET}" if color else msg, flush=True)


def rule(title: str) -> None:
    say()
    say(f"{BOLD}{'=' * 78}{RESET}")
    say(f"{BOLD}  {title}{RESET}")
    say(f"{BOLD}{'=' * 78}{RESET}")


async def create_demo_run() -> str:
    async with session_scope() as session:
        definition = await get_or_create_definition(
            session, name="idempotency-demo", model="fake-model", tools=["send_email"]
        )
        run, _step = await create_run(
            session,
            definition=definition,
            task=f"Email the customer their shipping confirmation. {encode_script(SCRIPT)}",
            input={"tool_faults": {"send_email": STALL}},
            first_step_kind=StepKind.AGENT_TURN,
            first_step_input={},
            timeout_seconds=600,
        )
        return str(run.id)


async def wait_until(predicate, *, timeout: float, interval: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def run_status(run_id: str) -> RunStatus:
    async with session_scope() as session:
        return RunStatus(
            (
                await session.execute(
                    sa.select(AgentRun.status).where(AgentRun.id == sa.cast(run_id, sa.Uuid))
                )
            ).scalar_one()
        )


async def main() -> int:
    rule("Idempotent side effects under worker failure")
    say(f"database      : {settings.database_url.split('@')[-1]}")
    say(f"lease TTL     : {settings.lease_ttl_seconds}s")
    say("scenario      : send an email, then lose the lease before recording it")

    run_id = await create_demo_run()
    say(f"run           : {run_id}")

    queue = PostgresQueue()
    workers = [Worker(queue, worker_id=f"demo-w{i}", concurrency=1) for i in range(2)]
    reaper = Reaper(PostgresQueue())
    tasks = [asyncio.create_task(w.run()) for w in workers]
    tasks.append(asyncio.create_task(reaper.run()))

    try:
        rule("1. The run stops at the approval gate")

        async def gated() -> bool:
            async with session_scope() as session:
                return (
                    await session.execute(
                        sa.select(sa.func.count())
                        .select_from(ApprovalRequest)
                        .where(
                            ApprovalRequest.run_id == sa.cast(run_id, sa.Uuid),
                            ApprovalRequest.decision == approvals.Decision.PENDING,
                        )
                    )
                ).scalar_one() > 0

        if not await wait_until(gated, timeout=60):
            say("the run never reached the approval gate", color=RED)
            return 1

        async with session_scope() as session:
            approval = (
                await session.execute(
                    sa.select(ApprovalRequest).where(
                        ApprovalRequest.run_id == sa.cast(run_id, sa.Uuid)
                    )
                )
            ).scalar_one()
        say(f"{approval.tool_name} is waiting for a human: {approval.reason}", color=YELLOW)
        say(f"arguments: to={approval.arguments['to']!r} subject={approval.arguments['subject']!r}")
        say("nothing has been sent yet.", color=DIM)

        rule("2. An operator approves it")
        async with session_scope() as session:
            decision = await approvals.decide(
                session,
                approval_id=approval.id,
                decision=approvals.Decision.APPROVED,
                decided_by="operator@example.test",
                decision_reason="verified the recipient",
            )
        if not decision.applied or decision.resume_step_id is None:
            say(f"could not approve: {decision.detail}", color=RED)
            return 1
        await queue.publish(step_id=decision.resume_step_id, run_id=sa.cast(run_id, sa.Uuid))
        say("approved; the run resumes", color=GREEN)

        rule("3. Wait for the email to actually be sent")

        async def sent() -> bool:
            async with session_scope() as session:
                return (
                    await session.execute(
                        sa.select(sa.func.count())
                        .select_from(EmailOutbox)
                        .where(EmailOutbox.run_id == sa.cast(run_id, sa.Uuid))
                    )
                ).scalar_one() > 0

        if not await wait_until(sent, timeout=60):
            say("the email was never sent", color=RED)
            return 1

        async with session_scope() as session:
            email = (
                await session.execute(
                    sa.select(EmailOutbox).where(EmailOutbox.run_id == sa.cast(run_id, sa.Uuid))
                )
            ).scalar_one()
        say(f"provider delivered {email.message_id} to {email.recipient}", color=GREEN)
        say("the worker is now stalled, holding an uncommitted result.", color=YELLOW)

        rule("4. The lease expires and the step is reassigned")
        say(f"waiting up to {settings.lease_ttl_seconds + 15}s ...")

        async def reclaimed() -> bool:
            async with session_scope() as session:
                return (
                    await session.execute(
                        sa.select(sa.func.max(Step.recoveries)).where(
                            Step.run_id == sa.cast(run_id, sa.Uuid)
                        )
                    )
                ).scalar_one() or 0 > 0

        if await wait_until(reclaimed, timeout=settings.lease_ttl_seconds + 15):
            say("the reaper reclaimed the orphaned step", color=GREEN)
        else:
            say("the step was never reclaimed", color=RED)

        rule("5. A second worker re-runs the tool call")
        await wait_until(lambda: _terminal(run_id), timeout=180, interval=0.5)
        status = await run_status(run_id)
        say(f"run status: {status}", color=GREEN if status is RunStatus.SUCCEEDED else RED)

        rule("6. What the provider actually saw")
        async with session_scope() as session:
            emails = (
                (
                    await session.execute(
                        sa.select(EmailOutbox).where(EmailOutbox.run_id == sa.cast(run_id, sa.Uuid))
                    )
                )
                .scalars()
                .all()
            )
            tool_calls = (
                (
                    await session.execute(
                        sa.select(ToolCall).where(ToolCall.run_id == sa.cast(run_id, sa.Uuid))
                    )
                )
                .scalars()
                .all()
            )
            steps = (
                (
                    await session.execute(
                        sa.select(Step)
                        .where(Step.run_id == sa.cast(run_id, sa.Uuid))
                        .order_by(Step.seq)
                    )
                )
                .scalars()
                .all()
            )

        say()
        say(f"{BOLD}{'seq':<4}{'kind':<12}{'status':<11}{'att':<5}{'rec':<5}{RESET}")
        say("-" * 78)
        for step in steps:
            colour = GREEN if str(step.status) == "succeeded" else RED
            flag = f"  {YELLOW}<- interrupted here{RESET}" if step.recoveries else ""
            say(
                f"{step.seq:<4}{step.kind!s:<12}{colour}{step.status!s:<11}{RESET}"
                f"{step.attempt:<5}{step.recoveries:<5}{flag}"
            )

        say()
        for email in emails:
            say(
                f"outbox: {email.message_id}  to={email.recipient}  "
                f"duplicate_attempts={email.duplicate_attempts}"
            )
        for call in tool_calls:
            say(
                f"ledger: {call.tool_name}  status={call.effect_status}  "
                f"key={call.idempotency_key[:16]}...  first_claimed_on_attempt="
                f"{call.attempt_observed}"
            )

        rule("7. Assertions")
        tool_step = next((s for s in steps if s.kind == StepKind.TOOL_CALL), None)
        checks = [
            ("run reached succeeded", status is RunStatus.SUCCEEDED),
            ("the send was gated until a human approved it", decision.applied),
            ("EXACTLY ONE email was sent", len(emails) == 1),
            (
                "the provider suppressed a duplicate send",
                bool(emails) and emails[0].duplicate_attempts >= 1,
            ),
            ("exactly one effect-ledger row", len(tool_calls) == 1),
            (
                "the ledger row is committed",
                bool(tool_calls) and tool_calls[0].effect_status == "committed",
            ),
            (
                "the step really did execute twice",
                tool_step is not None and tool_step.attempt == 2,
            ),
            (
                "the step was reassigned by the reaper",
                tool_step is not None and tool_step.recoveries >= 1,
            ),
        ]
        ok = True
        for name, passed in checks:
            ok &= passed
            say(f"  {GREEN}PASS{RESET}  {name}" if passed else f"  {RED}FAIL{RESET}  {name}")

        say()
        say(
            f"{DIM}The step executed twice and the effect happened once. That gap is\n"
            f"closed by the effect ledger plus a provider that honours the key — not\n"
            f"by preventing the re-execution, which is impossible to guarantee.{RESET}"
        )
        say()
        say(f"{BOLD}{GREEN}DEMO PASSED{RESET}" if ok else f"{BOLD}{RED}DEMO FAILED{RESET}")
        return 0 if ok else 1
    finally:
        for worker in workers:
            worker.request_stop()
        reaper.request_stop()
        await asyncio.wait(tasks, timeout=20)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await dispose_engine()


async def _terminal(run_id: str) -> bool:
    return await run_status(run_id) in {
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }


if __name__ == "__main__":
    raise SystemExit(run_async(main()))
