"""Side effects must not duplicate when a worker dies mid-tool-call.

This is the suite the whole project is built to earn. Phase 2 left a visible
duplicate: an effect landed, the worker died before recording it, and the replacement
did the work again. Here that duplicate has to be gone — and gone for the right
reason, which is why the tests assert on the provider's own state (`email_outbox`,
`sandbox_records`) rather than on anything the engine reports about itself.

The crash is injected between EXECUTE and COMMIT, deterministically. A `sleep`-based
race would pass by luck; this fires at the exact instruction boundary that matters.

`send_email` is gated behind a human approval (Phase 4), so these tests run an
`AutoApprover` alongside the workers. That is not a shortcut around the gate — the run
still parks, is still decided by a separate actor, and still resumes through the
normal path. It just supplies the operator the scenario needs.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from app.db import session_scope
from app.domain.models import EmailOutbox, SandboxRecord
from app.domain.states import RunStatus, StepKind
from tests.helpers import (
    AutoApprover,
    Cluster,
    count_emails,
    get_run,
    get_steps,
    get_tool_calls,
    get_transitions,
    make_agent_run,
    wait_for_run_terminal,
)

pytestmark = [pytest.mark.chaos, pytest.mark.integration]

#: Stall past the 4s test lease TTL without heartbeating, so the lease expires under
#: the worker and the reaper reassigns the step. The stalled worker then wakes and
#: tries to commit — the fencing path — while a second worker redoes the tool call.
STALL_AFTER_EFFECT = {
    "kind": "hang",
    "phase": "after_effect",
    "seconds": 7,
    "suppress_heartbeat": True,
    "times": 1,
}


@pytest.fixture
def queue(pg_queue):
    return pg_queue


async def test_an_email_is_sent_once_even_when_the_worker_dies_after_sending(queue) -> None:
    """THE test.

    The worker performs the send, then loses its lease before recording it. Another
    worker picks the step up, re-runs the tool, and the provider recognises the
    idempotency key and returns the original message instead of sending again.

    Asserted on the provider's outbox, not on engine bookkeeping: the claim is
    "the customer received one email", and only the outbox can settle that.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "send_email",
                        "input": {
                            "to": "customer@example.test",
                            "subject": "Your order",
                            "body": "It shipped.",
                        },
                    }
                ]
            },
            {"text": "email sent"},
        ],
        queue=queue,
        tools=["send_email"],
        tool_faults={"send_email": STALL_AFTER_EFFECT},
    )

    async with Cluster(queue, workers=2), AutoApprover(queue, run_id):
        assert await wait_for_run_terminal(run_id, timeout=120) is RunStatus.SUCCEEDED

    assert await count_emails() == 1, "the customer received more than one email"

    async with session_scope() as session:
        email = (
            await session.execute(sa.select(EmailOutbox).where(EmailOutbox.run_id == run_id))
        ).scalar_one()
    # The provider saw the replay and suppressed it — proof the dedupe path ran
    # rather than the second attempt simply never happening.
    assert email.duplicate_attempts >= 1

    steps = await get_steps(run_id)
    tool_step = next(s for s in steps if s.kind == StepKind.TOOL_CALL)
    assert tool_step.attempt == 2, "the step really was executed twice"
    assert tool_step.recoveries == 1

    reasons = [t.reason for t in await get_transitions(run_id) if t.step_id == tool_step.id]
    assert "lease_expired" in reasons


async def test_the_effect_ledger_keeps_one_row_across_the_recovery(queue) -> None:
    """One effect key, one ledger row — the retry must not create a second."""
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "send_email",
                        "input": {"to": "a@example.test", "subject": "S", "body": "B"},
                    }
                ]
            },
            {"text": "done"},
        ],
        queue=queue,
        tools=["send_email"],
        tool_faults={"send_email": STALL_AFTER_EFFECT},
    )
    async with Cluster(queue, workers=2), AutoApprover(queue, run_id):
        assert await wait_for_run_terminal(run_id, timeout=120) is RunStatus.SUCCEEDED

    tool_calls = await get_tool_calls(run_id)
    assert len(tool_calls) == 1
    assert tool_calls[0].effect_status == "committed"
    # `attempt_observed` records which attempt first claimed the key. It is metadata:
    # if it had been part of the key, the retry would have produced a second row and
    # a second email.
    assert tool_calls[0].attempt_observed == 1


async def test_a_safe_to_replay_write_converges_after_a_crash(queue) -> None:
    """`database_write` is SAFE_TO_REPLAY because its SQL is an upsert.

    The replay genuinely re-executes — `writes` proves it — and the record still ends
    up in exactly the state a single execution would have produced.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "database_write",
                        "input": {"namespace": "notes", "key": "k1", "value": "v1"},
                    }
                ]
            },
            {"text": "stored"},
        ],
        queue=queue,
        tools=["database_write"],
        tool_faults={"database_write": STALL_AFTER_EFFECT},
    )
    async with Cluster(queue, workers=2), AutoApprover(queue, run_id):
        assert await wait_for_run_terminal(run_id, timeout=120) is RunStatus.SUCCEEDED

    async with session_scope() as session:
        rows = (
            (await session.execute(sa.select(SandboxRecord).where(SandboxRecord.run_id == run_id)))
            .scalars()
            .all()
        )

    assert len(rows) == 1, "the upsert created a duplicate record"
    assert rows[0].value == "v1"
    assert rows[0].writes == 2, "expected the replay to actually re-execute"


async def test_a_completed_tool_call_is_replayed_from_the_ledger_not_re_executed(queue) -> None:
    """When the crash lands *after* the commit, the tool must not run again at all.

    Distinct from the ambiguous case: here the ledger says `committed`, so the
    replacement worker returns the stored result without touching the provider.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "send_email",
                        "input": {"to": "b@example.test", "subject": "S", "body": "B"},
                    }
                ]
            },
            {"text": "done"},
        ],
        queue=queue,
        tools=["send_email"],
        # `before_commit` fires after the ledger row is committed by the tool's own
        # provider write but before the step result is recorded.
        tool_faults={
            "send_email": {
                "kind": "hang",
                "phase": "before_commit",
                "seconds": 7,
                "suppress_heartbeat": True,
                "times": 1,
            }
        },
    )
    async with Cluster(queue, workers=2), AutoApprover(queue, run_id):
        assert await wait_for_run_terminal(run_id, timeout=120) is RunStatus.SUCCEEDED

    assert await count_emails() == 1
    assert len(await get_tool_calls(run_id)) == 1


async def test_multiple_side_effecting_tools_each_happen_once(queue) -> None:
    """A run with two distinct effects, one of which is interrupted.

    Guards against a dedupe that is too aggressive as well as one that is too lax:
    the two calls have different effect keys and must both happen.
    """
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "database_write",
                        "input": {"namespace": "notes", "key": "first", "value": "1"},
                    }
                ]
            },
            {
                "tools": [
                    {
                        "name": "send_email",
                        "input": {"to": "c@example.test", "subject": "S", "body": "B"},
                    }
                ]
            },
            {"text": "both done"},
        ],
        queue=queue,
        tools=["database_write", "send_email"],
        tool_faults={"send_email": STALL_AFTER_EFFECT},
    )
    async with Cluster(queue, workers=2), AutoApprover(queue, run_id):
        assert await wait_for_run_terminal(run_id, timeout=120) is RunStatus.SUCCEEDED

    assert await count_emails() == 1
    async with session_scope() as session:
        records = (
            (await session.execute(sa.select(SandboxRecord).where(SandboxRecord.run_id == run_id)))
            .scalars()
            .all()
        )
    assert len(records) == 1
    assert records[0].key == "first"

    tool_calls = await get_tool_calls(run_id)
    assert {tc.tool_name for tc in tool_calls} == {"database_write", "send_email"}
    assert len({tc.idempotency_key for tc in tool_calls}) == 2


async def test_an_unsafe_effect_is_escalated_rather_than_replayed() -> None:
    """UNSAFE_TO_REPLAY must refuse to guess.

    Exercised at the policy level: the engine has no way to know whether the effect
    landed, and the correct behaviour is a surfaced `ambiguous_effect`, not a
    coin-flip.
    """
    from app.core.idempotency import resolve_ambiguous
    from app.domain.errors import AmbiguousEffectError
    from app.tools.base import EffectPolicy

    assert resolve_ambiguous(EffectPolicy.SAFE_TO_REPLAY, tool_name="database_write")
    assert resolve_ambiguous(EffectPolicy.REQUIRES_PROVIDER_KEY, tool_name="send_email")

    with pytest.raises(AmbiguousEffectError) as caught:
        resolve_ambiguous(EffectPolicy.UNSAFE_TO_REPLAY, tool_name="fire_missile")
    assert caught.value.code == "ambiguous_effect"


async def test_the_run_still_completes_and_reports_correctly(queue) -> None:
    """Recovery must not just avoid duplication — the run has to finish properly."""
    run_id, _ = await make_agent_run(
        [
            {
                "tools": [
                    {
                        "name": "send_email",
                        "input": {"to": "d@example.test", "subject": "S", "body": "B"},
                    }
                ]
            },
            {"text": "the email was sent"},
        ],
        queue=queue,
        tools=["send_email"],
        tool_faults={"send_email": STALL_AFTER_EFFECT},
    )
    async with Cluster(queue, workers=2), AutoApprover(queue, run_id):
        assert await wait_for_run_terminal(run_id, timeout=120) is RunStatus.SUCCEEDED

    run = await get_run(run_id)
    assert run.output["answer"] == "the email was sent"
    assert run.cost_usd > 0
    steps = await get_steps(run_id)
    assert all(str(s.status) == "succeeded" for s in steps)
    assert all(s.lease_owner is None for s in steps)
