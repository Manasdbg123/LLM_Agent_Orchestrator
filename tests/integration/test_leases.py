"""Lease mechanics against a real Postgres.

These are the tests that would be worthless against a mock: what is being asserted is
the behaviour of an atomic conditional UPDATE under concurrency, of a partial unique
index, and of `now()` evaluated server-side. A fake database would simply agree with
whatever the code did.
"""

from __future__ import annotations

import asyncio

import pytest
import sqlalchemy as sa

from app.core.leases import (
    claim_step,
    find_expired_leases,
    find_runnable_steps,
    heartbeat,
    reclaim_expired_lease,
)
from app.core.transitions import transition_step
from app.db import session_scope
from app.domain.models import StateTransition, Step
from app.domain.states import StepStatus
from tests.helpers import make_run

pytestmark = [pytest.mark.integration]


async def _expire_lease(step_id) -> None:
    """Push a lease into the past. Simulates the passage of `lease_ttl` without
    making the test wait for it."""
    async with session_scope() as s:
        await s.execute(
            sa.text(
                "UPDATE steps SET lease_expires_at = now() - interval '1 second' WHERE id = :i"
            ),
            {"i": step_id},
        )


async def test_claim_marks_the_step_running_and_takes_ownership(clean_db: None) -> None:
    _run_id, step_id = await make_run([{"label": "a"}])
    async with session_scope() as s:
        lease = await claim_step(s, step_id=step_id, worker_id="w1", ttl_seconds=30)

    assert lease is not None
    assert lease.attempt == 1
    assert lease.epoch == 1

    async with session_scope() as s:
        step = (await s.execute(sa.select(Step).where(Step.id == step_id))).scalar_one()
    assert StepStatus(step.status) is StepStatus.RUNNING
    assert step.lease_owner == "w1"
    assert step.lease_expires_at is not None


async def test_only_one_of_two_concurrent_claims_wins(clean_db: None) -> None:
    """The core anti-double-execution property.

    Two workers race for the same step in genuinely concurrent transactions; Postgres
    row locks must serialize them so exactly one comes away with a lease.
    """
    _run_id, step_id = await make_run([{"label": "a"}])

    async def attempt(worker_id: str):
        async with session_scope() as s:
            return await claim_step(s, step_id=step_id, worker_id=worker_id, ttl_seconds=30)

    first, second = await asyncio.gather(attempt("w1"), attempt("w2"))
    winners = [lease for lease in (first, second) if lease is not None]
    assert len(winners) == 1, "two workers claimed the same step"
    assert winners[0].attempt == 1


async def test_many_concurrent_claims_still_yield_exactly_one_winner(clean_db: None) -> None:
    _run_id, step_id = await make_run([{"label": "a"}])

    async def attempt(i: int):
        async with session_scope() as s:
            return await claim_step(s, step_id=step_id, worker_id=f"w{i}", ttl_seconds=30)

    results = await asyncio.gather(*(attempt(i) for i in range(12)))
    assert sum(1 for r in results if r is not None) == 1


async def test_a_live_lease_blocks_further_claims(clean_db: None) -> None:
    _run_id, step_id = await make_run([{"label": "a"}])
    async with session_scope() as s:
        assert await claim_step(s, step_id=step_id, worker_id="w1", ttl_seconds=30) is not None
    async with session_scope() as s:
        assert await claim_step(s, step_id=step_id, worker_id="w2", ttl_seconds=30) is None


async def test_an_expired_lease_can_be_claimed_by_someone_else(clean_db: None) -> None:
    _run_id, step_id = await make_run([{"label": "a"}])
    async with session_scope() as s:
        first = await claim_step(s, step_id=step_id, worker_id="w1", ttl_seconds=30)
    await _expire_lease(step_id)

    # Still RUNNING, but unowned in practice. The reaper normally resets it; this
    # asserts the claim predicate alone already refuses to trust a dead lease.
    async with session_scope() as s:
        rows = await find_expired_leases(s, limit=10)
        assert [r.id for r in rows] == [step_id]
        assert await reclaim_expired_lease(s, row=rows[0]) == str(StepStatus.PENDING)

    async with session_scope() as s:
        second = await claim_step(s, step_id=step_id, worker_id="w2", ttl_seconds=30)
    assert second is not None
    assert second.epoch == first.epoch + 1, "epoch must advance so the old owner is fenced"
    assert second.attempt == 2


async def test_heartbeat_extends_only_for_the_current_owner_and_epoch(clean_db: None) -> None:
    _run_id, step_id = await make_run([{"label": "a"}])
    async with session_scope() as s:
        lease = await claim_step(s, step_id=step_id, worker_id="w1", ttl_seconds=30)

    async with session_scope() as s:
        assert await heartbeat(
            s, step_id=step_id, worker_id="w1", epoch=lease.epoch, ttl_seconds=30
        )
        assert not await heartbeat(
            s, step_id=step_id, worker_id="w2", epoch=lease.epoch, ttl_seconds=30
        )
        # A stalled worker waking with a stale epoch must not resurrect its lease.
        assert not await heartbeat(
            s, step_id=step_id, worker_id="w1", epoch=lease.epoch - 1, ttl_seconds=30
        )


async def test_a_fenced_worker_cannot_commit_its_result(clean_db: None) -> None:
    """The failure mode most naive lease implementations miss.

    w1 stalls, its lease expires, w2 takes over. w1 then wakes up and tries to write
    the result it computed. That write must not land.
    """
    _run_id, step_id = await make_run([{"label": "a"}])
    async with session_scope() as s:
        stale = await claim_step(s, step_id=step_id, worker_id="w1", ttl_seconds=30)

    await _expire_lease(step_id)
    async with session_scope() as s:
        rows = await find_expired_leases(s, limit=10)
        await reclaim_expired_lease(s, row=rows[0])
    async with session_scope() as s:
        current = await claim_step(s, step_id=step_id, worker_id="w2", ttl_seconds=30)

    async with session_scope() as s:
        rejected = await transition_step(
            s,
            step_id=step_id,
            to=StepStatus.SUCCEEDED,
            reason="completed",
            actor="w1",
            expect=StepStatus.RUNNING,
            require_owner="w1",
            require_epoch=stale.epoch,
            values={"output": {"from": "the zombie"}},
        )
    assert not rejected.applied
    assert rejected.blocked_by == "fenced"

    async with session_scope() as s:
        accepted = await transition_step(
            s,
            step_id=step_id,
            to=StepStatus.SUCCEEDED,
            reason="completed",
            actor="w2",
            expect=StepStatus.RUNNING,
            require_owner="w2",
            require_epoch=current.epoch,
            values={"output": {"from": "the rightful owner"}},
        )
    assert accepted.applied

    async with session_scope() as s:
        step = (await s.execute(sa.select(Step).where(Step.id == step_id))).scalar_one()
    assert step.output == {"from": "the rightful owner"}


async def test_recovery_budget_is_separate_from_the_retry_budget(clean_db: None) -> None:
    """A step that keeps killing its worker must not silently eat the user's retries,
    and must eventually be abandoned instead of cycling forever."""
    _run_id, step_id = await make_run([{"label": "poison"}])

    async with session_scope() as s:
        await s.execute(
            sa.update(Step).where(Step.id == step_id).values(max_recoveries=2, max_attempts=5)
        )

    for expected in (1, 2):
        async with session_scope() as s:
            assert await claim_step(s, step_id=step_id, worker_id="w", ttl_seconds=30)
        await _expire_lease(step_id)
        async with session_scope() as s:
            rows = await find_expired_leases(s, limit=10)
            assert await reclaim_expired_lease(s, row=rows[0]) == str(StepStatus.PENDING)
        async with session_scope() as s:
            step = (await s.execute(sa.select(Step).where(Step.id == step_id))).scalar_one()
        assert step.recoveries == expected

    # Third death exceeds max_recoveries: abandoned, not retried forever.
    async with session_scope() as s:
        assert await claim_step(s, step_id=step_id, worker_id="w", ttl_seconds=30)
    await _expire_lease(step_id)
    async with session_scope() as s:
        rows = await find_expired_leases(s, limit=10)
        assert await reclaim_expired_lease(s, row=rows[0]) == str(StepStatus.FAILED)

    async with session_scope() as s:
        step = (await s.execute(sa.select(Step).where(Step.id == step_id))).scalar_one()
    assert StepStatus(step.status) is StepStatus.FAILED
    assert step.error["code"] == "too_many_recoveries"


async def test_runnable_scan_respects_the_enqueue_grace(clean_db: None) -> None:
    """The reaper must not race the enqueue that just happened inline."""
    run_id, step_id = await make_run([{"label": "a"}])

    async with session_scope() as s:
        fresh = await find_runnable_steps(s, limit=10, grace_seconds=5)
    assert fresh == [], "a just-created step should not be republished immediately"

    async with session_scope() as s:
        await s.execute(
            sa.text("UPDATE steps SET available_at = now() - interval '10 seconds' WHERE id = :i"),
            {"i": step_id},
        )
        stale = await find_runnable_steps(s, limit=10, grace_seconds=5)
    assert [r.id for r in stale] == [step_id]
    assert stale[0].run_id == run_id


async def test_every_status_change_is_recorded_in_the_audit_log(clean_db: None) -> None:
    run_id, step_id = await make_run([{"label": "a"}])
    async with session_scope() as s:
        lease = await claim_step(s, step_id=step_id, worker_id="w1", ttl_seconds=30)
        await transition_step(
            s,
            step_id=step_id,
            to=StepStatus.SUCCEEDED,
            reason="completed",
            actor="w1",
            expect=StepStatus.RUNNING,
            require_owner="w1",
            require_epoch=lease.epoch,
            values={"output": {}},
        )

    async with session_scope() as s:
        rows = (
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

    step_rows = [r for r in rows if r.step_id == step_id]
    assert [(r.from_status, r.to_status) for r in step_rows] == [
        (None, "pending"),
        ("pending", "running"),
        ("running", "succeeded"),
    ]
    assert step_rows[1].actor == "w1"
    assert step_rows[1].reason == "claimed"


async def test_illegal_transitions_are_refused(clean_db: None) -> None:
    from app.domain.errors import IllegalTransitionError

    _run_id, step_id = await make_run([{"label": "a"}])
    async with session_scope() as s:
        lease = await claim_step(s, step_id=step_id, worker_id="w1", ttl_seconds=30)
        await transition_step(
            s,
            step_id=step_id,
            to=StepStatus.SUCCEEDED,
            reason="completed",
            actor="w1",
            expect=StepStatus.RUNNING,
            require_owner="w1",
            require_epoch=lease.epoch,
        )

    with pytest.raises(IllegalTransitionError):
        async with session_scope() as s:
            await transition_step(
                s,
                step_id=step_id,
                to=StepStatus.RUNNING,
                reason="resurrection",
                actor="w1",
            )


async def test_a_run_cannot_have_two_active_steps(clean_db: None) -> None:
    """The run-serialization invariant, enforced by the database.

    Belt and braces: if a future bug in the engine tries to fan a run out, this index
    turns it into an integrity error instead of two workers advancing one run.
    """
    from sqlalchemy.exc import IntegrityError

    run_id, _step_id = await make_run([{"label": "a"}])
    with pytest.raises(IntegrityError):
        async with session_scope() as s:
            s.add(
                Step(
                    run_id=run_id,
                    seq=99,
                    kind="dummy",
                    status=StepStatus.PENDING,
                    input={},
                )
            )
            await s.flush()
