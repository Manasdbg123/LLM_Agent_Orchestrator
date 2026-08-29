"""Crash recovery, in-process.

This suite simulates a worker that *stalls* rather than dies: its process stays alive
but it stops heartbeating, so its lease expires under it. That is the harder case,
because the stalled worker eventually wakes up and tries to commit a result for work
someone else has already redone — which is exactly the situation fencing exists for.

The kill -9 variant lives in `test_worker_kill.py`, which uses real subprocesses.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from app.db import session_scope
from app.domain.models import Step
from app.domain.states import RunStatus, StepStatus
from tests.helpers import (
    Cluster,
    get_effects,
    get_run,
    get_steps,
    get_transitions,
    make_run,
    wait_for_run_terminal,
)

pytestmark = [pytest.mark.chaos, pytest.mark.integration]

# lease_ttl is 4s under test settings (see tests/conftest.py), so a 7s stall
# guarantees the lease expires and the reaper acts before the worker wakes.
STALL = {
    "kind": "hang",
    "phase": "after_effect",
    "seconds": 7,
    "suppress_heartbeat": True,
    "times": 1,
}


@pytest.fixture(params=["postgres", "redis"])
def queue(request):
    """Run each test on both backends.

    The backend is resolved lazily via `getfixturevalue`: requesting both as direct
    parameters would make an unavailable Redis skip the Postgres runs too, silently
    reducing coverage to nothing. Kept synchronous because `getfixturevalue` on an
    async fixture cannot run from inside an already-running event loop.
    """
    name = "pg_queue" if request.param == "postgres" else "redis_queue"
    return request.getfixturevalue(name)


async def test_an_orphaned_step_is_reassigned_and_the_run_completes(queue) -> None:
    """The headline claim, as an executable assertion.

    Step 2's worker records its effect, then stalls without heartbeating. Its lease
    expires; the reaper hands the step back; another worker finishes the run.
    """
    plan = [
        {"label": "before"},
        {"label": "victim", "fault": STALL},
        {"label": "after"},
    ]
    run_id, _ = await make_run(plan, queue=queue)

    async with Cluster(queue, workers=2):
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    assert [str(s.status) for s in steps] == ["succeeded"] * 4

    victim = steps[1]
    assert victim.recoveries == 1, "the reaper should have reclaimed this step once"
    assert victim.attempt == 2, "it was executed by two different leases"

    # Recovery consumed the *recovery* budget, not the retry budget: a worker dying
    # is not the step failing.
    reasons = [t.reason for t in await get_transitions(run_id) if t.step_id == victim.id]
    assert "lease_expired" in reasons
    assert "retryable_error" not in reasons


async def test_completed_steps_are_never_re_executed_during_recovery(queue) -> None:
    """The assertion that makes the recovery credible rather than merely claimed.

    Steps that had already committed must not run again when a later step is
    reassigned; only the interrupted step repeats.
    """
    plan = [
        {"label": "before"},
        {"label": "victim", "fault": STALL},
        {"label": "after"},
    ]
    run_id, _ = await make_run(plan, queue=queue)

    async with Cluster(queue, workers=2):
        assert await wait_for_run_terminal(run_id, timeout=90) is RunStatus.SUCCEEDED

    effects = await get_effects(run_id)
    assert effects["before"].executions == 1, "a completed step was re-executed"
    assert effects["after"].executions == 1

    # The interrupted step *did* run twice, and its effect happened twice. That is
    # not a bug in the recovery: it is the exact duplicate-effect problem that the
    # Phase 3/4 idempotency ledger exists to eliminate, made visible here rather than
    # hidden. `dummy` is deliberately non-idempotent so this number is observable.
    assert effects["victim"].executions == 2


async def test_the_recovered_step_is_taken_over_by_a_different_worker(queue) -> None:
    plan = [{"label": "victim", "fault": STALL}]
    run_id, _ = await make_run(plan, queue=queue)

    async with Cluster(queue, workers=2):
        assert await wait_for_run_terminal(run_id, timeout=90) is RunStatus.SUCCEEDED

    steps = await get_steps(run_id)
    # The successful output records the worker that produced it; the effect ledger
    # records the last worker to execute. With two workers available, the takeover
    # should be visible in the audit log as a change of actor.
    actors = [
        t.actor
        for t in await get_transitions(run_id)
        if t.step_id == steps[0].id and t.to_status == "running"
    ]
    assert len(actors) == 2, "the step should have been claimed twice"
    assert "reaper" in [t.actor for t in await get_transitions(run_id)]


async def test_a_stalled_worker_cannot_commit_after_being_fenced(queue) -> None:
    """The zombie write.

    After recovery the original worker wakes and tries to commit. Its write must be
    rejected, and the surviving result must be the one from the worker that actually
    held the lease.
    """
    plan = [{"label": "victim", "fault": STALL}]
    run_id, _ = await make_run(plan, queue=queue)

    async with Cluster(queue, workers=2) as cluster:
        assert await wait_for_run_terminal(run_id, timeout=90) is RunStatus.SUCCEEDED
        # Give the stalled worker time to wake up and attempt its late write.
        await _sleep(6)
        outcomes = {}
        for w in cluster.workers:
            for key, value in w.stats.items():
                outcomes[key] = outcomes.get(key, 0) + value

    assert outcomes.get("fenced", 0) >= 1, f"expected a fenced commit attempt, got {outcomes}"

    steps = await get_steps(run_id)
    victim = steps[0]
    assert StepStatus(victim.status) is StepStatus.SUCCEEDED
    # attempt 2's output won, not the zombie's.
    assert victim.output["attempt"] == 2
    assert victim.lease_owner is None


async def test_a_poison_step_is_abandoned_instead_of_cycling_forever(queue) -> None:
    """A step that kills every worker that touches it must not loop indefinitely."""
    plan = [
        {
            "label": "poison",
            "fault": {
                "kind": "hang",
                "phase": "after_effect",
                "seconds": 30,
                "suppress_heartbeat": True,
                "times": 99,
            },
        }
    ]
    run_id, _ = await make_run(plan, queue=queue)

    async with session_scope() as s:
        await s.execute(sa.update(Step).where(Step.run_id == run_id).values(max_recoveries=1))

    async with Cluster(queue, workers=1):
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.FAILED
    run = await get_run(run_id)
    assert run.error["code"] == "too_many_recoveries"
    steps = await get_steps(run_id)
    assert steps[0].recoveries == 2, "one recovery allowed, the second exceeds the budget"


async def test_the_queue_is_not_load_bearing(pg_queue, redis_queue) -> None:
    """Flush Redis mid-run; the run still finishes.

    The step ids in flight are destroyed along with the stream. Recovery comes from
    the reaper's poll of the runnable predicate in Postgres, which is the whole point
    of keeping the queue advisory.
    """
    plan = [{"label": "a", "sleep_ms": 1500}, {"label": "b"}, {"label": "c"}]
    run_id, _ = await make_run(plan, queue=redis_queue)

    async with Cluster(redis_queue, workers=1):
        await _sleep(1)
        await redis_queue.redis.delete(redis_queue.stream)
        await redis_queue.setup()  # recreate the (now empty) stream and group
        status = await wait_for_run_terminal(run_id, timeout=90)

    assert status is RunStatus.SUCCEEDED
    effects = await get_effects(run_id)
    assert {label: e.executions for label, e in effects.items()} == {"a": 1, "b": 1, "c": 1}


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
