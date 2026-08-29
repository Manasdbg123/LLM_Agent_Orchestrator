"""Properties of the state machines themselves.

These assert on the transition tables rather than on any particular code path, so a
future edit that quietly adds an edge (say, succeeded -> running) fails here instead
of showing up as an impossible state in production.
"""

from __future__ import annotations

import pytest

from app.domain.states import (
    ACTIVE_STEP_STATUSES,
    RUN_TRANSITIONS,
    STEP_TRANSITIONS,
    TERMINAL_RUN_STATUSES,
    TERMINAL_STEP_STATUSES,
    RunStatus,
    StepStatus,
    run_transition_legal,
    step_transition_legal,
)


def test_every_status_appears_in_its_table() -> None:
    assert set(RUN_TRANSITIONS) == set(RunStatus)
    assert set(STEP_TRANSITIONS) == set(StepStatus)


@pytest.mark.parametrize("status", sorted(TERMINAL_STEP_STATUSES))
def test_terminal_step_statuses_are_absorbing(status: StepStatus) -> None:
    assert STEP_TRANSITIONS[status] == frozenset()


@pytest.mark.parametrize("status", sorted(TERMINAL_RUN_STATUSES))
def test_terminal_run_statuses_are_absorbing(status: RunStatus) -> None:
    assert RUN_TRANSITIONS[status] == frozenset()


def test_active_and_terminal_step_statuses_partition_the_space() -> None:
    assert ACTIVE_STEP_STATUSES | TERMINAL_STEP_STATUSES == set(StepStatus)
    assert not (ACTIVE_STEP_STATUSES & TERMINAL_STEP_STATUSES)


def test_every_non_terminal_status_is_reachable_from_pending() -> None:
    seen = {StepStatus.PENDING}
    frontier = [StepStatus.PENDING]
    while frontier:
        for nxt in STEP_TRANSITIONS[frontier.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(StepStatus), f"unreachable step statuses: {set(StepStatus) - seen}"


def test_worker_death_returns_a_step_to_the_pool() -> None:
    # running -> pending is crash recovery. It must NOT be running -> retrying:
    # a worker dying is not the step failing, and it draws on a different budget.
    assert step_transition_legal(StepStatus.RUNNING, StepStatus.PENDING)


def test_retrying_is_directly_claimable() -> None:
    # A worker claims straight out of `retrying` once the backoff elapses; there is
    # no retrying -> pending hop to add a reaper interval of latency.
    assert step_transition_legal(StepStatus.RETRYING, StepStatus.RUNNING)


def test_a_succeeded_step_cannot_be_revived() -> None:
    for status in StepStatus:
        assert not step_transition_legal(StepStatus.SUCCEEDED, status)


def test_a_run_cannot_leave_a_terminal_state() -> None:
    assert not run_transition_legal(RunStatus.SUCCEEDED, RunStatus.RUNNING)
    assert not run_transition_legal(RunStatus.CANCELLED, RunStatus.RUNNING)
    assert not run_transition_legal(RunStatus.FAILED, RunStatus.SUCCEEDED)


def test_approval_gate_can_resume_or_end_a_run() -> None:
    assert run_transition_legal(RunStatus.AWAITING_APPROVAL, RunStatus.RUNNING)
    assert run_transition_legal(RunStatus.AWAITING_APPROVAL, RunStatus.FAILED)
    assert step_transition_legal(StepStatus.AWAITING_APPROVAL, StepStatus.PENDING)
