"""The state machines, as data.

The transition tables below are the single authority on what is legal. `app.core.
transitions.transition_*` is the only code permitted to write a status column, and it
consults these tables, so the diagrams in DESIGN.md are executable rather than
aspirational.
"""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepKind(StrEnum):
    # Phase 2 only: a deterministic step used to prove durability before any LLM
    # code exists. Kept in the enum permanently because the chaos suite runs on it.
    DUMMY = "dummy"
    AGENT_TURN = "agent_turn"
    TOOL_CALL = "tool_call"
    FINALIZE = "finalize"


class TransitionReason(StrEnum):
    """Why a transition happened. Recorded on every row of `state_transitions`."""

    CREATED = "created"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    RETRYABLE_ERROR = "retryable_error"
    TERMINAL_ERROR = "terminal_error"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    BACKOFF_ELAPSED = "backoff_elapsed"
    LEASE_EXPIRED = "lease_expired"
    TOO_MANY_RECOVERIES = "too_many_recoveries"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPROVAL_EXPIRED = "approval_expired"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    STEP_FAILED = "step_failed"
    RUN_STARTED = "run_started"


TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
)
TERMINAL_STEP_STATUSES: frozenset[StepStatus] = frozenset(
    {StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.CANCELLED}
)

#: Statuses that keep a step occupying the run's single active-step slot.
ACTIVE_STEP_STATUSES: frozenset[StepStatus] = frozenset(
    {
        StepStatus.PENDING,
        StepStatus.RUNNING,
        StepStatus.RETRYING,
        StepStatus.AWAITING_APPROVAL,
    }
)

#: Statuses from which a worker may claim a lease.
CLAIMABLE_STEP_STATUSES: frozenset[StepStatus] = frozenset(
    {StepStatus.PENDING, StepStatus.RETRYING}
)


RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.AWAITING_APPROVAL,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.AWAITING_APPROVAL: frozenset(
        {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset(
        {
            StepStatus.RUNNING,
            StepStatus.AWAITING_APPROVAL,
            StepStatus.CANCELLED,
            # A step can fail before ever running: run deadline blew past, or the
            # run was failed by a guardrail while this step sat in the queue.
            StepStatus.FAILED,
        }
    ),
    StepStatus.RUNNING: frozenset(
        {
            StepStatus.SUCCEEDED,
            StepStatus.RETRYING,
            StepStatus.FAILED,
            StepStatus.CANCELLED,
            # Not a failure: the worker died. The reaper hands the step back to the
            # pool. Governed by `recoveries`, not by the retry budget.
            StepStatus.PENDING,
        }
    ),
    # `retrying` means "failed, waiting out its backoff, claimable once available_at
    # passes". A worker claims straight out of it rather than going through a
    # separate retrying -> pending hop, which would add a reaper interval of latency
    # to every retry and add a state with no observable meaning of its own.
    StepStatus.RETRYING: frozenset({StepStatus.RUNNING, StepStatus.CANCELLED, StepStatus.FAILED}),
    StepStatus.AWAITING_APPROVAL: frozenset(
        {StepStatus.PENDING, StepStatus.FAILED, StepStatus.CANCELLED}
    ),
    StepStatus.SUCCEEDED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.CANCELLED: frozenset(),
}


def run_transition_legal(src: RunStatus, dst: RunStatus) -> bool:
    return dst in RUN_TRANSITIONS[src]


def step_transition_legal(src: StepStatus, dst: StepStatus) -> bool:
    return dst in STEP_TRANSITIONS[src]


__all__ = [
    "ACTIVE_STEP_STATUSES",
    "CLAIMABLE_STEP_STATUSES",
    "RUN_TRANSITIONS",
    "STEP_TRANSITIONS",
    "TERMINAL_RUN_STATUSES",
    "TERMINAL_STEP_STATUSES",
    "RunStatus",
    "StepKind",
    "StepStatus",
    "TransitionReason",
    "run_transition_legal",
    "step_transition_legal",
]
