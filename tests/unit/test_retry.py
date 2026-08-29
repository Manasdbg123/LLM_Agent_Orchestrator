from __future__ import annotations

import dataclasses
import random

import pytest

from app.core.retry import RetryAction, RetryPolicy, decide
from app.domain.errors import (
    AmbiguousEffectError,
    ErrorClass,
    ErrorCode,
    RetryableError,
    TerminalError,
    classify,
)

POLICY = RetryPolicy(
    max_attempts=3, initial_backoff_seconds=1.0, multiplier=2.0, max_backoff_seconds=10.0
)


def test_backoff_grows_and_is_capped() -> None:
    no_jitter = dataclasses.replace(POLICY, jitter=False)
    assert no_jitter.backoff_for(1) == 1.0
    assert no_jitter.backoff_for(2) == 2.0
    assert no_jitter.backoff_for(3) == 4.0
    assert no_jitter.backoff_for(50) == 10.0  # capped, not astronomical


def test_full_jitter_stays_within_the_cap() -> None:
    rng = random.Random(1234)
    for attempt in range(1, 8):
        for _ in range(50):
            delay = POLICY.backoff_for(attempt, rng=rng)
            assert 0.0 <= delay <= POLICY.max_backoff_seconds


def test_jitter_actually_varies() -> None:
    # Without jitter, every worker that failed against the same dependency at the
    # same instant retries in lockstep and re-creates the thundering herd.
    rng = random.Random(7)
    samples = {POLICY.backoff_for(4, rng=rng) for _ in range(20)}
    assert len(samples) > 1


def test_retry_after_from_upstream_wins_but_is_still_capped() -> None:
    assert POLICY.backoff_for(1, retry_after_seconds=5.0) == 5.0
    # A hostile or buggy Retry-After cannot park a step for a day.
    assert POLICY.backoff_for(1, retry_after_seconds=86_400) == POLICY.max_backoff_seconds
    assert POLICY.backoff_for(1, retry_after_seconds=-3) == 0.0


def test_retryable_error_within_budget_retries() -> None:
    decision = decide(RetryableError("flaky"), attempt=1, policy=POLICY)
    assert decision.action is RetryAction.RETRY
    assert decision.should_retry


def test_retryable_error_out_of_budget_fails_and_says_so() -> None:
    decision = decide(RetryableError("flaky"), attempt=3, policy=POLICY)
    assert decision.action is RetryAction.FAIL
    # The recorded reason must be "we ran out of attempts", not "unretryable bug".
    assert decision.code is ErrorCode.ATTEMPTS_EXHAUSTED
    assert decision.error["attempts"] == 3
    assert decision.error["max_attempts"] == 3


def test_terminal_error_never_retries_even_on_attempt_one() -> None:
    decision = decide(TerminalError("bad input"), attempt=1, policy=POLICY)
    assert decision.action is RetryAction.FAIL
    assert decision.delay_seconds == 0.0


def test_ambiguous_effect_is_escalated_not_retried() -> None:
    # Blindly retrying an effect that may already have happened is how a customer
    # gets two emails. It must never fall into the retry branch.
    decision = decide(AmbiguousEffectError("did it send?"), attempt=1, policy=POLICY)
    assert decision.action is RetryAction.ESCALATE_AMBIGUOUS


@pytest.mark.parametrize(
    "exc",
    [TimeoutError(), TimeoutError(), ConnectionError(), ConnectionResetError()],
)
def test_transient_stdlib_failures_are_retryable(exc: BaseException) -> None:
    assert classify(exc) is ErrorClass.RETRYABLE


@pytest.mark.parametrize("exc", [ValueError("nope"), KeyError("k"), TypeError()])
def test_unknown_failures_default_to_terminal(exc: BaseException) -> None:
    # Deliberate: an unclassified failure that silently retries is how a bug becomes
    # an outage. Unknown means stop.
    assert classify(exc) is ErrorClass.TERMINAL


def test_policy_from_mapping_falls_back_per_field() -> None:
    policy = RetryPolicy.from_mapping({"max_attempts": 9})
    assert policy.max_attempts == 9
    assert policy.multiplier == RetryPolicy.default().multiplier
