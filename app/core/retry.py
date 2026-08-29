"""Retry policy and the retry/fail decision.

Backoff is *persisted*, never slept on: a retry sets `available_at = now() + delay`
and releases the lease. An in-process `await asyncio.sleep(delay)` would hold a
worker slot hostage and would evaporate if the worker restarted; a timestamp in
Postgres survives both.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.config import settings
from app.domain.errors import (
    ErrorClass,
    ErrorCode,
    classify,
    code_of,
    error_to_json,
    retry_after_of,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    initial_backoff_seconds: float
    multiplier: float
    max_backoff_seconds: float
    #: Full jitter (AWS "Exponential Backoff and Jitter"). Without it, N workers that
    #: fail against the same dependency at the same instant retry in lockstep forever.
    jitter: bool = True

    @classmethod
    def default(cls) -> RetryPolicy:
        return cls(
            max_attempts=settings.default_max_attempts,
            initial_backoff_seconds=settings.retry_initial_backoff_seconds,
            multiplier=settings.retry_backoff_multiplier,
            max_backoff_seconds=settings.retry_max_backoff_seconds,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> RetryPolicy:
        base = cls.default()
        if not data:
            return base
        return cls(
            max_attempts=int(data.get("max_attempts", base.max_attempts)),
            initial_backoff_seconds=float(
                data.get("initial_backoff_seconds", base.initial_backoff_seconds)
            ),
            multiplier=float(data.get("multiplier", base.multiplier)),
            max_backoff_seconds=float(data.get("max_backoff_seconds", base.max_backoff_seconds)),
            jitter=bool(data.get("jitter", base.jitter)),
        )

    def backoff_for(
        self,
        attempt: int,
        *,
        retry_after_seconds: float | None = None,
        rng: random.Random | None = None,
    ) -> float:
        """Delay before attempt `attempt + 1`.

        An explicit `Retry-After` from upstream wins over our own computation: the
        server knows when it will be ready and we do not. It is still capped, so a
        hostile or buggy header cannot park a step for a day.
        """
        capped = min(
            self.initial_backoff_seconds * (self.multiplier ** max(0, attempt - 1)),
            self.max_backoff_seconds,
        )
        if retry_after_seconds is not None:
            return min(max(retry_after_seconds, 0.0), self.max_backoff_seconds)
        if not self.jitter:
            return capped
        return (rng or random).uniform(0.0, capped)


class RetryAction(StrEnum):
    RETRY = "retry"
    FAIL = "fail"
    #: The side effect may have landed. Never a blind retry — Phase 4 resolves this
    #: against the tool's declared effect policy. Until then it is terminal, which is
    #: the safe direction to be wrong in.
    ESCALATE_AMBIGUOUS = "escalate_ambiguous"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    action: RetryAction
    delay_seconds: float
    error: dict[str, Any]
    code: ErrorCode

    @property
    def should_retry(self) -> bool:
        return self.action is RetryAction.RETRY


def decide(
    exc: BaseException,
    *,
    attempt: int,
    policy: RetryPolicy,
    rng: random.Random | None = None,
) -> RetryDecision:
    """Classify a failure and say what the state machine should do about it."""
    error_class = classify(exc)
    error = error_to_json(exc)
    code = code_of(exc)

    if error_class is ErrorClass.AMBIGUOUS:
        return RetryDecision(RetryAction.ESCALATE_AMBIGUOUS, 0.0, error, code)

    if error_class is ErrorClass.RETRYABLE and attempt < policy.max_attempts:
        delay = policy.backoff_for(attempt, retry_after_seconds=retry_after_of(exc), rng=rng)
        return RetryDecision(RetryAction.RETRY, delay, error, code)

    if error_class is ErrorClass.RETRYABLE:
        # Retryable, but out of budget. Record *why* it is terminal, so the run's
        # error does not read as an unretryable bug when it was a flaky dependency.
        error = dict(error) | {
            "code": str(ErrorCode.ATTEMPTS_EXHAUSTED),
            "attempts": attempt,
            "max_attempts": policy.max_attempts,
            "last_error": error.get("code"),
        }
        return RetryDecision(RetryAction.FAIL, 0.0, error, ErrorCode.ATTEMPTS_EXHAUSTED)

    return RetryDecision(RetryAction.FAIL, 0.0, error, code)


__all__ = ["RetryAction", "RetryDecision", "RetryPolicy", "decide"]
