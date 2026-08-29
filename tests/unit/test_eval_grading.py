"""Tests for the eval harness's own grading.

A harness that reports 100% because it never actually checks anything is worse than
no harness: it converts an unverified claim into a number that looks verified. So the
grader is tested the same way the engine is -- by feeding it observations that should
fail and asserting that they do.
"""

from __future__ import annotations

from typing import Any

from app.domain.states import RunStatus
from eval.harness import TaskResult, _grade
from eval.report import percentile, summarize
from eval.tasks import TASKS, EvalTask, Expectation, Tier


def _obs(**overrides: Any) -> dict[str, Any]:
    """A clean observation of a successful, unremarkable run."""
    base: dict[str, Any] = {
        "status": RunStatus.SUCCEEDED,
        "error_code": None,
        "answer": "the answer is 42",
        "tool_sequence": ["calculator"],
        "tool_call_steps": 1,
        "emails": 0,
        "records": {},
        "approvals": 0,
        "retries": 0,
        "recoveries": 0,
    }
    base.update(overrides)
    return base


def _task(expect: Expectation) -> EvalTask:
    return EvalTask(
        id="t",
        tier=Tier.SIMPLE,
        intent="test",
        task="test",
        script=[],
        tools=["calculator"],
        expect=expect,
    )


# --- the grader must pass what it should ------------------------------------------


def test_matching_observation_passes() -> None:
    task = _task(
        Expectation(
            answer_contains=("42",), tool_sequence=("calculator",), min_tool_calls=1
        )
    )
    assert _grade(task, _obs()) == []


def test_unstated_expectations_are_not_checked() -> None:
    """A default Expectation asserts only the status, so unrelated facts cannot fail it."""
    task = _task(Expectation())
    assert _grade(task, _obs(emails=7, retries=3, tool_sequence=["web_search"])) == []


# --- and fail what it should ------------------------------------------------------


def test_wrong_status_fails() -> None:
    task = _task(Expectation())
    failures = _grade(task, _obs(status=RunStatus.FAILED))
    assert any("status" in f for f in failures)


def test_missing_answer_substring_fails() -> None:
    task = _task(Expectation(answer_contains=("541171",)))
    assert any("541171" in f for f in _grade(task, _obs()))


def test_answer_match_is_case_insensitive() -> None:
    task = _task(Expectation(answer_contains=("THE ANSWER",)))
    assert _grade(task, _obs()) == []


def test_wrong_tool_order_fails() -> None:
    """Order matters: 'called both tools' is a weaker claim than 'in this order'."""
    task = _task(Expectation(tool_sequence=("web_search", "calculator")))
    failures = _grade(task, _obs(tool_sequence=["calculator", "web_search"]))
    assert any("tool sequence" in f for f in failures)


def test_extra_email_fails() -> None:
    """The check the whole idempotency story rests on."""
    task = _task(Expectation(emails_sent=1))
    failures = _grade(task, _obs(emails=2))
    assert any("email" in f for f in failures)


def test_missing_email_fails() -> None:
    task = _task(Expectation(emails_sent=1))
    assert any("email" in f for f in _grade(task, _obs(emails=0)))


def test_missing_record_fails() -> None:
    task = _task(Expectation(records=(("scratch", "k", "v"),)))
    assert any("scratch/k" in f for f in _grade(task, _obs()))


def test_wrong_record_value_fails() -> None:
    task = _task(Expectation(records=(("scratch", "k", "v"),)))
    failures = _grade(task, _obs(records={("scratch", "k"): "other"}))
    assert any("!= expected" in f for f in failures)


def test_unexpected_approval_gate_fails() -> None:
    """Gating too much is a bug too, not just gating too little."""
    task = _task(Expectation(approvals=0))
    assert any("approval" in f for f in _grade(task, _obs(approvals=1)))


def test_missing_retry_fails() -> None:
    task = _task(Expectation(min_retries=1))
    assert any("retries" in f for f in _grade(task, _obs(retries=0)))


def test_missing_recovery_fails() -> None:
    task = _task(Expectation(min_recoveries=1))
    assert any("recoveries" in f for f in _grade(task, _obs(recoveries=0)))


def test_wrong_error_code_fails() -> None:
    task = _task(Expectation(status=RunStatus.FAILED, error_code="budget_exceeded"))
    failures = _grade(task, _obs(status=RunStatus.FAILED, error_code="max_steps_exceeded"))
    assert any("error code" in f for f in failures)


def test_all_failures_are_reported_not_just_the_first() -> None:
    task = _task(
        Expectation(
            answer_contains=("nope",),
            tool_sequence=("web_search",),
            emails_sent=3,
        )
    )
    assert len(_grade(task, _obs())) == 3


# --- the catalog itself -----------------------------------------------------------


def test_task_ids_are_unique() -> None:
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_every_task_states_an_expectation_beyond_status() -> None:
    """A task whose only assertion is 'it finished' is not evidence of anything."""
    for task in TASKS:
        expect = task.expect
        asserts_something = (
            expect.answer_contains
            or expect.tool_sequence is not None
            or expect.min_tool_calls
            or expect.error_code
            or expect.emails_sent is not None
            or expect.records
            or expect.approvals
            or expect.min_retries
            or expect.min_recoveries
        )
        assert asserts_something, f"{task.id} asserts nothing but its run status"


def test_every_task_declares_only_tools_it_scripts() -> None:
    """A scripted call to a tool the run was not granted would fail for the wrong reason."""
    for task in TASKS:
        for turn in task.script:
            for call in turn.get("tools", []):
                assert call["name"] in task.tools, (
                    f"{task.id} scripts {call['name']!r} but grants {task.tools}"
                )


# --- reporting --------------------------------------------------------------------


def test_percentile_returns_an_observed_value() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert percentile(values, 50) in values
    assert percentile(values, 99) == 100.0
    assert percentile(values, 0) == 1.0


def test_percentile_of_empty_is_zero() -> None:
    assert percentile([], 95) == 0.0


def _result(task_id: str, *, passed: bool, tier: str = "simple", **kw: Any) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        tier=tier,
        intent="",
        run_id=None,
        passed=passed,
        status="succeeded" if passed else "failed",
        **kw,
    )


def test_summary_counts_and_rates() -> None:
    results = [
        _result("a", passed=True, steps_used=4, cost_usd=0.01, latency_s=1.0),
        _result("b", passed=False, steps_used=6, cost_usd=0.03, latency_s=3.0),
        _result("c", passed=True, tier="complex", steps_used=8, cost_usd=0.02, latency_s=2.0),
    ]

    summary = summarize(results)

    assert summary["tasks"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["success_rate"] == 2 / 3
    assert summary["avg_steps"] == 6.0
    assert round(summary["total_cost_usd"], 4) == 0.06
    assert summary["by_tier"]["simple"]["tasks"] == 2
    assert summary["by_tier"]["complex"]["success_rate"] == 1.0
