"""The evaluation catalog.

Seventeen tasks spanning four tiers, from a single tool call to deliberate crashes.

Two things about the shape are deliberate.

**Expectations assert on ground truth, not on the engine's own report.** "The email
was sent once" is checked by counting rows in `email_outbox` -- the mock provider's
store -- not by reading a field the engine set about itself. A bug that makes the
engine misreport its own behaviour would have to survive that check, and it cannot.

**Every task carries both a natural-language instruction and a script.** Against the
scripted provider (the default: deterministic and free, so a red result means the
engine broke rather than that the model had an off day) the script drives the turns.
Against a live model only `task` is sent, and the same expectations grade the
outcome. The catalog is therefore one artefact rather than two that drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.domain.states import RunStatus


class Tier(StrEnum):
    """Roughly, how much of the engine a task exercises."""

    SIMPLE = "simple"  # one tool call, or none
    MULTI_STEP = "multi_step"  # 2-3 calls with a sequential dependency
    COMPLEX = "complex"  # 4+ calls, parallel tool use, or self-correction
    RELIABILITY = "reliability"  # approvals, retries, crashes, guardrails


#: Let the tool's side effect land, then stall the worker without heartbeating so its
#: lease expires under it. The reaper hands the step to another worker, which must
#: NOT redo the effect.
STALL_AFTER_EFFECT: dict[str, Any] = {
    "kind": "hang",
    "phase": "after_effect",
    "seconds": 30,
    "suppress_heartbeat": True,
    "times": 1,
}

#: A transient failure before the effect: the classic retryable case.
TRANSIENT_BEFORE_EFFECT: dict[str, Any] = {
    "kind": "error",
    "phase": "before_effect",
    "retryable": True,
    "times": 1,
}


@dataclass(frozen=True, slots=True)
class Expectation:
    """What must be true once the run reaches a terminal state.

    Every field defaults to "don't care", so a task states only what it is actually
    testing and an unrelated engine change cannot produce a spurious failure.
    """

    status: RunStatus = RunStatus.SUCCEEDED
    #: Substrings that must appear in the run's final answer (case-insensitive).
    answer_contains: tuple[str, ...] = ()
    #: The exact ordered sequence of tools invoked. None means "don't care".
    tool_sequence: tuple[str, ...] | None = None
    min_tool_calls: int = 0
    #: `run.error["code"]` when the task is expected to fail.
    error_code: str | None = None
    #: Rows in `email_outbox` for this run. Ground truth for "sent exactly once".
    emails_sent: int | None = None
    #: (namespace, key, value) triples that must be present in `sandbox_records`.
    records: tuple[tuple[str, str, str], ...] = ()
    #: How many approval gates the run must have opened.
    approvals: int = 0
    #: At least one step must have reached this attempt number.
    min_retries: int = 0
    #: The reaper must have reclaimed at least this many step executions.
    min_recoveries: int = 0


@dataclass(frozen=True, slots=True)
class EvalTask:
    id: str
    tier: Tier
    #: What the task is testing. Printed in the report.
    intent: str
    #: The instruction a real model would receive.
    task: str
    #: Scripted assistant turns for the deterministic provider.
    script: list[dict[str, Any]]
    tools: list[str]
    expect: Expectation
    #: Auto-decide any approval this run opens. Stands in for an operator; the
    #: decision goes through the same `approvals.decide` that the API endpoint calls.
    approve: bool = False
    reject: bool = False
    tool_faults: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_steps: int | None = None
    max_cost_usd: float | None = None
    timeout_seconds: int = 120


def _answer(text: str) -> dict[str, Any]:
    return {"text": text}


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"tools": [{"name": name, "input": args}]}


# --- the catalog ------------------------------------------------------------------

TASKS: list[EvalTask] = [
    # --- tier 1: single tool ------------------------------------------------------
    EvalTask(
        id="calc_single",
        tier=Tier.SIMPLE,
        intent="One tool call, one answer: the shortest complete path through the loop.",
        task="What is 1847 * 293? Use the calculator and report the number.",
        script=[_call("calculator", expression="1847 * 293"), _answer("1847 * 293 = 541171.")],
        tools=["calculator"],
        expect=Expectation(
            answer_contains=("541171",),
            tool_sequence=("calculator",),
            min_tool_calls=1,
        ),
    ),
    EvalTask(
        id="search_single",
        tier=Tier.SIMPLE,
        intent="Retrieval path: a read-only tool whose output must reach the next turn.",
        task="Search for what a fencing token is and summarise it in one sentence.",
        script=[
            _call("web_search", query="lease fencing token distributed lock"),
            _answer(
                "A fencing token is a monotonic number that lets a resource reject a "
                "stale lock holder's writes."
            ),
        ],
        tools=["web_search"],
        expect=Expectation(answer_contains=("fencing",), tool_sequence=("web_search",)),
    ),
    EvalTask(
        id="db_write_scratch",
        tier=Tier.SIMPLE,
        intent="A side-effecting tool on a namespace that does not trip the approval gate.",
        task="Store the value 'ok' under the key 'healthcheck' in the 'scratch' namespace.",
        script=[
            _call("database_write", namespace="scratch", key="healthcheck", value="ok"),
            _answer("Stored scratch/healthcheck = ok."),
        ],
        tools=["database_write"],
        expect=Expectation(
            tool_sequence=("database_write",),
            records=(("scratch", "healthcheck", "ok"),),
            approvals=0,
        ),
    ),
    EvalTask(
        id="no_tool_answer",
        tier=Tier.SIMPLE,
        intent="The model answers directly. Proves tools are an option, not a forced path.",
        task="Answer from your own knowledge: what does 'idempotent' mean?",
        script=[
            _answer(
                "Idempotent means applying the operation twice has the same effect as "
                "applying it once."
            )
        ],
        tools=["calculator", "web_search"],
        expect=Expectation(answer_contains=("idempotent",), tool_sequence=(), min_tool_calls=0),
    ),
    # --- tier 2: multi-step -------------------------------------------------------
    EvalTask(
        id="calc_chain",
        tier=Tier.MULTI_STEP,
        intent="Two dependent calls: the second consumes the first's observation.",
        task="Compute 12 * 12, then add 44 to the result.",
        script=[
            _call("calculator", expression="12 * 12"),
            _call("calculator", expression="144 + 44"),
            _answer("12 * 12 = 144, and 144 + 44 = 188."),
        ],
        tools=["calculator"],
        expect=Expectation(
            answer_contains=("188",),
            tool_sequence=("calculator", "calculator"),
            min_tool_calls=2,
        ),
    ),
    EvalTask(
        id="search_then_calc",
        tier=Tier.MULTI_STEP,
        intent="Cross-tool dependency: a retrieved fact feeds a computation.",
        task="Look up the speed of light, then compute how far light travels in 60 seconds.",
        script=[
            _call("web_search", query="speed of light in vacuum constant metres"),
            _call("calculator", expression="299792458 * 60"),
            _answer("Light travels 17987547480 metres in 60 seconds."),
        ],
        tools=["web_search", "calculator"],
        expect=Expectation(
            answer_contains=("17987547480",),
            tool_sequence=("web_search", "calculator"),
        ),
    ),
    EvalTask(
        id="search_calc_store",
        tier=Tier.MULTI_STEP,
        intent="Read, compute, persist -- the shape of most real agent work.",
        task=(
            "Find the Earth-Moon distance, convert it to metres, and store it under "
            "'moon_distance_m' in the 'scratch' namespace."
        ),
        script=[
            _call("web_search", query="earth to moon distance lunar kilometres"),
            _call("calculator", expression="384400 * 1000"),
            _call(
                "database_write", namespace="scratch", key="moon_distance_m", value="384400000"
            ),
            _answer("Stored 384400000 metres under scratch/moon_distance_m."),
        ],
        tools=["web_search", "calculator", "database_write"],
        expect=Expectation(
            tool_sequence=("web_search", "calculator", "database_write"),
            records=(("scratch", "moon_distance_m", "384400000"),),
        ),
    ),
    # --- tier 3: complex ----------------------------------------------------------
    EvalTask(
        id="four_tool_pipeline",
        tier=Tier.COMPLEX,
        intent="Four calls across four tools, including a gated one, in a single run.",
        task=(
            "Research retry backoff, compute the total delay of 5 exponential attempts at "
            "a 1s base, store the total under 'backoff_total_s' in 'scratch', then email "
            "the summary to ops@example.test."
        ),
        script=[
            _call("web_search", query="exponential backoff jitter retry thundering herd"),
            _call("calculator", expression="1 + 2 + 4 + 8 + 16"),
            _call("database_write", namespace="scratch", key="backoff_total_s", value="31"),
            _call(
                "send_email",
                to="ops@example.test",
                subject="Backoff analysis",
                body="Five attempts at a 1s base with doubling totals 31 seconds of delay.",
            ),
            _answer("Researched, computed 31s, stored it, and emailed ops@example.test."),
        ],
        tools=["web_search", "calculator", "database_write", "send_email"],
        approve=True,
        expect=Expectation(
            tool_sequence=("web_search", "calculator", "database_write", "send_email"),
            min_tool_calls=4,
            records=(("scratch", "backoff_total_s", "31"),),
            emails_sent=1,
            approvals=1,
        ),
    ),
    EvalTask(
        id="parallel_tool_uses",
        tier=Tier.COMPLEX,
        intent=(
            "One assistant turn emitting two tool_use blocks. Each becomes its own durable "
            "step, chained rather than fanned out."
        ),
        task="Compute 2^10 and separately search for what SKIP LOCKED does, then report both.",
        script=[
            {
                "tools": [
                    {"name": "calculator", "input": {"expression": "2 ** 10"}},
                    {"name": "web_search", "input": {"query": "postgres skip locked queue"}},
                ]
            },
            _answer(
                "2^10 = 1024. SKIP LOCKED lets concurrent workers claim distinct queue rows "
                "without blocking on each other."
            ),
        ],
        tools=["calculator", "web_search"],
        expect=Expectation(
            answer_contains=("1024",),
            tool_sequence=("calculator", "web_search"),
            min_tool_calls=2,
        ),
    ),
    EvalTask(
        id="malformed_then_recover",
        tier=Tier.COMPLEX,
        intent=(
            "Invalid tool arguments come back as an errored tool_result and the model "
            "corrects itself. A bad turn must not kill a run."
        ),
        task="Compute the square root of 144 using the calculator.",
        script=[
            # `expression` is what the schema requires; `formula` is not a field at all.
            {"tools": [{"name": "calculator", "input": {"formula": "sqrt(144)"}}]},
            _call("calculator", expression="144 ** 0.5"),
            _answer("The square root of 144 is 12."),
        ],
        tools=["calculator"],
        expect=Expectation(
            answer_contains=("12",),
            tool_sequence=("calculator", "calculator"),
            min_tool_calls=2,
        ),
    ),
    # --- tier 4: reliability ------------------------------------------------------
    EvalTask(
        id="approval_granted",
        tier=Tier.RELIABILITY,
        intent=(
            "A gated tool parks the run in awaiting_approval; approving resumes it and the "
            "effect happens exactly once."
        ),
        task="Email finance@example.test confirming the quarterly figures are ready.",
        script=[
            _call(
                "send_email",
                to="finance@example.test",
                subject="Q figures ready",
                body="The quarterly figures have been published.",
            ),
            _answer("Email sent to finance@example.test."),
        ],
        tools=["send_email"],
        approve=True,
        expect=Expectation(tool_sequence=("send_email",), emails_sent=1, approvals=1),
    ),
    EvalTask(
        id="approval_rejected",
        tier=Tier.RELIABILITY,
        intent=(
            "A rejected gate produces no side effect, and the refusal is fed back so the "
            "model can finish honestly rather than the run simply dying."
        ),
        task="Email everyone@example.test announcing the outage.",
        script=[
            _call(
                "send_email",
                to="everyone@example.test",
                subject="Outage",
                body="We are currently down.",
            ),
            _answer("The email was not approved, so nothing was sent."),
        ],
        tools=["send_email"],
        reject=True,
        expect=Expectation(answer_contains=("not approved",), emails_sent=0, approvals=1),
    ),
    EvalTask(
        id="approval_sensitive_namespace",
        tier=Tier.RELIABILITY,
        intent=(
            "Approval is data-dependent: the same tool is ungated on 'scratch' and gated on "
            "'billing'."
        ),
        task="Record invoice INV-8891 as paid in the billing namespace.",
        script=[
            _call("database_write", namespace="scratch", key="invoice_seen", value="INV-8891"),
            _call("database_write", namespace="billing", key="INV-8891", value="paid"),
            _answer("Invoice INV-8891 recorded as paid."),
        ],
        tools=["database_write"],
        approve=True,
        expect=Expectation(
            tool_sequence=("database_write", "database_write"),
            records=(("scratch", "invoice_seen", "INV-8891"), ("billing", "INV-8891", "paid")),
            # Exactly one: the scratch write must NOT have opened a gate.
            approvals=1,
        ),
    ),
    EvalTask(
        id="retry_transient",
        tier=Tier.RELIABILITY,
        intent=(
            "A retryable failure before the effect is retried with backoff and then "
            "succeeds; the run as a whole never notices."
        ),
        task="Compute 99 * 99 and report it.",
        script=[_call("calculator", expression="99 * 99"), _answer("99 * 99 = 9801.")],
        tools=["calculator"],
        tool_faults={"calculator": TRANSIENT_BEFORE_EFFECT},
        expect=Expectation(answer_contains=("9801",), min_retries=1),
    ),
    EvalTask(
        id="crash_recovery_no_duplicate",
        tier=Tier.RELIABILITY,
        intent=(
            "The headline claim, graded: a worker's lease expires after the email is sent, "
            "the reaper reassigns the step, and the customer still gets ONE email."
        ),
        task="Email customer@example.test their shipping confirmation.",
        script=[
            _call(
                "send_email",
                to="customer@example.test",
                subject="Your order has shipped",
                body="Tracking number 1Z999AA10123456784.",
            ),
            _answer("Shipping confirmation sent."),
        ],
        tools=["send_email"],
        approve=True,
        tool_faults={"send_email": STALL_AFTER_EFFECT},
        expect=Expectation(
            emails_sent=1,
            approvals=1,
            min_recoveries=1,
            tool_sequence=("send_email",),
        ),
        timeout_seconds=180,
    ),
    EvalTask(
        id="guardrail_max_steps",
        tier=Tier.RELIABILITY,
        intent=(
            "A model that never stops is stopped by the engine, under a named error rather "
            "than by silently exhausting a budget."
        ),
        task="Keep computing 1 + 1 forever.",
        script=[_call("calculator", expression="1 + 1")] * 30,
        tools=["calculator"],
        max_steps=6,
        expect=Expectation(status=RunStatus.FAILED, error_code="max_steps_exceeded"),
    ),
    EvalTask(
        id="guardrail_budget",
        tier=Tier.RELIABILITY,
        intent="Cost is enforced before the call, not apologised for after it.",
        task="Do expensive research on durable execution.",
        script=[
            {
                "tools": [{"name": "web_search", "input": {"query": "durable execution"}}],
                "input_tokens": 400_000,
                "output_tokens": 40_000,
            },
            {
                "tools": [{"name": "web_search", "input": {"query": "durable execution again"}}],
                "input_tokens": 400_000,
                "output_tokens": 40_000,
            },
            _answer("done"),
        ],
        tools=["web_search"],
        max_cost_usd=0.05,
        expect=Expectation(status=RunStatus.FAILED, error_code="budget_exceeded"),
    ),
]

BY_ID: dict[str, EvalTask] = {t.id: t for t in TASKS}


def select(ids: list[str] | None = None, tiers: list[str] | None = None) -> list[EvalTask]:
    """Filter the catalog. An unknown id is an error, not a silently empty run."""
    tasks = TASKS
    if ids:
        missing = [i for i in ids if i not in BY_ID]
        if missing:
            raise SystemExit(f"unknown task id(s): {', '.join(missing)}")
        tasks = [BY_ID[i] for i in ids]
    if tiers:
        wanted = {Tier(t) for t in tiers}
        tasks = [t for t in tasks if t.tier in wanted]
    return tasks


__all__ = [
    "BY_ID",
    "STALL_AFTER_EFFECT",
    "TASKS",
    "TRANSIENT_BEFORE_EFFECT",
    "EvalTask",
    "Expectation",
    "Tier",
    "select",
]
