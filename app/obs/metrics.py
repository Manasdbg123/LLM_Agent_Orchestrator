"""Prometheus metrics.

Two kinds of number live here and they are collected differently, which is the only
subtle thing in this module:

**Counters and histograms** are incremented in-process at the moment the thing
happens. They are per-process, and Prometheus sums them across scraped instances —
which is exactly right for "how many steps failed" but wrong for anything that
describes a shared, current state.

**Gauges describing current state** — runs in progress, queue depth, oldest pending
approval — are *not* incremented in-process. Three workers each keeping their own
idea of "runs in progress" would produce three partial answers that Prometheus would
then add together into a fourth wrong one. Instead they are read from Postgres by a
refresher task, because Postgres is the only component that knows the true value.
"""

from __future__ import annotations

import asyncio
from typing import Any

import sqlalchemy as sa
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

from app.obs.logging import get_logger

log = get_logger("metrics")

REGISTRY = CollectorRegistry()

# --- counters and histograms (per process, summed by Prometheus) ---------------

RUNS_STARTED = Counter(
    "agentorc_runs_started_total", "Runs that began executing", registry=REGISTRY
)
RUNS_COMPLETED = Counter(
    "agentorc_runs_completed_total",
    "Runs that reached a terminal state",
    ["status", "error_code"],
    registry=REGISTRY,
)
STEPS_COMPLETED = Counter(
    "agentorc_steps_total",
    "Steps that reached a terminal state",
    ["kind", "status"],
    registry=REGISTRY,
)
STEP_RETRIES = Counter(
    "agentorc_step_retries_total",
    "Retries scheduled after a retryable failure",
    ["kind", "error_code"],
    registry=REGISTRY,
)
LEASE_EXPIRATIONS = Counter(
    "agentorc_lease_expirations_total",
    "Leases the reaper found expired (a worker died or stalled)",
    registry=REGISTRY,
)
STEPS_ABANDONED = Counter(
    "agentorc_steps_abandoned_total",
    "Steps abandoned after exhausting their recovery budget (poison pills)",
    registry=REGISTRY,
)
FENCED_WRITES = Counter(
    "agentorc_fenced_writes_total",
    "Commits rejected because the worker had lost its lease",
    registry=REGISTRY,
)
STEP_DURATION = Histogram(
    "agentorc_step_duration_seconds",
    "Wall-clock duration of a step execution",
    ["kind"],
    # Tuned for this workload: tool calls land in the tens of milliseconds, model
    # turns in single-digit seconds. The default buckets bunch both into one.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)
LLM_COST = Counter(
    "agentorc_llm_cost_usd_total", "Estimated USD spent", ["model"], registry=REGISTRY
)
LLM_TOKENS = Counter(
    "agentorc_llm_tokens_total", "Tokens billed", ["model", "kind"], registry=REGISTRY
)
LLM_LATENCY = Histogram(
    "agentorc_llm_latency_seconds",
    "Provider call latency",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)
TOOL_CALLS = Counter(
    "agentorc_tool_calls_total",
    "Tool executions by outcome",
    ["tool", "outcome"],
    registry=REGISTRY,
)
EFFECT_DEDUPES = Counter(
    "agentorc_effect_dedupes_total",
    "Tool calls served from the effect ledger instead of re-executing",
    ["tool", "reason"],
    registry=REGISTRY,
)
APPROVAL_DECISIONS = Counter(
    "agentorc_approval_decisions_total",
    "Approval gate outcomes",
    ["decision"],
    registry=REGISTRY,
)
APPROVAL_WAIT = Histogram(
    "agentorc_approval_wait_seconds",
    "How long a run waited for a human decision",
    buckets=(1, 5, 15, 60, 300, 900, 3600, 21600, 86400),
    registry=REGISTRY,
)

# --- gauges (refreshed from Postgres; see the module docstring) ----------------

RUNS_IN_PROGRESS = Gauge(
    "agentorc_runs_in_progress",
    "Runs that are not yet terminal",
    ["status"],
    registry=REGISTRY,
)
QUEUE_DEPTH = Gauge("agentorc_queue_depth", "Steps runnable but not yet claimed", registry=REGISTRY)
STEPS_RUNNING = Gauge(
    "agentorc_steps_running", "Steps currently held by a worker", registry=REGISTRY
)
PENDING_APPROVALS = Gauge(
    "agentorc_pending_approvals", "Approvals awaiting a human", registry=REGISTRY
)
OLDEST_PENDING_APPROVAL = Gauge(
    "agentorc_oldest_pending_approval_seconds",
    "Age of the longest-waiting approval",
    registry=REGISTRY,
)
GAUGE_REFRESH_FAILURES = Counter(
    "agentorc_gauge_refresh_failures_total",
    "Failed attempts to refresh state gauges from Postgres",
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """The /metrics payload."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# --- instrumentation helpers ---------------------------------------------------


def observe_step_finished(kind: str, status: str, duration_seconds: float | None) -> None:
    STEPS_COMPLETED.labels(kind=kind, status=status).inc()
    if duration_seconds is not None and duration_seconds >= 0:
        STEP_DURATION.labels(kind=kind).observe(duration_seconds)


def observe_llm_call(model: str, usage: Any, cost: Any, latency_ms: int | None) -> None:
    LLM_COST.labels(model=model).inc(float(cost))
    LLM_TOKENS.labels(model=model, kind="input").inc(usage.input_tokens)
    LLM_TOKENS.labels(model=model, kind="output").inc(usage.output_tokens)
    LLM_TOKENS.labels(model=model, kind="cache_read").inc(usage.cache_read_tokens)
    LLM_TOKENS.labels(model=model, kind="cache_write").inc(usage.cache_write_tokens)
    if latency_ms is not None:
        LLM_LATENCY.labels(model=model).observe(latency_ms / 1000)


# --- gauge refresher -----------------------------------------------------------

_GAUGE_SQL = sa.text(
    """
SELECT
  (SELECT count(*) FROM agent_runs WHERE status = 'pending')            AS runs_pending,
  (SELECT count(*) FROM agent_runs WHERE status = 'running')            AS runs_running,
  (SELECT count(*) FROM agent_runs WHERE status = 'awaiting_approval')  AS runs_awaiting,
  (SELECT count(*) FROM steps
     WHERE status IN ('pending','retrying') AND available_at <= now())  AS queue_depth,
  (SELECT count(*) FROM steps WHERE status = 'running')                 AS steps_running,
  (SELECT count(*) FROM approval_requests WHERE decision = 'pending')   AS approvals_pending,
  (SELECT COALESCE(EXTRACT(epoch FROM now() - min(requested_at)), 0)
     FROM approval_requests WHERE decision = 'pending')                 AS oldest_approval
"""
)


async def refresh_gauges() -> None:
    """Read current state from the one component that actually knows it."""
    from app.db import session_scope

    async with session_scope() as session:
        row = (await session.execute(_GAUGE_SQL)).one()

    RUNS_IN_PROGRESS.labels(status="pending").set(row.runs_pending)
    RUNS_IN_PROGRESS.labels(status="running").set(row.runs_running)
    RUNS_IN_PROGRESS.labels(status="awaiting_approval").set(row.runs_awaiting)
    QUEUE_DEPTH.set(row.queue_depth)
    STEPS_RUNNING.set(row.steps_running)
    PENDING_APPROVALS.set(row.approvals_pending)
    OLDEST_PENDING_APPROVAL.set(float(row.oldest_approval or 0))


async def gauge_refresh_loop(interval_seconds: float = 5.0) -> None:
    """Background task; runs in the API process only.

    Every replica scraping this would report the same cluster-wide numbers, so it is
    deliberately not started in workers — duplicate series that Prometheus would sum
    into nonsense.
    """
    while True:
        try:
            await refresh_gauges()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Metrics must never take the API down with them.
            GAUGE_REFRESH_FAILURES.inc()
            log.warning("gauge_refresh_failed", error=str(exc))
        await asyncio.sleep(interval_seconds)


__all__ = [
    "APPROVAL_DECISIONS",
    "APPROVAL_WAIT",
    "EFFECT_DEDUPES",
    "FENCED_WRITES",
    "LEASE_EXPIRATIONS",
    "REGISTRY",
    "RUNS_COMPLETED",
    "RUNS_STARTED",
    "STEPS_ABANDONED",
    "STEP_RETRIES",
    "TOOL_CALLS",
    "gauge_refresh_loop",
    "observe_llm_call",
    "observe_step_finished",
    "refresh_gauges",
    "render",
]
