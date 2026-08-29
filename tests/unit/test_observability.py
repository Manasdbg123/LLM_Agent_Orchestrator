"""Metrics and tracing that can be checked without infrastructure."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest

from app.obs import metrics, tracing

DASHBOARD = Path(__file__).resolve().parents[2] / "ops/grafana/dashboards/agent-orchestrator.json"


def _registered_metric_names() -> set[str]:
    """Base names in the registry, including the suffixes Prometheus derives."""
    names: set[str] = set()
    for metric in metrics.REGISTRY.collect():
        names.add(metric.name)
        for sample in metric.samples:
            names.add(sample.name)
            # Histograms expose _bucket/_count/_sum; PromQL queries reference those.
            for suffix in ("_bucket", "_count", "_sum", "_total", "_created"):
                if sample.name.endswith(suffix):
                    names.add(sample.name)
    # Counters are collected without the _total suffix that queries use.
    names |= {f"{n}_total" for n in list(names)}
    names |= {f"{n}_bucket" for n in list(names)}
    return names


def test_dashboard_references_only_metrics_that_exist() -> None:
    """A dashboard panel querying a metric nobody emits is a blank panel.

    Blank panels are worse than missing ones: they look like "the system is quiet"
    rather than "this query is wrong", so the mistake survives review.
    """
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    exprs = [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]
    referenced = {m for e in exprs for m in re.findall(r"agentorc_[a-z_]+", e)}
    assert referenced, "dashboard queries no orchestrator metrics at all"

    known = _registered_metric_names()
    missing = sorted(referenced - known)
    assert not missing, f"dashboard references metrics that are never emitted: {missing}"


def test_dashboard_is_valid_and_has_the_panels_the_design_promises() -> None:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    titles = {p["title"] for p in dashboard["panels"]}
    # The four the design commits to: throughput, success rate, retry rate, latency.
    assert "Step throughput by outcome" in titles
    assert "Run success rate" in titles
    assert "Retries and recoveries" in titles
    assert "Step latency percentiles" in titles
    assert all("gridPos" in p for p in dashboard["panels"])


def test_latency_panel_queries_p50_p95_p99() -> None:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    panel = next(p for p in dashboard["panels"] if p["title"] == "Step latency percentiles")
    quantiles = {t["legendFormat"] for t in panel["targets"]}
    assert quantiles == {"p50", "p95", "p99"}


def test_metrics_render_in_prometheus_format() -> None:
    metrics.STEPS_COMPLETED.labels(kind="agent_turn", status="succeeded").inc()
    payload, content_type = metrics.render()
    text = payload.decode()
    assert "text/plain" in content_type or "openmetrics" in content_type
    assert "agentorc_steps_total" in text
    # HELP/TYPE lines are what make a metric self-describing in Grafana's browser.
    assert "# HELP agentorc_steps_total" in text
    assert "# TYPE agentorc_steps_total" in text


def test_step_duration_uses_buckets_suited_to_this_workload() -> None:
    """Default buckets bunch tool calls and model turns into the same one.

    Tool calls land in tens of milliseconds, model turns in seconds; without buckets
    covering both, p50 and p99 are indistinguishable.
    """
    metrics.STEP_DURATION.labels(kind="tool_call").observe(0.02)
    metrics.STEP_DURATION.labels(kind="agent_turn").observe(3.0)
    text = metrics.render()[0].decode()
    assert 'le="0.05"' in text
    assert 'le="5.0"' in text


def test_state_gauges_are_not_incremented_in_process() -> None:
    """Guards the collection strategy, not just the numbers.

    `runs_in_progress` must be set from Postgres, never counted locally: three
    workers each keeping their own tally would give Prometheus three partial answers
    to sum into a fourth wrong one. Gauges have `.set`; if someone converts one to a
    Counter to `.inc()` it, this fails.
    """
    from prometheus_client import Gauge

    for gauge in (
        metrics.RUNS_IN_PROGRESS,
        metrics.QUEUE_DEPTH,
        metrics.STEPS_RUNNING,
        metrics.PENDING_APPROVALS,
    ):
        assert isinstance(gauge, Gauge)


# --- tracing -------------------------------------------------------------------


def test_a_run_context_round_trips_through_a_string() -> None:
    """The property the whole recovery story depends on.

    A step executing in a different process, minutes later, must be able to rejoin
    the trace its run started — from nothing but a string in a database column.
    """
    traceparent = tracing.new_run_context(uuid.uuid4(), "a task")
    assert traceparent is not None

    with tracing.step_span(name="step.agent_turn", parent_traceparent=traceparent):
        step_trace_id = tracing.current_trace_id()

    # W3C format: version-traceid-spanid-flags
    run_trace_id = traceparent.split("-")[1]
    assert step_trace_id == run_trace_id


def test_two_steps_from_the_same_run_share_one_trace() -> None:
    """A crash and its recovery must read as one trace, not two failures."""
    traceparent = tracing.new_run_context(uuid.uuid4(), "a task")

    with tracing.step_span(name="step.tool_call", parent_traceparent=traceparent):
        first = tracing.current_trace_id()
    # A different worker, a different process, later in time.
    with tracing.step_span(name="step.tool_call", parent_traceparent=traceparent):
        second = tracing.current_trace_id()

    assert first == second


def test_child_spans_stay_inside_their_step() -> None:
    traceparent = tracing.new_run_context(uuid.uuid4(), "a task")
    with tracing.step_span(name="step.agent_turn", parent_traceparent=traceparent):
        step_trace = tracing.current_trace_id()
        with tracing.child_span("llm.complete", {"gen_ai.request.model": "fake-model"}):
            assert tracing.current_trace_id() == step_trace


@pytest.mark.parametrize("traceparent", [None, "", "not-a-traceparent"])
def test_a_missing_or_malformed_context_does_not_break_a_step(traceparent) -> None:
    """Tracing must never be able to fail a run.

    A run created before this column existed has no traceparent; a corrupted value is
    equally possible. Either way the step still executes — it just starts a new trace.
    """
    with tracing.step_span(name="step.dummy", parent_traceparent=traceparent):
        assert tracing.current_trace_id() is not None


# --- ops config ----------------------------------------------------------------

OPS = Path(__file__).resolve().parents[2]


def _load_yaml(relative: str):
    import yaml

    return yaml.safe_load((OPS / relative).read_text(encoding="utf-8"))


def test_compose_declares_services_not_nested_under_volumes() -> None:
    """Structure, not just syntax.

    A services block accidentally indented under `volumes:` is still valid YAML —
    it just silently defines no services. Checking that it parses proves nothing.
    """
    pytest.importorskip("yaml")
    compose = _load_yaml("docker-compose.yml")

    services = compose["services"]
    assert {"postgres", "redis", "prometheus", "grafana", "otel-collector"} <= set(services)

    # Volumes must be names, not service definitions that drifted into the section.
    for name, value in (compose.get("volumes") or {}).items():
        assert value is None or isinstance(value, dict), name
        assert "image" not in (value or {}), f"{name} looks like a service, not a volume"

    referenced = {
        v.split(":")[0]
        for svc in services.values()
        for v in (svc.get("volumes") or [])
        if not v.startswith(".") and not v.startswith("/")
    }
    assert referenced <= set(compose.get("volumes") or {}), "a named volume is undeclared"


def test_prometheus_scrapes_the_api_for_gauges() -> None:
    pytest.importorskip("yaml")
    prom = _load_yaml("ops/prometheus/prometheus.yml")
    jobs = {j["job_name"] for j in prom["scrape_configs"]}
    assert "agentorc-api" in jobs, "state gauges are only refreshed by the API process"


def test_grafana_provisioning_points_at_the_dashboard_directory() -> None:
    pytest.importorskip("yaml")
    providers = _load_yaml("ops/grafana/provisioning/dashboards/dashboards.yml")
    path = providers["providers"][0]["options"]["path"]
    assert path == "/var/lib/grafana/dashboards"

    compose = _load_yaml("docker-compose.yml")
    mounts = compose["services"]["grafana"]["volumes"]
    assert any(m.endswith(f"{path}:ro") for m in mounts), (
        "the provisioning path is not actually mounted, so no dashboard would load"
    )
