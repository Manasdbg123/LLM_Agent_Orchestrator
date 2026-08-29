"""OpenTelemetry tracing across a workflow that outlives its processes.

A run can last minutes and span several workers, one of which may be killed halfway
through. That breaks the usual assumption behind tracing — that a parent span is a
live object in memory that child spans can attach to. It is not: by the time a
recovered step runs, the process that would have held the parent span is gone.

So the trace context is **persisted, not held**. At run creation a root context is
minted and its W3C `traceparent` is written to `agent_runs.traceparent`. Every step,
in whatever process, restores that context and starts its span as a child of it. The
result is one trace containing every step of the run, including the step that died
and the step that replaced it — which is exactly the picture that makes a recovery
legible instead of looking like two unrelated failures.

Exporting is optional and off by default. Without an OTLP endpoint configured the
tracer still produces real span contexts (so `traceparent` propagation and the
`trace_id` recorded on every state transition still work), it just does not ship
them anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanKind

from app.obs.logging import get_logger

log = get_logger("tracing")

_configured = False
TRACER_NAME = "agent-orchestrator"


def configure_tracing(service_name: str | None = None) -> None:
    """Idempotent; safe to call from every entry point."""
    global _configured
    if _configured:
        return

    from app.config import settings

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name or settings.service_name,
                "service.version": "0.5.0",
            }
        )
    )

    if settings.otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
            )
            log.info("otlp_exporter_configured", endpoint=settings.otlp_endpoint)
        except Exception as exc:
            # A missing collector must not stop the engine from running.
            log.warning("otlp_exporter_failed", error=str(exc))

    trace.set_tracer_provider(provider)
    _configured = True


def tracer() -> trace.Tracer:
    configure_tracing()
    return trace.get_tracer(TRACER_NAME)


def current_traceparent() -> str | None:
    """Serialise the active context to a W3C traceparent header."""
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")


def context_from(traceparent: str | None) -> Context | None:
    """Rebuild a context from a stored traceparent.

    This is what lets a step in a *different process*, minutes later, join the trace
    its run started.
    """
    if not traceparent:
        return None
    return extract({"traceparent": traceparent})


def new_run_context(run_id: Any, task: str) -> str | None:
    """Mint the root context for a run and return its traceparent.

    The root span is opened and closed immediately: nothing can hold a span open for
    the lifetime of a run that may outlive the process, so the run's identity travels
    as a serialised context instead.
    """
    with tracer().start_as_current_span(
        "run",
        kind=SpanKind.SERVER,
        attributes={"agentorc.run_id": str(run_id), "agentorc.task": task[:200]},
    ):
        return current_traceparent()


@contextmanager
def step_span(
    *,
    name: str,
    parent_traceparent: str | None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span]:
    """A span for one step, parented to the run's persisted context."""
    context = context_from(parent_traceparent)
    with tracer().start_as_current_span(
        name, context=context, kind=SpanKind.CONSUMER, attributes=attributes or {}
    ) as span:
        yield span


@contextmanager
def child_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    """A span nested inside whatever is currently active (an LLM or tool call)."""
    with tracer().start_as_current_span(
        name, kind=SpanKind.CLIENT, attributes=attributes or {}
    ) as span:
        yield span


def current_trace_id() -> str | None:
    """Hex trace id of the active span, recorded on every state transition.

    That is what joins the audit log in Postgres to the traces in Grafana: given a
    failed run you can go from its transition rows straight to the trace.
    """
    span = trace.get_current_span()
    if isinstance(span, NonRecordingSpan) and span.get_span_context().trace_id == 0:
        return None
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")


def record_exception(span: trace.Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))


__all__ = [
    "child_span",
    "configure_tracing",
    "context_from",
    "current_trace_id",
    "current_traceparent",
    "new_run_context",
    "record_exception",
    "step_span",
    "tracer",
]
