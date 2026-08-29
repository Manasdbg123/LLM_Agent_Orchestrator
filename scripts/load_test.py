"""Concurrent load against the HTTP API.

    python scripts/load_test.py --runs 200 --concurrency 25

Submits N agent runs through `POST /v1/runs` at a fixed in-flight concurrency, then
polls each to completion and reports submission latency, end-to-end run latency, and
throughput.

Unlike the eval harness, this one deliberately goes over HTTP: submission is the path
a real client takes, and the question here is what happens to it when many clients
arrive at once. It drives the deployment you point it at and starts nothing itself,
so the worker count in the report is whatever you actually have running -- which is
the number that makes the throughput figure mean something.

Prerequisites (three terminals):

    make api
    make worker      # run this a few times to scale out
    make reaper

Then:

    python scripts/load_test.py --runs 200 --concurrency 25 --workers 2

The `--workers` flag is documentation only: it is recorded in the report so the
throughput number can be read against the fleet that produced it.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import httpx  # noqa: E402

from app.llm.fake import encode_script  # noqa: E402
from app.runtime import run as run_async  # noqa: E402

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)

TERMINAL = {"succeeded", "failed", "cancelled"}

#: A three-turn run: two tool calls and an answer. Six steps of real durable work
#: (agent_turn, tool_call, agent_turn, tool_call, agent_turn, finalize) per run, which
#: is representative enough that the steps/second figure is not measuring a trivial
#: single-hop path.
SCRIPT: list[dict[str, Any]] = [
    {"tools": [{"name": "calculator", "input": {"expression": "1847 * 293"}}]},
    {"tools": [{"name": "web_search", "input": {"query": "postgres skip locked queue"}}]},
    {"text": "1847 * 293 = 541171, and SKIP LOCKED lets workers claim distinct rows."},
]
TASK_TEXT = "Compute 1847 * 293 and look up what SKIP LOCKED does."


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: always a value that actually occurred."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass(slots=True)
class RunRecord:
    run_id: str | None = None
    submit_ms: float = 0.0
    submit_error: str | None = None
    status: str = "unknown"
    steps: int = 0
    cost_usd: float = 0.0
    #: Engine-side: created_at to ended_at, as the server recorded them.
    engine_latency_s: float = 0.0
    #: Client-side: submission sent to the poll that saw it terminal. Includes the
    #: poll interval, so it is always the larger of the two and always the honest one
    #: to quote as "what a client experienced".
    client_latency_s: float = 0.0


@dataclass(slots=True)
class LoadResult:
    records: list[RunRecord] = field(default_factory=list)
    wall_s: float = 0.0


async def _submit(client: httpx.AsyncClient, record: RunRecord) -> None:
    body = {
        "agent": "loadtest",
        "task": TASK_TEXT + encode_script(SCRIPT),
        "first_step_kind": "agent_turn",
        "tools": ["calculator", "web_search"],
        "timeout_seconds": 300,
    }
    started = time.perf_counter()
    try:
        response = await client.post("/v1/runs", json=body)
        record.submit_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 201:
            record.submit_error = f"HTTP {response.status_code}: {response.text[:120]}"
            return
        record.run_id = response.json()["id"]
    except Exception as exc:
        record.submit_ms = (time.perf_counter() - started) * 1000
        record.submit_error = f"{type(exc).__name__}: {exc}"


async def _poll(
    client: httpx.AsyncClient, record: RunRecord, *, submitted_at: float, timeout: float
) -> None:
    """Poll one run to a terminal state."""
    if record.run_id is None:
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            response = await client.get(f"/v1/runs/{record.run_id}")
            payload = response.json()
        except Exception:
            # A transient read failure is not the measurement; keep polling and let
            # the timeout be the thing that gives up.
            await asyncio.sleep(0.5)
            continue

        record.status = payload.get("status", "unknown")
        if record.status in TERMINAL:
            record.client_latency_s = time.perf_counter() - submitted_at
            record.steps = int(payload.get("steps_used", 0))
            record.cost_usd = float(payload.get("cost_usd", 0.0))
            created, ended = payload.get("created_at"), payload.get("ended_at")
            if created and ended:
                record.engine_latency_s = (
                    dt.datetime.fromisoformat(ended) - dt.datetime.fromisoformat(created)
                ).total_seconds()
            return
        await asyncio.sleep(0.5)

    record.status = f"timeout_in_{record.status}"


async def drive(
    base_url: str, *, runs: int, concurrency: int, timeout: float
) -> LoadResult:
    """Submit `runs` runs with at most `concurrency` in flight, then wait for all."""
    result = LoadResult(records=[RunRecord() for _ in range(runs)])
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)

    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0, limits=limits) as client:

        async def one(record: RunRecord) -> None:
            async with semaphore:
                submitted_at = time.perf_counter()
                await _submit(client, record)
            # Polling happens outside the semaphore: the concurrency knob controls
            # submission pressure, which is the thing being load tested. Holding the
            # slot through a multi-second run would turn it into a throughput cap set
            # by the client rather than by the system.
            await _poll(client, record, submitted_at=submitted_at, timeout=timeout)

        await asyncio.gather(*(one(r) for r in result.records))
    result.wall_s = time.perf_counter() - started
    return result


def summarize(result: LoadResult) -> dict[str, Any]:
    records = result.records
    submitted = [r for r in records if r.submit_error is None]
    completed = [r for r in records if r.status in TERMINAL]
    succeeded = [r for r in records if r.status == "succeeded"]
    submit_ms = [r.submit_ms for r in submitted]
    client_latency = [r.client_latency_s for r in completed]
    engine_latency = [r.engine_latency_s for r in completed if r.engine_latency_s > 0]
    steps = sum(r.steps for r in completed)

    return {
        "runs_requested": len(records),
        "submit_failures": len(records) - len(submitted),
        "completed": len(completed),
        "succeeded": len(succeeded),
        "not_terminal": len(records) - len(completed),
        "wall_s": result.wall_s,
        "runs_per_s": len(completed) / result.wall_s if result.wall_s else 0.0,
        "steps_per_s": steps / result.wall_s if result.wall_s else 0.0,
        "total_steps": steps,
        "submit_p50_ms": percentile(submit_ms, 50),
        "submit_p95_ms": percentile(submit_ms, 95),
        "submit_p99_ms": percentile(submit_ms, 99),
        "submit_mean_ms": statistics.fmean(submit_ms) if submit_ms else 0.0,
        "run_p50_s": percentile(client_latency, 50),
        "run_p95_s": percentile(client_latency, 95),
        "run_p99_s": percentile(client_latency, 99),
        "run_mean_s": statistics.fmean(client_latency) if client_latency else 0.0,
        "engine_p50_s": percentile(engine_latency, 50),
        "engine_p95_s": percentile(engine_latency, 95),
        "engine_p99_s": percentile(engine_latency, 99),
        "total_cost_usd": sum(r.cost_usd for r in completed),
    }


def render_console(summary: dict[str, Any]) -> str:
    ok = summary["succeeded"] == summary["runs_requested"]
    colour = GREEN if ok else (YELLOW if summary["succeeded"] else RED)
    return "\n".join(
        [
            "",
            f"{BOLD}Results{RESET}",
            f"  runs            {summary['runs_requested']} requested, "
            f"{colour}{summary['succeeded']} succeeded{RESET}, "
            f"{summary['submit_failures']} submit failures, "
            f"{summary['not_terminal']} never finished",
            f"  wall clock      {summary['wall_s']:.2f}s",
            f"  throughput      {summary['runs_per_s']:.2f} runs/s   "
            f"{summary['steps_per_s']:.2f} steps/s   "
            f"({summary['total_steps']} steps total)",
            f"  submit latency  mean {summary['submit_mean_ms']:.1f}ms  "
            f"p50 {summary['submit_p50_ms']:.1f}  "
            f"p95 {summary['submit_p95_ms']:.1f}  "
            f"p99 {summary['submit_p99_ms']:.1f}ms",
            f"  run latency     mean {summary['run_mean_s']:.2f}s  "
            f"p50 {summary['run_p50_s']:.2f}  "
            f"p95 {summary['run_p95_s']:.2f}  "
            f"p99 {summary['run_p99_s']:.2f}s  (client-observed)",
            f"  engine latency  p50 {summary['engine_p50_s']:.2f}  "
            f"p95 {summary['engine_p95_s']:.2f}  "
            f"p99 {summary['engine_p99_s']:.2f}s  (created_at to ended_at)",
            "",
        ]
    )


def render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Load test report",
            "",
            f"Generated {meta['generated_at']}.",
            "",
            "## Setup",
            "",
            "| | |",
            "|---|---|",
            f"| Target | `{meta['base_url']}` |",
            f"| Runs submitted | {summary['runs_requested']} |",
            f"| Submission concurrency | {meta['concurrency']} |",
            f"| Workers | {meta['workers']} |",
            f"| Queue backend | {meta['queue']} |",
            f"| LLM provider | {meta['provider']} |",
            f"| Host | {meta['host']} |",
            "",
            "Each run is three model turns and two tool calls, so six durable steps: "
            "`agent_turn -> tool_call -> agent_turn -> tool_call -> agent_turn -> finalize`. "
            "The scripted provider removes network variance from the model, which is what "
            "leaves the engine itself as the thing being measured.",
            "",
            "## Results",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Runs succeeded | {summary['succeeded']} / {summary['runs_requested']} |",
            f"| Submit failures | {summary['submit_failures']} |",
            f"| Runs that never reached a terminal state | {summary['not_terminal']} |",
            f"| Wall clock | {summary['wall_s']:.2f}s |",
            f"| Throughput | **{summary['runs_per_s']:.2f} runs/s** "
            f"({summary['steps_per_s']:.2f} steps/s) |",
            f"| Steps executed | {summary['total_steps']} |",
            f"| Submit latency (mean / p50 / p95 / p99) | {summary['submit_mean_ms']:.1f} / "
            f"{summary['submit_p50_ms']:.1f} / {summary['submit_p95_ms']:.1f} / "
            f"{summary['submit_p99_ms']:.1f} ms |",
            f"| Run latency, client-observed (mean / p50 / p95 / p99) | "
            f"{summary['run_mean_s']:.2f} / {summary['run_p50_s']:.2f} / "
            f"{summary['run_p95_s']:.2f} / {summary['run_p99_s']:.2f} s |",
            f"| Run latency, engine-recorded (p50 / p95 / p99) | "
            f"{summary['engine_p50_s']:.2f} / {summary['engine_p95_s']:.2f} / "
            f"{summary['engine_p99_s']:.2f} s |",
            "",
            "Client-observed latency includes the 500ms poll interval and is therefore "
            "always the larger of the two. It is the honest number to quote for what a "
            "caller experiences; the engine-recorded figure is the honest one for what the "
            "system spent.",
            "",
        ]
        + (["## Notes", "", meta["note"], ""] if meta.get("note") else [])
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="load_test.py", description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="per-run wait for a terminal state"
    )
    parser.add_argument("--workers", default="unspecified", help="recorded in the report")
    parser.add_argument(
        "--note",
        default="",
        help="free-text line recorded in the report, e.g. a comparison against another fleet size",
    )
    parser.add_argument("--markdown", default="docs/LOAD_TEST.md")
    parser.add_argument("--json", default="docs/load_test_results.json")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> tuple[int, dict[str, Any] | None]:
    """Drive the load. Returns (exit code, summary) -- writing the report is the
    caller's job, so no file I/O happens on the event loop."""
    async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as client:
        try:
            health = await client.get("/readyz")
            health.raise_for_status()
        except Exception as exc:
            print(f"{RED}API not reachable at {args.base_url}{RESET}: {exc}")
            print(f"{DIM}Start it with `make api`, and at least one `make worker`.{RESET}")
            return 2, None

    print(
        f"{BOLD}Load test{RESET}  {args.runs} runs, concurrency {args.concurrency}, "
        f"target {args.base_url}\n{DIM}submitting...{RESET}"
    )
    result = await drive(
        args.base_url, runs=args.runs, concurrency=args.concurrency, timeout=args.timeout
    )
    summary = summarize(result)
    print(render_console(summary))

    for record in result.records:
        if record.submit_error:
            print(f"  {RED}submit failed{RESET}: {record.submit_error}")

    return (0 if summary["succeeded"] == summary["runs_requested"] else 1), summary


def write_artifacts(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    from app.config import settings

    meta = {
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "workers": args.workers,
        "queue": settings.queue_backend,
        "provider": settings.llm_provider,
        "host": f"{sys.platform}, python {sys.version.split()[0]}",
        "note": args.note,
    }
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(render_markdown(summary, meta), encoding="utf-8")
    Path(args.json).write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.markdown} and {args.json}")


if __name__ == "__main__":
    cli_args = _parse_args()
    exit_code, load_summary = run_async(main(cli_args))
    if load_summary is not None and not cli_args.no_write:
        write_artifacts(cli_args, load_summary)
    raise SystemExit(exit_code)
