"""Aggregation and rendering for eval results.

Kept separate from the runner so the numbers can be recomputed from a saved
`results.json` without re-running the suite -- which matters when the interesting
question is "did this change move the numbers", and the previous run is a file.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eval.harness import TaskResult

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile.

    Not `statistics.quantiles`: that interpolates and needs at least two points, and
    a seventeen-task suite reporting an interpolated p99 would be inventing precision
    it does not have. Nearest-rank always returns a value that actually occurred.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    """Headline numbers plus a per-tier breakdown."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    latencies = [r.latency_s for r in results]
    costs = [r.cost_usd for r in results]
    steps = [r.steps_used for r in results]

    by_tier: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        grouped[r.tier].append(r)
    for tier, rows in grouped.items():
        by_tier[tier] = {
            "tasks": len(rows),
            "passed": sum(1 for r in rows if r.passed),
            "success_rate": sum(1 for r in rows if r.passed) / len(rows),
            "avg_steps": statistics.fmean(r.steps_used for r in rows),
            "avg_cost_usd": statistics.fmean(r.cost_usd for r in rows),
            "avg_latency_s": statistics.fmean(r.latency_s for r in rows),
        }

    return {
        "tasks": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed / total if total else 0.0,
        "avg_steps": statistics.fmean(steps) if steps else 0.0,
        "avg_cost_usd": statistics.fmean(costs) if costs else 0.0,
        "total_cost_usd": sum(costs),
        "avg_latency_s": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "p99_latency_s": percentile(latencies, 99),
        "total_tool_calls": sum(r.tool_calls for r in results),
        "total_retries": sum(r.retries for r in results),
        "total_recoveries": sum(r.recoveries for r in results),
        "total_approvals": sum(r.approvals for r in results),
        "by_tier": by_tier,
    }


def render_console(results: list[TaskResult], summary: dict[str, Any]) -> str:
    """The table you read while it is running."""
    lines: list[str] = []
    header = (
        f"{'task':<32} {'tier':<12} {'res':<5} "
        f"{'steps':>5} {'tools':>5} {'cost $':>9} {'lat s':>7}"
    )
    lines.append(BOLD + header + RESET)
    lines.append(DIM + "-" * len(header) + RESET)

    for r in results:
        mark = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
        lines.append(
            f"{r.task_id:<32} {r.tier:<12} {mark:<14} "
            f"{r.steps_used:>5} {r.tool_calls:>5} {r.cost_usd:>9.5f} {r.latency_s:>7.2f}"
        )
        for failure in r.failures:
            lines.append(f"    {RED}x{RESET} {failure}")

    rate = summary["success_rate"]
    colour = GREEN if rate == 1.0 else (YELLOW if rate >= 0.8 else RED)
    lines.append("")
    lines.append(
        f"{BOLD}{summary['passed']}/{summary['tasks']} passed{RESET}  "
        f"success rate {colour}{rate:.0%}{RESET}"
    )
    lines.append(
        f"  avg steps {summary['avg_steps']:.1f}   "
        f"avg cost ${summary['avg_cost_usd']:.5f}   "
        f"total cost ${summary['total_cost_usd']:.4f}"
    )
    lines.append(
        f"  latency  avg {summary['avg_latency_s']:.2f}s  "
        f"p50 {summary['p50_latency_s']:.2f}s  "
        f"p95 {summary['p95_latency_s']:.2f}s  "
        f"p99 {summary['p99_latency_s']:.2f}s"
    )
    lines.append(
        f"  retries {summary['total_retries']}   "
        f"lease recoveries {summary['total_recoveries']}   "
        f"approval gates {summary['total_approvals']}"
    )
    return "\n".join(lines)


def render_markdown(
    results: list[TaskResult], summary: dict[str, Any], meta: dict[str, Any]
) -> str:
    """The committed report."""
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"Generated {meta['generated_at']} — provider `{meta['provider']}`, "
        f"queue `{meta['queue']}`, {meta['workers']} worker(s), "
        f"concurrency {meta['concurrency']}.",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tasks | {summary['tasks']} |",
        f"| Success rate | **{summary['success_rate']:.0%}** "
        f"({summary['passed']}/{summary['tasks']}) |",
        f"| Avg steps per task | {summary['avg_steps']:.1f} |",
        f"| Avg cost per task | ${summary['avg_cost_usd']:.5f} |",
        f"| Total cost | ${summary['total_cost_usd']:.4f} |",
        f"| Avg latency | {summary['avg_latency_s']:.2f}s |",
        f"| p50 / p95 / p99 latency | {summary['p50_latency_s']:.2f}s / "
        f"{summary['p95_latency_s']:.2f}s / {summary['p99_latency_s']:.2f}s |",
        f"| Tool calls executed | {summary['total_tool_calls']} |",
        f"| Retries | {summary['total_retries']} |",
        f"| Lease recoveries | {summary['total_recoveries']} |",
        f"| Approval gates opened | {summary['total_approvals']} |",
        "",
        "## By tier",
        "",
        "| Tier | Tasks | Success | Avg steps | Avg cost | Avg latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tier in ("simple", "multi_step", "complex", "reliability"):
        row = summary["by_tier"].get(tier)
        if row is None:
            continue
        lines.append(
            f"| {tier} | {row['tasks']} | {row['success_rate']:.0%} | "
            f"{row['avg_steps']:.1f} | ${row['avg_cost_usd']:.5f} | {row['avg_latency_s']:.2f}s |"
        )

    lines += [
        "",
        "## Per task",
        "",
        "| Task | Tier | Result | Status | Steps | Tools | Cost | Latency |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r.task_id}` | {r.tier} | {'PASS' if r.passed else '**FAIL**'} | {r.status} | "
            f"{r.steps_used} | {r.tool_calls} | ${r.cost_usd:.5f} | {r.latency_s:.2f}s |"
        )

    failed = [r for r in results if r.failures]
    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            lines.append(f"**`{r.task_id}`** — {r.intent}")
            lines += [f"- {f}" for f in r.failures]
            lines.append("")

    lines += ["", "## What each task covers", ""]
    for r in results:
        lines.append(f"- **`{r.task_id}`** ({r.tier}) — {r.intent}")

    lines.append("")
    return "\n".join(lines)


def write_artifacts(
    results: list[TaskResult],
    summary: dict[str, Any],
    meta: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": summary,
        "results": [
            {**asdict(r), "run_id": str(r.run_id) if r.run_id else None} for r in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(results, summary, meta), encoding="utf-8")


def now_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


__all__ = [
    "now_stamp",
    "percentile",
    "render_console",
    "render_markdown",
    "summarize",
    "write_artifacts",
]
