"""Run the evaluation suite.

    python -m eval                      # all 17 tasks, scripted provider, no Redis
    python -m eval --tier simple        # one tier
    python -m eval --task calc_single   # one task
    python -m eval --live               # against the real Anthropic API (costs money)
    python -m eval --queue redis --workers 4 --concurrency 4

Writes `docs/EVAL_REPORT.md` and `docs/eval_results.json`. Exits non-zero if any task
fails, so it is usable as a CI gate rather than only as a thing to look at.

Requires Postgres. Redis only if you ask for the redis queue backend.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    parser.add_argument("--task", action="append", dest="tasks", help="task id (repeatable)")
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        choices=["simple", "multi_step", "complex", "reliability"],
        help="tier to include (repeatable)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the configured real LLM provider instead of the scripted one (costs money)",
    )
    parser.add_argument("--queue", choices=["postgres", "redis"], default="postgres")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="tasks in flight at once; >1 makes the latency column contended",
    )
    parser.add_argument("--json", default="docs/eval_results.json")
    parser.add_argument("--markdown", default="docs/EVAL_REPORT.md")
    parser.add_argument("--no-write", action="store_true", help="console output only")
    return parser.parse_args()


ARGS = _parse_args()

# Must be set before app.config is imported anywhere below.
os.environ.setdefault("AGENTORC_QUEUE_BACKEND", ARGS.queue)
# The reliability tier deliberately stalls a worker; without this those tasks would
# fail on "fault injection is disabled" rather than exercising recovery.
os.environ.setdefault("AGENTORC_ENABLE_FAULT_INJECTION", "true")
if not ARGS.live:
    os.environ.setdefault("AGENTORC_LLM_PROVIDER", "fake")
# Compressed so an expired lease is observable inside a task's lifetime rather than a
# minute after it. Every relationship the config validates still holds.
os.environ.setdefault("AGENTORC_LEASE_TTL_SECONDS", "5")
os.environ.setdefault("AGENTORC_REAPER_INTERVAL_SECONDS", "1")
os.environ.setdefault("AGENTORC_ENQUEUE_GRACE_SECONDS", "2")
os.environ.setdefault("AGENTORC_STALLED_MESSAGE_IDLE_MS", "20000")
os.environ.setdefault("AGENTORC_LOG_LEVEL", "WARNING")
os.environ.setdefault("AGENTORC_LOG_JSON", "false")

from pathlib import Path  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import dispose_engine  # noqa: E402
from app.queue.factory import build_queue  # noqa: E402
from app.runtime import run as run_async  # noqa: E402
from eval import report  # noqa: E402
from eval.harness import Cluster, TaskResult, run_suite  # noqa: E402
from eval.tasks import select  # noqa: E402


def _progress(result: TaskResult) -> None:
    mark = (
        f"{report.GREEN}PASS{report.RESET}"
        if result.passed
        else f"{report.RED}FAIL{report.RESET}"
    )
    print(f"  {mark}  {result.task_id:<32} {result.status:<12} {result.latency_s:>6.2f}s")
    for failure in result.failures:
        print(f"        {report.RED}x{report.RESET} {failure}")


async def main() -> int:
    tasks = select(ARGS.tasks, ARGS.tiers)
    print(
        f"{report.BOLD}Running {len(tasks)} task(s){report.RESET}  "
        f"provider={settings.llm_provider} queue={settings.queue_backend} "
        f"workers={ARGS.workers} concurrency={ARGS.concurrency}\n"
    )

    queue = build_queue()
    await queue.setup()
    try:
        async with Cluster(queue, workers=ARGS.workers):
            results = await run_suite(
                tasks,
                queue,
                live=ARGS.live,
                concurrency=ARGS.concurrency,
                on_result=_progress,
            )
    finally:
        await queue.close()
        await dispose_engine()

    # Back into catalog order: with concurrency > 1 they complete out of order, and a
    # report whose row order changes run to run is a nuisance to diff.
    order = {t.id: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: order.get(r.task_id, 0))

    summary = report.summarize(results)
    print("\n" + report.render_console(results, summary))

    if not ARGS.no_write:
        meta = {
            "generated_at": report.now_stamp(),
            "provider": settings.llm_provider,
            "queue": settings.queue_backend,
            "workers": ARGS.workers,
            "concurrency": ARGS.concurrency,
            "live": ARGS.live,
        }
        report.write_artifacts(
            results,
            summary,
            meta,
            json_path=Path(ARGS.json),
            markdown_path=Path(ARGS.markdown),
        )
        print(f"\nwrote {ARGS.markdown} and {ARGS.json}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_async(main()))
