"""Evaluation harness.

Runs a catalog of tasks through the real engine — real Postgres, real workers, real
reaper — and reports success rate, steps, cost and latency. See `eval/tasks.py` for
the catalog and `eval/harness.py` for the runner.
"""
