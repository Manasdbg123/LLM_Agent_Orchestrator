# Load test report

Generated 2026-08-29 03:13 UTC.

## Setup

| | |
|---|---|
| Target | `http://localhost:8000` |
| Runs submitted | 200 |
| Submission concurrency | 40 |
| Workers | 4 (concurrency 8 each) |
| Queue backend | postgres |
| LLM provider | fake |
| Host | win32, python 3.14.3 |

Each run is three model turns and two tool calls, so six durable steps: `agent_turn -> tool_call -> agent_turn -> tool_call -> agent_turn -> finalize`. The scripted provider removes network variance from the model, which is what leaves the engine itself as the thing being measured.

## Results

| Metric | Value |
|---|---|
| Runs succeeded | 200 / 200 |
| Submit failures | 0 |
| Runs that never reached a terminal state | 0 |
| Wall clock | 17.17s |
| Throughput | **11.65 runs/s** (69.87 steps/s) |
| Steps executed | 1200 |
| Submit latency (mean / p50 / p95 / p99) | 1153.2 / 776.2 / 2697.8 / 2956.2 ms |
| Run latency, client-observed (mean / p50 / p95 / p99) | 12.93 / 13.45 / 14.09 / 14.54 s |
| Run latency, engine-recorded (p50 / p95 / p99) | 12.08 / 12.85 / 12.92 s |

Client-observed latency includes the 500ms poll interval and is therefore always the larger of the two. It is the honest number to quote for what a caller experiences; the engine-recorded figure is the honest one for what the system spent.

## Notes

**Horizontal scaling, and its limit here.** The identical workload against **2 workers** (`docs/load_test_2workers.json`) completed 200/200 runs in 18.33s: 10.91 runs/s, 65.5 steps/s, submit p95 870ms, run p95 15.95s. Doubling the fleet to 4 workers produced the table above — more throughput and lower run latency, but sublinearly, and submit latency got *worse*.

That is the honest shape of this result, and the reason is worth stating: everything here shares one local Postgres and a single uvicorn process on a laptop. On the `postgres` queue backend every worker polls the same table, so adding workers adds contention on the store that is also serving submissions — which is exactly what a rising submit p95 alongside a falling run p95 looks like. The engine is not worker-CPU-bound at this size; it is bound by the one Postgres underneath all of it. The `redis` backend replaces that poll with a push and would be the first thing to change before reading more into these numbers, followed by separating the API's connection pool from the workers'.

**What the run does prove, and it is the important part:** across 400 runs in the two configurations, every run completed exactly 6 steps. Duplicate execution under concurrency would show up immediately as a step count above 6, and it does not appear at any fleet size.
