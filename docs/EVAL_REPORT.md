# Evaluation report

Generated 2026-08-29 03:27 UTC — provider `fake`, queue `postgres`, 2 worker(s), concurrency 1.

## Headline

| Metric | Value |
|---|---|
| Tasks | 17 |
| Success rate | **100%** (17/17) |
| Avg steps per task | 4.9 |
| Avg cost per task | $0.04012 |
| Total cost | $0.6820 |
| Avg latency | 1.55s |
| p50 / p95 / p99 latency | 1.21s / 6.66s / 6.66s |
| Tool calls executed | 28 |
| Retries | 2 |
| Lease recoveries | 1 |
| Approval gates opened | 5 |

## By tier

| Tier | Tasks | Success | Avg steps | Avg cost | Avg latency |
|---|---:|---:|---:|---:|---:|
| simple | 4 | 100% | 3.5 | $0.00350 | 0.94s |
| multi_step | 3 | 100% | 6.7 | $0.00667 | 1.71s |
| complex | 3 | 100% | 7.0 | $0.00667 | 1.90s |
| reliability | 7 | 100% | 4.1 | $0.08971 | 1.67s |

## Per task

| Task | Tier | Result | Status | Steps | Tools | Cost | Latency |
|---|---|---|---|---:|---:|---:|---:|
| `calc_single` | simple | PASS | succeeded | 4 | 1 | $0.00400 | 1.21s |
| `search_single` | simple | PASS | succeeded | 4 | 1 | $0.00400 | 1.14s |
| `db_write_scratch` | simple | PASS | succeeded | 4 | 1 | $0.00400 | 0.96s |
| `no_tool_answer` | simple | PASS | succeeded | 2 | 0 | $0.00200 | 0.47s |
| `calc_chain` | multi_step | PASS | succeeded | 6 | 2 | $0.00600 | 1.59s |
| `search_then_calc` | multi_step | PASS | succeeded | 6 | 2 | $0.00600 | 1.45s |
| `search_calc_store` | multi_step | PASS | succeeded | 8 | 3 | $0.00800 | 2.09s |
| `four_tool_pipeline` | complex | PASS | succeeded | 10 | 4 | $0.01000 | 2.85s |
| `parallel_tool_uses` | complex | PASS | succeeded | 5 | 2 | $0.00400 | 1.36s |
| `malformed_then_recover` | complex | PASS | succeeded | 6 | 2 | $0.00600 | 1.48s |
| `approval_granted` | reliability | PASS | succeeded | 4 | 1 | $0.00400 | 1.04s |
| `approval_rejected` | reliability | PASS | succeeded | 3 | 1 | $0.00400 | 0.79s |
| `approval_sensitive_namespace` | reliability | PASS | succeeded | 6 | 2 | $0.00600 | 1.63s |
| `retry_transient` | reliability | PASS | succeeded | 4 | 1 | $0.00400 | 1.17s |
| `crash_recovery_no_duplicate` | reliability | PASS | succeeded | 4 | 1 | $0.00400 | 6.66s |
| `guardrail_max_steps` | reliability | PASS | failed | 6 | 3 | $0.00600 | 0.31s |
| `guardrail_budget` | reliability | PASS | failed | 2 | 1 | $0.60000 | 0.11s |

## What each task covers

- **`calc_single`** (simple) — One tool call, one answer: the shortest complete path through the loop.
- **`search_single`** (simple) — Retrieval path: a read-only tool whose output must reach the next turn.
- **`db_write_scratch`** (simple) — A side-effecting tool on a namespace that does not trip the approval gate.
- **`no_tool_answer`** (simple) — The model answers directly. Proves tools are an option, not a forced path.
- **`calc_chain`** (multi_step) — Two dependent calls: the second consumes the first's observation.
- **`search_then_calc`** (multi_step) — Cross-tool dependency: a retrieved fact feeds a computation.
- **`search_calc_store`** (multi_step) — Read, compute, persist -- the shape of most real agent work.
- **`four_tool_pipeline`** (complex) — Four calls across four tools, including a gated one, in a single run.
- **`parallel_tool_uses`** (complex) — One assistant turn emitting two tool_use blocks. Each becomes its own durable step, chained rather than fanned out.
- **`malformed_then_recover`** (complex) — Invalid tool arguments come back as an errored tool_result and the model corrects itself. A bad turn must not kill a run.
- **`approval_granted`** (reliability) — A gated tool parks the run in awaiting_approval; approving resumes it and the effect happens exactly once.
- **`approval_rejected`** (reliability) — A rejected gate produces no side effect, and the refusal is fed back so the model can finish honestly rather than the run simply dying.
- **`approval_sensitive_namespace`** (reliability) — Approval is data-dependent: the same tool is ungated on 'scratch' and gated on 'billing'.
- **`retry_transient`** (reliability) — A retryable failure before the effect is retried with backoff and then succeeds; the run as a whole never notices.
- **`crash_recovery_no_duplicate`** (reliability) — The headline claim, graded: a worker's lease expires after the email is sent, the reaper reassigns the step, and the customer still gets ONE email.
- **`guardrail_max_steps`** (reliability) — A model that never stops is stopped by the engine, under a named error rather than by silently exhausting a budget.
- **`guardrail_budget`** (reliability) — Cost is enforced before the call, not apologised for after it.
