# Agent Orchestration Engine

[![CI](https://github.com/Manasdbg123/LLM_Agent_Orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/Manasdbg123/LLM_Agent_Orchestrator/actions/workflows/ci.yml)

A durable execution engine for LLM agents. Submit a task; the engine runs a
ReAct-style loop where **every state transition is committed to Postgres before it is
acted on**. Workers are stateless and disposable — kill one mid-step and another
picks the run up exactly where it stopped.

The interesting part is not the agent loop. It is the reliability layer underneath:
leases with fencing tokens, idempotent effects, explicit retry classification, and
human approval gates. See [DESIGN.md](DESIGN.md) for the architecture, the state
machine diagrams, and the reasoning behind each trade-off.

> **Status: all six phases complete.** The durable engine, the ReAct loop, four
> tools, the idempotency ledger, cost tracking, approval gates, every guardrail,
> observability (structured logs, OpenTelemetry, Prometheus, Grafana, a run
> dashboard), a 17-task evaluation harness and a load test are built and tested.
> 194 tests pass; the eval suite reports 17/17.

---

## What works today

| | |
|---|---|
| **Durable state machine** | Runs and steps, every transition validated against a transition table and recorded in an append-only audit log |
| **Leases + fencing** | One atomic conditional `UPDATE` per claim; a stalled worker's late write is rejected by its epoch |
| **Crash recovery** | A reaper reclaims expired leases and re-dispatches; demonstrated by killing real processes |
| **Retries** | Retryable vs. terminal classification, exponential backoff with full jitter, persisted as a timestamp so it survives a restart |
| **Guardrails** | `max_steps`, wall-clock deadline, per-step recovery budget (poison-pill protection) |
| **Two queue backends** | Redis Streams, and a Redis-free Postgres mode that proves the queue is not load-bearing |
| **ReAct loop** | Plan -> tool call -> observe -> answer, where each model turn and each tool call is its own durable step |
| **Idempotent effects** | A ledger keyed on (run, step, tool, args) — deliberately *not* attempt — so a side effect happens at most once per key |
| **4 tools** | Restricted-AST calculator, fixture-backed search, upserting DB write, email through a provider that honours an idempotency key |
| **Structured output** | Tools sent with `strict: true`; invalid arguments come back to the model as an errored `tool_result` rather than killing the run |
| **Cost tracking** | Per-call token and USD ledger from a versioned price table, rolled up per run, exposed at `/v1/runs/{id}/cost` |
| **Approval gates** | Risky calls park the run in `awaiting_approval` and are never enqueued; approving resumes it, rejecting feeds the refusal back to the model |
| **Guardrails** | `max_steps`, USD budget, wall-clock deadline, consecutive-invalid-call cap, per-step recovery budget — each failing with its own error code |
| **API** | Create / inspect / cancel runs, step timeline, audit log, cost breakdown, approval queue and decisions, OpenAPI docs |
| **Observability** | One log line per transition from the function that writes it, OTel traces that survive a crash via a `traceparent` on the step, Prometheus metrics, a provisioned Grafana dashboard, and a server-rendered run/approval UI |
| **Eval harness** | 17 tasks across four tiers, graded against ground truth in the database rather than the engine's self-report — `python -m eval` |
| **Load test** | Concurrent submission over HTTP with throughput and latency percentiles; it found a real concurrency bug the unit tests could not |

---

## Setup

Requires Python 3.12+ and PostgreSQL 14+. Redis is **optional** — the engine runs
fully without it on the Postgres queue backend.

### 1. Database

With Docker:

```bash
docker compose up -d
```

Without Docker, against a local PostgreSQL install — creates the `orch` role and the
`orch` / `orch_test` databases, prompting for your postgres superuser password so the
application never needs it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_local_db.ps1
```

### 2. Application

```bash
python -m venv .venv
.venv\Scripts\activate           # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env             # set AGENTORC_QUEUE_BACKEND=postgres if no Redis
alembic upgrade head
```

### 3. Run the three processes

Each in its own terminal:

```bash
python -m app.api                # API on :8000, docs at /docs
python -m app.worker             # start as many as you like
python -m app.reaper             # crash recovery + retry dispatch
```

By default the engine uses a **scripted fake model provider**, so nothing calls a
paid API until you opt in. To use Claude:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # or: ant auth login
export AGENTORC_LLM_PROVIDER=anthropic
export AGENTORC_LLM_MODEL=claude-opus-5
```

Use `python -m app.api` rather than `uvicorn app.api.main:app`. On Windows uvicorn
builds its event loop before this package gets a say, and the default
`ProactorEventLoop` is one psycopg's async driver refuses to run on. The module entry
point installs a selector loop first (see `app/runtime.py`).

### 4. Submit a run

```bash
curl -X POST localhost:8000/v1/runs \
  -H 'content-type: application/json' \
  -H 'Idempotency-Key: demo-1' \
  -d '{"agent":"demo","task":"three steps",
       "input":{"plan":[{"label":"a"},{"label":"b"},{"label":"c"}]}}'

curl localhost:8000/v1/runs/<id>/steps        # the timeline
curl localhost:8000/v1/runs/<id>/transitions  # the audit log: who changed what, and why
```

Re-sending the same `Idempotency-Key` returns the original run instead of starting a
second one.

```bash
curl localhost:8000/v1/runs/<id>/cost   # per-call tokens, USD, price version
```

A run that wants to send an email parks itself for a human:

```bash
curl localhost:8000/v1/approvals                       # the queue
curl -X POST localhost:8000/v1/approvals/<id>/decision \
  -H 'content-type: application/json' \
  -d '{"decision":"approve","decided_by":"you@example.com","reason":"checked it"}'
```

Rejecting does not kill the run: the refusal comes back to the model as an errored
tool result so it can choose another course. Set the agent's `on_approval_rejected`
to `fail_run` if you want a rejection to be terminal.

---

## The dashboard

`make api` serves an operator dashboard at **http://localhost:8000/ui** — server-rendered,
no build step, light and dark themes.

| Page | What you can do |
|---|---|
| **Runs** (`/ui`) | See running / awaiting / succeeded / failed counts and total spend; filter by status; search task text; **start a run** with an optional step and budget cap |
| **Run detail** (`/ui/runs/{id}`) | Step and budget meters, the answer or failure code, a step timeline with durations, per-step cost, retries and lease recoveries, the full audit log; **cancel** a live run |
| **Approvals** (`/ui/approvals`) | Review a paused tool call's arguments and **approve or reject** it, with a reason the model sees on rejection |

Every action goes through the same core service call as its API endpoint, so the
dashboard cannot disagree with the API. Pages refresh themselves every three seconds
(toggle **Live** in the header) by swapping the page body in place, and pause while you
are typing in a form; a finished run stops refreshing.

---

## The crash-recovery demo

This is the part worth actually running. It is not a mock and not a narrated log —
it starts real worker processes, kills one of them with no warning while it is
holding a lease mid-step, and then shows you the database recovering.

```bash
python scripts/demo_crash_recovery.py
```

What it does, in order:

1. Starts **two real worker processes** and submits a three-step run.
2. Waits until a worker is *provably* mid-step: it has committed its side effect but
   has **not** committed its result. (It polls the database for that exact state
   rather than sleeping and hoping.)
3. **Kills that process** — `Popen.kill()`, i.e. `SIGKILL` / `TerminateProcess`. No
   atexit hooks, no `finally` blocks, no lease release. The lease just goes stale.
4. Waits for the lease to expire and the reaper to move the step back to `pending`.
5. Watches the surviving worker pick it up and finish the run.
6. Prints the step timeline and the full state-transition audit trail, then asserts:
   - the run reached `succeeded`
   - **step 1 was not re-executed** (`executions == 1`)
   - the interrupted step was reassigned (`recoveries >= 1`, `attempt == 2`)
   - no step is left holding a lease

It exits non-zero if any assertion fails, so it is usable in CI.

The demo runs on the **Postgres queue backend with Redis switched off**, so what you
are watching cannot be explained away as queue redelivery.

It also prints one honest caveat: the interrupted step's side effect ran *twice*.
That is not a recovery bug — the effect landed and the process died before the result
could be recorded, so the replacement worker had no way to know. Removing that
duplicate is precisely what the Phase 3/4 idempotency ledger does. Phase 2 leaves it
visible rather than hiding it.

---

## The idempotency demo

Phase 2's crash demo ended on an honest caveat: the interrupted step's side effect
ran *twice*. This demo is that caveat being closed.

```bash
python scripts/demo_idempotency.py
```

An agent tries to send an email. The run parks at the approval gate; an operator
approves it; the worker sends it, then stalls before recording the result. Its lease
expires and a second worker re-runs the tool call — and the provider recognises the
idempotency key and returns the original message instead of sending again.

It asserts against the provider's own outbox table, not against engine bookkeeping —
the claim is "the customer received one email", and only the outbox settles that:

```
send_email is waiting for a human: tool is marked as requiring approval
approved; the run resumes
...
outbox: msg_e8f75b08d0def38b  to=customer@example.test  duplicate_attempts=1
ledger: send_email  status=committed  key=ef810c33c771704c...  first_claimed_on_attempt=1

  PASS  the send was gated until a human approved it
  PASS  EXACTLY ONE email was sent
  PASS  the provider suppressed a duplicate send
  PASS  the step really did execute twice
  PASS  the step was reassigned by the reaper
```

The step executed twice and the effect happened once. That gap is closed by the
ledger plus a provider that honours the key — not by preventing the re-execution,
which is not something any engine can guarantee.

---

## The evaluation harness

Seventeen tasks across four tiers, from a single tool call to a deliberate worker
crash, run through the real engine — real Postgres, real workers, a real reaper — and
graded.

```bash
python -m eval                      # everything (~35s, no Redis, no API key)
python -m eval --tier reliability   # one tier
python -m eval --task crash_recovery_no_duplicate
python -m eval --live               # against the real Anthropic API (costs money)
```

It writes [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md) and `docs/eval_results.json`, and
exits non-zero if any task fails, so it works as a CI gate rather than only as
something to look at.

Latest run, scripted provider, Postgres queue, two workers:

| | |
|---|---|
| Success rate | **100%** (17/17) |
| Avg steps per task | 4.9 |
| Avg cost per task | $0.04012 |
| Avg latency | 1.55s (p50 1.21s, p95 6.66s) |
| Retries / lease recoveries / approval gates exercised | 2 / 1 / 5 |

Read the cost column with one caveat: these are the scripted provider's synthetic
token counts, and `guardrail_budget` deliberately declares a 440k-token turn to drive
the budget guardrail. It alone accounts for $0.60 of the $0.68 total, so the *median*
task costs $0.004, not the $0.040 mean. Run `--live` for figures that mean money.

**What the tiers cover.** `simple` is one tool call or none. `multi_step` chains 2–3
dependent calls across tools. `complex` runs a four-tool pipeline, a single model turn
emitting two tool_use blocks, and a malformed tool call the model has to correct.
`reliability` is the interesting one: approval granted, approval rejected,
data-dependent gating on a sensitive namespace, a transient failure that must be
retried, a worker whose lease expires mid-send, and two guardrails driven past their
limits.

**Grading reads ground truth, not the engine's self-report.** "The email was sent
once" is a `SELECT count(*)` against the mock provider's `email_outbox`; "the record
was written" is a row in `sandbox_records`. What the run says about itself is recorded
in the results file but never decides pass or fail — an engine bug that also corrupts
the engine's account of itself is a real failure mode, and grading on self-reported
fields is blind to exactly that.

The grader is itself tested (`tests/unit/test_eval_grading.py`): it is fed
observations that *should* fail and asserted to fail on them. A harness that reports
100% because it never checks anything is worse than no harness.

**One catalog, two providers.** Every task carries both a natural-language instruction
and a script. The default scripted provider makes the suite deterministic and free, so
a red result means the engine broke rather than that the model had an off day;
`--live` sends only the instruction to Claude and grades with the same expectations.
The trade-off is explicit: the default suite cannot catch a bad prompt or a poor tool
choice — only `--live` can — but it catches every engine-level regression in about
thirty seconds, which is the one that can run on every commit.

---

## The load test

```bash
# three terminals
make api
make worker          # run this a few times to scale out
make reaper

python scripts/load_test.py --runs 200 --concurrency 40 --workers 4
```

Submits N runs through `POST /v1/runs` at a fixed in-flight concurrency, polls each to
completion, and writes [docs/LOAD_TEST.md](docs/LOAD_TEST.md). Each run is six durable
steps, so the steps/second figure is not measuring a trivial single-hop path.

Unlike the eval harness this one goes over HTTP on purpose: submission under
concurrency is where the transport is the subject. It starts nothing itself, so the
worker count in the report is the fleet that actually produced the number.

**It found a real bug on its first run.** At concurrency 20, 13% of submissions
returned HTTP 500: `get_or_create_definition` was a read-then-insert, so several
requests naming the same new agent all missed, all inserted, and every one but the
winner took a unique violation. The fix is `ON CONFLICT DO NOTHING` plus a re-read;
`tests/integration/test_definition_concurrency.py` is the regression test, and against
the old shape nine of its twelve concurrent callers raise.

No existing test could have caught it: tests create uniquely-named definitions and
therefore never collide. It needed concurrent submission of the *same* name — which is
the normal case in production, where an agent is defined once and called constantly.

**Results, and their honest limits.** 200 runs at concurrency 40, everything on one
laptop: ~11–13 runs/s and ~65–77 steps/s, 200/200 succeeding. Doubling the fleet from
2 to 4 workers raised throughput and lowered run latency, but *sublinearly*, and made
submit latency worse — everything shares one Postgres and one uvicorn process, and on
the Postgres queue backend every worker polls the same table it is being submitted
through. The engine is not worker-CPU-bound at this size; it is bound by the one
Postgres under all of it. Switching to the Redis backend replaces that poll with a
push and is the first thing to change before reading more into the numbers.

What the run does establish: across 400 runs, **every run executed exactly six
steps**, max attempt 1, zero duplicate committed effects. Duplicate execution under
concurrency would show up immediately as a seventh step.

---

## Tests

```bash
pytest tests/unit                    # no infrastructure needed
pytest -m integration                # real Postgres
pytest -m chaos                      # kills processes, expires leases, injects faults
pytest                               # everything
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff and mypy, then
the unit and integration suites against real Postgres 16 and Redis 7 service
containers, then the 17-task evaluation suite, on every pull request.

Current status on a machine with Postgres but no Redis: **194 passed, 15 skipped**
(every skip is a Redis-backed test parameter). Most integration and chaos tests are
parameterised over both queue backends, so the Redis adapter's own code path
(`XREADGROUP` / `XAUTOCLAIM` / ack handling) is the one part not yet covered by a
passing test.

Integration tests run against a **real** database — mocking it would mock away the
entire subject of the tests, since what is being asserted is the behaviour of atomic
conditional `UPDATE`s under concurrency, partial unique indexes, and `now()` evaluated
server-side. The schema is created by `alembic upgrade head`, never
`metadata.create_all`, so migration drift fails a test instead of reaching production.

Tests that matter most:

| Test | What it proves |
|---|---|
| `test_only_one_of_two_concurrent_claims_wins` | Two workers racing for one step: exactly one wins |
| `test_a_fenced_worker_cannot_commit_its_result` | A stalled worker's late write is rejected after takeover |
| `test_completed_steps_are_never_re_executed_during_recovery` | Recovery replays the interrupted step and nothing else |
| `test_a_worker_that_dies_mid_step_does_not_lose_the_run` | Real `kill -9` on a real process, real recovery |
| `test_recovery_budget_is_separate_from_the_retry_budget` | A poison step is abandoned, not cycled forever |
| `test_the_queue_is_not_load_bearing` | Redis is flushed mid-run; the run still completes |
| `test_multiple_workers_never_execute_a_step_twice` | Horizontal scaling doesn't duplicate work |
| `test_an_email_is_sent_once_even_when_the_worker_dies_after_sending` | The headline: one email, asserted on the provider's outbox |
| `test_a_safe_to_replay_write_converges_after_a_crash` | The replay really re-executes, and the upsert still converges |
| `test_malformed_tool_arguments_are_corrected_by_the_model` | Bad tool args are recoverable, not fatal |
| `test_key_is_stable_across_attempts` | The idempotency key ignores `attempt` — the guard against silent duplicate effects |
| `test_refuses_anything_that_is_not_arithmetic` | The calculator's allowlist blocks the usual sandbox escapes |
| `test_cost_rollup_equals_the_sum_of_its_ledger` | The denormalised cost rollup matches its ledger |
| `test_a_risky_tool_parks_the_run_instead_of_running` | A gated step is never claimed — `attempt == 0`, no email sent |
| `test_a_decision_cannot_be_reversed` | Approvals are idempotent; a second decision returns 409 |
| `test_a_run_survives_a_full_worker_restart_while_gated` | Approval state lives in Postgres, not in a worker |
| `test_max_steps_stops_the_run_exactly_at_its_budget` | The cap stops the run *at* the limit, not past it |
| `test_the_budget_check_reads_the_ledger_not_a_cached_value` | Spend from a prior step is visible to the next |
| `test_recovering_from_invalid_calls_resets_the_counter` | The invalid-call cap counts consecutive, not lifetime, failures |

---

## Layout

```
app/
  domain/     states.py (the transition tables), errors.py (taxonomy), models.py
  core/       transitions.py, leases.py, retry.py, runs.py
  queue/      base.py (port), redis_streams.py, postgres.py
  engine/     executor.py, types.py, faults.py, handlers/
  api/        FastAPI surface
  worker/     python -m app.worker
  reaper/     python -m app.reaper
  obs/        logging.py, tracing.py, metrics.py
  dashboard/  server-rendered run list, run detail, approval queue
  tools/      calculator, web_search, database_write, send_email
  llm/        provider port, Anthropic adapter, scripted provider, pricing
eval/       tasks.py (the catalog), harness.py (runner + grading), report.py
alembic/  tests/{unit,integration,chaos}/  scripts/  ops/  docs/
```

Two rules the codebase depends on:

- **`app/core/transitions.py` is the only code that writes a status column.** It
  validates against the transition table and writes the audit row in the same
  transaction, so the log cannot drift from the state it describes. (The one
  sanctioned exception is `leases.claim_step`, which must be a single atomic
  statement; it records its own transition.)
- **Postgres is the source of truth; the queue is advisory.** A step is runnable iff
  one SQL predicate says so, and the reaper polls that same predicate. Losing the
  queue costs latency, never work.
- **The transcript is derived, never stored twice.** The `messages` array sent to the
  model is rebuilt from the step timeline each turn, so it cannot drift from the state
  machine and a recovered run reconstructs identical context.
- **The idempotency key never includes `attempt`.** Including it would mean a retry
  computes a different key and the ledger forgets the first attempt — which is exactly
  how a customer gets two emails.
- **A gated step is never enqueued.** It is created directly in `awaiting_approval`,
  so there is no window in which a worker could claim it before a human has seen it.
- **Every guardrail fails with its own error code.** "The run failed" is not
  actionable; `budget_exceeded`, `max_steps_exceeded`, `deadline_exceeded` and
  `model_output_invalid` are four different bugs to go and fix.

## Configuration

Everything is `AGENTORC_`-prefixed; see [.env.example](.env.example). The timing
knobs are interdependent and validated at startup — `stalled_message_idle_ms` must
exceed `lease_ttl_seconds`, and `enqueue_grace_seconds` must exceed
`reaper_interval_seconds`, or the process refuses to start rather than misbehaving
subtly under load.

`AGENTORC_ENABLE_FAULT_INJECTION` lets a step deliberately crash, stall, or fail its
own worker. It is off by default, and a step that requests a fault without it raises
rather than silently ignoring the request.
