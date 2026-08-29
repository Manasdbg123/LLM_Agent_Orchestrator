"""Process configuration.

Every knob that affects reliability semantics lives here rather than being scattered
as literals, because the correctness argument depends on the relationship *between*
several of them (see `Settings.validate_timing`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENTORC_", extra="ignore", case_sensitive=False
    )

    # --- infrastructure -------------------------------------------------------
    database_url: str = "postgresql+psycopg://orch:orch@localhost:5432/orch"
    redis_url: str = "redis://localhost:6379/0"
    db_pool_size: int = 10
    db_max_overflow: int = 10

    # --- queue ----------------------------------------------------------------
    # "postgres" drops Redis entirely and polls the runnable predicate. It exists
    # both as a low-dependency deployment mode and as a live proof that Redis is
    # not load-bearing for correctness (DESIGN.md section 2).
    queue_backend: Literal["redis", "postgres"] = "redis"
    stream_name: str = "steps:ready"
    consumer_group: str = "workers"
    # An entry sitting unacked in the consumer group's PEL for longer than this is
    # assumed orphaned and reclaimed via XAUTOCLAIM. Must exceed lease_ttl_seconds,
    # otherwise a worker legitimately holding a lease has its message stolen.
    stalled_message_idle_ms: int = 90_000

    # --- leases / recovery ----------------------------------------------------
    # Must exceed the slowest legitimate step, or healthy workers get reaped.
    lease_ttl_seconds: int = 60
    # Ratio, not an absolute: 3 heartbeats per TTL tolerates two consecutive misses.
    heartbeat_divisor: int = 3
    reaper_interval_seconds: float = 2.0
    # A step is only re-published by the reaper once it has been runnable for this
    # long, so the reaper does not race the enqueue that just happened inline.
    enqueue_grace_seconds: float = 5.0
    reaper_batch_size: int = 100

    # --- worker ---------------------------------------------------------------
    worker_concurrency: int = 4
    worker_poll_block_ms: int = 2_000
    shutdown_grace_seconds: float = 20.0

    # --- run defaults (snapshotted onto each run at creation) ------------------
    default_max_steps: int = 20
    default_max_cost_usd: float = 1.0
    default_timeout_seconds: int = 900
    default_max_attempts: int = 3
    default_max_recoveries: int = 2

    # --- retry backoff --------------------------------------------------------
    retry_initial_backoff_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    retry_max_backoff_seconds: float = 60.0

    # --- guardrails -----------------------------------------------------------
    # How long a pending approval waits before the run is failed. Without an expiry
    # a forgotten approval pins a run open forever and its budget with it.
    approval_ttl_seconds: int = 86_400
    # A model that keeps emitting unusable tool calls is looping, not recovering.
    max_consecutive_invalid_tool_calls: int = 3
    # When a run is one step from its cap, the final turn is made with tools removed
    # so it produces an answer instead of another tool call it cannot execute.
    reserve_final_answer_step: bool = True

    # --- LLM ------------------------------------------------------------------
    # "fake" is a scripted, deterministic provider. It is the default so that a
    # misconfigured deployment cannot silently start spending money, and so tests and
    # the eval harness are reproducible.
    llm_provider: Literal["fake", "anthropic"] = "fake"
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 8_000
    # low | medium | high | xhigh | max. Steers thinking depth and token spend.
    llm_effort: str = "high"
    llm_timeout_seconds: float = 120.0
    # Server-side refusal fallbacks: on a policy decline the API retries on another
    # model within the same call, which for an autonomous agent is the difference
    # between a hiccup and a dead run.
    llm_enable_fallbacks: bool = True

    # --- safety ---------------------------------------------------------------
    # Fault injection lets a step deliberately crash its own worker. It is a test
    # affordance and is refused unless explicitly enabled.
    enable_fault_injection: bool = False

    # --- observability --------------------------------------------------------
    # Empty disables export. The tracer still runs and still produces trace ids, so
    # traceparent propagation and the trace_id on every transition keep working with
    # no collector present.
    otlp_endpoint: str = ""
    metrics_enabled: bool = True
    #: Port each worker/reaper exposes /metrics on. 0 disables.
    worker_metrics_port: int = 0
    gauge_refresh_seconds: float = 5.0

    log_level: str = "INFO"
    log_json: bool = True
    service_name: str = "agent-orchestrator"

    @property
    def heartbeat_interval_seconds(self) -> float:
        return self.lease_ttl_seconds / self.heartbeat_divisor

    @property
    def sync_database_url(self) -> str:
        """Alembic runs synchronously; strip the async driver marker."""
        return self.database_url.replace("+psycopg_async", "+psycopg")

    @model_validator(mode="after")
    def validate_timing(self) -> Settings:
        if self.heartbeat_divisor < 2:
            raise ValueError("heartbeat_divisor must be >= 2 to tolerate a missed heartbeat")
        if self.stalled_message_idle_ms <= self.lease_ttl_seconds * 1000:
            raise ValueError(
                "stalled_message_idle_ms must exceed lease_ttl_seconds, otherwise a "
                "worker holding a valid lease has its queue message reclaimed under it"
            )
        if self.enqueue_grace_seconds <= self.reaper_interval_seconds:
            raise ValueError("enqueue_grace_seconds must exceed reaper_interval_seconds")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()


def reload_settings() -> Settings:
    """Test hook: re-read the environment."""
    global settings
    get_settings.cache_clear()
    settings = get_settings()
    return settings


__all__ = ["Settings", "get_settings", "reload_settings", "settings"]
