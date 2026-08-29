"""Provider selection.

Defaults to the fake provider. That is a safety default, not a convenience one: an
engine that silently reached for a paid API because a config value was missing would
be an expensive way to discover a typo.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.llm.base import LLMProvider


@lru_cache(maxsize=1)
def default_provider() -> LLMProvider:
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            timeout_seconds=settings.llm_timeout_seconds,
            enable_fallbacks=settings.llm_enable_fallbacks,
        )

    from app.llm.fake import FakeProvider

    return FakeProvider()


def reset_provider_cache() -> None:
    """Test hook: pick up a changed `llm_provider` setting."""
    default_provider.cache_clear()


__all__ = ["default_provider", "reset_provider_cache"]
