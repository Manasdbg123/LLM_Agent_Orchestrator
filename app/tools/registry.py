"""Tool registry.

Instances are built once per process and reused, because tools hold backends
(a search corpus, an email provider) that are expensive or stateful to construct.
"""

from __future__ import annotations

from functools import lru_cache

from app.domain.errors import ErrorCode, TerminalError
from app.llm.base import ToolSchema
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.database_write import DatabaseWriteTool
from app.tools.send_email import SendEmailTool
from app.tools.web_search import WebSearchTool

_TOOL_CLASSES: tuple[type[BaseTool], ...] = (
    CalculatorTool,
    WebSearchTool,
    DatabaseWriteTool,
    SendEmailTool,
)


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        instances = tools if tools is not None else [cls() for cls in _TOOL_CLASSES]
        self._tools: dict[str, BaseTool] = {t.name: t for t in instances}

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            # Terminal, not retryable: an unknown tool will still be unknown next
            # time. The agent loop turns this into an errored tool_result so the
            # model can pick a real tool instead.
            raise TerminalError(
                f"unknown tool {name!r}; available: {', '.join(self.names)}",
                code=ErrorCode.UNKNOWN_TOOL,
            ) from None

    def schemas(self, names: list[str] | None = None) -> list[ToolSchema]:
        """Schemas for the model, in a stable order.

        Order matters more than it looks: `tools` is rendered before `system` and
        `messages` in the cached prefix, so a set that reorders between calls
        invalidates the prompt cache on every request.
        """
        selected = self.names if not names else [n for n in names if n in self._tools]
        return [self._tools[n].schema() for n in sorted(selected)]


@lru_cache(maxsize=1)
def default_registry() -> ToolRegistry:
    return ToolRegistry()


__all__ = ["ToolRegistry", "default_registry"]
