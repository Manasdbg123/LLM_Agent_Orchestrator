"""Read-only search.

The backend is pluggable and defaults to a **fixture corpus** rather than a live
search API. That is a deliberate choice for the eval harness: a benchmark whose
inputs change under you measures the weather, not the system. With a fixed corpus,
"success rate dropped from 93% to 78%" means the engine changed.

A live backend is a drop-in implementation of `SearchBackend`; nothing above this
line knows the difference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import Field

from app.tools.base import BaseTool, EffectPolicy, ToolArgs, ToolContext, ToolResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_corpus.json"


@dataclass(frozen=True, slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str

    def render(self) -> str:
        return f"{self.title}\n{self.url}\n{self.snippet}"


class SearchBackend(Protocol):
    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class FixtureBackend:
    """Deterministic corpus lookup.

    Matching is bag-of-words overlap rather than exact string equality, so an agent
    that phrases a query slightly differently still finds the document — which keeps
    the eval measuring the orchestration rather than the agent's phrasing luck.
    """

    def __init__(self, corpus: list[dict[str, Any]] | None = None) -> None:
        if corpus is None:
            corpus = (
                json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
                if FIXTURE_PATH.exists()
                else []
            )
        self.documents = corpus

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t}

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        query_tokens = self._tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.documents:
            searchable = " ".join(
                [doc.get("title", ""), doc.get("snippet", ""), *doc.get("keywords", [])]
            )
            doc_tokens = self._tokens(searchable)
            overlap = len(query_tokens & doc_tokens)
            if overlap:
                # Normalised so a long document does not outrank a precise one.
                scored.append((overlap / max(len(query_tokens), 1), doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1].get("title", "")))
        return [
            SearchHit(
                title=doc.get("title", ""), url=doc.get("url", ""), snippet=doc.get("snippet", "")
            )
            for _score, doc in scored[:max_results]
        ]


class WebSearchArgs(ToolArgs):
    query: str = Field(..., min_length=1, max_length=300, description="Search query")
    max_results: int = Field(
        default=3, ge=1, le=10, description="How many results to return (1-10)"
    )


class WebSearchTool(BaseTool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search for factual information. Returns titles, URLs and snippets. "
        "Use it when the answer depends on information you do not already have."
    )
    args_model: ClassVar[type[ToolArgs]] = WebSearchArgs
    # Read-only: replaying a search cannot change the world.
    effect_policy: ClassVar[EffectPolicy] = EffectPolicy.SAFE_TO_REPLAY
    requires_approval: ClassVar[bool] = False
    timeout_seconds: ClassVar[float] = 15.0

    def __init__(self, backend: SearchBackend | None = None) -> None:
        self.backend = backend or FixtureBackend()

    async def execute(self, ctx: ToolContext, args: WebSearchArgs) -> ToolResult:
        hits = await self.backend.search(args.query, args.max_results)
        if not hits:
            # An empty result is information, not a failure: the model should be able
            # to say "I could not find that" rather than retry a doomed search.
            return ToolResult(
                content=f"No results found for {args.query!r}.",
                data={"query": args.query, "result_count": 0},
            )
        return ToolResult(
            content="\n\n".join(hit.render() for hit in hits),
            data={
                "query": args.query,
                "result_count": len(hits),
                "urls": [hit.url for hit in hits],
            },
        )


__all__ = ["FixtureBackend", "SearchBackend", "SearchHit", "WebSearchArgs", "WebSearchTool"]
