from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    path: str
    score: float
    code: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    name: str

    def index(self, corpus: list[Any] | tuple[Any, ...]) -> None:
        """Build the retrieval index. Called once per corpus."""

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return top-k results for a query."""

    def latency_ms(self) -> dict[str, float]:
        """Return p50/p95/p99 latency stats from the last search batch."""
