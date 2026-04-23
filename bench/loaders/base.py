from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class EvalExample:
    query: str
    ground_truth_code: str
    ground_truth_path: str
    language: str
    source: str
    corpus_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeDocument:
    document_id: str
    path: str
    code: str
    language: str
    index_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedBenchmark:
    source: str
    language: str
    revision: str
    corpus_id: str
    corpus: tuple[CodeDocument, ...]
    examples: tuple[EvalExample, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class Loader(Protocol):
    name: str

    def load(self, sample_size: int | None = None) -> LoadedBenchmark:
        """Load a retrieval corpus plus eval examples for one source/language."""


def ensure_non_empty(items: Iterable[Any], *, label: str) -> tuple[Any, ...]:
    values = tuple(items)
    if not values:
        raise ValueError(f"Expected non-empty iterable for {label}")
    return values
