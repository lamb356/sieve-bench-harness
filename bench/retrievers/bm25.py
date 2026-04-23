from __future__ import annotations

import re
import time
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from bench.contamination.normalize import normalize_for_search
from bench.loaders.base import CodeDocument
from bench.metrics.performance import summarize_latency
from bench.retrievers.base import SearchResult

_IDENTIFIER_BOUNDARY_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_IDENTIFIER_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_ALPHA_NUM_BOUNDARY = re.compile(r"([A-Za-z])([0-9])|([0-9])([A-Za-z])")
_TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+")


def tokenize_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    spaced = text.replace("_", " ").replace("-", " ")
    spaced = _IDENTIFIER_BOUNDARY_1.sub(r"\1 \2", spaced)
    spaced = _IDENTIFIER_BOUNDARY_2.sub(r"\1 \2", spaced)
    spaced = _ALPHA_NUM_BOUNDARY.sub(lambda match: " ".join(part for part in match.groups() if part), spaced)
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(spaced)]


def _document_text(document: CodeDocument) -> str:
    if document.index_text:
        return document.index_text
    return normalize_for_search(document.code, language=document.language)


class BM25Retriever:
    name = "bm25"
    display_name = "BM25"

    def __init__(self) -> None:
        self._documents: tuple[CodeDocument, ...] = ()
        self._tokenized_corpus: list[list[str]] = []
        self._index: BM25Okapi | None = None
        self._latency_samples_ms: list[float] = []

    def index(self, corpus: Sequence[CodeDocument]) -> None:
        self._documents = tuple(corpus)
        if not self._documents:
            raise ValueError("BM25Retriever requires a non-empty corpus")
        self._tokenized_corpus = [tokenize_text(_document_text(document)) for document in self._documents]
        self._index = BM25Okapi(self._tokenized_corpus)
        self._latency_samples_ms.clear()

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("BM25Retriever.search() called before index()")
        if k <= 0:
            return []
        query_tokens = tokenize_text(query)
        if not query_tokens:
            self._latency_samples_ms.append(0.0)
            return []

        started = time.perf_counter()
        scores = self._index.get_scores(query_tokens)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)

        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores) if float(score) > 0.0),
            key=lambda item: (-item[1], self._documents[item[0]].document_id),
        )[:k]
        return [
            SearchResult(
                document_id=self._documents[index].document_id,
                path=self._documents[index].path,
                score=score,
                code=self._documents[index].code,
                metadata={"token_count": len(self._tokenized_corpus[index])},
            )
            for index, score in ranked
        ]

    def latency_ms(self) -> dict[str, float]:
        if not self._latency_samples_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return summarize_latency(self._latency_samples_ms)
