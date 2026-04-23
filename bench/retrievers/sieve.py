from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Sequence

from bench.constants import GLOBAL_RANDOM_SEED
from bench.loaders.base import CodeDocument
from bench.metrics.performance import summarize_latency
from bench.retrievers.base import SearchResult


class SieveStubRetriever:
    name = "sieve-stub"
    display_name = "SIEVE (stub)"

    def __init__(self, *, seed: int = GLOBAL_RANDOM_SEED) -> None:
        self.seed = int(seed)
        self._documents: tuple[CodeDocument, ...] = ()
        self._latency_samples_ms: list[float] = []

    def index(self, corpus: Sequence[CodeDocument]) -> None:
        self._documents = tuple(corpus)
        if not self._documents:
            raise ValueError("SieveStubRetriever requires a non-empty corpus")
        self._latency_samples_ms.clear()

    def _rng_for_query(self, query: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{query}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:16], byteorder="big", signed=False))

    def search(self, query: str, k: int) -> list[SearchResult]:
        if not self._documents:
            raise RuntimeError("SieveStubRetriever.search() called before index()")
        if k <= 0:
            return []
        started = time.perf_counter()
        indices = list(range(len(self._documents)))
        self._rng_for_query(query).shuffle(indices)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)
        return [
            SearchResult(
                document_id=self._documents[index].document_id,
                path=self._documents[index].path,
                score=1.0 / float(rank + 1),
                code=self._documents[index].code,
                metadata={"stub": "deterministic-query-hash-random", "seed": self.seed},
            )
            for rank, index in enumerate(indices[:k])
        ]

    def latency_ms(self) -> dict[str, float]:
        if not self._latency_samples_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return summarize_latency(self._latency_samples_ms)
