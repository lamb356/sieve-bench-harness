from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from bench.contamination.normalize import normalize_for_search
from bench.loaders.base import CodeDocument
from bench.metrics.performance import summarize_latency
from bench.retrievers.base import SearchResult

_IDENTIFIER_BOUNDARY_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_IDENTIFIER_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_ALPHA_NUM_BOUNDARY = re.compile(r"([A-Za-z])([0-9])|([0-9])([A-Za-z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self._doc_lengths = np.array([], dtype=np.float32)
        self._length_norm = np.array([], dtype=np.float32)
        self._avgdl = 0.0
        self._k1 = 1.5
        self._b = 0.75
        self._epsilon = 0.25
        self._latency_samples_ms: list[float] = []

    def index(self, corpus: Sequence[CodeDocument]) -> None:
        self._documents = tuple(corpus)
        if not self._documents:
            raise ValueError("BM25Retriever requires a non-empty corpus")

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequencies: dict[str, int] = defaultdict(int)
        doc_lengths: list[int] = []
        for document_index, document in enumerate(self._documents):
            token_counts = Counter(tokenize_text(_document_text(document)))
            doc_lengths.append(sum(token_counts.values()))
            for term, frequency in token_counts.items():
                document_frequencies[term] += 1
                postings[term].append((document_index, frequency))

        self._doc_lengths = np.asarray(doc_lengths, dtype=np.float32)
        self._avgdl = float(self._doc_lengths.mean()) if len(self._doc_lengths) else 0.0
        if self._avgdl > 0.0:
            self._length_norm = self._k1 * (1.0 - self._b + self._b * self._doc_lengths / self._avgdl)
        else:
            self._length_norm = np.zeros(len(self._documents), dtype=np.float32)

        corpus_size = len(self._documents)
        raw_idf: dict[str, float] = {}
        negative_terms: list[str] = []
        idf_sum = 0.0
        for term, containing_documents in document_frequencies.items():
            # Match rank_bm25.BM25Okapi's ATIRE-style idf exactly, including its
            # epsilon floor for terms that appear in more than half the corpus.
            idf = float(np.log(corpus_size - containing_documents + 0.5) - np.log(containing_documents + 0.5))
            raw_idf[term] = idf
            idf_sum += idf
            if idf < 0.0:
                negative_terms.append(term)
        average_idf = idf_sum / len(raw_idf) if raw_idf else 0.0
        epsilon_idf = self._epsilon * average_idf
        for term in negative_terms:
            raw_idf[term] = epsilon_idf
        self._idf = raw_idf
        self._postings = {
            term: (
                np.asarray([document_index for document_index, _frequency in entries], dtype=np.int32),
                np.asarray([frequency for _document_index, frequency in entries], dtype=np.float32),
            )
            for term, entries in postings.items()
        }
        self._latency_samples_ms.clear()

    def search(self, query: str, k: int) -> list[SearchResult]:
        if not self._documents:
            raise RuntimeError("BM25Retriever.search() called before index()")
        if k <= 0:
            return []
        query_tokens = tokenize_text(query)
        if not query_tokens:
            self._latency_samples_ms.append(0.0)
            return []

        started = time.perf_counter()
        scores = np.zeros(len(self._documents), dtype=np.float32)
        if self._avgdl > 0.0:
            for query_token in query_tokens:
                postings = self._postings.get(query_token)
                if postings is None:
                    continue
                document_indices, term_frequencies = postings
                denominator = term_frequencies + self._length_norm[document_indices]
                scores[document_indices] += self._idf.get(query_token, 0.0) * (term_frequencies * (self._k1 + 1.0) / denominator)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)

        ranked = sorted(
            ((int(index), float(scores[index])) for index in np.flatnonzero(scores > 0.0)),
            key=lambda item: (-item[1], self._documents[item[0]].document_id),
        )[:k]
        return [
            SearchResult(
                document_id=self._documents[index].document_id,
                path=self._documents[index].path,
                score=score,
                code=self._documents[index].code,
                metadata={"token_count": int(self._doc_lengths[index])},
            )
            for index, score in ranked
        ]

    def latency_ms(self) -> dict[str, float]:
        if not self._latency_samples_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return summarize_latency(self._latency_samples_ms)
