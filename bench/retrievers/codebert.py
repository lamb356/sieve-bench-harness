from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bench.constants import BENCH_MODEL_CACHE_DIR, CODEBERT_MODEL_ID, CODEBERT_MODEL_REVISION
from bench.contamination.normalize import normalize_for_search
from bench.loaders.base import CodeDocument
from bench.metrics.performance import summarize_latency
from bench.retrievers.base import SearchResult

HEAD_TAIL_TRUNCATION_STRATEGY = "documents=head+tail-balanced-to-512;queries=head-only-to-512;mean-pool-attention-mask"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _document_text(document: CodeDocument) -> str:
    if document.index_text:
        return document.index_text
    return normalize_for_search(document.code, language=document.language)


class TransformerEmbeddingBackend:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.model_revision = model_revision
        self.cache_dir = Path(cache_dir)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.truncation_strategy = HEAD_TAIL_TRUNCATION_STRATEGY
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=str(self.cache_dir),
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=str(self.cache_dir),
        ).to(self.device)
        self.model.eval()

    def _token_ids(self, text: str, *, for_document: bool) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        special_budget = self.tokenizer.num_special_tokens_to_add(pair=False)
        token_budget = max(1, self.max_length - special_budget)
        if len(token_ids) <= token_budget:
            return token_ids
        if not for_document:
            return token_ids[:token_budget]
        head = token_budget // 2
        tail = token_budget - head
        return token_ids[:head] + token_ids[-tail:]

    def _encode(self, texts: Sequence[str], *, for_document: bool) -> np.ndarray:
        vectors: list[np.ndarray] = []
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded_items = [
                    self.tokenizer.prepare_for_model(
                        self._token_ids(text, for_document=for_document),
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=False,
                    )
                    for text in batch_texts
                ]
                encoded = self.tokenizer.pad(
                    encoded_items,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(vectors, axis=0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, for_document=True)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, for_document=False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_documents(texts)


class TransformerEmbeddingRetriever:
    name = "transformer"
    display_name = "Transformer"

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        embedding_backend: Any | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_cache_dir = Path(model_cache_dir)
        self.batch_size = int(batch_size)
        self.embedding_backend = embedding_backend or TransformerEmbeddingBackend(
            model_id=model_id,
            model_revision=model_revision,
            cache_dir=self.model_cache_dir,
            batch_size=batch_size,
        )
        self._documents: tuple[CodeDocument, ...] = ()
        self._document_embeddings: np.ndarray | None = None
        self._latency_samples_ms: list[float] = []

    def embedding_metadata(self) -> dict[str, Any]:
        return {
            "model_id": str(getattr(self.embedding_backend, "model_id", self.model_id)),
            "model_revision": str(getattr(self.embedding_backend, "model_revision", self.model_revision)),
            "max_length": int(getattr(self.embedding_backend, "max_length", 512)),
            "truncation_strategy": str(getattr(self.embedding_backend, "truncation_strategy", HEAD_TAIL_TRUNCATION_STRATEGY)),
            "cache_dir": str(self.model_cache_dir),
            "device": str(getattr(self.embedding_backend, "device", "unknown")),
        }

    def _encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.embedding_backend, "encode_documents"):
            return np.asarray(self.embedding_backend.encode_documents(texts), dtype=np.float32)
        return np.asarray(self.embedding_backend.encode(texts), dtype=np.float32)

    def _encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.embedding_backend, "encode_queries"):
            return np.asarray(self.embedding_backend.encode_queries(texts), dtype=np.float32)
        return np.asarray(self.embedding_backend.encode(texts), dtype=np.float32)

    def index(self, corpus: Sequence[CodeDocument]) -> None:
        self._documents = tuple(corpus)
        if not self._documents:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty corpus")
        texts = [_document_text(document) for document in self._documents]
        embeddings = self._encode_documents(texts)
        if embeddings.shape[0] != len(self._documents):
            raise ValueError(f"Embedding backend returned {embeddings.shape[0]} vectors for {len(self._documents)} documents")
        self._document_embeddings = _l2_normalize(embeddings)
        self._latency_samples_ms.clear()

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._document_embeddings is None:
            raise RuntimeError(f"{self.__class__.__name__}.search() called before index()")
        if k <= 0:
            return []
        started = time.perf_counter()
        query_embedding = _l2_normalize(self._encode_queries([query]))[0]
        scores = self._document_embeddings @ query_embedding
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)

        top_n = min(k, len(self._documents))
        if top_n == 0:
            return []
        candidate_indices = np.argpartition(-scores, kth=top_n - 1)[:top_n]
        ranked_indices = sorted(candidate_indices.tolist(), key=lambda index: (-float(scores[index]), self._documents[index].document_id))
        metadata = self.embedding_metadata()
        return [
            SearchResult(
                document_id=self._documents[index].document_id,
                path=self._documents[index].path,
                score=float(scores[index]),
                code=self._documents[index].code,
                metadata={**metadata, "ranker": self.name},
            )
            for index in ranked_indices
        ]

    def latency_ms(self) -> dict[str, float]:
        if not self._latency_samples_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return summarize_latency(self._latency_samples_ms)


class CodeBERTRetriever(TransformerEmbeddingRetriever):
    name = "codebert"
    display_name = "CodeBERT"

    def __init__(
        self,
        *,
        model_cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        embedding_backend: Any | None = None,
        batch_size: int = 16,
    ) -> None:
        super().__init__(
            model_id=CODEBERT_MODEL_ID,
            model_revision=CODEBERT_MODEL_REVISION,
            model_cache_dir=model_cache_dir,
            embedding_backend=embedding_backend,
            batch_size=batch_size,
        )
