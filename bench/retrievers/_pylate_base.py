from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bench.constants import BENCH_MODEL_CACHE_DIR
from bench.loaders.base import CodeDocument
from bench.metrics.performance import summarize_latency
from bench.retrievers.base import SearchResult

PYLATE_BRUTE_FORCE_TRUNCATION_STRATEGY = (
    "pylate-colbert;queries=model-query-length;documents=model-document-length;"
    "score=bruteforce-maxsim;no-plaid-index"
)


def _document_text(document: CodeDocument) -> str:
    return document.code


class PyLateEmbeddingBackend:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        import torch
        from pylate import models

        self.model_id = model_id
        self.model_revision = model_revision
        self.cache_dir = Path(cache_dir)
        self.batch_size = int(batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.model = models.ColBERT(
            model_name_or_path=model_id,
            revision=model_revision,
            cache_folder=str(self.cache_dir),
            device=self.device,
        )
        self.model.eval()
        self.query_length = int(getattr(self.model, "query_length", 32))
        self.document_length = int(getattr(self.model, "document_length", 180))
        self.truncation_strategy = PYLATE_BRUTE_FORCE_TRUNCATION_STRATEGY

    def _as_tensor_list(self, embeddings: Any) -> list[Any]:
        torch = self._torch
        if isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 2:
                return [embeddings.detach()]
            if embeddings.ndim == 3:
                return [row.detach() for row in embeddings]
        return [torch.as_tensor(embedding).detach() for embedding in embeddings]

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> list[Any]:
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            is_query=is_query,
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=False,
            normalize_embeddings=True,
            padding=False,
        )
        return self._as_tensor_list(embeddings)

    def encode_documents(self, texts: Sequence[str]) -> list[Any]:
        return self._encode(texts, is_query=False)

    def encode_queries(self, texts: Sequence[str]) -> list[Any]:
        return self._encode(texts, is_query=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "cache_dir": str(self.cache_dir),
            "device": self.device,
            "truncation_strategy": self.truncation_strategy,
            "max_tokens": {"query": self.query_length, "document": self.document_length},
        }


class PyLateBruteForceMaxSimRetriever:
    name = "pylate-bruteforce"
    display_name = "PyLate brute-force MaxSim"

    def __init__(
        self,
        *,
        name: str | None = None,
        display_name: str | None = None,
        model_id: str,
        model_revision: str,
        model_cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        embedding_backend: Any | None = None,
        batch_size: int = 16,
        score_chunk_size: int = 256,
    ) -> None:
        if name is not None:
            self.name = name
        if display_name is not None:
            self.display_name = display_name
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_cache_dir = Path(model_cache_dir)
        self.batch_size = int(batch_size)
        self.score_chunk_size = int(score_chunk_size)
        self.embedding_backend = embedding_backend or PyLateEmbeddingBackend(
            model_id=model_id,
            model_revision=model_revision,
            cache_dir=self.model_cache_dir,
            batch_size=batch_size,
        )
        self._documents: tuple[CodeDocument, ...] = ()
        self._document_embeddings: tuple[Any, ...] = ()
        self._latency_samples_ms: list[float] = []

    def embedding_metadata(self) -> dict[str, Any]:
        if hasattr(self.embedding_backend, "metadata"):
            metadata = dict(self.embedding_backend.metadata())
        else:
            metadata = {
                "model_id": str(getattr(self.embedding_backend, "model_id", self.model_id)),
                "model_revision": str(getattr(self.embedding_backend, "model_revision", self.model_revision)),
                "device": str(getattr(self.embedding_backend, "device", "unknown")),
                "truncation_strategy": str(
                    getattr(self.embedding_backend, "truncation_strategy", PYLATE_BRUTE_FORCE_TRUNCATION_STRATEGY)
                ),
                "max_tokens": {
                    "query": int(getattr(self.embedding_backend, "query_length", 32)),
                    "document": int(getattr(self.embedding_backend, "document_length", 180)),
                },
            }
        metadata.setdefault("model_id", str(getattr(self.embedding_backend, "model_id", self.model_id)))
        metadata.setdefault("model_revision", str(getattr(self.embedding_backend, "model_revision", self.model_revision)))
        metadata.setdefault("cache_dir", str(self.model_cache_dir))
        metadata.setdefault("device", str(getattr(self.embedding_backend, "device", "unknown")))
        metadata.setdefault("truncation_strategy", PYLATE_BRUTE_FORCE_TRUNCATION_STRATEGY)
        metadata.setdefault(
            "max_tokens",
            {
                "query": int(getattr(self.embedding_backend, "query_length", 32)),
                "document": int(getattr(self.embedding_backend, "document_length", 180)),
            },
        )
        metadata["scoring"] = "bruteforce-maxsim"
        metadata["plaid_index"] = False
        return metadata

    def _to_token_tensor(self, embedding: Any) -> Any:
        import torch

        tensor = embedding if isinstance(embedding, torch.Tensor) else torch.as_tensor(embedding)
        tensor = tensor.detach().to(dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D token embedding matrix, got shape={tuple(tensor.shape)}")
        if tensor.shape[0] == 0:
            raise ValueError("PyLate token embedding matrix must contain at least one token")
        return tensor

    def _encode_documents(self, texts: Sequence[str]) -> list[Any]:
        return [self._to_token_tensor(embedding).cpu() for embedding in self.embedding_backend.encode_documents(texts)]

    def _encode_query(self, query: str) -> Any:
        embeddings = self.embedding_backend.encode_queries([query])
        if len(embeddings) != 1:
            raise ValueError(f"PyLate backend returned {len(embeddings)} query embeddings for one query")
        return self._to_token_tensor(embeddings[0])

    def index(self, corpus: Sequence[CodeDocument]) -> None:
        self._documents = tuple(corpus)
        if not self._documents:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty corpus")
        texts = [_document_text(document) for document in self._documents]
        embeddings = self._encode_documents(texts)
        if len(embeddings) != len(self._documents):
            raise ValueError(f"PyLate backend returned {len(embeddings)} vectors for {len(self._documents)} documents")
        self._document_embeddings = tuple(embeddings)
        self._latency_samples_ms.clear()

    def _score_query(self, query_embedding: Any) -> list[float]:
        import torch

        if not self._document_embeddings:
            return []
        score_device = query_embedding.device
        if score_device.type == "cpu" and str(getattr(self.embedding_backend, "device", "cpu")).startswith("cuda") and torch.cuda.is_available():
            score_device = torch.device(str(getattr(self.embedding_backend, "device")))
            query_embedding = query_embedding.to(score_device)

        scores: list[float] = []
        for start in range(0, len(self._document_embeddings), self.score_chunk_size):
            chunk = [embedding.to(score_device) for embedding in self._document_embeddings[start : start + self.score_chunk_size]]
            lengths = torch.tensor([embedding.shape[0] for embedding in chunk], device=score_device)
            padded = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True, padding_value=0.0)
            token_positions = torch.arange(padded.shape[1], device=score_device).unsqueeze(0)
            document_mask = token_positions < lengths.unsqueeze(1)
            similarities = torch.einsum("qh,dth->dqt", query_embedding, padded)
            similarities = similarities.masked_fill(~document_mask.unsqueeze(1), torch.finfo(similarities.dtype).min)
            chunk_scores = similarities.max(dim=-1).values.sum(dim=-1)
            scores.extend(float(score) for score in chunk_scores.detach().cpu().tolist())
        return scores

    def search(self, query: str, k: int) -> list[SearchResult]:
        if not self._document_embeddings:
            raise RuntimeError(f"{self.__class__.__name__}.search() called before index()")
        if k <= 0:
            return []
        started = time.perf_counter()
        query_embedding = self._encode_query(query)
        scores = self._score_query(query_embedding)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)

        top_n = min(k, len(self._documents))
        ranked_indices = sorted(range(len(self._documents)), key=lambda index: (-float(scores[index]), self._documents[index].document_id))[:top_n]
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
