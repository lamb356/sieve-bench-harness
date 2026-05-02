from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bench.constants import BENCH_MODEL_CACHE_DIR, BGE_SMALL_MODEL_ID, BGE_SMALL_MODEL_REVISION
from bench.retrievers.codebert import TransformerEmbeddingBackend, TransformerEmbeddingRetriever

BGE_SMALL_TRUNCATION_STRATEGY = (
    "documents=head+tail-balanced-to-512;queries=head-only-to-512;"
    "cls-pool-first-token;normalize-l2;query_instruction=none"
)


class BgeSmallEmbeddingBackend(TransformerEmbeddingBackend):
    """BAAI/bge-small-en-v1.5 backend using the model's CLS pooling contract."""

    def __init__(
        self,
        *,
        model_id: str = BGE_SMALL_MODEL_ID,
        model_revision: str = BGE_SMALL_MODEL_REVISION,
        cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_revision=model_revision,
            cache_dir=cache_dir,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        self.truncation_strategy = BGE_SMALL_TRUNCATION_STRATEGY

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
                pooled = outputs.last_hidden_state[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


class BgeSmallRetriever(TransformerEmbeddingRetriever):
    name = "bge-small"
    display_name = "bge-small-en-v1.5"

    def __init__(
        self,
        *,
        model_cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        embedding_backend: Any | None = None,
        batch_size: int = 16,
    ) -> None:
        model_cache_dir = Path(model_cache_dir)
        super().__init__(
            model_id=BGE_SMALL_MODEL_ID,
            model_revision=BGE_SMALL_MODEL_REVISION,
            model_cache_dir=model_cache_dir,
            embedding_backend=embedding_backend
            or BgeSmallEmbeddingBackend(
                model_id=BGE_SMALL_MODEL_ID,
                model_revision=BGE_SMALL_MODEL_REVISION,
                cache_dir=model_cache_dir,
                batch_size=batch_size,
            ),
            batch_size=batch_size,
        )
