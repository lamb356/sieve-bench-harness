from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.constants import BENCH_MODEL_CACHE_DIR, UNIXCODER_MODEL_ID, UNIXCODER_MODEL_REVISION
from bench.retrievers.codebert import TransformerEmbeddingRetriever


class UniXcoderRetriever(TransformerEmbeddingRetriever):
    name = "unixcoder"
    display_name = "UniXcoder"

    def __init__(
        self,
        *,
        model_cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        embedding_backend: Any | None = None,
        batch_size: int = 16,
    ) -> None:
        super().__init__(
            model_id=UNIXCODER_MODEL_ID,
            model_revision=UNIXCODER_MODEL_REVISION,
            model_cache_dir=model_cache_dir,
            embedding_backend=embedding_backend,
            batch_size=batch_size,
        )
