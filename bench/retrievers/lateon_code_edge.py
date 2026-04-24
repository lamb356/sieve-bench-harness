from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.constants import BENCH_MODEL_CACHE_DIR, LATEON_CODE_EDGE_MODEL_ID, LATEON_CODE_EDGE_MODEL_REVISION
from bench.retrievers._pylate_base import PyLateBruteForceMaxSimRetriever


class LateOnCodeEdgeRetriever(PyLateBruteForceMaxSimRetriever):
    name = "lateon-code-edge"
    display_name = "LateOn-Code-edge"

    def __init__(
        self,
        *,
        model_cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        embedding_backend: Any | None = None,
        batch_size: int = 16,
        score_chunk_size: int = 256,
    ) -> None:
        super().__init__(
            model_id=LATEON_CODE_EDGE_MODEL_ID,
            model_revision=LATEON_CODE_EDGE_MODEL_REVISION,
            model_cache_dir=model_cache_dir,
            embedding_backend=embedding_backend,
            batch_size=batch_size,
            score_chunk_size=score_chunk_size,
        )
