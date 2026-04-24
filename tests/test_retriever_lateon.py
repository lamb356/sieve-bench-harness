from __future__ import annotations

from collections.abc import Sequence

import torch

from bench.constants import (
    LATEON_CODE_EDGE_MODEL_ID,
    LATEON_CODE_EDGE_MODEL_REVISION,
    LATEON_CODE_MODEL_ID,
    LATEON_CODE_MODEL_REVISION,
)
from bench.loaders.base import CodeDocument
from bench.retrievers._pylate_base import PyLateBruteForceMaxSimRetriever
from bench.retrievers.lateon_code import LateOnCodeRetriever
from bench.retrievers.lateon_code_edge import LateOnCodeEdgeRetriever


class FakePyLateBackend:
    model_id = "test/fake-pylate"
    model_revision = "fake-revision"
    device = "cpu"
    query_length = 32
    document_length = 180
    truncation_strategy = "fake-pylate-maxsim"

    def __init__(self) -> None:
        self.document_texts: list[str] = []
        self.query_texts: list[str] = []

    def _embedding_for_text(self, text: str) -> torch.Tensor:
        lowered = text.lower()
        if "parse" in lowered and "http" in lowered:
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        if "json" in lowered:
            return torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        if "gradient" in lowered:
            return torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
        return torch.tensor([[0.1, 0.1]], dtype=torch.float32)

    def encode_documents(self, texts: Sequence[str]) -> list[torch.Tensor]:
        self.document_texts.extend(texts)
        return [self._embedding_for_text(text) for text in texts]

    def encode_queries(self, texts: Sequence[str]) -> list[torch.Tensor]:
        self.query_texts.extend(texts)
        return [self._embedding_for_text(text) for text in texts]

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "device": self.device,
            "truncation_strategy": self.truncation_strategy,
            "max_tokens": {"query": self.query_length, "document": self.document_length},
        }


def _toy_corpus() -> list[CodeDocument]:
    return [
        CodeDocument(
            document_id="doc-json",
            path="python/json.py",
            code="def parse_http_request(raw):\n    return raw.headers  # raw code should not be indexed here\n",
            language="python",
            index_text="write json file serialize data path return dumps",
        ),
        CodeDocument(
            document_id="doc-http",
            path="python/http.py",
            code="def unrelated_raw_code():\n    return 'raw code is ignored when index_text exists'\n",
            language="python",
            index_text="parse http request raw request headers return",
        ),
        CodeDocument(
            document_id="doc-gradient",
            path="python/gradient.py",
            code="def gradient_norm(tensor):\n    return tensor.grad.norm()\n",
            language="python",
            index_text="gradient norm tensor grad return norm",
        ),
    ]


def test_pylate_bruteforce_retriever_ranks_by_maxsim_over_normalized_index_text(tmp_path) -> None:
    backend = FakePyLateBackend()
    retriever = PyLateBruteForceMaxSimRetriever(
        name="fake-lateon",
        display_name="Fake LateOn",
        model_id="test/fake-pylate",
        model_revision="fake-revision",
        model_cache_dir=tmp_path / "models",
        embedding_backend=backend,
        score_chunk_size=2,
    )

    retriever.index(_toy_corpus())
    results = retriever.search("parse http request headers", k=2)

    assert backend.document_texts == [
        "write json file serialize data path return dumps",
        "parse http request raw request headers return",
        "gradient norm tensor grad return norm",
    ]
    assert backend.query_texts == ["parse http request headers"]
    assert [result.document_id for result in results] == ["doc-http", "doc-gradient"]
    assert results[0].metadata["model_id"] == "test/fake-pylate"
    assert results[0].metadata["model_revision"] == "fake-revision"
    assert results[0].metadata["truncation_strategy"] == "fake-pylate-maxsim"
    assert results[0].metadata["max_tokens"] == {"query": 32, "document": 180}
    assert results[0].metadata["scoring"] == "bruteforce-maxsim"
    assert results[0].metadata["plaid_index"] is False
    assert retriever.latency_ms()["p50"] >= 0.0


def test_lateon_thin_retrievers_expose_pinned_model_metadata(tmp_path) -> None:
    edge = LateOnCodeEdgeRetriever(model_cache_dir=tmp_path / "models", embedding_backend=FakePyLateBackend())
    full = LateOnCodeRetriever(model_cache_dir=tmp_path / "models", embedding_backend=FakePyLateBackend())

    assert edge.name == "lateon-code-edge"
    assert edge.display_name == "LateOn-Code-edge"
    assert edge.model_id == LATEON_CODE_EDGE_MODEL_ID
    assert edge.model_revision == LATEON_CODE_EDGE_MODEL_REVISION
    assert full.name == "lateon-code"
    assert full.display_name == "LateOn-Code"
    assert full.model_id == LATEON_CODE_MODEL_ID
    assert full.model_revision == LATEON_CODE_MODEL_REVISION
    assert edge.embedding_metadata()["scoring"] == "bruteforce-maxsim"
    assert full.embedding_metadata()["plaid_index"] is False
