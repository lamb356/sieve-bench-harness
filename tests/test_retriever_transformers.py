from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from bench.loaders.base import CodeDocument
from bench.retrievers.codebert import CodeBERTRetriever
from bench.retrievers.unixcoder import UniXcoderRetriever


class KeywordEmbeddingBackend:
    model_id = "test/mock-code-encoder"
    model_revision = "test-revision"
    max_length = 512
    truncation_strategy = "test-keyword-no-truncation"
    device = "cpu"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(sum(token in lowered for token in ("parse", "http", "request", "header"))),
                    float(sum(token in lowered for token in ("gradient", "tensor", "norm"))),
                    float(sum(token in lowered for token in ("json", "serialize", "file"))),
                    0.01,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def _toy_corpus() -> list[CodeDocument]:
    corpus: list[CodeDocument] = []
    for index in range(100):
        index_text = f"generic helper value {index}"
        code = f"def helper_{index}():\n    return {index}\n"
        if index == 17:
            code = "def write_json_file(data, path):\n    return json.dumps(data)\n"
            index_text = "write json file serialize data path return dumps"
        elif index == 42:
            code = "def parse_http_request(raw_request):\n    return raw_request.headers\n"
            index_text = "parse http request raw request headers return"
        elif index == 88:
            code = "def gradient_norm(tensor):\n    return tensor.grad.norm()\n"
            index_text = "gradient norm tensor grad return norm"
        corpus.append(
            CodeDocument(
                document_id=f"doc-{index}",
                path=f"python/doc_{index}.py",
                code=code,
                language="python",
                index_text=index_text,
            )
        )
    return corpus


@pytest.mark.parametrize("retriever_cls,target", [(CodeBERTRetriever, "doc-42"), (UniXcoderRetriever, "doc-42")])
def test_transformer_retrievers_rank_by_mean_pooled_cosine_embeddings_on_100_doc_toy_corpus(
    retriever_cls: type[CodeBERTRetriever] | type[UniXcoderRetriever], target: str, tmp_path
) -> None:
    retriever = retriever_cls(model_cache_dir=tmp_path / "models", embedding_backend=KeywordEmbeddingBackend())
    retriever.index(_toy_corpus())

    results = retriever.search("parse http request headers", k=5)

    assert results[0].document_id == target
    assert results[0].metadata["model_id"] == "test/mock-code-encoder"
    assert results[0].metadata["truncation_strategy"] == "test-keyword-no-truncation"
    assert retriever.latency_ms()["p50"] >= 0.0


def test_transformer_retrievers_expose_pinned_model_metadata(tmp_path) -> None:
    retriever = CodeBERTRetriever(model_cache_dir=tmp_path / "models", embedding_backend=KeywordEmbeddingBackend())

    assert retriever.embedding_metadata()["model_id"] == "test/mock-code-encoder"
    assert retriever.embedding_metadata()["max_length"] == 512
    assert "truncation_strategy" in retriever.embedding_metadata()
