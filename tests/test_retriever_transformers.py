from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from bench.loaders.base import CodeDocument
from bench.retrievers import RETRIEVER_REPORT_METADATA
from bench.retrievers import codebert as codebert_module
from bench.retrievers.bge_small import BgeSmallRetriever
from bench.retrievers.codebert import CodeBERTRetriever
from bench.retrievers.unixcoder import UniXcoderRetriever


class KeywordEmbeddingBackend:
    model_id = "test/mock-code-encoder"
    model_revision = "test-revision"
    max_length = 512
    truncation_strategy = "test-keyword-no-truncation"
    device = "cpu"

    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.encoded_batches.append(list(texts))
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


@pytest.mark.parametrize("retriever_cls", [CodeBERTRetriever, UniXcoderRetriever])
def test_transformer_retrievers_encode_raw_code_not_index_text(
    retriever_cls: type[CodeBERTRetriever] | type[UniXcoderRetriever], tmp_path
) -> None:
    backend = KeywordEmbeddingBackend()
    retriever = retriever_cls(model_cache_dir=tmp_path / "models", embedding_backend=backend)
    corpus = [
        CodeDocument(
            document_id="raw-doc",
            path="python/raw.py",
            code="def rawonlysignal():\n    return 'target'\n",
            language="python",
            index_text="metadata decoy only",
        ),
        CodeDocument(
            document_id="index-text-decoy",
            path="python/decoy.py",
            code="def unrelated():\n    return 'noise'\n",
            language="python",
            index_text="rawonlysignal rawonlysignal rawonlysignal",
        ),
    ]

    retriever.index(corpus)

    assert backend.encoded_batches[0] == [document.code for document in corpus]
    assert all("rawonlysignal rawonlysignal" not in text for text in backend.encoded_batches[0])


def test_bge_small_retriever_encodes_raw_code_not_index_text(tmp_path) -> None:
    backend = KeywordEmbeddingBackend()
    retriever = BgeSmallRetriever(model_cache_dir=tmp_path / "models", embedding_backend=backend)
    corpus = [
        CodeDocument(
            document_id="raw-doc",
            path="python/raw.py",
            code="def parse_http_request(raw_request):\n    return raw_request.headers\n",
            language="python",
            index_text="metadata decoy only",
        ),
        CodeDocument(
            document_id="index-text-decoy",
            path="python/decoy.py",
            code="def unrelated():\n    return 'noise'\n",
            language="python",
            index_text="parse http request headers parse http request headers",
        ),
    ]

    retriever.index(corpus)

    assert backend.encoded_batches[0] == [document.code for document in corpus]
    assert all("parse http request headers parse" not in text for text in backend.encoded_batches[0])
    assert retriever.search("parse http request headers", k=1)[0].document_id == "raw-doc"


def test_bge_small_is_annotated_as_default_dense_backend() -> None:
    metadata = RETRIEVER_REPORT_METADATA["bge-small"]

    assert metadata.role == "default_dense_backend"
    assert metadata.table == "hero"
    assert metadata.params == "33M"
    assert "Default dense backend" in metadata.role_label


def test_codebert_is_annotated_as_null_baseline_for_extended_table() -> None:
    metadata = RETRIEVER_REPORT_METADATA["codebert"]

    assert metadata.role == "null_baseline"
    assert metadata.table == "extended"
    assert metadata.display_name == "CodeBERT (pretrained features only)"
    assert "NULL BASELINE" in metadata.role_label

    assert "NULL BASELINE" in (codebert_module.__doc__ or "")
    assert "Do not cite this row as a CodeBERT-vs-SIEVE comparison" in (codebert_module.__doc__ or "")
