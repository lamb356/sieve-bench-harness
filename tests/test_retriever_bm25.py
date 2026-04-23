from __future__ import annotations

from bench.loaders.base import CodeDocument
from bench.retrievers.bm25 import BM25Retriever, tokenize_text


def _toy_corpus() -> list[CodeDocument]:
    corpus: list[CodeDocument] = []
    for index in range(100):
        code = f"def helper_{index}():\n    return {index}\n"
        index_text = f"helper {index} return {index}"
        if index == 73:
            code = "def parse_http_request(raw_request: str) -> dict:\n    header_map = {}\n    return header_map\n"
            index_text = "parse http request raw request dict header map return header map"
        elif index == 31:
            code = "def compute_gradient_norm(tensor):\n    return tensor.grad.norm()\n"
            index_text = "compute gradient norm tensor grad norm return"
        corpus.append(
            CodeDocument(
                document_id=f"doc-{index}",
                path=f"python/doc_{index}.py",
                code=code,
                language="python",
                index_text=index_text,
                metadata={"rank": index},
            )
        )
    return corpus


def test_bm25_retriever_finds_known_answers_on_100_doc_toy_corpus() -> None:
    retriever = BM25Retriever()
    retriever.index(_toy_corpus())

    http_results = retriever.search("parse http request headers", k=5)
    gradient_results = retriever.search("compute tensor gradient norm", k=5)

    assert http_results[0].document_id == "doc-73"
    assert gradient_results[0].document_id == "doc-31"
    assert http_results[0].path == "python/doc_73.py"
    assert "p50" in retriever.latency_ms()


def test_bm25_tokenizer_splits_code_identifiers_for_bag_of_tokens() -> None:
    assert tokenize_text("parse_httpRequest HTTPResponse200") == [
        "parse",
        "http",
        "request",
        "http",
        "response",
        "200",
    ]
