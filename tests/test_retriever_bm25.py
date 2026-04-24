from __future__ import annotations

import time

from rank_bm25 import BM25Okapi

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


def test_bm25_scores_match_rank_bm25_reference_on_deterministic_fixture() -> None:
    corpus = [
        CodeDocument(
            document_id="doc-a",
            path="a.py",
            code="",
            language="python",
            index_text="alpha beta beta gamma",
            metadata={},
        ),
        CodeDocument(
            document_id="doc-b",
            path="b.py",
            code="",
            language="python",
            index_text="alpha rare rare beta",
            metadata={},
        ),
        CodeDocument(
            document_id="doc-c",
            path="c.py",
            code="",
            language="python",
            index_text="gamma delta epsilon",
            metadata={},
        ),
    ]
    query_tokens = tokenize_text("alpha beta rare")
    reference = BM25Okapi([tokenize_text(document.index_text) for document in corpus])
    reference_scores = reference.get_scores(query_tokens)
    expected = sorted(
        (
            (document.document_id, float(reference_scores[index]))
            for index, document in enumerate(corpus)
            if reference_scores[index] > 0.0
        ),
        key=lambda item: (-item[1], item[0]),
    )[:3]

    retriever = BM25Retriever()
    retriever.index(corpus)
    actual = [(result.document_id, result.score) for result in retriever.search("alpha beta rare", k=3)]

    assert [document_id for document_id, _score in actual] == [document_id for document_id, _score in expected]
    for (_actual_id, actual_score), (_expected_id, expected_score) in zip(actual, expected, strict=True):
        assert abs(actual_score - expected_score) < 1e-5


def test_bm25_latency_under_20ms_p50_on_corpus_of_15k() -> None:
    shared_terms = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    corpus = [
        CodeDocument(
            document_id=f"doc-{index:05d}",
            path=f"python/doc_{index:05d}.py",
            code=f"def helper_{index}():\n    return {index}\n",
            language="python",
            index_text=f"helper_{index} {shared_terms} {'rare_target' if index % 137 == 0 else ''}",
            metadata={},
        )
        for index in range(15_000)
    ]
    retriever = BM25Retriever()
    retriever.index(corpus)
    query = f"{shared_terms} rare_target"

    latencies_ms: list[float] = []
    for _ in range(7):
        started = time.perf_counter()
        results = retriever.search(query, k=10)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        assert results

    p50_ms = sorted(latencies_ms)[len(latencies_ms) // 2]
    assert p50_ms < 20.0
