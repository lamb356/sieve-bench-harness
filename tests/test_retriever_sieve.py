from __future__ import annotations

from bench.loaders.base import CodeDocument
from bench.retrievers.sieve import SieveStubRetriever


def _toy_corpus() -> list[CodeDocument]:
    return [
        CodeDocument(
            document_id=f"doc-{index}",
            path=f"python/doc_{index}.py",
            code=f"def helper_{index}():\n    return {index}\n",
            language="python",
            index_text=f"helper {index}",
        )
        for index in range(100)
    ]


def test_sieve_stub_is_deterministic_for_same_query_hash() -> None:
    retriever = SieveStubRetriever(seed=1337)
    retriever.index(_toy_corpus())

    first = retriever.search("parse http request headers", k=10)
    second = retriever.search("parse http request headers", k=10)
    different = retriever.search("compute tensor gradient norm", k=10)

    assert [result.document_id for result in first] == [result.document_id for result in second]
    assert [result.document_id for result in first] != [result.document_id for result in different]
    assert all(result.metadata["stub"] == "deterministic-query-hash-random" for result in first)
