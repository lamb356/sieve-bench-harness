from bench.retrievers.ripgrep import RipgrepRetriever
from bench.loaders.base import CodeDocument


def test_ripgrep_retriever_finds_relevant_doc_in_top_five(tmp_path) -> None:
    corpus = []
    for index in range(100):
        code = f"def helper_{index}():\n    return {index}\n"
        if index == 73:
            code = "def parse_http_request(raw_request: str) -> dict:\n    header_map = {}\n    return header_map\n"
        corpus.append(
            CodeDocument(
                document_id=f"doc-{index}",
                path=f"python/doc_{index}.py",
                code=code,
                language="python",
                metadata={"rank": index},
            )
        )

    retriever = RipgrepRetriever(index_root=tmp_path / "ripgrep-index")
    retriever.index(corpus)
    results = retriever.search("parse http request headers", k=5)

    assert any(result.document_id == "doc-73" for result in results)
    latency = retriever.latency_ms()
    assert "p50" in latency
    assert "p95" in latency
