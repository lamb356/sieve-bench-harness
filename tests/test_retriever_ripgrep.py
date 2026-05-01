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


def test_ripgrep_retriever_handles_non_ascii_multiline_queries_without_regex_error(tmp_path) -> None:
    document = CodeDocument(
        document_id="doc-1",
        path="typescript/doc_1.ts",
        code="export function formatLabel(value: number): string { return value.toString(); }",
        language="typescript",
        metadata={},
    )
    retriever = RipgrepRetriever(index_root=tmp_path / "ripgrep-index")
    retriever.index((document,))

    results = retriever.search("格式化显示值\n支持本地化", k=5)

    assert results == []


def test_ripgrep_materializes_and_searches_raw_code_not_index_text(tmp_path) -> None:
    retriever = RipgrepRetriever(index_root=tmp_path / "ripgrep-index")
    retriever.index(
        [
            CodeDocument(
                document_id="raw-doc",
                path="python/raw.py",
                code="def rawonlysignal():\n    return 'target'\n",
                language="python",
                index_text="metadata decoy only",
                metadata={},
            ),
            CodeDocument(
                document_id="index-text-decoy",
                path="python/decoy.py",
                code="def unrelated():\n    return 'noise'\n",
                language="python",
                index_text="rawonlysignal rawonlysignal rawonlysignal",
                metadata={},
            ),
        ]
    )

    assert (tmp_path / "ripgrep-index" / "python" / "raw.py").read_text(encoding="utf-8") == "def rawonlysignal():\n    return 'target'\n"
    assert "rawonlysignal" not in (tmp_path / "ripgrep-index" / "python" / "decoy.py").read_text(encoding="utf-8")
    assert [result.document_id for result in retriever.search("rawonlysignal", k=2)] == ["raw-doc"]
