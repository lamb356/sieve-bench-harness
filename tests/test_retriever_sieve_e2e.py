from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from bench.loaders.base import CodeDocument
from bench.retrievers.sieve import SieveRetriever


@pytest.mark.skipif(os.environ.get("SIEVE_RUN_REAL_E2E") != "1", reason="set SIEVE_RUN_REAL_E2E=1 to run real Rust SIEVE smoke")
def test_sieve_retriever_real_engine_e2e_smoke() -> None:
    query_onnx = os.environ.get("SIEVE_ENCODER_QUERY_ONNX")
    doc_onnx = os.environ.get("SIEVE_ENCODER_DOC_ONNX")
    if not query_onnx or not doc_onnx:
        pytest.skip("real SIEVE smoke requires SIEVE_ENCODER_QUERY_ONNX and SIEVE_ENCODER_DOC_ONNX")
    binary_path = os.environ.get("SIEVE_BINARY")
    repo_path = os.environ.get("SIEVE_REPO")
    if binary_path is None and repo_path is None:
        pytest.skip("real SIEVE smoke requires SIEVE_BINARY or SIEVE_REPO")
    if binary_path is None and repo_path is not None and not Path(repo_path, "Cargo.toml").is_file():
        pytest.skip("real SIEVE smoke requires SIEVE_REPO to point at a SIEVE checkout")

    corpus: list[CodeDocument] = []
    for index in range(100):
        if index == 42:
            code = "def known_correct_answer():\n    return 'needle42 target'\n"
        else:
            code = f"def synthetic_doc_{index}():\n    return {index}\n# filler token_{index}\n"
        corpus.append(
            CodeDocument(
                document_id=f"doc-{index}",
                path=f"python/doc_{index:03}.py",
                code=code,
                language="python",
                index_text=code,
            )
        )

    retriever = SieveRetriever(
        binary_path=binary_path,
        repo_path=repo_path,
        query_onnx_path=query_onnx,
        doc_onnx_path=doc_onnx,
        build_release=binary_path is None,
    )
    retriever.index(corpus)

    found_known_top5 = False
    for query_index in range(10):
        query = "known_correct_answer needle42 target" if query_index == 0 else f"synthetic_doc_{query_index} token_{query_index}"
        results = retriever.search(query, k=10)
        assert len(results) >= 5
        assert all(math.isfinite(result.score) for result in results)
        if query_index == 0 and any(result.document_id == "doc-42" for result in results[:5]):
            found_known_top5 = True
    assert found_known_top5
