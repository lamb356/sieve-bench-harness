from __future__ import annotations

from pathlib import Path

from bench.loaders.base import CodeDocument
from bench.retrievers.sieve import SieveRetriever


def _toy_corpus() -> list[CodeDocument]:
    return [
        CodeDocument(
            document_id=f"doc-{index}",
            path=f"python/doc_{index}.py",
            code=f"def helper_{index}():\n    return {index}\n",
            language="python",
            index_text=f"helper {index}",
        )
        for index in range(10)
    ]


def _fake_sieve_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake_sieve.py"
    binary.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

args = sys.argv[1:]
if args[0] == "index":
    source_root = Path(args[1])
    (source_root / ".sieve").mkdir(parents=True, exist_ok=True)
    print("indexed")
    raise SystemExit(0)
if args[0] == "search":
    query = args[1]
    top = int(args[args.index("--top") + 1])
    match = re.search(r"helper[_ ](\\d+)", query)
    first = int(match.group(1)) if match else 0
    rows = []
    seen = set()
    for idx in [first, *range(10)]:
        if idx in seen:
            continue
        seen.add(idx)
        rows.append({
            "path": f"docs/{idx:06d}_doc_{idx}.py",
            "line": 1,
            "chunk_id": 0,
            "byte_range": [0, 32],
            "snippet": f"def helper_{idx}():",
            "score": 1.0 / float(len(rows) + 1),
            "layer": "hot-vector",
        })
        if len(rows) >= top:
            break
    print(json.dumps(rows))
    raise SystemExit(0)
raise SystemExit(f"unexpected command: {args}")
""",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def test_sieve_retriever_uses_cli_and_maps_results(tmp_path: Path) -> None:
    retriever = SieveRetriever(binary_path=_fake_sieve_binary(tmp_path), build_release=False)
    retriever.index(_toy_corpus())

    results = retriever.search("find helper_7", k=5)

    assert [result.document_id for result in results[:2]] == ["doc-7", "doc-0"]
    assert results[0].path == "python/doc_7.py"
    assert results[0].metadata["sieve_path"] == "docs/000007_doc_7.py"
    assert results[0].metadata["layer"] == "hot-vector"
    assert retriever.latency_ms()["p50"] > 0.0
    assert retriever.embedding_metadata()["interface"] == "sieve-cli-subprocess"


def test_sieve_retriever_materializes_raw_code_not_index_text(tmp_path: Path) -> None:
    retriever = SieveRetriever(binary_path=_fake_sieve_binary(tmp_path), build_release=False)
    retriever.index(
        [
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
    )

    assert retriever._source_root is not None
    raw_file = retriever._source_root / "docs" / "000000_raw.py"
    decoy_file = retriever._source_root / "docs" / "000001_decoy.py"
    assert raw_file.read_text(encoding="utf-8") == "def rawonlysignal():\n    return 'target'\n"
    assert "rawonlysignal" not in decoy_file.read_text(encoding="utf-8")


def test_sieve_retriever_requires_explicit_repo_or_binary_when_not_on_path(monkeypatch) -> None:
    monkeypatch.delenv("SIEVE_BINARY", raising=False)
    monkeypatch.delenv("SIEVE_REPO", raising=False)
    monkeypatch.setenv("PATH", "")
    retriever = SieveRetriever(build_release=False)

    try:
        retriever._resolve_binary()
    except FileNotFoundError as exc:
        assert "set SIEVE_BINARY or SIEVE_REPO" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("missing binary/repo should fail with an explicit configuration error")


def test_sieve_retriever_requires_non_empty_corpus(tmp_path: Path) -> None:
    retriever = SieveRetriever(binary_path=_fake_sieve_binary(tmp_path), build_release=False)
    try:
        retriever.index([])
    except ValueError as exc:
        assert "non-empty corpus" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("empty corpus should fail")
