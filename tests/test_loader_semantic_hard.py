from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark
from bench.loaders.semantic_hard import (
    SEMANTIC_HARD_DIR,
    SEMANTIC_HARD_MIN_ENTRIES,
    build_semantic_hard_benchmark,
    load_semantic_hard_entries,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry(*, language: str = "python", query_id: str = "q1", doc_id: str = "d1", code: str = "def answer():\n    return 42\n") -> dict[str, object]:
    query = "original easy query"
    return {
        "schema_version": 1,
        "selection": "semantic-hard-v1",
        "hardness_definition": "ripgrep_recall_at_5_eq_0_against_full_eval_corpus",
        "selection_ordinal": 0,
        "language": language,
        "source": "unit-source",
        "corpus_id": "unit-corpus",
        "dataset_id": "unit/dataset",
        "dataset_revision": "unit-revision",
        "eval_split": "test",
        "query_id": query_id,
        "ground_truth_document_id": doc_id,
        "ground_truth_path": f"src/{doc_id}.py",
        "query": query,
        "ground_truth_code": code,
        "query_sha256": _sha256(query),
        "ground_truth_code_sha256": _sha256(code),
        "ripgrep": {"top_k": 5, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0},
        "provenance": {"source_metadata": {"query_id": query_id, "ground_truth_document_id": doc_id}},
    }


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")


def _base_loaded() -> LoadedBenchmark:
    docs = (
        CodeDocument(document_id="d1", path="src/d1.py", code="def answer():\n    return 42\n", language="python"),
        CodeDocument(document_id="d2", path="src/d2.py", code="def fallback():\n    return 0\n", language="python"),
        CodeDocument(document_id="d3", path="src/d3.py", code="def helper():\n    return 1\n", language="python"),
    )
    examples = (
        EvalExample(
            query="original easy query",
            ground_truth_code=docs[0].code,
            ground_truth_path=docs[0].path,
            language="python",
            source="unit-source",
            corpus_id="unit-corpus",
            metadata={"query_id": "q1", "ground_truth_document_id": "d1"},
        ),
        EvalExample(
            query="second original query",
            ground_truth_code=docs[1].code,
            ground_truth_path=docs[1].path,
            language="python",
            source="unit-source",
            corpus_id="unit-corpus",
            metadata={"query_id": "q2", "ground_truth_document_id": "d2"},
        ),
    )
    return LoadedBenchmark(
        source="unit-source",
        language="python",
        revision="unit-revision",
        corpus_id="unit-corpus",
        corpus=docs,
        examples=examples,
        metadata={"dataset_id": "unit/dataset", "eval_split": "test", "full_example_count": len(examples)},
    )


def test_load_semantic_hard_entries_fails_loudly_for_small_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    _write_manifest(manifest, [_entry()])

    with pytest.raises(ValueError, match="Expected at least 2 semantic-hard entries"):
        load_semantic_hard_entries("python", manifest_path=manifest, min_entries=2)


def test_load_semantic_hard_entries_rejects_wrong_language(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    _write_manifest(manifest, [_entry(language="go")])

    with pytest.raises(ValueError, match="language=go"):
        load_semantic_hard_entries("python", manifest_path=manifest, min_entries=1)


def test_load_semantic_hard_entries_rejects_missing_required_schema_field(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    row = _entry()
    del row["hardness_definition"]
    _write_manifest(manifest, [row])

    with pytest.raises(ValueError, match="hardness_definition"):
        load_semantic_hard_entries("python", manifest_path=manifest, min_entries=1)


def test_load_semantic_hard_entries_rejects_missing_provenance_source_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    row = _entry()
    row["provenance"] = {}
    _write_manifest(manifest, [row])

    with pytest.raises(ValueError, match="provenance.source_metadata"):
        load_semantic_hard_entries("python", manifest_path=manifest, min_entries=1)


def test_load_semantic_hard_entries_rejects_duplicate_query_hashes(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    first = _entry(query_id="q1", doc_id="d1")
    second = _entry(query_id="q2", doc_id="d2", code="def fallback():\n    return 0\n")
    second["selection_ordinal"] = 1
    second["query"] = first["query"]
    second["query_sha256"] = first["query_sha256"]
    _write_manifest(manifest, [first, second])

    with pytest.raises(ValueError, match="Duplicate query_sha256"):
        load_semantic_hard_entries("python", manifest_path=manifest, min_entries=2)


def test_build_semantic_hard_benchmark_uses_manifest_queries_and_keeps_required_positives(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    first = _entry(query_id="q1", doc_id="d1", code="def answer():\n    return 42\n")
    second = _entry(query_id="q2", doc_id="d2", code="def fallback():\n    return 0\n")
    second["selection_ordinal"] = 1
    second["query"] = "second original query"
    second["query_sha256"] = _sha256(str(second["query"]))
    _write_manifest(manifest, [first, second])

    loaded = build_semantic_hard_benchmark(
        base_loaded=_base_loaded(),
        language="python",
        sample_size=1,
        corpus_sample_size=2,
        manifest_path=manifest,
        min_entries=2,
    )

    assert loaded.source == "unit-source-semantic-hard-v1"
    assert loaded.metadata["semantic_hard_manifest_count"] == 2
    assert loaded.metadata["semantic_hard_sample_size"] == 1
    assert loaded.metadata["corpus_sample_size"] == 2
    assert len(loaded.examples) == 1
    assert len(loaded.corpus) == 2
    assert loaded.examples[0].query == "original easy query"
    assert loaded.examples[0].metadata["semantic_hard"]["ripgrep"]["recall@5"] == 0.0
    assert {loaded.examples[0].metadata["ground_truth_document_id"]} <= {document.document_id for document in loaded.corpus}


def test_build_semantic_hard_benchmark_fails_when_manifest_code_hash_does_not_match(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    bad = _entry()
    bad["ground_truth_code_sha256"] = _sha256("different code")
    _write_manifest(manifest, [bad])

    with pytest.raises(ValueError, match="code hash mismatch"):
        build_semantic_hard_benchmark(
            base_loaded=_base_loaded(),
            language="python",
            manifest_path=manifest,
            min_entries=1,
        )


def test_build_semantic_hard_benchmark_fails_when_manifest_query_pair_drifts(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    bad = _entry()
    bad["query_id"] = "stale-query-id"
    _write_manifest(manifest, [bad])

    with pytest.raises(ValueError, match="query_id mismatch"):
        build_semantic_hard_benchmark(
            base_loaded=_base_loaded(),
            language="python",
            manifest_path=manifest,
            min_entries=1,
        )


def test_build_semantic_hard_benchmark_fails_when_manifest_query_text_drifts(tmp_path: Path) -> None:
    manifest = tmp_path / "python.jsonl"
    bad = _entry()
    bad["query"] = "stale semantic query"
    bad["query_sha256"] = _sha256(str(bad["query"]))
    _write_manifest(manifest, [bad])

    with pytest.raises(ValueError, match="query text drift"):
        build_semantic_hard_benchmark(
            base_loaded=_base_loaded(),
            language="python",
            manifest_path=manifest,
            min_entries=1,
        )


def test_committed_semantic_hard_manifests_have_minimum_unique_zero_recall_entries() -> None:
    for language in ("python", "typescript", "go", "rust"):
        entries = load_semantic_hard_entries(language, manifest_path=SEMANTIC_HARD_DIR / f"{language}.jsonl")
        assert len(entries) >= SEMANTIC_HARD_MIN_ENTRIES
        assert len({entry["query_id"] for entry in entries}) == len(entries)
        assert len({entry["ground_truth_document_id"] for entry in entries}) == len(entries)
        assert len({entry["ground_truth_code_sha256"] for entry in entries}) == len(entries)
        assert all(entry["ripgrep"]["recall@5"] == 0.0 for entry in entries)
