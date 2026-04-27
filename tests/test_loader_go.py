from __future__ import annotations

import pytest

from bench.constants import (
    GO_DATASET_ID,
    GO_DATASET_REVISION,
    GO_EVAL_FULL,
    GO_EVAL_FULL_QUERY_COUNT,
)
from bench.loaders.go import CoIRGoLoader


def _go_code_row(index: int, *, partition: str = "test") -> dict[str, object]:
    return {
        "_id": f"q{index}",
        "title": f"formatExample{index}",
        "partition": partition,
        "text": f"func formatExample{index}(value int) string {{\n\treturn fmt.Sprint(value)\n}}",
        "language": "go",
        "meta_information": {"resource": f"pkg/example_{index}.go"},
    }


def _go_query_row(index: int, *, partition: str = "test") -> dict[str, object]:
    return {
        "_id": f"c{index}",
        "title": "",
        "partition": partition,
        "text": f"Format example value {index} for display.",
        "language": "",
        "meta_information": {"resource": ""},
    }


def _go_qrel(index: int) -> dict[str, object]:
    return {"query-id": f"q{index}", "corpus-id": f"c{index}", "score": 1}


def test_loader_go_returns_official_coir_go_test_qrels_metadata() -> None:
    loaded = CoIRGoLoader._build_loaded_benchmark(
        query_rows=[_go_code_row(i) for i in range(3)],
        corpus_rows=[_go_query_row(i) for i in range(3)],
        qrel_rows=[_go_qrel(i) for i in range(3)],
        sample_size=None,
        eval_split=GO_EVAL_FULL,
        expected_example_count=3,
    )

    assert loaded.source == "coir-go"
    assert loaded.language == "go"
    assert loaded.revision == GO_DATASET_REVISION
    assert loaded.metadata["dataset_id"] == GO_DATASET_ID
    assert loaded.metadata["dataset_revision"] == GO_DATASET_REVISION
    assert loaded.metadata["dataset_language"] == "go"
    assert loaded.metadata["eval_split"] == GO_EVAL_FULL
    assert loaded.metadata["qrel_row_count"] == 3
    assert loaded.metadata["full_example_count"] == 3
    assert len(loaded.corpus) == 3
    assert len(loaded.examples) == 3
    assert loaded.corpus[0].path.endswith(".go")
    assert loaded.corpus[0].document_id == "q0"
    assert loaded.examples[0].query.startswith("Format example value")
    assert loaded.examples[0].ground_truth_code.startswith("func formatExample")
    assert loaded.examples[0].metadata["ground_truth_document_id"] == "q0"
    assert loaded.metadata["methodology"].startswith("Official CoIR/CodeSearchNet Go")


def test_loader_go_can_bound_corpus_for_practical_validation_runs() -> None:
    loaded = CoIRGoLoader._build_loaded_benchmark(
        query_rows=[_go_code_row(i) for i in range(20)],
        corpus_rows=[_go_query_row(i) for i in range(20)],
        qrel_rows=[_go_qrel(i) for i in range(20)],
        sample_size=5,
        corpus_sample_size=10,
        eval_split=GO_EVAL_FULL,
    )

    positive_ids = {str(example.metadata["ground_truth_document_id"]) for example in loaded.examples}
    corpus_ids = {document.document_id for document in loaded.corpus}
    assert len(loaded.examples) == 5
    assert len(loaded.corpus) == 10
    assert positive_ids <= corpus_ids
    assert loaded.metadata["corpus_sampling_note"]


def test_loader_go_validates_expected_count_before_sampling() -> None:
    with pytest.raises(ValueError, match="Expected 2 examples"):
        CoIRGoLoader._build_loaded_benchmark(
            query_rows=[_go_code_row(1)],
            corpus_rows=[_go_query_row(1)],
            qrel_rows=[_go_qrel(1)],
            sample_size=1,
            eval_split=GO_EVAL_FULL,
            expected_example_count=2,
        )


def test_loader_go_preserves_custom_revision_metadata() -> None:
    loaded = CoIRGoLoader._build_loaded_benchmark(
        query_rows=[_go_code_row(1)],
        corpus_rows=[_go_query_row(1)],
        qrel_rows=[_go_qrel(1)],
        sample_size=None,
        eval_split=GO_EVAL_FULL,
        revision="custom-go-revision",
    )

    assert GO_EVAL_FULL_QUERY_COUNT == 8122
    assert loaded.revision == "custom-go-revision"
    assert loaded.metadata["dataset_revision"] == "custom-go-revision"
    assert loaded.corpus[0].metadata["dataset_revision"] == "custom-go-revision"
    assert loaded.examples[0].metadata["dataset_revision"] == "custom-go-revision"


def test_loader_go_treats_test_qrels_as_authoritative_for_non_test_query_text_rows() -> None:
    loaded = CoIRGoLoader._build_loaded_benchmark(
        query_rows=[_go_code_row(1, partition="test")],
        corpus_rows=[_go_query_row(1, partition="train")],
        qrel_rows=[_go_qrel(1)],
        sample_size=None,
        eval_split=GO_EVAL_FULL,
        expected_example_count=1,
    )

    assert len(loaded.examples) == 1
    assert loaded.metadata["full_example_count"] == 1
    assert loaded.examples[0].query.startswith("Format example value 1")


def test_loader_go_generates_unique_file_paths_for_duplicate_titles_or_resources() -> None:
    duplicate_code_rows = [_go_code_row(i) for i in range(2)]
    for row in duplicate_code_rows:
        row["title"] = "duplicateTitle"
        row["meta_information"] = {"resource": "pkg/duplicate.go"}

    loaded = CoIRGoLoader._build_loaded_benchmark(
        query_rows=duplicate_code_rows,
        corpus_rows=[_go_query_row(i) for i in range(2)],
        qrel_rows=[_go_qrel(i) for i in range(2)],
        sample_size=None,
        eval_split=GO_EVAL_FULL,
        expected_example_count=2,
    )

    paths = [document.path for document in loaded.corpus]
    assert len(paths) == len(set(paths))
    assert all(path.endswith(".go") for path in paths)
    assert loaded.examples[0].ground_truth_path in paths
