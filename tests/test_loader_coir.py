from bench.constants import PYTHON_EVAL_FULL, PYTHON_EVAL_FULL_QUERY_COUNT
from bench.loaders.coir import CoIRPythonLoader


def test_coir_python_loader_uses_official_test_pairs_only() -> None:
    query_rows = [
        {"_id": "q-train", "title": "train_fn", "partition": "train", "text": "def train_fn():\n    return 'train'", "language": "python", "meta_information": {"resource": "train.py"}},
        {"_id": "q-test-1", "title": "test_one", "partition": "test", "text": "def test_one():\n    return 1", "language": "python", "meta_information": {"resource": "pkg/test_one.py"}},
        {"_id": "q-test-2", "title": "test_two", "partition": "test", "text": "def test_two():\n    return 2", "language": "python", "meta_information": {"resource": "pkg/test_two.py"}},
    ]
    corpus_rows = [
        {"_id": "c-train", "title": "", "partition": "train", "text": "training helper text", "language": "python", "meta_information": {"resource": "train.txt"}},
        {"_id": "c-test-1", "title": "", "partition": "test", "text": "find the function that returns one", "language": "python", "meta_information": {"resource": "docs/one.txt"}},
        {"_id": "c-test-2", "title": "", "partition": "test", "text": "find the function that returns two", "language": "python", "meta_information": {"resource": "docs/two.txt"}},
    ]
    qrel_rows = [
        {"query-id": "q-train", "corpus-id": "c-train", "score": 1},
        {"query-id": "q-test-1", "corpus-id": "c-test-1", "score": 1},
        {"query-id": "q-test-2", "corpus-id": "c-test-2", "score": 1},
    ]

    loaded = CoIRPythonLoader._build_loaded_benchmark(
        query_rows=query_rows,
        corpus_rows=corpus_rows,
        qrel_rows=qrel_rows,
        sample_size=10,
    )

    assert loaded.source == "coir"
    assert loaded.language == "python"
    assert len(loaded.corpus) == 2
    assert len(loaded.examples) == 2
    assert loaded.examples[0].query in {"find the function that returns one", "find the function that returns two"}
    assert loaded.examples[0].ground_truth_code.startswith("def test_")
    assert all(example.metadata["split"] == "test" for example in loaded.examples)


def test_phase_b5_loader_returns_correct_split_size() -> None:
    query_rows = [
        {
            "_id": f"q{i}",
            "title": f"fn_{i}",
            "partition": "test",
            "text": f"def fn_{i}():\n    return {i}",
            "language": "python",
            "meta_information": {"resource": f"pkg/fn_{i}.py"},
        }
        for i in range(PYTHON_EVAL_FULL_QUERY_COUNT)
    ]
    corpus_rows = [
        {
            "_id": f"c{i}",
            "title": "",
            "partition": "test",
            "text": f"find function {i}",
            "language": "python",
            "meta_information": {"resource": f"docs/fn_{i}.txt"},
        }
        for i in range(PYTHON_EVAL_FULL_QUERY_COUNT)
    ]
    qrel_rows = [
        {"query-id": f"q{i}", "corpus-id": f"c{i}", "score": 1}
        for i in range(PYTHON_EVAL_FULL_QUERY_COUNT)
    ]

    loaded = CoIRPythonLoader._build_loaded_benchmark(
        query_rows=query_rows,
        corpus_rows=corpus_rows,
        qrel_rows=qrel_rows,
        sample_size=None,
        eval_split=PYTHON_EVAL_FULL,
        expected_example_count=PYTHON_EVAL_FULL_QUERY_COUNT,
    )

    assert loaded.metadata["eval_split"] == PYTHON_EVAL_FULL
    assert loaded.metadata["expected_example_count"] == PYTHON_EVAL_FULL_QUERY_COUNT
    assert len(loaded.examples) == PYTHON_EVAL_FULL_QUERY_COUNT
