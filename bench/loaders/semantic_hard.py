from __future__ import annotations

"""Canonical semantic-hard subset loader.

The canonical source of truth is ``bench/data/semantic-hard-v1/{language}.jsonl``.
Each row is a previously validated full-eval example whose paired code document
was not found by the harness ripgrep baseline in the top 5.  This module keeps
that subset data-driven rather than encoding query rewrites in loader code.
"""

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bench.constants import GLOBAL_RANDOM_SEED
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark

SEMANTIC_HARD_SELECTION = "semantic-hard-v1"
SEMANTIC_HARD_HARDNESS_DEFINITION = "ripgrep_recall_at_5_eq_0_against_full_eval_corpus"
SEMANTIC_HARD_MIN_ENTRIES = 300
SEMANTIC_HARD_DIR = Path(__file__).resolve().parents[1] / "data" / SEMANTIC_HARD_SELECTION
SUPPORTED_SEMANTIC_HARD_LANGUAGES = ("python", "typescript", "go", "rust")
REQUIRED_MANIFEST_STRING_FIELDS = (
    "language",
    "source",
    "corpus_id",
    "dataset_id",
    "dataset_revision",
    "eval_split",
    "hardness_definition",
    "query_id",
    "ground_truth_document_id",
    "ground_truth_path",
    "query",
    "ground_truth_code",
    "query_sha256",
    "ground_truth_code_sha256",
)

SemanticHardEntry = dict[str, Any]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_str(entry: Mapping[str, Any], key: str, *, path: Path, line_no: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_no}: expected non-empty string field {key!r}")
    return value


def _validate_entry(entry: Mapping[str, Any], *, language: str, path: Path, line_no: int) -> SemanticHardEntry:
    if entry.get("schema_version") != 1:
        raise ValueError(f"{path}:{line_no}: expected schema_version=1")
    if entry.get("selection") != SEMANTIC_HARD_SELECTION:
        raise ValueError(f"{path}:{line_no}: expected selection={SEMANTIC_HARD_SELECTION!r}")

    required = {field: _required_str(entry, field, path=path, line_no=line_no) for field in REQUIRED_MANIFEST_STRING_FIELDS}
    observed_language = required["language"]
    if observed_language != language:
        raise ValueError(f"{path}:{line_no}: expected language={language}, got language={observed_language}")
    if required["hardness_definition"] != SEMANTIC_HARD_HARDNESS_DEFINITION:
        raise ValueError(
            f"{path}:{line_no}: expected hardness_definition={SEMANTIC_HARD_HARDNESS_DEFINITION!r}, "
            f"got {required['hardness_definition']!r}"
        )

    query = required["query"]
    code = required["ground_truth_code"]
    query_hash = required["query_sha256"]
    code_hash = required["ground_truth_code_sha256"]
    if query_hash != _sha256(query):
        raise ValueError(f"{path}:{line_no}: query hash mismatch for semantic-hard entry")
    if code_hash != _sha256(code):
        raise ValueError(f"{path}:{line_no}: code hash mismatch for semantic-hard entry")

    ripgrep = entry.get("ripgrep")
    if not isinstance(ripgrep, Mapping):
        raise ValueError(f"{path}:{line_no}: expected ripgrep metadata object")
    recall_at_5 = ripgrep.get("recall@5")
    if not isinstance(recall_at_5, (int, float)) or isinstance(recall_at_5, bool) or recall_at_5 != 0.0:
        raise ValueError(f"{path}:{line_no}: semantic-hard entry has ripgrep recall@5={recall_at_5}, expected 0.0")

    provenance = entry.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{path}:{line_no}: expected provenance metadata object")
    source_metadata = provenance.get("source_metadata")
    if not isinstance(source_metadata, Mapping) or not source_metadata:
        raise ValueError(f"{path}:{line_no}: expected non-empty provenance.source_metadata object")

    ordinal = entry.get("selection_ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise ValueError(f"{path}:{line_no}: expected non-negative integer selection_ordinal")

    return dict(entry)


def load_semantic_hard_entries(
    language: str,
    *,
    manifest_path: Path | None = None,
    min_entries: int = SEMANTIC_HARD_MIN_ENTRIES,
) -> tuple[SemanticHardEntry, ...]:
    """Load and validate a canonical semantic-hard JSONL manifest.

    Validation intentionally fails loudly on missing manifests, malformed rows,
    wrong language rows, hash drift, non-hard rows, or duplicate query/document
    identities.  The returned entries are ordered by ``selection_ordinal``.
    """

    if language not in SUPPORTED_SEMANTIC_HARD_LANGUAGES:
        raise ValueError(f"Unsupported semantic-hard language {language!r}; expected one of {SUPPORTED_SEMANTIC_HARD_LANGUAGES}")

    path = manifest_path or SEMANTIC_HARD_DIR / f"{language}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing semantic-hard manifest for {language} at {path}")

    entries: list[SemanticHardEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_no}: expected JSON object row")
            entries.append(_validate_entry(raw, language=language, path=path, line_no=line_no))

    if len(entries) < min_entries:
        raise ValueError(f"Expected at least {min_entries} semantic-hard entries for {language} at {path}, got {len(entries)}")

    duplicate_fields = (
        "selection_ordinal",
        "query_id",
        "ground_truth_document_id",
        "ground_truth_code_sha256",
        "query_sha256",
    )
    for field in duplicate_fields:
        values = [entry[field] for entry in entries]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field} values in semantic-hard manifest {path}")

    return tuple(sorted(entries, key=lambda entry: int(entry["selection_ordinal"])))


def _sample_entries(entries: tuple[SemanticHardEntry, ...], sample_size: int | None) -> tuple[SemanticHardEntry, ...]:
    if sample_size is None:
        return entries
    if sample_size > len(entries):
        raise ValueError(f"sample_size={sample_size} exceeds semantic-hard manifest size {len(entries)}")
    return entries[:sample_size]


def load_semantic_hard_benchmark(
    language: str,
    *,
    sample_size: int | None = None,
    corpus_sample_size: int | None = None,
    manifest_path: Path | None = None,
    min_entries: int = SEMANTIC_HARD_MIN_ENTRIES,
) -> LoadedBenchmark:
    """Load a full-language corpus and project it onto the semantic-hard manifest."""

    if language == "python":
        from bench.loaders.coir import CoIRPythonLoader

        base_loaded = CoIRPythonLoader().load_full_eval(sample_size=None)
    elif language == "typescript":
        from bench.loaders.typescript import TypeScriptEvalLoader

        base_loaded = TypeScriptEvalLoader().load_full_eval(sample_size=None, corpus_sample_size=None)
    elif language == "go":
        from bench.loaders.go import CoIRGoLoader

        base_loaded = CoIRGoLoader().load_full_eval(sample_size=None, corpus_sample_size=None)
    elif language == "rust":
        from bench.loaders.rust import RustEvalLoader

        base_loaded = RustEvalLoader().load_full_eval(sample_size=None, corpus_sample_size=None)
    else:
        raise ValueError(f"Unsupported semantic-hard language {language!r}; expected one of {SUPPORTED_SEMANTIC_HARD_LANGUAGES}")

    return build_semantic_hard_benchmark(
        base_loaded=base_loaded,
        language=language,
        sample_size=sample_size,
        corpus_sample_size=corpus_sample_size,
        manifest_path=manifest_path,
        min_entries=min_entries,
    )


def build_semantic_hard_benchmark(
    *,
    base_loaded: LoadedBenchmark,
    language: str,
    sample_size: int | None = None,
    corpus_sample_size: int | None = None,
    manifest_path: Path | None = None,
    min_entries: int = SEMANTIC_HARD_MIN_ENTRIES,
) -> LoadedBenchmark:
    """Project a full-language benchmark onto the canonical semantic-hard rows."""

    if base_loaded.language != language:
        raise ValueError(f"Base benchmark language mismatch: expected {language}, got {base_loaded.language}")

    entries = load_semantic_hard_entries(language, manifest_path=manifest_path, min_entries=min_entries)
    selected_entries = _sample_entries(entries, sample_size)

    documents_by_id = {document.document_id: document for document in base_loaded.corpus}
    examples_by_doc_id = {
        str(example.metadata.get("ground_truth_document_id")): example
        for example in base_loaded.examples
        if example.metadata.get("ground_truth_document_id") is not None
    }
    examples: list[EvalExample] = []
    qrels: dict[str, dict[str, int]] = {}
    for entry in selected_entries:
        doc_id = str(entry["ground_truth_document_id"])
        document = documents_by_id.get(doc_id)
        if document is None:
            raise ValueError(f"Semantic-hard manifest references document {doc_id!r} not present in {base_loaded.source}/{language} corpus")
        observed_hash = _sha256(document.code)
        if observed_hash != entry["ground_truth_code_sha256"]:
            raise ValueError(
                f"Semantic-hard manifest code hash mismatch for document {doc_id}: manifest={entry['ground_truth_code_sha256']} corpus={observed_hash}"
            )
        if document.path != entry["ground_truth_path"]:
            raise ValueError(
                f"Semantic-hard manifest path mismatch for document {doc_id}: manifest={entry['ground_truth_path']} corpus={document.path}"
            )

        base_example = examples_by_doc_id.get(doc_id)
        if base_example is None:
            raise ValueError(f"Semantic-hard manifest references example for document {doc_id!r} not present in {base_loaded.source}/{language} examples")

        query_id = str(entry["query_id"])
        base_query_id = str(base_example.metadata.get("query_id"))
        if base_query_id != query_id:
            raise ValueError(f"Semantic-hard manifest query_id mismatch for document {doc_id}: manifest={query_id} base={base_query_id}")
        if base_example.query != entry["query"]:
            raise ValueError(f"Semantic-hard manifest query text drift for document {doc_id}: manifest query does not match base eval query")

        base_metadata = dict(base_example.metadata)
        metadata = {
            **base_metadata,
            "query_id": query_id,
            "ground_truth_document_id": doc_id,
            "semantic_hard": {
                "selection": SEMANTIC_HARD_SELECTION,
                "selection_ordinal": entry["selection_ordinal"],
                "hardness_definition": SEMANTIC_HARD_HARDNESS_DEFINITION,
                "query_sha256": entry["query_sha256"],
                "ground_truth_code_sha256": entry["ground_truth_code_sha256"],
                "ripgrep": entry["ripgrep"],
                "provenance": entry.get("provenance", {}),
            },
        }
        examples.append(
            EvalExample(
                query=str(entry["query"]),
                ground_truth_code=document.code,
                ground_truth_path=document.path,
                language=language,
                source=f"{base_loaded.source}-{SEMANTIC_HARD_SELECTION}",
                corpus_id=base_loaded.corpus_id,
                metadata=metadata,
            )
        )
        qrels[query_id] = {doc_id: 1}

    corpus = list(base_loaded.corpus)
    if corpus_sample_size is not None and corpus_sample_size < len(corpus):
        required_document_ids = {str(example.metadata["ground_truth_document_id"]) for example in examples}
        if len(required_document_ids) > corpus_sample_size:
            raise ValueError(
                f"corpus_sample_size={corpus_sample_size} is smaller than the {len(required_document_ids)} required semantic-hard positives"
            )
        required = [document for document in corpus if document.document_id in required_document_ids]
        negatives = [document for document in corpus if document.document_id not in required_document_ids]
        rng = random.Random(GLOBAL_RANDOM_SEED + 17)
        sampled_negative_ids = {
            document.document_id
            for document in rng.sample(negatives, k=min(len(negatives), corpus_sample_size - len(required)))
        }
        selected_ids = required_document_ids | sampled_negative_ids
        corpus = [document for document in corpus if document.document_id in selected_ids]

    metadata = dict(base_loaded.metadata)
    metadata.update(
        {
            "semantic_hard_selection": SEMANTIC_HARD_SELECTION,
            "semantic_hard_manifest_path": str(manifest_path or SEMANTIC_HARD_DIR / f"{language}.jsonl"),
            "semantic_hard_manifest_count": len(entries),
            "semantic_hard_min_entries": min_entries,
            "semantic_hard_sample_size": sample_size,
            "semantic_hard_query_count": len(examples),
            "semantic_hard_hardness_definition": "ripgrep_recall_at_5_eq_0_against_full_eval_corpus",
            "base_source": base_loaded.source,
            "base_full_example_count": base_loaded.metadata.get("full_example_count", len(base_loaded.examples)),
            "full_example_count": len(entries),
            "document_count": len(corpus),
            "corpus_sample_size": corpus_sample_size,
            "corpus_sampling_note": (
                "Corpus was deterministically reduced to semantic-hard positives plus random negatives for a bounded validation run."
                if corpus_sample_size is not None and corpus_sample_size < len(base_loaded.corpus)
                else None
            ),
            "qrel_count": len(qrels),
            "qrels": qrels,
            "methodology": (
                "Canonical semantic-hard JSONL subset selected from full-language eval examples where the harness ripgrep baseline "
                "missed the paired code document in top-5; corpus remains the full source-language eval corpus unless bounded."
            ),
        }
    )

    return LoadedBenchmark(
        source=f"{base_loaded.source}-{SEMANTIC_HARD_SELECTION}",
        language=language,
        revision=base_loaded.revision,
        corpus_id=base_loaded.corpus_id,
        corpus=tuple(corpus),
        examples=tuple(examples),
        metadata=metadata,
    )
