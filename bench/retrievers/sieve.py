from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bench.loaders.base import CodeDocument
from bench.metrics.performance import ru_maxrss_to_mebibytes, summarize_latency
from bench.retrievers.base import SearchResult


class SieveRetriever:
    name = "sieve"
    display_name = "SIEVE"

    def __init__(
        self,
        *,
        binary_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        query_onnx_path: str | Path | None = None,
        doc_onnx_path: str | Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        build_release: bool = True,
    ) -> None:
        self._binary_path_arg = Path(binary_path) if binary_path is not None else None
        repo_path_value = repo_path or os.environ.get("SIEVE_REPO")
        self._repo_path = Path(repo_path_value) if repo_path_value else None
        self._query_onnx_path = Path(query_onnx_path) if query_onnx_path is not None else None
        self._doc_onnx_path = Path(doc_onnx_path) if doc_onnx_path is not None else None
        self._extra_env = dict(extra_env or {})
        self._build_release = build_release
        self._binary_path: Path | None = None
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self._source_root: Path | None = None
        self._index_root: Path | None = None
        self._documents_by_path: dict[str, CodeDocument] = {}
        self._latency_samples_ms: list[float] = []
        self._child_peak_rss_mb = 0.0
        self._index_seconds = 0.0

    def index(self, corpus: Sequence[CodeDocument]) -> None:
        documents = tuple(corpus)
        if not documents:
            raise ValueError("SieveRetriever requires a non-empty corpus")
        self._latency_samples_ms.clear()
        self._documents_by_path.clear()
        self._workdir = tempfile.TemporaryDirectory(prefix="sieve-bench-")
        self._source_root = Path(self._workdir.name) / "corpus"
        self._source_root.mkdir(parents=True)
        for ordinal, document in enumerate(documents):
            relative_path = self._materialized_relative_path(document, ordinal)
            output_path = self._source_root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(document.code, encoding="utf-8")
            self._documents_by_path[relative_path.as_posix()] = document
        self._index_root = self._source_root / ".sieve"
        started = time.perf_counter()
        self._run_sieve(["index", str(self._source_root)])
        self._index_seconds = time.perf_counter() - started

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._index_root is None:
            raise RuntimeError("SieveRetriever.search() called before index()")
        if k <= 0:
            return []
        started = time.perf_counter()
        completed = self._run_sieve(
            [
                "search",
                query,
                "--index",
                str(self._index_root),
                "--top",
                str(k),
                "--format",
                "json",
            ]
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"sieve search did not return valid JSON: {exc}\nstdout={completed.stdout!r}\nstderr={completed.stderr!r}") from exc
        if not isinstance(payload, list):
            raise RuntimeError(f"sieve search JSON must be a list, got {type(payload).__name__}")
        results: list[SearchResult] = []
        for item in payload[:k]:
            if not isinstance(item, dict):
                continue
            relative_path = str(item.get("path", ""))
            document = self._documents_by_path.get(relative_path)
            if document is None:
                continue
            score = float(item.get("score", 0.0))
            results.append(
                SearchResult(
                    document_id=document.document_id,
                    path=document.path,
                    score=score,
                    code=document.code,
                    metadata={
                        "sieve_path": relative_path,
                        "layer": item.get("layer"),
                        "line": item.get("line"),
                        "chunk_id": item.get("chunk_id"),
                        "byte_range": item.get("byte_range"),
                        "snippet": item.get("snippet"),
                    },
                )
            )
        return results

    def latency_ms(self) -> dict[str, float]:
        if not self._latency_samples_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return summarize_latency(self._latency_samples_ms)

    def embedding_metadata(self) -> dict[str, Any]:
        return {
            "interface": "sieve-cli-subprocess",
            "binary_path": str(self._resolve_binary()),
            "repo_path": str(self._repo_path),
            "index_build_seconds": self._index_seconds,
            "child_process_peak_rss_mb": self._child_peak_rss_mb,
        }

    def _run_sieve(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = [str(self._resolve_binary()), *args]
        before_rusage = _children_maxrss_mb()
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=self._env())
        after_rusage = _children_maxrss_mb()
        # ru_maxrss for RUSAGE_CHILDREN is a high-water mark for waited children on Linux.
        self._child_peak_rss_mb = max(self._child_peak_rss_mb, before_rusage, after_rusage)
        if completed.returncode != 0:
            raise RuntimeError(
                "sieve subprocess failed with exit code "
                f"{completed.returncode}: {' '.join(command)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
            )
        return completed

    def _resolve_binary(self) -> Path:
        if self._binary_path is not None:
            return self._binary_path
        if self._binary_path_arg is not None:
            if not self._binary_path_arg.is_file():
                raise FileNotFoundError(f"configured sieve binary does not exist: {self._binary_path_arg}")
            self._binary_path = self._binary_path_arg
            return self._binary_path
        env_binary = os.environ.get("SIEVE_BINARY")
        if env_binary:
            binary = Path(env_binary)
            if not binary.is_file():
                raise FileNotFoundError(f"SIEVE_BINARY does not exist: {binary}")
            self._binary_path = binary
            return self._binary_path
        which_binary = shutil.which("sieve")
        if which_binary is not None:
            self._binary_path = Path(which_binary)
            return self._binary_path
        if self._repo_path is None:
            raise FileNotFoundError(
                "could not locate sieve binary on PATH; set SIEVE_BINARY or SIEVE_REPO"
            )
        if not self._repo_path.joinpath("Cargo.toml").is_file():
            raise FileNotFoundError(
                "could not locate sieve binary; set SIEVE_BINARY or SIEVE_REPO "
                f"(tried repo {self._repo_path})"
            )
        if self._build_release:
            subprocess.run(
                ["cargo", "build", "--release", "-p", "sieve-cli", "--features", "semantic"],
                cwd=self._repo_path,
                text=True,
                check=True,
            )
        metadata = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        target_directory = Path(json.loads(metadata.stdout)["target_directory"])
        binary = target_directory / "release" / "sieve"
        if not binary.is_file():
            raise FileNotFoundError(f"built sieve binary not found at {binary}")
        self._binary_path = binary
        return self._binary_path

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self._extra_env)
        if self._query_onnx_path is not None:
            env["SIEVE_ENCODER_QUERY_ONNX"] = str(self._query_onnx_path)
        if self._doc_onnx_path is not None:
            env["SIEVE_ENCODER_DOC_ONNX"] = str(self._doc_onnx_path)
        return env

    @staticmethod
    def _materialized_relative_path(document: CodeDocument, ordinal: int) -> Path:
        source_path = Path(document.path)
        suffix = source_path.suffix or _language_suffix(document.language)
        stem = _safe_stem(source_path.stem or document.document_id)
        return Path("docs") / f"{ordinal:06d}_{stem}{suffix}"


def _language_suffix(language: str) -> str:
    return {
        "python": ".py",
        "rust": ".rs",
        "typescript": ".ts",
        "javascript": ".js",
        "java": ".java",
        "go": ".go",
    }.get(language.lower(), ".txt")


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:80] or "document"


def _children_maxrss_mb() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix fallback
        return 0.0
    return ru_maxrss_to_mebibytes(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


# Backward-compatible import name for callers that still import the Phase B stub class.
SieveStubRetriever = SieveRetriever
