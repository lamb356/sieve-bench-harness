from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from bench.loaders.base import CodeDocument
from bench.retrievers.base import SearchResult


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "arguments",
    "be",
    "by",
    "code",
    "def",
    "example",
    "for",
    "from",
    "function",
    "get",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "return",
    "returns",
    "set",
    "should",
    "that",
    "the",
    "this",
    "to",
    "use",
    "with",
}


def _safe_relative_path(path: str) -> Path:
    candidate = Path(path)
    safe_parts = [part for part in candidate.parts if part not in {"", ".", ".."}]
    if not safe_parts:
        return Path("doc.py")
    return Path(*safe_parts)


def tokenize_query(query: str, *, limit: int = 12) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(query.lower()):
        token = match.group(0)
        if token in _STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    if tokens:
        return tokens
    fallback = " ".join(query.strip().lower().split())
    return [fallback[:128]] if fallback else []


class RipgrepRetriever:
    name = "ripgrep"

    def __init__(self, *, index_root: Path) -> None:
        self.binary = shutil.which("rg")
        if self.binary is None:
            raise FileNotFoundError("ripgrep binary 'rg' is required")
        self.index_root = index_root
        self._documents_by_path: dict[str, CodeDocument] = {}
        self._document_frequency: dict[str, int] = defaultdict(int)
        self._document_count = 0
        self._latency_samples_ms: list[float] = []

    def index(self, corpus: list[CodeDocument] | tuple[CodeDocument, ...]) -> None:
        if self.index_root.exists():
            shutil.rmtree(self.index_root)
        self.index_root.mkdir(parents=True, exist_ok=True)
        self._documents_by_path.clear()
        self._document_frequency = defaultdict(int)
        self._document_count = len(corpus)
        self._latency_samples_ms.clear()
        for document in corpus:
            relative_path = _safe_relative_path(document.path)
            destination = self.index_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(document.code, encoding="utf-8")
            self._documents_by_path[str(relative_path)] = document
            unique_tokens = set(tokenize_query(document.code, limit=10_000))
            for token in unique_tokens:
                self._document_frequency[token] += 1

    def search(self, query: str, k: int) -> list[SearchResult]:
        tokens = tokenize_query(query)
        if not tokens:
            self._latency_samples_ms.append(0.0)
            return []
        token_weights = {
            token: math.log((self._document_count + 1.0) / (self._document_frequency.get(token, 0) + 1.0)) + 1.0
            for token in tokens
        }
        weighted_tokens = sorted(token_weights.items(), key=lambda item: (-item[1], item[0]))[:6]
        active_tokens = [token for token, _weight in weighted_tokens]
        active_weights = {token: token_weights[token] for token in active_tokens}

        command = [
            self.binary,
            "--json",
            "--fixed-strings",
            "--no-heading",
            "--color",
            "never",
            "--line-number",
            "-i",
        ]
        for token in active_tokens:
            command.extend(["-e", token])
        command.append(str(self.index_root))

        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"ripgrep search failed: {completed.stderr.strip()}")

        file_scores: dict[str, float] = defaultdict(float)
        file_token_hits: dict[str, set[str]] = defaultdict(set)
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") != "match":
                continue
            relative_path = str(Path(payload["data"]["path"]["text"]).relative_to(self.index_root))
            matched_tokens = {
                str(submatch["match"]["text"]).lower()
                for submatch in payload["data"]["submatches"]
                if str(submatch["match"]["text"]).lower() in active_weights
            }
            for token in matched_tokens - file_token_hits[relative_path]:
                file_scores[relative_path] += active_weights[token]
            file_token_hits[relative_path].update(matched_tokens)

        ranked_paths = sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            SearchResult(
                document_id=self._documents_by_path[path].document_id,
                path=self._documents_by_path[path].path,
                score=score,
                code=self._documents_by_path[path].code,
                metadata={"matched_path": path},
            )
            for path, score in ranked_paths
        ]

    def latency_ms(self) -> dict[str, float]:
        if not self._latency_samples_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(self._latency_samples_ms)

        def percentile(fraction: float) -> float:
            if len(ordered) == 1:
                return ordered[0]
            index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
            return ordered[index]

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }
