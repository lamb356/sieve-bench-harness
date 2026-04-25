"""Retriever implementations and report-table metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrieverReportMetadata:
    table: Literal["hero", "extended"]
    role: str
    role_label: str
    params: str
    display_name: str
    order: int


RETRIEVER_REPORT_METADATA: dict[str, RetrieverReportMetadata] = {
    "ripgrep": RetrieverReportMetadata(
        table="hero",
        role="latency_floor",
        role_label="Latency floor, exact-match baseline",
        params="—",
        display_name="ripgrep",
        order=10,
    ),
    "bm25": RetrieverReportMetadata(
        table="hero",
        role="classical_ir",
        role_label="Classical IR baseline",
        params="—",
        display_name="BM25",
        order=20,
    ),
    "unixcoder": RetrieverReportMetadata(
        table="hero",
        role="mid_size_neural_encoder",
        role_label="Mid-size neural encoder",
        params="126M",
        display_name="UniXcoder",
        order=30,
    ),
    "lateon-code-edge": RetrieverReportMetadata(
        table="hero",
        role="size_matched_retrieval_trained_primary",
        role_label="Size-matched retrieval-trained (primary)",
        params="17M",
        display_name="LateOn-Code-edge",
        order=40,
    ),
    "sieve": RetrieverReportMetadata(
        table="hero",
        role="our_model",
        role_label="Our model",
        params="4.2M",
        display_name="SIEVE",
        order=50,
    ),
    "codebert": RetrieverReportMetadata(
        table="extended",
        role="null_baseline",
        role_label="NULL BASELINE: off-the-shelf encoder without retrieval fine-tuning",
        params="125M",
        display_name="CodeBERT (pretrained features only)",
        order=110,
    ),
    "lateon-code": RetrieverReportMetadata(
        table="extended",
        role="sota_reference_larger_scale",
        role_label="SOTA reference at 10x scale; not a fair comparison with SIEVE's 4.2M param class",
        params="149M",
        display_name="LateOn-Code",
        order=120,
    ),
}
