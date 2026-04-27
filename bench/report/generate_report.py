from __future__ import annotations

import csv
import html
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import orjson
from jinja2 import Environment, FileSystemLoader, select_autoescape

from bench.retrievers import RETRIEVER_REPORT_METADATA, RetrieverReportMetadata


def _environment(template_root: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_root)),
        autoescape=select_autoescape(enabled_extensions=("html",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def write_results_json(payload: dict[str, Any], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n")


def write_hero_table(payload: dict[str, Any], *, template_root: Path, output_path: Path) -> None:
    summary = payload["summary"]
    environment = _environment(template_root)
    template = environment.get_template("hero_table.md.j2")
    rendered = template.render(
        source=summary["source"],
        language=summary["language"],
        retriever=summary["retriever"],
        query_count=summary["query_count"],
        contamination_rejected_count=summary["contamination_rejected_count"],
        recall_at_5=f"{summary['recall@5']:.3f}",
        contamination_flag="yes" if summary["contamination_flag"] else "no",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def _format_metric(value: Any) -> str:
    return f"{float(value):.3f}"


def _format_latency(value: Any) -> str:
    return f"{float(value):.2f} ms"


def _format_throughput(value: Any) -> str:
    return f"{float(value):.2f} q/s"


def _format_memory(value: Any) -> str:
    return f"{float(value):.2f} MB"


def _phase_label(payload: dict[str, Any]) -> str:
    phase = str(payload.get("benchmark", {}).get("phase", "B-v2"))
    if phase == "B":
        return "Phase B"
    if phase.startswith("B-"):
        return "Phase " + phase.replace("-", " ", 1)
    return f"Phase {phase}"


def _is_cpu_subprocess_memory_row(row: dict[str, Any]) -> bool:
    measurement = row.get("memory_measurement")
    if not isinstance(measurement, dict):
        return False
    process = measurement.get("process")
    if not isinstance(process, dict) or process.get("mode") != "subprocess":
        return False
    total = measurement.get("total")
    return isinstance(total, dict) and str(total.get("backend", "")).startswith("cpu-")


def _memory_diagnostic_warnings(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("retriever_summaries", [])
    warnings: list[str] = []
    zero_cpu_rows = [
        row
        for row in rows
        if _is_cpu_subprocess_memory_row(row) and float(row.get("memory_mb", 0.0)) == 0.0
    ]
    if len(zero_cpu_rows) >= 2:
        names = ", ".join(_metadata_for(row).display_name for row in zero_cpu_rows)
        warnings.append(
            "possible memory measurement bug: "
            f"CPU subprocess retrievers {names} report 0.00 MB delta RSS; verify current-RSS baselines are not hiding retriever allocations"
        )
    for left, right in combinations(rows, 2):
        if "memory_mb" not in left or "memory_mb" not in right:
            continue
        left_memory = float(left["memory_mb"])
        right_memory = float(right["memory_mb"])
        if left_memory < 1.0 or right_memory < 1.0:
            continue
        if abs(left_memory - right_memory) < 1.0:
            left_name = _metadata_for(left).display_name
            right_name = _metadata_for(right).display_name
            warnings.append(
                "possible memory measurement bug: "
                f"retriever {left_name}={left_memory:.2f} MB and {right_name}={right_memory:.2f} MB report nearly-identical memory"
            )
    return warnings


def _append_metric_row(lines: list[str], row: dict[str, Any], metadata: RetrieverReportMetadata) -> None:
    lines.append(
        "| "
        + " | ".join(
            [
                metadata.display_name,
                metadata.role_label,
                metadata.params,
                _format_metric(row["recall@1"]),
                _format_metric(row["recall@5"]),
                _format_metric(row["recall@10"]),
                _format_metric(row["mrr@10"]),
                _format_metric(row["ndcg@10"]),
                _format_latency(row["p50"]),
                _format_latency(row["p95"]),
                _format_throughput(row["throughput_qps"]),
                _format_memory(row.get("memory_mb", 0.0)),
            ]
        )
        + " |"
    )


def _fallback_metadata(row: dict[str, Any]) -> RetrieverReportMetadata:
    return RetrieverReportMetadata(
        table=str(row.get("table", "hero")),  # type: ignore[arg-type]
        role=str(row.get("role", "baseline")),
        role_label=str(row.get("role_label", row.get("role", "Baseline"))),
        params=str(row.get("params", "—")),
        display_name=str(row.get("display_name", row.get("retriever", "unknown"))),
        order=int(row.get("order", 999)),
    )


def _metadata_for(row: dict[str, Any]) -> RetrieverReportMetadata:
    base = RETRIEVER_REPORT_METADATA.get(str(row.get("retriever")))
    if base is None:
        base = _fallback_metadata(row)
    return RetrieverReportMetadata(
        table=str(row.get("table_override", base.table)),  # type: ignore[arg-type]
        role=str(row.get("role_override", base.role)),
        role_label=str(row.get("role_label_override", base.role_label)),
        params=str(row.get("params_override", base.params)),
        display_name=str(row.get("display_name_override", base.display_name)),
        order=int(row.get("order_override", base.order)),
    )


def _language_title(language: str) -> str:
    if language.lower() == "typescript":
        return "TypeScript"
    return language.capitalize()


def _rows_for_table(payload: dict[str, Any], table: str) -> list[tuple[dict[str, Any], RetrieverReportMetadata]]:
    rows = []
    for row in payload["retriever_summaries"]:
        metadata = _metadata_for(row)
        row_table = str(row.get("table", metadata.table))
        if row_table == table:
            rows.append((row, metadata))
    return sorted(rows, key=lambda item: item[1].order)


def render_phase_b_hero_table(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    phase_label = _phase_label(payload)
    language_title = _language_title(str(summary["language"]))
    benchmark = payload.get("benchmark", {})
    sieve_summary = next((row for row in payload.get("retriever_summaries", []) if row.get("retriever") == "sieve"), {})
    sieve_embedding = sieve_summary.get("embedding") if isinstance(sieve_summary.get("embedding"), dict) else {}
    if str(summary.get("language")) == "typescript" and (
        sieve_embedding.get("interface") == "pending-sieve-placeholder" or sieve_embedding.get("route_status") == "sieve-cli-unavailable"
    ):
        sieve_note = (
            "The SIEVE TypeScript row is a zero-recall pending placeholder because the SIEVE CLI route was unavailable; "
            "set SIEVE_BINARY or SIEVE_REPO and replace random/local ONNX exports with Phase 1 weights before quality claims."
        )
    else:
        sieve_note = (
            "The SIEVE row calls the real Rust engine via the existing `sieve index`/`sieve search --format json` CLI; "
            "Phase 1 weights should replace random/local ONNX exports before final quality claims."
        )
    lines = [
        f"# {phase_label} {language_title} benchmark — {summary['source']} / {summary['language']}",
        "",
        f"Queries: {summary['query_count']}  ",
        f"Corpus documents: {summary['corpus_document_count']}  ",
        f"Contamination rejected: {summary['contamination_rejected_count']}",
        "",
        "This benchmark frames SIEVE as a local-first, small-model code retrieval system for agent deployment. The hero table compares deployment-relevant competitors on the size/latency/memory Pareto frontier rather than treating unbounded model size as a fair baseline.",
        "",
        "## Hero Table: Size-Matched Competitors",
        "",
        "| Retriever | Role | Params | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | p50 latency | p95 latency | Throughput | Memory |",
        "|---|---|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for row, metadata in _rows_for_table(payload, "hero"):
        _append_metric_row(lines, row, metadata)

    lines.extend(
        [
            "",
            "Quality claims for SIEVE should be read from the hero table. The extended rows below are reference points only, not apples-to-apples competitors against SIEVE's 4.2M-parameter deployment class.",
            "",
            "Note: LateOn's multi-vector MaxSim brute-force scoring is inherently slower than single-vector cosine; production deployment would use PLAID indexing, which is intentionally not benchmarked in Phase B v3.",
            "",
            "## Extended Table: Reference Baselines",
            "",
            "| Retriever | Role | Params | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | p50 latency | p95 latency | Throughput | Memory |",
            "|---|---|---:|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row, metadata in _rows_for_table(payload, "extended"):
        _append_metric_row(lines, row, metadata)

    diagnostic_warnings = _memory_diagnostic_warnings(payload)
    if diagnostic_warnings:
        lines.extend(["", "## Diagnostics", ""])
        lines.extend(f"- {warning}" for warning in diagnostic_warnings)

    findings = summary.get("findings") or []
    if findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.extend(["", "## Findings", "", "- None."])

    lines.extend(["", "## Notes", ""])
    if str(payload.get("benchmark", {}).get("phase")) == "B.5" and str(summary.get("language")) == "python":
        lines.extend(
            [
                "- Methodology: Phase B v3 is the semantic-hard subset: queries resistant to literal match.",
                "- Methodology: Phase B.5 is the full CodeSearchNet Python eval distribution, a mixed semantic-hard + raw-surface workload representative of real agent search.",
                "- Real-world agent retrieval performance lives between these two numbers, weighted toward Phase B.5 when agents see typical query distributions.",
            ]
        )
    if str(summary.get("language")) == "typescript":
        lines.extend(
            [
                "- Methodology: `Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2` is used as the public canonical TypeScript `.ts` NL-to-code route.",
                "- Methodology: An official CodeXGLUE/CoIR/CodeSearchNet TypeScript split was not available with comparable public retrieval qrels; this route uses the pinned permissively licensed test split directly with one paired function relevant per query.",
                f"- Methodology: Dataset language is `{benchmark.get('dataset_language', 'typescript')}`; eval split is `{benchmark.get('eval_split', 'typescript-treesitter-dedupe-test')}`.",
                "- SIEVE TypeScript rows are labeled Phase 1 weights pending until real trained ONNX exports replace the current random/local weights.",
            ]
        )
        row_license_set = benchmark.get("row_license_set") or []
        if row_license_set:
            lines.append(
                f"- Provenance: Dataset card license is `{benchmark.get('dataset_card_license', 'apache-2.0')}`; row license set is `{', '.join(str(item) for item in row_license_set)}` across {benchmark.get('unique_repo_count', 'unknown')} repositories."
            )
        if benchmark.get("corpus_sampling_note"):
            lines.append(
                f"- Sampling: {benchmark['corpus_sampling_note']} Full eval examples: {benchmark.get('full_example_count')}; benchmark corpus documents: {summary.get('corpus_document_count')}; corpus sample size: {benchmark.get('corpus_sample_size')}."
            )
    lines.extend(
        [
            f"- {phase_label} uses normalized `document.index_text` for all retrievers to keep the benchmark surface aligned with Phase B v1.",
            "- CodeBERT is an explicit null baseline: base pretrained features without retrieval fine-tuning, routed to the extended table.",
            "- UniXcoder uses the required `<encoder-only>` token wrapper before mean pooling and cosine ranking.",
            "- LateOn-Code-edge and LateOn-Code use pinned public Hugging Face revisions with PyLate ColBERT-style multi-vector embeddings and brute-force MaxSim scoring for this phase.",
            "- CPU-only rows report retriever marginal delta RSS from isolated subprocesses; `results.json` includes baseline, index, search, total, and subprocess PID details.",
            f"- {sieve_note}",
        ]
    )
    return "\n".join(lines) + "\n"


def _json_cell(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, sort_keys=True)


def write_benchmark_csv(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(row.get(key)) for key in fieldnames})


def write_interactive_html(payload: dict[str, Any], *, output_path: Path) -> None:
    table = render_phase_b_hero_table(payload)
    phase_label = _phase_label(payload)
    escaped_json = html.escape(json.dumps(payload, indent=2, sort_keys=True))
    escaped_table = html.escape(table)
    language_title = _language_title(str(payload.get("summary", {}).get("language", "python")))
    html_body = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>{phase_label} {language_title} benchmark</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    pre {{ background: #111827; color: #f9fafb; padding: 1rem; overflow-x: auto; border-radius: 0.5rem; }}
  </style>
</head>
<body>
  <h1>{phase_label} {language_title} benchmark</h1>
  <h2>Markdown table</h2>
  <pre>{escaped_table}</pre>
  <h2>Raw JSON</h2>
  <pre>{escaped_json}</pre>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_body, encoding="utf-8")


def write_phase_b_reports(payload: dict[str, Any], *, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "results.json"
    benchmark_table = output_dir / "benchmark-table.md"
    benchmark_csv = output_dir / "benchmark-full.csv"
    interactive_html = output_dir / "interactive.html"

    write_results_json(payload, output_path=results_json)
    benchmark_table.write_text(render_phase_b_hero_table(payload), encoding="utf-8")
    write_benchmark_csv(payload.get("rows", []), output_path=benchmark_csv)
    write_interactive_html(payload, output_path=interactive_html)
    return {
        "results_json": results_json,
        "benchmark_table": benchmark_table,
        "benchmark_csv": benchmark_csv,
        "interactive_html": interactive_html,
    }
