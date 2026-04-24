from __future__ import annotations

import csv
import html
import json
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
    return RETRIEVER_REPORT_METADATA.get(str(row.get("retriever")), _fallback_metadata(row))


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
    lines = [
        f"# Phase B v2 Python benchmark — {summary['source']} / {summary['language']}",
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

    lines.extend(
        [
            "",
            "Quality claims for SIEVE should be read from the hero table. The extended rows below are reference points only, not apples-to-apples competitors against SIEVE's 4.2M-parameter deployment class.",
            "",
            "Note: LateOn's multi-vector MaxSim brute-force scoring is inherently slower than single-vector cosine; production deployment would use PLAID indexing, which is intentionally not benchmarked in Phase B v2.",
            "",
            "## Extended Table: Reference Baselines",
            "",
            "| Retriever | Role | Params | Recall@5 | MRR@10 | p50 latency | Memory |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for row, metadata in _rows_for_table(payload, "extended"):
        lines.append(
            "| "
            + " | ".join(
                [
                    metadata.display_name,
                    metadata.role_label,
                    metadata.params,
                    _format_metric(row["recall@5"]),
                    _format_metric(row["mrr@10"]),
                    _format_latency(row["p50"]),
                    _format_memory(row.get("memory_mb", 0.0)),
                ]
            )
            + " |"
        )

    findings = summary.get("findings") or []
    if findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.extend(["", "## Findings", "", "- None."])

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Phase B v2 uses normalized `document.index_text` for all retrievers to keep the benchmark surface aligned with Phase B v1.",
            "- CodeBERT is an explicit null baseline: base pretrained features without retrieval fine-tuning, routed to the extended table.",
            "- UniXcoder uses the required `<encoder-only>` token wrapper before mean pooling and cosine ranking.",
            "- LateOn-Code-edge and LateOn-Code use pinned public Hugging Face revisions with PyLate ColBERT-style multi-vector embeddings and brute-force MaxSim scoring for this phase.",
            "- CPU-only rows report Linux/WSL process RSS high-water via `resource.getrusage`; detailed baseline/delta measurements are preserved in `results.json`.",
            "- The SIEVE row is a deterministic query-hash-seeded random stub, not the real SIEVE integration.",
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
    escaped_json = html.escape(json.dumps(payload, indent=2, sort_keys=True))
    escaped_table = html.escape(table)
    html_body = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Phase B v2 Python benchmark</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    pre {{ background: #111827; color: #f9fafb; padding: 1rem; overflow-x: auto; border-radius: 0.5rem; }}
  </style>
</head>
<body>
  <h1>Phase B v2 Python benchmark</h1>
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
