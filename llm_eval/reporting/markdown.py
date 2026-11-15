"""Markdown formatting for benchmark runs and regression summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_eval.core.metrics import BenchmarkSummary
    from llm_eval.reporting.comparison import ComparisonResult


def render_benchmark_markdown(summary: BenchmarkSummary, title: str = "LLM Inference Benchmark") -> str:
    """Renders a single benchmark summary into a GitHub-Flavored Markdown report."""
    lines = [
        f"# {title}",
        "",
        "## Throughput & Request Overview",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Requests** | {summary.total_requests:,} |",
        f"| **Successful Requests** | {summary.successful_requests:,} |",
        f"| **Failed Requests** | {summary.failed_requests:,} |",
        f"| **Total Prompt Tokens** | {summary.total_prompt_tokens:,} |",
        f"| **Total Output Tokens** | {summary.total_output_tokens:,} |",
        f"| **Duration** | {summary.duration_seconds:.2f} s |",
        f"| **Request Throughput** | {summary.request_throughput:.2f} req/s |",
        f"| **Token Throughput** | {summary.token_throughput:.2f} tok/s |",
        "",
        "## Latency Decomposition (ms)",
        "",
        "| Metric | Mean (ms) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **TTFT** (Time to First Token) | {summary.ttft_mean*1000:.2f} | {summary.ttft_p50*1000:.2f} | {summary.ttft_p90*1000:.2f} | {summary.ttft_p95*1000:.2f} | {summary.ttft_p99*1000:.2f} |",
        f"| **TPOT** (Time per Output Token) | {summary.tpot_mean*1000:.2f} | {summary.tpot_p50*1000:.2f} | {summary.tpot_p90*1000:.2f} | {summary.tpot_p95*1000:.2f} | {summary.tpot_p99*1000:.2f} |",
        f"| **ITL** (Inter-Token Latency) | {summary.itl_mean*1000:.2f} | {summary.itl_p50*1000:.2f} | {summary.itl_p90*1000:.2f} | {summary.itl_p95*1000:.2f} | {summary.itl_p99*1000:.2f} |",
        "",
    ]
    return "\n".join(lines)


def render_comparison_markdown(result: ComparisonResult) -> str:
    """Renders a comparative evaluation between baseline and candidate benchmarks."""
    status_icon = "Passed" if not result.has_regressions else "Regression Detected"
    lines = [
        f"# Benchmark Comparison Report: {status_icon}",
        "",
        f"- **Baseline Duration**: {result.baseline.duration_seconds:.2f} s ({result.baseline.successful_requests} reqs)",
        f"- **Candidate Duration**: {result.candidate.duration_seconds:.2f} s ({result.candidate.successful_requests} reqs)",
        "",
        "## Throughput Comparison",
        "",
        "| Metric | Baseline | Candidate | Delta (%) | Status |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for item in result.throughput_items:
        delta_str = f"{item.delta_pct:+.1f}%" if item.baseline_val != 0 else "N/A"
        lines.append(
            f"| **{item.name}** | {item.baseline_val:.2f} | {item.candidate_val:.2f} | {delta_str} | {item.status_badge} |"
        )

    lines.extend([
        "",
        "## Latency Comparison (p50 / p95 / p99)",
        "",
        "| Metric | Baseline (ms) | Candidate (ms) | Delta (%) | Status |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for item in result.latency_items:
        delta_str = f"{item.delta_pct:+.1f}%" if item.baseline_val != 0 else "N/A"
        lines.append(
            f"| **{item.name}** | {item.baseline_val*1000:.2f} | {item.candidate_val*1000:.2f} | {delta_str} | {item.status_badge} |"
        )

    lines.append("")
    return "\n".join(lines)
