"""Markdown formatting for benchmark runs and regression summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_eval.reporting.histogram import render_ascii_histogram

if TYPE_CHECKING:
    from llm_eval.core.metrics import BenchmarkSummary
    from llm_eval.reporting.comparison import ComparisonResult


def render_benchmark_markdown(
    summary: BenchmarkSummary, title: str = "LLM Inference Benchmark"
) -> str:
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
        f"| **Queue Delay (Mean)** | {summary.queue_delay_mean * 1000:.2f} ms |",
        f"| **Queue Delay (p95)** | {summary.queue_delay_p95 * 1000:.2f} ms |",
        "",
        "## Latency Decomposition (ms)",
        "",
        "| Metric | Mean (ms) | Std (ms) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **TTFT** (Time to First Token) | {summary.ttft_mean * 1000:.2f} | {summary.ttft_std * 1000:.2f} | {summary.ttft_p50 * 1000:.2f} | {summary.ttft_p90 * 1000:.2f} | {summary.ttft_p95 * 1000:.2f} | {summary.ttft_p99 * 1000:.2f} |",
        f"| **TPOT** (Time per Output Token) | {summary.tpot_mean * 1000:.2f} | {summary.tpot_std * 1000:.2f} | {summary.tpot_p50 * 1000:.2f} | {summary.tpot_p90 * 1000:.2f} | {summary.tpot_p95 * 1000:.2f} | {summary.tpot_p99 * 1000:.2f} |",
        f"| **ITL** (Inter-Token Latency) | {summary.itl_mean * 1000:.2f} | {summary.itl_std * 1000:.2f} | {summary.itl_p50 * 1000:.2f} | {summary.itl_p90 * 1000:.2f} | {summary.itl_p95 * 1000:.2f} | {summary.itl_p99 * 1000:.2f} |",
        "",
    ]

    if summary.raw_ttft_list:
        lines.extend(
            [
                "## TTFT Tail Latency Distribution",
                "",
                "```text",
                render_ascii_histogram(summary.raw_ttft_list, num_bins=5, unit="ms"),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def render_comparison_markdown(result: ComparisonResult) -> str:
    """Renders a comparative evaluation between baseline and candidate benchmarks."""
    status_icon = "Passed" if not result.has_regressions else "Regression Detected"
    has_p_values = any(i.p_value is not None for i in result.latency_items)

    lines = [
        f"# Benchmark Comparison Report: {status_icon}",
        "",
        f"- **Baseline Duration**: {result.baseline.duration_seconds:.2f} s ({result.baseline.successful_requests} reqs)",
        f"- **Candidate Duration**: {result.candidate.duration_seconds:.2f} s ({result.candidate.successful_requests} reqs)",
    ]
    if result.alpha is not None:
        lines.append(f"- **Significance Alpha**: {result.alpha:.3f}")
    lines.extend(
        [
            "",
            "## Throughput Comparison",
            "",
            "| Metric | Baseline | Candidate | Delta (%) | Status |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for item in result.throughput_items:
        delta_str = f"{item.delta_pct:+.1f}%" if item.baseline_val != 0 else "N/A"
        lines.append(
            f"| **{item.name}** | {item.baseline_val:.2f} | {item.candidate_val:.2f} | {delta_str} | {item.status_badge} |"
        )

    lines.append("")
    lines.append("## Latency Comparison (p50 / p95 / p99)")
    lines.append("")

    if has_p_values:
        lines.append(
            "| Metric | Baseline (ms) | Candidate (ms) | Delta (%) | p-value | Effect Size (d) | Status |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for item in result.latency_items:
            delta_str = f"{item.delta_pct:+.1f}%" if item.baseline_val != 0 else "N/A"
            p_val_str = f"{item.p_value:.4f}" if item.p_value is not None else "-"
            d_str = f"{item.effect_size_d:+.2f}" if item.effect_size_d is not None else "-"
            lines.append(
                f"| **{item.name}** | {item.baseline_val * 1000:.2f} | {item.candidate_val * 1000:.2f} | {delta_str} | {p_val_str} | {d_str} | {item.status_badge} |"
            )
    else:
        lines.append("| Metric | Baseline (ms) | Candidate (ms) | Delta (%) | Status |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for item in result.latency_items:
            delta_str = f"{item.delta_pct:+.1f}%" if item.baseline_val != 0 else "N/A"
            lines.append(
                f"| **{item.name}** | {item.baseline_val * 1000:.2f} | {item.candidate_val * 1000:.2f} | {delta_str} | {item.status_badge} |"
            )

    lines.append("")
    return "\n".join(lines)
