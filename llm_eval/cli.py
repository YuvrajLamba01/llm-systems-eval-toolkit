"""Command-line interface for the LLM inference systems evaluation toolkit."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from llm_eval.adapters.mock_engine import MockEngineAdapter
from llm_eval.adapters.vllm_client import VLLMEngineAdapter
from llm_eval.core.metrics import BenchmarkSummary
from llm_eval.core.workload import SyntheticPromptGenerator
from llm_eval.reporting.comparison import compare_summaries
from llm_eval.reporting.markdown import render_benchmark_markdown, render_comparison_markdown
from llm_eval.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="High-precision LLM inference performance, latency decomposition, and evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run streaming inference benchmarks.")
    bench_parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Target model identifier.",
    )
    bench_parser.add_argument(
        "--endpoint", type=str, default="http://localhost:8000/v1", help="vLLM server API base URL."
    )
    bench_parser.add_argument("--api-key", type=str, default="EMPTY", help="Optional API key.")
    bench_parser.add_argument(
        "--mock", action="store_true", help="Use deterministic mock engine without a live server."
    )
    bench_parser.add_argument(
        "--concurrency", type=int, default=4, help="Concurrent request limit."
    )
    bench_parser.add_argument(
        "--num-requests", type=int, default=16, help="Total benchmark requests."
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=32, help="Maximum generated tokens per request."
    )
    bench_parser.add_argument(
        "--request-rate",
        type=float,
        default=None,
        help="Target request arrival rate (req/s) for Poisson schedule.",
    )
    bench_parser.add_argument(
        "--synthetic-prompts",
        action="store_true",
        help="Generate synthetic prompts with variable context length.",
    )
    bench_parser.add_argument(
        "--prompt-tokens", type=int, default=128, help="Mean token length for synthetic prompts."
    )
    bench_parser.add_argument(
        "--output", type=Path, default=None, help="Filepath to write JSON summary results."
    )
    bench_parser.add_argument(
        "--report", type=Path, default=None, help="Filepath to write Markdown report."
    )
    bench_parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Filepath to export per-request timings to CSV.",
    )
    bench_parser.add_argument(
        "--prometheus-port",
        type=int,
        default=None,
        help="Port to host live Prometheus /metrics HTTP exposition server.",
    )

    # Subcommand: report
    report_parser = subparsers.add_parser(
        "report", help="Generate a Markdown report from a JSON benchmark file."
    )
    report_parser.add_argument(
        "--input", type=Path, required=True, help="Input benchmark JSON file."
    )
    report_parser.add_argument(
        "--output", type=Path, default=None, help="Output Markdown report file."
    )

    # Subcommand: compare
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two benchmark runs and detect regressions."
    )
    compare_parser.add_argument(
        "--baseline", type=Path, required=True, help="Baseline benchmark JSON file."
    )
    compare_parser.add_argument(
        "--candidate", type=Path, required=True, help="Candidate benchmark JSON file."
    )
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5.0%%).",
    )
    compare_parser.add_argument(
        "--p-value",
        type=float,
        default=None,
        help="Statistical significance alpha (e.g., 0.05). If set, regressions are gated on p < alpha.",
    )
    compare_parser.add_argument(
        "--output", type=Path, default=None, help="Filepath to save comparison Markdown report."
    )
    compare_parser.add_argument(
        "--fail-on-regression", action="store_true", help="Exit code 1 if regressions are detected."
    )

    return parser


def _write_trace_csv(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id",
                "prompt_tokens",
                "output_tokens",
                "queue_delay_ms",
                "ttft_ms",
                "total_latency_ms",
                "error",
            ]
        )
        for r in records:
            ttft_ms = round(r.ttft * 1000, 2) if r.ttft is not None else ""
            lat_ms = round(r.total_latency * 1000, 2) if r.total_latency is not None else ""
            q_ms = round(r.queue_delay_s * 1000, 2)
            writer.writerow(
                [
                    r.request_id,
                    r.prompt_tokens,
                    r.output_tokens,
                    q_ms,
                    ttft_ms,
                    lat_ms,
                    r.error or "",
                ]
            )


async def _run_benchmark_command(args: argparse.Namespace) -> int:
    if args.mock:
        adapter = MockEngineAdapter(
            prefill_latency_s=0.03,
            decode_latency_per_token_s=0.008,
        )
    else:
        adapter = VLLMEngineAdapter(
            base_url=args.endpoint,
            model_name=args.model,
            api_key=args.api_key,
        )

    runner = BenchmarkRunner(adapter=adapter, model_name=args.model)

    if args.synthetic_prompts:
        gen = SyntheticPromptGenerator()
        prompts = gen.generate_prompt_batch(
            num_requests=args.num_requests,
            mean_tokens=args.prompt_tokens,
        )
    else:
        prompts = [
            f"Performance evaluation prompt #{i} for throughput testing"
            for i in range(args.num_requests)
        ]

    rate_info = f", arrival_rate={args.request_rate} req/s" if args.request_rate else ""
    print(
        f"Executing benchmark: {len(prompts)} requests, concurrency={args.concurrency}{rate_info}, model={args.model}..."
    )

    if args.prometheus_port:
        from prometheus_client import CollectorRegistry

        from llm_eval.telemetry.prometheus_exporter import start_prometheus_server

        prom_reg = CollectorRegistry()
        start_prometheus_server(port=args.prometheus_port, registry=prom_reg)
        print(
            f"Prometheus metrics HTTP exporter initialized on port {args.prometheus_port} (/metrics)"
        )
    else:
        prom_reg = None

    summary = await runner.run_benchmark(
        prompts=prompts,
        concurrency=args.concurrency,
        max_tokens_per_request=args.max_tokens,
        request_rate=args.request_rate,
    )

    summary_dict = summary.to_dict()
    if prom_reg is not None:
        from llm_eval.telemetry.prometheus_exporter import export_summary_to_prometheus

        export_summary_to_prometheus(summary, model_name=args.model, registry=prom_reg)
        print(
            f"Summary metrics exported to live Prometheus registry on port {args.prometheus_port}."
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
        print(f"Benchmark JSON saved to {args.output}")

    if args.export_csv:
        _write_trace_csv(args.export_csv, summary.records)
        print(f"Request trace CSV saved to {args.export_csv}")

    md_report = render_benchmark_markdown(summary, title=f"LLM Benchmark: {args.model}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md_report, encoding="utf-8")
        print(f"Benchmark Markdown report saved to {args.report}")

    print("\n" + md_report)
    return 0


def _run_report_command(args: argparse.Namespace) -> int:
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    summary = BenchmarkSummary.from_dict(raw)
    report_md = render_benchmark_markdown(summary, title=f"Benchmark Report ({args.input.stem})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_md, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report_md)
    return 0


def _run_compare_command(args: argparse.Namespace) -> int:
    base_raw = json.loads(args.baseline.read_text(encoding="utf-8"))
    cand_raw = json.loads(args.candidate.read_text(encoding="utf-8"))

    baseline = BenchmarkSummary.from_dict(base_raw)
    candidate = BenchmarkSummary.from_dict(cand_raw)

    result = compare_summaries(
        baseline,
        candidate,
        regression_threshold_pct=args.threshold,
        alpha=args.p_value,
    )
    md = render_comparison_markdown(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Comparison report written to {args.output}")

    print("\n" + md)

    if args.fail_on_regression and result.has_regressions:
        print("ERROR: Performance regression threshold exceeded!", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return asyncio.run(_run_benchmark_command(args))
    elif args.command == "report":
        return _run_report_command(args)
    elif args.command == "compare":
        return _run_compare_command(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
