from __future__ import annotations

import json
from pathlib import Path

from llm_eval.cli import build_parser, main


def test_cli_parser_structure():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--mock",
            "--concurrency",
            "2",
            "--num-requests",
            "4",
            "--prometheus-port",
            "9100",
        ]
    )
    assert args.command == "benchmark"
    assert args.mock is True
    assert args.concurrency == 2
    assert args.num_requests == 4
    assert args.prometheus_port == 9100


def test_cli_benchmark_mock_execution(tmp_path: Path):
    output_json = tmp_path / "bench.json"
    output_md = tmp_path / "bench.md"

    exit_code = main(
        [
            "benchmark",
            "--mock",
            "--concurrency",
            "2",
            "--num-requests",
            "3",
            "--max-tokens",
            "4",
            "--output",
            str(output_json),
            "--report",
            str(output_md),
        ]
    )

    assert exit_code == 0
    assert output_json.exists()
    assert output_md.exists()

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["total_requests"] == 3
    assert data["successful_requests"] == 3
    assert "ttft_ms" in data


def test_cli_report_subcommand(tmp_path: Path):
    bench_json = tmp_path / "bench.json"
    bench_md = tmp_path / "generated_report.md"

    main(
        [
            "benchmark",
            "--mock",
            "--num-requests",
            "2",
            "--max-tokens",
            "2",
            "--output",
            str(bench_json),
        ]
    )

    exit_code = main(["report", "--input", str(bench_json), "--output", str(bench_md)])
    assert exit_code == 0
    assert bench_md.exists()
    content = bench_md.read_text(encoding="utf-8")
    assert "LLM Inference Benchmark" in content or "Benchmark Report" in content


def test_cli_compare_regression_exit_code(tmp_path: Path):
    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    report_file = tmp_path / "comparison.md"

    # Base: fast
    base_summary = {
        "total_requests": 10,
        "successful_requests": 10,
        "failed_requests": 0,
        "total_prompt_tokens": 100,
        "total_output_tokens": 200,
        "duration_seconds": 1.0,
        "request_throughput_rps": 10.0,
        "token_throughput_tps": 200.0,
        "ttft_ms": {"mean": 20.0, "p50": 20.0, "p90": 25.0, "p95": 28.0, "p99": 30.0},
        "tpot_ms": {"mean": 5.0, "p50": 5.0, "p90": 6.0, "p95": 7.0, "p99": 8.0},
        "itl_ms": {"mean": 5.0, "p50": 5.0, "p90": 6.0, "p95": 7.0, "p99": 8.0},
    }
    base_file.write_text(json.dumps(base_summary), encoding="utf-8")

    # Cand: 50% slower TTFT (severe regression)
    cand_summary = dict(base_summary)
    cand_summary["ttft_ms"] = {"mean": 35.0, "p50": 35.0, "p90": 40.0, "p95": 45.0, "p99": 50.0}
    cand_file.write_text(json.dumps(cand_summary), encoding="utf-8")

    # With --fail-on-regression, must return 1
    exit_code = main(
        [
            "compare",
            "--baseline",
            str(base_file),
            "--candidate",
            str(cand_file),
            "--threshold",
            "5.0",
            "--output",
            str(report_file),
            "--fail-on-regression",
        ]
    )

    assert exit_code == 1
    assert report_file.exists()
    assert "REGRESSION" in report_file.read_text(encoding="utf-8")


def test_cli_compare_with_p_value_gating(tmp_path: Path):
    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    report_file = tmp_path / "comp_pval.md"

    base_summary = {
        "total_requests": 20,
        "successful_requests": 20,
        "failed_requests": 0,
        "total_prompt_tokens": 100,
        "total_output_tokens": 200,
        "duration_seconds": 1.0,
        "request_throughput_rps": 20.0,
        "token_throughput_tps": 200.0,
        "ttft_ms": {"mean": 20.0, "std": 1.0, "p50": 20.0, "p90": 22.0, "p95": 23.0, "p99": 24.0},
        "tpot_ms": {"mean": 5.0, "std": 0.5, "p50": 5.0, "p90": 6.0, "p95": 6.5, "p99": 7.0},
        "itl_ms": {"mean": 5.0, "std": 0.5, "p50": 5.0, "p90": 6.0, "p95": 6.5, "p99": 7.0},
    }
    base_file.write_text(json.dumps(base_summary), encoding="utf-8")

    # Candidate has slight 8% mean TTFT increase (21.6ms), but large std so p > 0.01 (insignificant)
    cand_summary = dict(base_summary)
    cand_summary["ttft_ms"] = {
        "mean": 21.6,
        "std": 8.0,
        "p50": 21.6,
        "p90": 25.0,
        "p95": 27.0,
        "p99": 30.0,
    }
    cand_file.write_text(json.dumps(cand_summary), encoding="utf-8")

    # With strict alpha=0.001, insignificant regression should NOT fail
    exit_code = main(
        [
            "compare",
            "--baseline",
            str(base_file),
            "--candidate",
            str(cand_file),
            "--threshold",
            "5.0",
            "--p-value",
            "0.001",
            "--output",
            str(report_file),
            "--fail-on-regression",
        ]
    )
    assert exit_code == 0
    assert report_file.exists()
    assert "p-value" in report_file.read_text(encoding="utf-8")


def test_cli_benchmark_poisson_and_csv(tmp_path: Path):
    output_json = tmp_path / "poisson_bench.json"
    output_csv = tmp_path / "trace.csv"

    exit_code = main(
        [
            "benchmark",
            "--mock",
            "--concurrency",
            "2",
            "--num-requests",
            "3",
            "--max-tokens",
            "3",
            "--request-rate",
            "100.0",
            "--synthetic-prompts",
            "--prompt-tokens",
            "32",
            "--output",
            str(output_json),
            "--export-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    assert output_json.exists()
    assert output_csv.exists()

    csv_lines = output_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 4  # Header + 3 requests
    assert "request_id" in csv_lines[0]
