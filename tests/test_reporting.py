from __future__ import annotations

from llm_eval.core.metrics import BenchmarkSummary
from llm_eval.reporting.comparison import compare_summaries
from llm_eval.reporting.markdown import render_benchmark_markdown, render_comparison_markdown


def _create_sample_summary(req_tps: float = 10.0, tok_tps: float = 200.0, ttft_p50: float = 0.04) -> BenchmarkSummary:
    return BenchmarkSummary(
        total_requests=20,
        successful_requests=20,
        failed_requests=0,
        total_prompt_tokens=200,
        total_output_tokens=640,
        duration_seconds=2.0,
        request_throughput=req_tps,
        token_throughput=tok_tps,
        ttft_mean=ttft_p50 * 1.1,
        ttft_p50=ttft_p50,
        ttft_p90=ttft_p50 * 1.5,
        ttft_p95=ttft_p50 * 1.8,
        ttft_p99=ttft_p50 * 2.0,
        tpot_mean=0.015,
        tpot_p50=0.012,
        tpot_p90=0.018,
        tpot_p95=0.020,
        tpot_p99=0.025,
        itl_mean=0.014,
        itl_p50=0.012,
        itl_p90=0.017,
        itl_p95=0.019,
        itl_p99=0.022,
    )


def test_markdown_rendering():
    summary = _create_sample_summary()
    md = render_benchmark_markdown(summary, title="Test Benchmark")
    assert "# Test Benchmark" in md
    assert "TTFT" in md
    assert "40.00" in md  # 0.04s * 1000 = 40.0ms
    assert "200.00 tok/s" in md


def test_summary_serialization_roundtrip():
    original = _create_sample_summary()
    data = original.to_dict()
    reconstructed = BenchmarkSummary.from_dict(data)

    assert reconstructed.total_requests == original.total_requests
    assert reconstructed.successful_requests == original.successful_requests
    assert round(reconstructed.ttft_p50 * 1000, 2) == round(original.ttft_p50 * 1000, 2)
    assert round(reconstructed.token_throughput, 2) == round(original.token_throughput, 2)


def test_comparison_detection():
    baseline = _create_sample_summary(req_tps=10.0, tok_tps=200.0, ttft_p50=0.04)
    # Candidate has slower TTFT (regression) and lower throughput (regression)
    candidate_regressed = _create_sample_summary(req_tps=8.0, tok_tps=160.0, ttft_p50=0.06)

    comp = compare_summaries(baseline, candidate_regressed, regression_threshold_pct=5.0)
    assert comp.has_regressions is True

    report = render_comparison_markdown(comp)
    assert "REGRESSION" in report
    assert "Benchmark Comparison Report" in report

    # Candidate has faster TTFT and higher throughput (improvement)
    candidate_improved = _create_sample_summary(req_tps=15.0, tok_tps=300.0, ttft_p50=0.02)
    comp_improved = compare_summaries(baseline, candidate_improved, regression_threshold_pct=5.0)
    assert comp_improved.has_regressions is False
    assert any(item.is_improvement for item in comp_improved.throughput_items)
