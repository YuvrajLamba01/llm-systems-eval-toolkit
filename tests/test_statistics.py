"""Tests for statistical hypothesis testing and regression significance gating."""

import numpy as np

from llm_eval.core.metrics import BenchmarkSummary
from llm_eval.core.statistics import (
    compare_empirical_distributions,
    compute_cohens_d,
    welch_t_test_from_stats,
)
from llm_eval.reporting.comparison import compare_summaries
from llm_eval.reporting.markdown import render_comparison_markdown


def test_compute_cohens_d():
    # Two identical distributions have effect size 0
    d_zero = compute_cohens_d(mean1=10.0, std1=2.0, n1=30, mean2=10.0, std2=2.0, n2=30)
    assert abs(d_zero) < 1e-6

    # Large difference
    d_large = compute_cohens_d(mean1=10.0, std1=2.0, n1=30, mean2=15.0, std2=2.0, n2=30)
    assert d_large > 2.0


def test_welch_t_test_from_stats_significance():
    # Clear significant difference (mean 10 vs mean 20 with small std)
    res = welch_t_test_from_stats(
        mean1=10.0,
        std1=1.0,
        n1=50,
        mean2=20.0,
        std2=1.0,
        n2=50,
        alpha=0.01,
    )
    assert res.p_value < 1e-5
    assert res.is_significant is True
    assert res.effect_size_d > 5.0

    # Noise / insignificant difference (mean 10.0 vs 10.05 with large std)
    res_noise = welch_t_test_from_stats(
        mean1=10.0,
        std1=5.0,
        n1=20,
        mean2=10.05,
        std2=5.0,
        n2=20,
        alpha=0.05,
    )
    assert res_noise.p_value > 0.05
    assert res_noise.is_significant is False


def test_compare_empirical_distributions():
    rng = np.random.default_rng(42)
    sample_a = list(rng.normal(loc=0.05, scale=0.005, size=100))
    # Sample B is heavily shifted to simulate 2x regression
    sample_b = list(rng.normal(loc=0.10, scale=0.005, size=100))

    res = compare_empirical_distributions(sample_a, sample_b, alpha=0.01)
    assert res.p_value < 1e-4
    assert res.is_significant is True
    assert res.effect_size_d > 0


def test_compare_summaries_with_p_value_gating():
    base = BenchmarkSummary(
        total_requests=50,
        successful_requests=50,
        failed_requests=0,
        total_prompt_tokens=5000,
        total_output_tokens=2500,
        duration_seconds=10.0,
        request_throughput=5.0,
        token_throughput=250.0,
        ttft_mean=0.040,
        ttft_std=0.002,
        ttft_p50=0.040,
        ttft_p90=0.042,
        ttft_p95=0.043,
        ttft_p99=0.045,
        tpot_mean=0.010,
        tpot_std=0.001,
        tpot_p50=0.010,
        tpot_p90=0.011,
        tpot_p95=0.012,
        tpot_p99=0.013,
        itl_mean=0.010,
        itl_std=0.001,
        itl_p50=0.010,
        itl_p90=0.011,
        itl_p95=0.012,
        itl_p99=0.013,
    )

    # Candidate has +10% higher TTFT (0.044 vs 0.040), statistically significant with n=50, std=0.002
    cand_regressed = BenchmarkSummary(
        total_requests=50,
        successful_requests=50,
        failed_requests=0,
        total_prompt_tokens=5000,
        total_output_tokens=2500,
        duration_seconds=10.0,
        request_throughput=5.0,
        token_throughput=250.0,
        ttft_mean=0.044,
        ttft_std=0.002,
        ttft_p50=0.044,
        ttft_p90=0.046,
        ttft_p95=0.047,
        ttft_p99=0.050,
        tpot_mean=0.010,
        tpot_std=0.001,
        tpot_p50=0.010,
        tpot_p90=0.011,
        tpot_p95=0.012,
        tpot_p99=0.013,
        itl_mean=0.010,
        itl_std=0.001,
        itl_p50=0.010,
        itl_p90=0.011,
        itl_p95=0.012,
        itl_p99=0.013,
    )

    # With alpha=0.01, TTFT p50 regression is flagged
    res_sig = compare_summaries(base, cand_regressed, regression_threshold_pct=5.0, alpha=0.01)
    ttft_item = next(i for i in res_sig.latency_items if i.name == "TTFT p50")
    assert ttft_item.is_regression is True
    assert ttft_item.p_value is not None
    assert ttft_item.p_value < 0.01

    md_output = render_comparison_markdown(res_sig)
    assert "p-value" in md_output
    assert "Effect Size (d)" in md_output
    assert "**Significance Alpha**: 0.010" in md_output
