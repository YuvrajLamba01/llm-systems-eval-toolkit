"""Statistical comparison and regression detection between benchmark summaries."""

from __future__ import annotations

from dataclasses import dataclass

from llm_eval.core.metrics import BenchmarkSummary
from llm_eval.core.statistics import (
    compare_empirical_distributions,
    welch_t_test_from_stats,
)


@dataclass
class ComparisonMetricItem:
    """Individual metric diff between baseline and candidate runs."""

    name: str
    baseline_val: float
    candidate_val: float
    higher_is_better: bool = False
    regression_threshold_pct: float = 5.0
    p_value: float | None = None
    effect_size_d: float | None = None
    alpha: float | None = None

    @property
    def delta_pct(self) -> float:
        if self.baseline_val == 0:
            return 0.0
        return ((self.candidate_val - self.baseline_val) / self.baseline_val) * 100.0

    @property
    def is_statistically_significant(self) -> bool:
        if self.p_value is None or self.alpha is None:
            return True
        return self.p_value < self.alpha

    @property
    def is_regression(self) -> bool:
        magnitude_check = (
            self.delta_pct < -self.regression_threshold_pct
            if self.higher_is_better
            else self.delta_pct > self.regression_threshold_pct
        )
        return magnitude_check and self.is_statistically_significant

    @property
    def is_improvement(self) -> bool:
        magnitude_check = (
            self.delta_pct > self.regression_threshold_pct
            if self.higher_is_better
            else self.delta_pct < -self.regression_threshold_pct
        )
        return magnitude_check and self.is_statistically_significant

    @property
    def status_badge(self) -> str:
        if self.is_regression:
            return "REGRESSION"
        if self.is_improvement:
            return "IMPROVED"
        return "SIMILAR"


@dataclass
class ComparisonResult:
    """Complete comparative analysis of two benchmark runs."""

    baseline: BenchmarkSummary
    candidate: BenchmarkSummary
    throughput_items: list[ComparisonMetricItem]
    latency_items: list[ComparisonMetricItem]
    alpha: float | None = None

    @property
    def has_regressions(self) -> bool:
        return any(i.is_regression for i in self.throughput_items + self.latency_items)


def compare_summaries(
    baseline: BenchmarkSummary,
    candidate: BenchmarkSummary,
    regression_threshold_pct: float = 5.0,
    alpha: float | None = None,
) -> ComparisonResult:
    """Calculates relative differences between two benchmarks and flags statistically significant regressions."""
    significance_alpha = alpha if alpha is not None else 0.05

    # Run statistical tests if sample data or variance is available
    ttft_pval = None
    ttft_d = None
    if len(baseline.raw_ttft_list) >= 2 and len(candidate.raw_ttft_list) >= 2:
        ttft_test = compare_empirical_distributions(
            baseline.raw_ttft_list, candidate.raw_ttft_list, alpha=significance_alpha
        )
        ttft_pval = ttft_test.p_value
        ttft_d = ttft_test.effect_size_d
    elif baseline.successful_requests > 1 and candidate.successful_requests > 1:
        ttft_test = welch_t_test_from_stats(
            baseline.ttft_mean,
            baseline.ttft_std,
            baseline.successful_requests,
            candidate.ttft_mean,
            candidate.ttft_std,
            candidate.successful_requests,
            alpha=significance_alpha,
        )
        ttft_pval = ttft_test.p_value
        ttft_d = ttft_test.effect_size_d

    tpot_pval = None
    tpot_d = None
    if len(baseline.raw_tpot_list) >= 2 and len(candidate.raw_tpot_list) >= 2:
        tpot_test = compare_empirical_distributions(
            baseline.raw_tpot_list, candidate.raw_tpot_list, alpha=significance_alpha
        )
        tpot_pval = tpot_test.p_value
        tpot_d = tpot_test.effect_size_d
    elif baseline.successful_requests > 1 and candidate.successful_requests > 1:
        tpot_test = welch_t_test_from_stats(
            baseline.tpot_mean,
            baseline.tpot_std,
            baseline.successful_requests,
            candidate.tpot_mean,
            candidate.tpot_std,
            candidate.successful_requests,
            alpha=significance_alpha,
        )
        tpot_pval = tpot_test.p_value
        tpot_d = tpot_test.effect_size_d

    itl_pval = None
    itl_d = None
    if len(baseline.raw_itl_list) >= 2 and len(candidate.raw_itl_list) >= 2:
        itl_test = compare_empirical_distributions(
            baseline.raw_itl_list, candidate.raw_itl_list, alpha=significance_alpha
        )
        itl_pval = itl_test.p_value
        itl_d = itl_test.effect_size_d
    elif (
        baseline.successful_requests > 1
        and candidate.successful_requests > 1
        and (baseline.itl_std > 0 or candidate.itl_std > 0)
    ):
        itl_test = welch_t_test_from_stats(
            baseline.itl_mean,
            baseline.itl_std,
            baseline.successful_requests,
            candidate.itl_mean,
            candidate.itl_std,
            candidate.successful_requests,
            alpha=significance_alpha,
        )
        itl_pval = itl_test.p_value
        itl_d = itl_test.effect_size_d

    throughput_items = [
        ComparisonMetricItem(
            name="Request Throughput (req/s)",
            baseline_val=baseline.request_throughput,
            candidate_val=candidate.request_throughput,
            higher_is_better=True,
            regression_threshold_pct=regression_threshold_pct,
            alpha=alpha,
        ),
        ComparisonMetricItem(
            name="Token Throughput (tok/s)",
            baseline_val=baseline.token_throughput,
            candidate_val=candidate.token_throughput,
            higher_is_better=True,
            regression_threshold_pct=regression_threshold_pct,
            alpha=alpha,
        ),
    ]

    latency_items = [
        ComparisonMetricItem(
            name="TTFT p50",
            baseline_val=baseline.ttft_p50,
            candidate_val=candidate.ttft_p50,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
            p_value=ttft_pval,
            effect_size_d=ttft_d,
            alpha=alpha,
        ),
        ComparisonMetricItem(
            name="TTFT p95",
            baseline_val=baseline.ttft_p95,
            candidate_val=candidate.ttft_p95,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
            p_value=ttft_pval,
            effect_size_d=ttft_d,
            alpha=alpha,
        ),
        ComparisonMetricItem(
            name="TTFT p99",
            baseline_val=baseline.ttft_p99,
            candidate_val=candidate.ttft_p99,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
            p_value=ttft_pval,
            effect_size_d=ttft_d,
            alpha=alpha,
        ),
        ComparisonMetricItem(
            name="TPOT p50",
            baseline_val=baseline.tpot_p50,
            candidate_val=candidate.tpot_p50,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
            p_value=tpot_pval,
            effect_size_d=tpot_d,
            alpha=alpha,
        ),
        ComparisonMetricItem(
            name="TPOT p95",
            baseline_val=baseline.tpot_p95,
            candidate_val=candidate.tpot_p95,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
            p_value=tpot_pval,
            effect_size_d=tpot_d,
            alpha=alpha,
        ),
        ComparisonMetricItem(
            name="ITL p50",
            baseline_val=baseline.itl_p50,
            candidate_val=candidate.itl_p50,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
            p_value=itl_pval,
            effect_size_d=itl_d,
            alpha=alpha,
        ),
        ComparisonMetricItem(
            name="ITL p95",
            baseline_val=baseline.itl_p95,
            candidate_val=candidate.itl_p95,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
            p_value=itl_pval,
            effect_size_d=itl_d,
            alpha=alpha,
        ),
    ]

    return ComparisonResult(
        baseline=baseline,
        candidate=candidate,
        throughput_items=throughput_items,
        latency_items=latency_items,
        alpha=alpha,
    )
