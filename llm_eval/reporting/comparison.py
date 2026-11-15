"""Statistical comparison and regression detection between benchmark summaries."""

from __future__ import annotations

from dataclasses import dataclass

from llm_eval.core.metrics import BenchmarkSummary


@dataclass
class ComparisonMetricItem:
    """Individual metric diff between baseline and candidate runs."""

    name: str
    baseline_val: float
    candidate_val: float
    higher_is_better: bool = False
    regression_threshold_pct: float = 5.0

    @property
    def delta_pct(self) -> float:
        if self.baseline_val == 0:
            return 0.0
        return ((self.candidate_val - self.baseline_val) / self.baseline_val) * 100.0

    @property
    def is_regression(self) -> bool:
        if self.higher_is_better:
            return self.delta_pct < -self.regression_threshold_pct
        return self.delta_pct > self.regression_threshold_pct

    @property
    def is_improvement(self) -> bool:
        if self.higher_is_better:
            return self.delta_pct > self.regression_threshold_pct
        return self.delta_pct < -self.regression_threshold_pct

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

    @property
    def has_regressions(self) -> bool:
        return any(i.is_regression for i in self.throughput_items + self.latency_items)


def compare_summaries(
    baseline: BenchmarkSummary,
    candidate: BenchmarkSummary,
    regression_threshold_pct: float = 5.0,
) -> ComparisonResult:
    """Calculates relative differences between two benchmarks and flags regressions."""
    throughput_items = [
        ComparisonMetricItem(
            name="Request Throughput (req/s)",
            baseline_val=baseline.request_throughput,
            candidate_val=candidate.request_throughput,
            higher_is_better=True,
            regression_threshold_pct=regression_threshold_pct,
        ),
        ComparisonMetricItem(
            name="Token Throughput (tok/s)",
            baseline_val=baseline.token_throughput,
            candidate_val=candidate.token_throughput,
            higher_is_better=True,
            regression_threshold_pct=regression_threshold_pct,
        ),
    ]

    latency_items = [
        ComparisonMetricItem(
            name="TTFT p50",
            baseline_val=baseline.ttft_p50,
            candidate_val=candidate.ttft_p50,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
        ),
        ComparisonMetricItem(
            name="TTFT p95",
            baseline_val=baseline.ttft_p95,
            candidate_val=candidate.ttft_p95,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
        ),
        ComparisonMetricItem(
            name="TTFT p99",
            baseline_val=baseline.ttft_p99,
            candidate_val=candidate.ttft_p99,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
        ),
        ComparisonMetricItem(
            name="TPOT p50",
            baseline_val=baseline.tpot_p50,
            candidate_val=candidate.tpot_p50,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
        ),
        ComparisonMetricItem(
            name="TPOT p95",
            baseline_val=baseline.tpot_p95,
            candidate_val=candidate.tpot_p95,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
        ),
        ComparisonMetricItem(
            name="ITL p50",
            baseline_val=baseline.itl_p50,
            candidate_val=candidate.itl_p50,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
        ),
        ComparisonMetricItem(
            name="ITL p95",
            baseline_val=baseline.itl_p95,
            candidate_val=candidate.itl_p95,
            higher_is_better=False,
            regression_threshold_pct=regression_threshold_pct,
        ),
    ]

    return ComparisonResult(
        baseline=baseline,
        candidate=candidate,
        throughput_items=throughput_items,
        latency_items=latency_items,
    )
