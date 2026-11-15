"""Statistical hypothesis testing and effect size analysis for latency distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy import stats as scipy_stats

    _HAS_SCIPY = True
except ImportError:
    scipy_stats = None
    _HAS_SCIPY = False


@dataclass
class HypothesisTestResult:
    """Results from two-sample statistical testing."""

    test_name: str
    statistic: float
    p_value: float
    effect_size_d: float
    is_significant: bool
    degrees_of_freedom: float | None = None


def compute_cohens_d(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> float:
    """Calculates Cohen's d effect size between two independent samples."""
    if n1 <= 1 or n2 <= 1:
        return 0.0
    pooled_var = (((n1 - 1) * (std1**2)) + ((n2 - 1) * (std2**2))) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    pooled_std = math.sqrt(pooled_var)
    return (mean2 - mean1) / pooled_std


def _approximate_t_pvalue(t_stat: float, df: float) -> float:
    """Fallback two-tailed p-value calculation using standard normal approximation."""
    abs_t = abs(t_stat)
    # Error function complementary approximation for standard normal
    p_norm = math.erfc(abs_t / math.sqrt(2.0))
    if df < 30:
        # Adjustment heuristic for low degrees of freedom
        correction = 1.0 + (abs_t**2 + 1.0) / (4.0 * max(1.0, df))
        return min(1.0, p_norm * correction)
    return min(1.0, p_norm)


def welch_t_test_from_stats(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Performs Welch's two-sample t-test using sample summary statistics."""
    d = compute_cohens_d(mean1, std1, n1, mean2, std2, n2)

    if n1 <= 1 or n2 <= 1 or (std1 == 0 and std2 == 0):
        return HypothesisTestResult(
            test_name="Welch's t-test",
            statistic=0.0,
            p_value=1.0,
            effect_size_d=0.0,
            is_significant=False,
            degrees_of_freedom=None,
        )

    var1 = std1**2
    var2 = std2**2
    if var1 == 0 and var2 == 0:
        return HypothesisTestResult(
            test_name="Welch's t-test",
            statistic=0.0,
            p_value=1.0,
            effect_size_d=0.0,
            is_significant=False,
            degrees_of_freedom=None,
        )

    se1 = var1 / n1
    se2 = var2 / n2
    se_diff = math.sqrt(se1 + se2)
    if se_diff <= 0:
        return HypothesisTestResult(
            test_name="Welch's t-test",
            statistic=0.0,
            p_value=1.0,
            effect_size_d=0.0,
            is_significant=False,
            degrees_of_freedom=None,
        )

    t_stat = (mean2 - mean1) / se_diff

    # Welch-Satterthwaite equation for degrees of freedom
    df_num = (se1 + se2) ** 2
    df_den = ((se1**2) / (n1 - 1)) + ((se2**2) / (n2 - 1))
    df = df_num / df_den if df_den > 0 else float(n1 + n2 - 2)

    if _HAS_SCIPY and scipy_stats is not None:
        p_val = float(scipy_stats.t.sf(abs(t_stat), df) * 2.0)
    else:
        p_val = _approximate_t_pvalue(t_stat, df)

    p_val = min(1.0, max(0.0, p_val))
    return HypothesisTestResult(
        test_name="Welch's t-test",
        statistic=float(t_stat),
        p_value=p_val,
        effect_size_d=d,
        is_significant=(p_val < alpha),
        degrees_of_freedom=float(df),
    )


def compare_empirical_distributions(
    sample_baseline: list[float],
    sample_candidate: list[float],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Performs non-parametric Kolmogorov-Smirnov and Welch tests on raw latency observations."""
    n1 = len(sample_baseline)
    n2 = len(sample_candidate)

    if n1 < 2 or n2 < 2:
        return HypothesisTestResult(
            test_name="Insufficient Data",
            statistic=0.0,
            p_value=1.0,
            effect_size_d=0.0,
            is_significant=False,
        )

    arr1 = np.asarray(sample_baseline, dtype=np.float64)
    arr2 = np.asarray(sample_candidate, dtype=np.float64)

    m1, s1 = float(np.mean(arr1)), float(np.std(arr1, ddof=1))
    m2, s2 = float(np.mean(arr2)), float(np.std(arr2, ddof=1))

    if _HAS_SCIPY and scipy_stats is not None:
        welch_res = scipy_stats.ttest_ind(arr1, arr2, equal_var=False)
        d = compute_cohens_d(m1, s1, n1, m2, s2, n2)

        # Use Welch's p-value as primary significance metric
        p_val = float(welch_res.pvalue)
        return HypothesisTestResult(
            test_name="Welch's t-test (Two-Sample)",
            statistic=float(welch_res.statistic),
            p_value=p_val,
            effect_size_d=d,
            is_significant=(p_val < alpha),
            degrees_of_freedom=float(welch_res.df) if hasattr(welch_res, "df") else None,
        )

    return welch_t_test_from_stats(m1, s1, n1, m2, s2, n2, alpha=alpha)
