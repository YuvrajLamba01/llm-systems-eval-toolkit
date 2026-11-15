"""Benchmark report generation and comparison utilities."""

from __future__ import annotations

from llm_eval.reporting.comparison import ComparisonResult, compare_summaries
from llm_eval.reporting.markdown import render_benchmark_markdown, render_comparison_markdown

__all__ = [
    "ComparisonResult",
    "compare_summaries",
    "render_benchmark_markdown",
    "render_comparison_markdown",
]
