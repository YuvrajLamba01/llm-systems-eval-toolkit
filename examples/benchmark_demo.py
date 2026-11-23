"""
End-to-End Demonstration: LLM Serving Latency Decomposition & Regression Gating.

Simulates two production LLM serving engine deployments (Baseline vs. Optimized Candidate)
under a realistic Poisson arrival schedule, decomposes microsecond TTFT/TPOT percentiles,
renders ASCII latency histograms, and executes Welch's t-test hypothesis gating.
"""

from __future__ import annotations

import asyncio

from llm_eval.adapters.mock_engine import MockEngineAdapter
from llm_eval.core.workload import SyntheticPromptGenerator
from llm_eval.reporting.comparison import compare_summaries
from llm_eval.reporting.histogram import render_ascii_histogram
from llm_eval.runner import BenchmarkRunner


async def main() -> None:
    print("=" * 80)
    print("LLM Systems Evaluation Toolkit — Live Performance Demonstration")
    print("=" * 80)

    # 1. Generate realistic synthetic workload prompts
    prompt_gen = SyntheticPromptGenerator(seed=42)
    prompts = prompt_gen.generate_prompt_batch(num_requests=30, mean_tokens=64, std_tokens=16)
    print(f"\n[1] Generated {len(prompts)} synthetic evaluation prompts with variable context spans.")

    # 2. Benchmark Baseline Serving Engine (e.g., standard unoptimized batch scheduler)
    print("\n[2] Executing Baseline Engine Benchmark under Poisson Arrival Process (lambda = 15 req/s)...")
    baseline_adapter = MockEngineAdapter(
        prefill_latency_s=0.045,
        decode_latency_per_token_s=0.014,
    )
    baseline_runner = BenchmarkRunner(baseline_adapter, model_name="llama-3-8b-baseline")
    baseline_summary = await baseline_runner.run_benchmark(
        prompts=prompts,
        concurrency=4,
        max_tokens_per_request=16,
        request_rate=15.0,
    )
    print(f"    Baseline Complete: TTFT p50={baseline_summary.ttft_p50*1000:.2f}ms | TPOT p50={baseline_summary.tpot_p50*1000:.2f}ms")

    # 3. Benchmark Candidate Engine (e.g., optimized with Radix Prefix Caching + PagedAttention)
    print("\n[3] Executing Candidate Engine Benchmark (Radix Prefix Caching Enabled)...")
    candidate_adapter = MockEngineAdapter(
        prefill_latency_s=0.022,
        decode_latency_per_token_s=0.009,
    )
    candidate_runner = BenchmarkRunner(candidate_adapter, model_name="llama-3-8b-candidate")
    candidate_summary = await candidate_runner.run_benchmark(
        prompts=prompts,
        concurrency=4,
        max_tokens_per_request=16,
        request_rate=15.0,
    )
    print(f"    Candidate Complete: TTFT p50={candidate_summary.ttft_p50*1000:.2f}ms | TPOT p50={candidate_summary.tpot_p50*1000:.2f}ms")

    # 4. Render ASCII Tail-Latency Histograms
    print("\n[4] Candidate TTFT Tail-Latency Distribution:")
    raw_ttfts = [r.ttft for r in candidate_summary.records if r.ttft is not None]
    print(render_ascii_histogram(raw_ttfts, num_bins=6))

    # 5. Statistical Comparison & Automated Hypothesis Gating
    print("\n[5] Statistical Hypothesis Testing (Welch's t-test with Cohen's d effect size):")
    comparison = compare_summaries(
        baseline=baseline_summary,
        candidate=candidate_summary,
        regression_threshold_pct=5.0,
        alpha=0.01,
    )

    all_items = comparison.throughput_items + comparison.latency_items
    print("\nMetric Comparison Table:")
    print(f"{'Metric':<30} | {'Baseline':<10} | {'Candidate':<10} | {'Delta (%)':<10} | {'Status':<12}")
    print("-" * 80)
    for item in all_items:
        print(f"{item.name:<30} | {item.baseline_val:<10.3f} | {item.candidate_val:<10.3f} | {item.delta_pct:>+8.2f}% | {item.status_badge:<12}")

    print("\n" + "=" * 80)
    verdict = "PASSED (Optimization Verified)" if not comparison.has_regressions else "FAILED (Regression Detected)"
    print(f"Overall Evaluation Verdict: {verdict}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
