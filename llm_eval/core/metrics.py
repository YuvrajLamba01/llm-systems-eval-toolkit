"""Latency decomposition and throughput metrics calculation for LLM inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RequestRecord:
    """Raw timing record for a single streaming inference request."""
    request_id: str
    prompt_tokens: int
    start_time: float
    first_token_time: float | None = None
    end_time: float | None = None
    token_timestamps: list[float] = field(default_factory=list)
    output_tokens: int = 0
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        """Time to First Token (TTFT) in seconds."""
        if self.first_token_time is not None:
            return self.first_token_time - self.start_time
        return None

    @property
    def total_latency(self) -> float | None:
        """End-to-end request duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def tpot(self) -> float | None:
        """Time per Output Token (TPOT) in seconds."""
        if self.end_time is not None and self.first_token_time is not None:
            generated_after_first = max(1, self.output_tokens - 1)
            return (self.end_time - self.first_token_time) / generated_after_first
        return None

    @property
    def inter_token_latencies(self) -> list[float]:
        """Inter-Token Latencies (ITL) between successive tokens in seconds."""
        if len(self.token_timestamps) < 2:
            return []
        arr = np.array(self.token_timestamps)
        return list(np.diff(arr))


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics across multiple requests."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_prompt_tokens: int
    total_output_tokens: int
    duration_seconds: float
    request_throughput: float  # req/s
    token_throughput: float    # tokens/s

    # TTFT distribution (seconds)
    ttft_mean: float
    ttft_p50: float
    ttft_p90: float
    ttft_p95: float
    ttft_p99: float

    # TPOT distribution (seconds)
    tpot_mean: float
    tpot_p50: float
    tpot_p90: float
    tpot_p95: float
    tpot_p99: float

    # ITL distribution (seconds)
    itl_mean: float
    itl_p50: float
    itl_p90: float
    itl_p95: float
    itl_p99: float

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "duration_seconds": round(self.duration_seconds, 4),
            "request_throughput_rps": round(self.request_throughput, 2),
            "token_throughput_tps": round(self.token_throughput, 2),
            "ttft_ms": {
                "mean": round(self.ttft_mean * 1000, 2),
                "p50": round(self.ttft_p50 * 1000, 2),
                "p90": round(self.ttft_p90 * 1000, 2),
                "p95": round(self.ttft_p95 * 1000, 2),
                "p99": round(self.ttft_p99 * 1000, 2),
            },
            "tpot_ms": {
                "mean": round(self.tpot_mean * 1000, 2),
                "p50": round(self.tpot_p50 * 1000, 2),
                "p90": round(self.tpot_p90 * 1000, 2),
                "p95": round(self.tpot_p95 * 1000, 2),
                "p99": round(self.tpot_p99 * 1000, 2),
            },
            "itl_ms": {
                "mean": round(self.itl_mean * 1000, 2),
                "p50": round(self.itl_p50 * 1000, 2),
                "p90": round(self.itl_p90 * 1000, 2),
                "p95": round(self.itl_p95 * 1000, 2),
                "p99": round(self.itl_p99 * 1000, 2),
            },
        }


def compute_benchmark_summary(records: list[RequestRecord], duration_seconds: float) -> BenchmarkSummary:
    """Calculate summary statistics from a list of request records."""
    successful = [r for r in records if r.error is None and r.output_tokens > 0]
    total_prompt_tokens = sum(r.prompt_tokens for r in records)
    total_output_tokens = sum(r.output_tokens for r in successful)

    ttfts = [r.ttft for r in successful if r.ttft is not None]
    tpots = [r.tpot for r in successful if r.tpot is not None]
    itls = [itl for r in successful for itl in r.inter_token_latencies]

    def _stats(arr: list[float]) -> tuple[float, float, float, float, float]:
        if not arr:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        np_arr = np.asarray(arr)
        return (
            float(np.mean(np_arr)),
            float(np.percentile(np_arr, 50)),
            float(np.percentile(np_arr, 90)),
            float(np.percentile(np_arr, 95)),
            float(np.percentile(np_arr, 99)),
        )

    ttft_mean, ttft_p50, ttft_p90, ttft_p95, ttft_p99 = _stats(ttfts)
    tpot_mean, tpot_p50, tpot_p90, tpot_p95, tpot_p99 = _stats(tpots)
    itl_mean, itl_p50, itl_p90, itl_p95, itl_p99 = _stats(itls)

    req_tps = len(successful) / max(0.001, duration_seconds)
    tok_tps = total_output_tokens / max(0.001, duration_seconds)

    return BenchmarkSummary(
        total_requests=len(records),
        successful_requests=len(successful),
        failed_requests=len(records) - len(successful),
        total_prompt_tokens=total_prompt_tokens,
        total_output_tokens=total_output_tokens,
        duration_seconds=duration_seconds,
        request_throughput=req_tps,
        token_throughput=tok_tps,
        ttft_mean=ttft_mean,
        ttft_p50=ttft_p50,
        ttft_p90=ttft_p90,
        ttft_p95=ttft_p95,
        ttft_p99=ttft_p99,
        tpot_mean=tpot_mean,
        tpot_p50=tpot_p50,
        tpot_p90=tpot_p90,
        tpot_p95=tpot_p95,
        tpot_p99=tpot_p99,
        itl_mean=itl_mean,
        itl_p50=itl_p50,
        itl_p90=itl_p90,
        itl_p95=itl_p95,
        itl_p99=itl_p99,
    )
