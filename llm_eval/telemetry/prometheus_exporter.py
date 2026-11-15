"""Prometheus metric exposition for cloud-native AI observability."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, generate_latest, start_http_server

from llm_eval.core.metrics import BenchmarkSummary


def export_summary_to_prometheus(
    summary: BenchmarkSummary,
    model_name: str = "default",
    registry: CollectorRegistry | None = None,
) -> str:
    """
    Exports BenchmarkSummary metrics into Prometheus text exposition format,
    matching cloud-native Kubernetes & Grafana dashboard standards.
    """
    reg = registry if registry is not None else CollectorRegistry()

    # Throughput metrics
    req_tps = Gauge(
        "llm_eval_request_throughput_rps",
        "Benchmark request throughput in requests per second",
        ["model"],
        registry=reg,
    )
    req_tps.labels(model=model_name).set(summary.request_throughput)

    tok_tps = Gauge(
        "llm_eval_token_throughput_tps",
        "Benchmark token throughput in tokens per second",
        ["model"],
        registry=reg,
    )
    tok_tps.labels(model=model_name).set(summary.token_throughput)

    # TTFT percentiles
    ttft_gauge = Gauge(
        "llm_eval_time_to_first_token_seconds",
        "Time to first token (TTFT) percentile distribution in seconds",
        ["model", "quantile"],
        registry=reg,
    )
    ttft_gauge.labels(model=model_name, quantile="0.50").set(summary.ttft_p50)
    ttft_gauge.labels(model=model_name, quantile="0.90").set(summary.ttft_p90)
    ttft_gauge.labels(model=model_name, quantile="0.95").set(summary.ttft_p95)
    ttft_gauge.labels(model=model_name, quantile="0.99").set(summary.ttft_p99)

    # TPOT percentiles
    tpot_gauge = Gauge(
        "llm_eval_time_per_output_token_seconds",
        "Time per output token (TPOT) percentile distribution in seconds",
        ["model", "quantile"],
        registry=reg,
    )
    tpot_gauge.labels(model=model_name, quantile="0.50").set(summary.tpot_p50)
    tpot_gauge.labels(model=model_name, quantile="0.90").set(summary.tpot_p90)
    tpot_gauge.labels(model=model_name, quantile="0.95").set(summary.tpot_p95)
    tpot_gauge.labels(model=model_name, quantile="0.99").set(summary.tpot_p99)

    # Queue delay
    queue_gauge = Gauge(
        "llm_eval_queue_delay_seconds",
        "Scheduling queue delay percentile distribution in seconds",
        ["model", "quantile"],
        registry=reg,
    )
    queue_gauge.labels(model=model_name, quantile="mean").set(summary.queue_delay_mean)
    queue_gauge.labels(model=model_name, quantile="0.95").set(summary.queue_delay_p95)

    return generate_latest(reg).decode("utf-8")


def start_prometheus_server(port: int, registry: CollectorRegistry | None = None) -> None:
    """Starts a standalone HTTP daemon exposing Prometheus metrics for scraping."""
    reg = registry if registry is not None else CollectorRegistry()
    start_http_server(port=port, registry=reg)
