from __future__ import annotations

from llm_eval.core.metrics import BenchmarkSummary
from llm_eval.telemetry.prometheus_exporter import export_summary_to_prometheus


def test_prometheus_metrics_export():
    summary = BenchmarkSummary(
        total_requests=10,
        successful_requests=10,
        failed_requests=0,
        total_prompt_tokens=100,
        total_output_tokens=320,
        duration_seconds=1.5,
        request_throughput=6.67,
        token_throughput=213.33,
        queue_delay_mean=0.002,
        queue_delay_p95=0.005,
        ttft_mean=0.035,
        ttft_p50=0.032,
        ttft_p90=0.045,
        ttft_p95=0.050,
        ttft_p99=0.060,
        tpot_mean=0.010,
        tpot_p50=0.009,
        tpot_p90=0.012,
        tpot_p95=0.014,
        tpot_p99=0.018,
        itl_mean=0.010,
        itl_p50=0.009,
        itl_p90=0.012,
        itl_p95=0.014,
        itl_p99=0.018,
    )

    output = export_summary_to_prometheus(summary, model_name="llama-3-8b")

    assert 'llm_eval_request_throughput_rps{model="llama-3-8b"} 6.67' in output
    assert 'llm_eval_token_throughput_tps{model="llama-3-8b"} 213.33' in output
    assert (
        'llm_eval_time_to_first_token_seconds{model="llama-3-8b",quantile="0.50"} 0.032' in output
    )
    assert 'llm_eval_queue_delay_seconds{model="llama-3-8b",quantile="0.95"} 0.005' in output


def test_start_prometheus_server_invocation(monkeypatch):
    from prometheus_client import CollectorRegistry

    from llm_eval.telemetry.prometheus_exporter import start_prometheus_server

    called_port = None

    def mock_start(port, registry):
        nonlocal called_port
        called_port = port

    monkeypatch.setattr("llm_eval.telemetry.prometheus_exporter.start_http_server", mock_start)
    reg = CollectorRegistry()
    start_prometheus_server(port=9200, registry=reg)

    assert called_port == 9200
