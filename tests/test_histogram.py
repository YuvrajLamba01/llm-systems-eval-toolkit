from __future__ import annotations

from llm_eval.reporting.histogram import render_ascii_histogram


def test_ascii_histogram_empty():
    res = render_ascii_histogram([])
    assert "No data points" in res


def test_ascii_histogram_uniform():
    res = render_ascii_histogram([0.05, 0.05, 0.05])
    assert "100.0%" in res
    assert "50.00 ms" in res


def test_ascii_histogram_distribution():
    # Latencies in seconds: 10ms, 20ms, 30ms, 40ms, 50ms
    latencies = [0.010, 0.020, 0.030, 0.040, 0.050]
    chart = render_ascii_histogram(latencies, num_bins=4, unit="ms", scale=1000.0)

    assert "ms" in chart
    assert "█" in chart
    lines = chart.strip().splitlines()
    assert len(lines) == 4
