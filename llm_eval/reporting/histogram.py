"""ASCII-based histogram generator for latency distribution visualization."""

from __future__ import annotations

import numpy as np


def render_ascii_histogram(
    values: list[float],
    num_bins: int = 5,
    unit: str = "ms",
    scale: float = 1000.0,
    bar_width: int = 24,
) -> str:
    """
    Renders an ASCII distribution histogram representing tail latency buckets.

    Args:
        values: List of raw values (e.g. durations in seconds).
        num_bins: Number of histogram buckets.
        unit: Display unit suffix.
        scale: Multiplier to convert raw values to display unit (default 1000 for s -> ms).
        bar_width: Maximum characters for the histogram bar.
    """
    if not values:
        return "No data points available for distribution."

    arr = np.asarray(values) * scale
    if len(arr) == 1 or np.all(arr == arr[0]):
        val = arr[0]
        return f"[{val:.2f} {unit}]: " + "█" * bar_width + " (100.0%)\n"

    counts, bin_edges = np.histogram(arr, bins=num_bins)
    total = len(arr)
    max_count = max(counts) if max(counts) > 0 else 1

    lines = []
    for i in range(len(counts)):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        count = counts[i]
        pct = (count / total) * 100.0
        bar_len = int((count / max_count) * bar_width)
        bar = "█" * bar_len
        lines.append(f"[{lower:6.2f} - {upper:6.2f} {unit}]: {bar:<{bar_width}} ({count:3d} | {pct:5.1f}%)")

    return "\n".join(lines)
