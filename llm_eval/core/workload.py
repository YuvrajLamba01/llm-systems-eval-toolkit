"""Synthetic workload generation, arrival distributions, and prompt schedules."""

from __future__ import annotations

import numpy as np


class PoissonArrivalSchedule:
    """
    Generates Poisson process request arrival intervals where inter-arrival times
    follow an exponential distribution: P(dt) = lambda * exp(-lambda * dt).
    """

    def __init__(self, request_rate_rps: float, seed: int | None = 42):
        if request_rate_rps <= 0:
            raise ValueError(f"Request rate must be strictly positive, got {request_rate_rps}")
        self.request_rate_rps = request_rate_rps
        self.rng = np.random.default_rng(seed)

    def next_interval_seconds(self) -> float:
        """Returns the delay in seconds before dispatching the next request."""
        # Mean interval is 1.0 / request_rate_rps
        return float(self.rng.exponential(scale=1.0 / self.request_rate_rps))

    def generate_schedule(self, num_requests: int) -> list[float]:
        """Generates relative dispatch offsets (in seconds) for a batch of requests."""
        if num_requests <= 0:
            return []
        intervals = self.rng.exponential(scale=1.0 / self.request_rate_rps, size=num_requests - 1)
        offsets = [0.0]
        cumulative = 0.0
        for interval in intervals:
            cumulative += float(interval)
            offsets.append(cumulative)
        return offsets


class SyntheticPromptGenerator:
    """Generates synthetic prompts of controlled token lengths for context-window scaling tests."""

    _VOCABULARY: tuple[str, ...] = (
        "quantum", "neural", "gradient", "transformer", "attention", "tensor",
        "matrix", "backpropagation", "pipeline", "cache", "latency", "throughput",
        "compiler", "kernel", "memory", "bandwidth", "vector", "parallel", "dispatch",
        "speculative", "prefix", "continuous", "batching", "scheduler", "token",
    )

    def __init__(self, seed: int | None = 42):
        self.rng = np.random.default_rng(seed)

    def generate_prompt(self, approximate_tokens: int) -> str:
        """Generates a synthetic sentence sequence with roughly the requested token count."""
        if approximate_tokens <= 0:
            approximate_tokens = 1
        words = self.rng.choice(self._VOCABULARY, size=approximate_tokens)
        return " ".join(words)

    def generate_prompt_batch(
        self,
        num_requests: int,
        mean_tokens: int = 128,
        std_tokens: int = 32,
    ) -> list[str]:
        """Generates a batch of prompts with normally distributed context lengths."""
        lengths = self.rng.normal(loc=mean_tokens, scale=std_tokens, size=num_requests)
        lengths = np.clip(lengths, a_min=8, a_max=None).astype(int)
        return [self.generate_prompt(length) for length in lengths]
