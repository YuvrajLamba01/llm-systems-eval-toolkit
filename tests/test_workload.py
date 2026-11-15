from __future__ import annotations

import pytest

from llm_eval.core.workload import PoissonArrivalSchedule, SyntheticPromptGenerator


def test_poisson_arrival_validation():
    with pytest.raises(ValueError, match="strictly positive"):
        PoissonArrivalSchedule(0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        PoissonArrivalSchedule(-5.0)


def test_poisson_arrival_schedule_monotonicity():
    schedule = PoissonArrivalSchedule(request_rate_rps=10.0, seed=123)
    offsets = schedule.generate_schedule(num_requests=25)

    assert len(offsets) == 25
    assert offsets[0] == 0.0

    # Ensure timestamps are strictly increasing
    for i in range(1, len(offsets)):
        assert offsets[i] >= offsets[i - 1]

    # At 10 req/s, 25 requests should take roughly 2.0 - 3.0 seconds
    assert 1.0 < offsets[-1] < 5.0


def test_synthetic_prompt_generator():
    gen = SyntheticPromptGenerator(seed=42)

    prompt = gen.generate_prompt(approximate_tokens=20)
    words = prompt.split()
    assert len(words) == 20

    batch = gen.generate_prompt_batch(num_requests=10, mean_tokens=50, std_tokens=5)
    assert len(batch) == 10
    for p in batch:
        assert len(p.split()) > 10
