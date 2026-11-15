"""Deterministic mock inference engine for offline evaluation and CI testing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from llm_eval.adapters.base import BaseEngineAdapter


class MockEngineAdapter(BaseEngineAdapter):
    """
    Simulates a streaming LLM inference engine with configurable TTFT prefill delay
    and per-token decode latency without requiring a physical GPU.
    """

    def __init__(
        self,
        prefill_latency_s: float = 0.05,
        decode_latency_per_token_s: float = 0.01,
        simulated_tokens: list[str] | None = None,
    ):
        self.prefill_latency_s = prefill_latency_s
        self.decode_latency_per_token_s = decode_latency_per_token_s
        self.simulated_tokens = simulated_tokens or [
            "The",
            " quick",
            " brown",
            " fox",
            " jumps",
            " over",
            " the",
            " lazy",
            " dog.",
        ]

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        # Simulate prefill / Time To First Token
        if self.prefill_latency_s > 0:
            await asyncio.sleep(self.prefill_latency_s)

        # Yield tokens with simulated decode step latency
        count = min(max_tokens, len(self.simulated_tokens))
        for i in range(count):
            token = self.simulated_tokens[i % len(self.simulated_tokens)]
            if i > 0 and self.decode_latency_per_token_s > 0:
                await asyncio.sleep(self.decode_latency_per_token_s)
            yield token
