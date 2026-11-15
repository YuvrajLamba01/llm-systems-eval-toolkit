"""Base interface for model-serving engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseEngineAdapter(ABC):
    """Abstract interface for interacting with an inference serving engine."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yields generated token text chunks as they arrive from the engine."""
