"""Async streaming client for vLLM OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from collections.abc import AsyncIterator

from llm_eval.adapters.base import BaseEngineAdapter


class VLLMEngineAdapter(BaseEngineAdapter):
    """Client for vLLM, SGLang, and OpenAI-compatible /v1/chat/completions endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "default",
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model_name
        self.api_key = api_key
        self.timeout = timeout

    @property
    def endpoint_url(self) -> str:
        """Construct normalized /v1/chat/completions URL without duplicate /v1 prefixes."""
        base = self.base_url
        base = base.removesuffix("/v1")
        return f"{base}/v1/chat/completions"

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Streams tokens from OpenAI-compatible SSE /v1/chat/completions."""
        url = self.endpoint_url
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Run blocking network call in thread executor
        loop = asyncio.get_running_loop()
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )

        def _open():
            return urllib.request.urlopen(req, timeout=self.timeout)

        response = await loop.run_in_executor(None, _open)

        try:
            while True:
                line = await loop.run_in_executor(None, response.readline)
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str.startswith(":"):
                    continue
                if line_str == "data: [DONE]":
                    break
                if line_str.startswith("data: "):
                    data_json = line_str[6:]
                    try:
                        chunk = json.loads(data_json)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()


# Cross-ecosystem aliases
SGLangEngineAdapter = VLLMEngineAdapter
OpenAICompatibleEngineAdapter = VLLMEngineAdapter
