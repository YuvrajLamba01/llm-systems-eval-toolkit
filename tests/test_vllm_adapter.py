"""Tests for VLLMEngineAdapter and cross-engine OpenAI streaming client."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from llm_eval.adapters.vllm_client import (
    OpenAICompatibleEngineAdapter,
    SGLangEngineAdapter,
    VLLMEngineAdapter,
)


def test_endpoint_url_normalization():
    """Verify endpoint_url strips duplicate /v1 prefixes and trailing slashes."""
    adapter1 = VLLMEngineAdapter(base_url="http://localhost:8000")
    assert adapter1.endpoint_url == "http://localhost:8000/v1/chat/completions"

    adapter2 = VLLMEngineAdapter(base_url="http://localhost:8000/")
    assert adapter2.endpoint_url == "http://localhost:8000/v1/chat/completions"

    adapter3 = VLLMEngineAdapter(base_url="http://localhost:8000/v1")
    assert adapter3.endpoint_url == "http://localhost:8000/v1/chat/completions"

    adapter4 = VLLMEngineAdapter(base_url="http://localhost:8000/v1/")
    assert adapter4.endpoint_url == "http://localhost:8000/v1/chat/completions"


def test_adapter_ecosystem_aliases():
    """Verify SGLang and OpenAICompatible aliases map to VLLMEngineAdapter."""
    assert SGLangEngineAdapter is VLLMEngineAdapter
    assert OpenAICompatibleEngineAdapter is VLLMEngineAdapter


@pytest.mark.asyncio
async def test_generate_stream_sse_parsing():
    """Verify generate_stream correctly parses OpenAI/vLLM/SGLang SSE chunks."""
    adapter = VLLMEngineAdapter(
        base_url="http://localhost:8000/v1",
        model_name="meta-llama/Llama-3-8B-Instruct",
        api_key="sk-test-token",
    )

    sse_data = (
        b": ping\n\n"
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "!"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    mock_resp = io.BytesIO(sse_data)
    mock_resp.close = MagicMock()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        tokens = []
        async for token in adapter.generate_stream(prompt="Hi", max_tokens=10):
            tokens.append(token)

        assert tokens == ["Hello", " world", "!"]
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:8000/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer sk-test-token"
        assert req.headers["Content-type"] == "application/json"
