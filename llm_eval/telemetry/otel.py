"""OpenTelemetry GenAI Semantic Conventions instrumentation for LLM inference."""

from __future__ import annotations

from opentelemetry import metrics, trace
from opentelemetry.metrics import Meter
from opentelemetry.trace import Span, Tracer


class GenAITelemetry:
    """Manages spans and metrics following OpenTelemetry GenAI semantic conventions."""

    def __init__(self, service_name: str = "llm-systems-eval"):
        self.tracer: Tracer = trace.get_tracer(service_name)
        self.meter: Meter = metrics.get_meter(service_name)

        # Standard GenAI metric instruments
        self.request_duration = self.meter.create_histogram(
            name="gen_ai.client.operation.duration",
            unit="s",
            description="GenAI request duration from invocation to completion.",
        )
        self.token_usage = self.meter.create_histogram(
            name="gen_ai.client.token.usage",
            unit="{token}",
            description="Number of tokens used in prompt and completion.",
        )
        self.ttft_histogram = self.meter.create_histogram(
            name="gen_ai.server.time_to_first_token",
            unit="s",
            description="Time to first generated token.",
        )

    def start_inference_span(
        self,
        model_name: str,
        system_name: str = "vllm",
        operation_name: str = "chat",
    ) -> Span:
        """Starts a span with standard GenAI semantic attributes."""
        span = self.tracer.start_span(
            f"{operation_name} {model_name}",
            attributes={
                "gen_ai.system": system_name,
                "gen_ai.request.model": model_name,
                "gen_ai.operation.name": operation_name,
            },
        )
        return span

    def record_metrics(
        self,
        model_name: str,
        duration_s: float,
        prompt_tokens: int,
        output_tokens: int,
        ttft_s: float | None = None,
        system_name: str = "vllm",
    ) -> None:
        """Records GenAI standard metrics for a completed inference call."""
        attrs = {
            "gen_ai.system": system_name,
            "gen_ai.response.model": model_name,
        }

        self.request_duration.record(duration_s, attributes=attrs)
        self.token_usage.record(prompt_tokens, attributes={**attrs, "gen_ai.token.type": "input"})
        self.token_usage.record(output_tokens, attributes={**attrs, "gen_ai.token.type": "output"})

        if ttft_s is not None:
            self.ttft_histogram.record(ttft_s, attributes=attrs)
