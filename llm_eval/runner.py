"""Benchmark execution engine orchestrating concurrent streaming requests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from llm_eval.adapters.base import BaseEngineAdapter
from llm_eval.core.metrics import BenchmarkSummary, RequestRecord, compute_benchmark_summary
from llm_eval.core.workload import PoissonArrivalSchedule
from llm_eval.telemetry.otel import GenAITelemetry


class BenchmarkRunner:
    """Orchestrates async streaming inference benchmark runs."""

    def __init__(
        self,
        adapter: BaseEngineAdapter,
        model_name: str = "mock-model",
        telemetry: GenAITelemetry | None = None,
    ):
        self.adapter = adapter
        self.model_name = model_name
        self.telemetry = telemetry or GenAITelemetry()

    async def _execute_single_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int = 10,
    ) -> RequestRecord:
        """Executes a single streaming inference request and records timestamps."""
        record = RequestRecord(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            start_time=time.perf_counter(),
        )

        span = self.telemetry.start_inference_span(self.model_name)
        try:
            async for token in self.adapter.generate_stream(prompt, max_tokens=max_tokens):
                now = time.perf_counter()
                if record.first_token_time is None:
                    record.first_token_time = now
                record.token_timestamps.append(now)
                record.output_tokens += 1
            record.end_time = time.perf_counter()
        except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
            record.error = str(exc)
            record.end_time = time.perf_counter()
        finally:
            span.end()
            if record.total_latency is not None and record.error is None:
                self.telemetry.record_metrics(
                    model_name=self.model_name,
                    duration_s=record.total_latency,
                    prompt_tokens=record.prompt_tokens,
                    output_tokens=record.output_tokens,
                    ttft_s=record.ttft,
                )

        return record

    async def run_benchmark(
        self,
        prompts: Sequence[str],
        concurrency: int = 4,
        max_tokens_per_request: int = 32,
        request_rate: float | None = None,
    ) -> BenchmarkSummary:
        """
        Runs benchmark across prompts with a fixed concurrency limit and optional
        Poisson arrival scheduling.
        """
        semaphore = asyncio.Semaphore(concurrency)
        records: list[RequestRecord] = []

        schedule_offsets = (
            PoissonArrivalSchedule(request_rate).generate_schedule(len(prompts))
            if request_rate is not None and request_rate > 0
            else [0.0] * len(prompts)
        )

        overall_start = time.perf_counter()

        async def _bounded_worker(idx: int, p: str, offset_s: float):
            if offset_s > 0:
                await asyncio.sleep(offset_s)

            queue_start = time.perf_counter()
            async with semaphore:
                queue_delay = time.perf_counter() - queue_start
                rec = await self._execute_single_request(
                    request_id=f"req-{idx}",
                    prompt=p,
                    max_tokens=max_tokens_per_request,
                )
                rec.queue_delay_s = queue_delay
                records.append(rec)

        tasks = [_bounded_worker(i, p, schedule_offsets[i]) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)
        overall_duration = time.perf_counter() - overall_start

        return compute_benchmark_summary(records, overall_duration)
