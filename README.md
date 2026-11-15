# LLM Systems Evaluation & Performance Toolkit (`llm-perf-eval`)

A reproducible, high-precision benchmarking and observability harness for Large Language Model (LLM) inference runtimes, serving engines, and continuous batching systems.

---

## Key Features

* **High-Precision Latency Decomposition:** Measures Time-to-First-Token (TTFT), Time-per-Output-Token (TPOT), and Inter-Token Latency (ITL) percentiles (p50, p90, p95, p99).
* **OpenTelemetry GenAI Semantic Conventions:** Emits spans and metrics natively conforming to OpenTelemetry GenAI standards (`gen_ai.client.operation.duration`, `gen_ai.client.token.usage`, `gen_ai.server.time_to_first_token`).
* **Pluggable Engine Adapters:**
  * `VLLMEngineAdapter`: Streaming SSE client for vLLM `/v1/chat/completions`.
  * `MockEngineAdapter`: Fully deterministic streaming engine for offline evaluation, stress testing, and continuous integration without physical GPU requirements.
* **Concurrency Scaling:** Measures throughput degradation, queueing effects, and tail latency across varying concurrency loads.

---

## Architecture

```text
llm-systems-eval-toolkit/
├── llm_eval/
│   ├── core/
│   │   └── metrics.py       # Latency decomposition & percentile accumulation
│   ├── telemetry/
│   │   └── otel.py          # OpenTelemetry GenAI span & metric instruments
│   ├── adapters/
│   │   ├── base.py          # Engine adapter abstract base class
│   │   ├── mock_engine.py   # Deterministic CPU mock for tests & simulation
│   │   └── vllm_client.py   # vLLM async streaming SSE client
│   └── runner.py            # Async concurrent execution harness
└── tests/
    ├── test_metrics.py
    ├── test_mock_engine.py
    └── test_runner.py
```

---

## Quickstart

```bash
# Run test suite
pytest tests -v

# Run linter
ruff check llm_eval tests
```
