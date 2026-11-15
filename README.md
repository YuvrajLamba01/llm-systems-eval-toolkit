# LLM Systems Evaluation & Performance Toolkit (`llm-perf-eval`)

[![CI](https://github.com/YuvrajLamba01/llm-systems-eval-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/YuvrajLamba01/llm-systems-eval-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible, high-precision benchmarking, latency decomposition, and observability harness for Large Language Model (LLM) inference runtimes, serving engines (vLLM, SGLang, TGI), and continuous batching schedulers.

---

## Key Capabilities

* **Full Latency Decomposition:** Measures microsecond-accurate Time-to-First-Token (TTFT), Time-per-Output-Token (TPOT), and Inter-Token Latency (ITL) percentiles (p50, p90, p95, p99).
* **OpenTelemetry GenAI Semantic Conventions:** Emits spans and metrics conforming strictly to OpenTelemetry GenAI standards (`gen_ai.client.operation.duration`, `gen_ai.client.token.usage`, `gen_ai.server.time_to_first_token`).
* **Poisson Process Traffic Modeling:** Simulates realistic production traffic arrivals with exponentially distributed inter-arrival times (`--request-rate`) to analyze queueing delays and scheduler contention.
* **Context Length Scaling:** Synthetic prompt generation (`--synthetic-prompts`) with variable token context windows.
* **ASCII Tail-Latency Distribution:** Visualizes tail-latency clustering directly in stdout and Markdown reports.
* **Automated Regression Gating:** Compares candidate inference runs against baselines and exits with code `1` if latency percentiles or throughput degrade beyond configurable tolerances (e.g., CI/CD gate).
* **Pluggable Engine Adapters:**
  * `VLLMEngineAdapter`: Async streaming SSE client for vLLM `/v1/chat/completions`.
  * `MockEngineAdapter`: Fully deterministic streaming engine for offline evaluation, stress testing, and CI testing without requiring physical GPUs.
* **Trace & CSV Export:** Exports per-request microsecond traces (`--export-csv`) for external visualization and telemetry ingestion.

---

## Architecture

```text
llm-systems-eval-toolkit/
├── llm_eval/
│   ├── cli.py               # Command-line interface (benchmark, report, compare)
│   ├── runner.py            # Async bounded-concurrency execution engine
│   ├── core/
│   │   ├── metrics.py       # Microsecond latency decomposition & percentile calculation
│   │   └── workload.py      # Poisson arrival scheduling & synthetic prompt generation
│   ├── telemetry/
│   │   └── otel.py          # OpenTelemetry GenAI span & metric instruments
│   ├── adapters/
│   │   ├── base.py          # Engine adapter abstract base class
│   │   ├── mock_engine.py   # Deterministic CPU mock for CI simulation
│   │   └── vllm_client.py   # vLLM async streaming SSE client
│   └── reporting/
│       ├── markdown.py      # GitHub-Flavored Markdown summary & comparison reports
│       ├── comparison.py    # Statistical diff & regression analysis
│       └── histogram.py     # ASCII tail-latency distribution renderer
└── tests/
    ├── test_cli.py
    ├── test_histogram.py
    ├── test_metrics.py
    ├── test_mock_engine.py
    ├── test_reporting.py
    ├── test_runner.py
    └── test_workload.py
```

---

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/YuvrajLamba01/llm-systems-eval-toolkit.git
cd llm-systems-eval-toolkit
pip install -e .
```

---

## CLI Usage

### 1. Run a Benchmark

```bash
# Benchmark against a live vLLM server
llm-eval benchmark \
  --model meta-llama/Llama-3-8B-Instruct \
  --endpoint http://localhost:8000/v1 \
  --concurrency 8 \
  --num-requests 100 \
  --output results/candidate.json \
  --report results/candidate.md

# Benchmark using deterministic mock engine (CPU / CI mode)
llm-eval benchmark --mock --num-requests 20 --output results/mock.json
```

### 2. Compare Against Baseline & Enforce CI Regression Gate

```bash
# Detect regressions with a 5% threshold (exits with code 1 if violated)
llm-eval compare \
  --baseline results/baseline.json \
  --candidate results/candidate.json \
  --threshold 5.0 \
  --output results/comparison.md \
  --fail-on-regression
```

### 3. Generate Markdown Report

```bash
llm-eval report --input results/candidate.json --output report.md
```

---

## Verification

```bash
# Run unit tests
pytest tests -v

# Run linter
ruff check .
```
