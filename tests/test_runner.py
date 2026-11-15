import unittest

from llm_eval.adapters.mock_engine import MockEngineAdapter
from llm_eval.runner import BenchmarkRunner


class TestBenchmarkRunner(unittest.IsolatedAsyncioTestCase):
    async def test_end_to_end_concurrent_benchmark(self):
        engine = MockEngineAdapter(
            prefill_latency_s=0.01,
            decode_latency_per_token_s=0.005,
            simulated_tokens=["A", "B", "C", "D"],
        )

        runner = BenchmarkRunner(adapter=engine, model_name="test-llm")
        prompts = [f"Prompt {i}" for i in range(8)]

        summary = await runner.run_benchmark(
            prompts=prompts,
            concurrency=4,
            max_tokens_per_request=4,
        )

        self.assertEqual(summary.total_requests, 8)
        self.assertEqual(summary.successful_requests, 8)
        self.assertEqual(summary.failed_requests, 0)
        self.assertEqual(summary.total_output_tokens, 32)
        self.assertGreater(summary.token_throughput, 0.0)

        # TTFT should reflect simulated prefill latency
        self.assertGreaterEqual(summary.ttft_mean, 0.008)
        # Summary dict exports cleanly
        report = summary.to_dict()
        self.assertEqual(report["total_output_tokens"], 32)
        self.assertIn("ttft_ms", report)


if __name__ == "__main__":
    unittest.main()
