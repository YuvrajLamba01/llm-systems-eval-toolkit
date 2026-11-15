import unittest

from llm_eval.core.metrics import RequestRecord, compute_benchmark_summary


class TestMetrics(unittest.TestCase):

    def test_single_token_request_metrics(self):
        rec = RequestRecord(
            request_id="req-1",
            prompt_tokens=15,
            start_time=10.0,
            first_token_time=10.05,
            end_time=10.05,
            token_timestamps=[10.05],
            output_tokens=1,
        )

        self.assertAlmostEqual(rec.ttft, 0.05, places=4)
        self.assertAlmostEqual(rec.total_latency, 0.05, places=4)
        self.assertEqual(len(rec.inter_token_latencies), 0)
        # generated_after_first is max(1, 0) = 1, so tpot is 0.0
        self.assertAlmostEqual(rec.tpot, 0.0, places=4)

    def test_multi_token_latency_decomposition(self):
        rec = RequestRecord(
            request_id="req-2",
            prompt_tokens=20,
            start_time=1.0,
            first_token_time=1.1,  # TTFT = 0.1s
            end_time=1.5,          # Total = 0.5s
            token_timestamps=[1.1, 1.2, 1.3, 1.4, 1.5],  # 5 tokens, ITLs = [0.1, 0.1, 0.1, 0.1]
            output_tokens=5,
        )

        self.assertAlmostEqual(rec.ttft, 0.1, places=4)
        self.assertAlmostEqual(rec.total_latency, 0.5, places=4)
        # 4 tokens generated after first in 0.4s -> TPOT = 0.1s
        self.assertAlmostEqual(rec.tpot, 0.1, places=4)
        self.assertEqual(len(rec.inter_token_latencies), 4)
        for itl in rec.inter_token_latencies:
            self.assertAlmostEqual(itl, 0.1, places=4)

    def test_benchmark_summary_aggregation(self):
        r1 = RequestRecord(
            request_id="r1",
            prompt_tokens=10,
            start_time=0.0,
            first_token_time=0.1,
            end_time=0.3,
            token_timestamps=[0.1, 0.2, 0.3],
            output_tokens=3,
        )
        r2 = RequestRecord(
            request_id="r2",
            prompt_tokens=10,
            start_time=0.0,
            first_token_time=0.2,
            end_time=0.6,
            token_timestamps=[0.2, 0.4, 0.6],
            output_tokens=3,
        )
        r_fail = RequestRecord(
            request_id="r3",
            prompt_tokens=5,
            start_time=0.0,
            end_time=0.05,
            error="ConnectionRefused",
        )

        summary = compute_benchmark_summary([r1, r2, r_fail], duration_seconds=1.0)

        self.assertEqual(summary.total_requests, 3)
        self.assertEqual(summary.successful_requests, 2)
        self.assertEqual(summary.failed_requests, 1)
        self.assertEqual(summary.total_output_tokens, 6)
        self.assertAlmostEqual(summary.request_throughput, 2.0, places=2)
        self.assertAlmostEqual(summary.token_throughput, 6.0, places=2)

        # TTFT mean = (0.1 + 0.2) / 2 = 0.15s
        self.assertAlmostEqual(summary.ttft_mean, 0.15, places=4)
        # Verify JSON export structure
        summary_dict = summary.to_dict()
        self.assertIn("ttft_ms", summary_dict)
        self.assertIn("tpot_ms", summary_dict)
        self.assertIn("itl_ms", summary_dict)
        self.assertEqual(summary_dict["ttft_ms"]["mean"], 150.0)


if __name__ == "__main__":
    unittest.main()
