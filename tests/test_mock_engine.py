import unittest

from llm_eval.adapters.mock_engine import MockEngineAdapter


class TestMockEngine(unittest.IsolatedAsyncioTestCase):

    async def test_mock_engine_streaming(self):
        engine = MockEngineAdapter(
            prefill_latency_s=0.01,
            decode_latency_per_token_s=0.005,
            simulated_tokens=["foo", "bar", "baz"],
        )

        tokens = []
        async for tok in engine.generate_stream(prompt="hello", max_tokens=3):
            tokens.append(tok)

        self.assertEqual(tokens, ["foo", "bar", "baz"])

    async def test_mock_engine_max_tokens_capping(self):
        engine = MockEngineAdapter(
            prefill_latency_s=0.0,
            decode_latency_per_token_s=0.0,
            simulated_tokens=["a", "b", "c", "d"],
        )

        tokens = []
        async for tok in engine.generate_stream(prompt="test", max_tokens=2):
            tokens.append(tok)

        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
