import unittest
import config


class TestConfig(unittest.TestCase):

    def test_model_name(self):
        self.assertEqual(config.OPENAI_MODEL, "gpt-4o-mini")

    def test_retrieval_constants(self):
        self.assertGreater(config.FAISS_TOP_K, 0)
        self.assertGreater(config.RERANK_TOP_K, 0)
        self.assertIsInstance(config.USE_HYBRID_SEARCH, bool)

    def test_generation_constants(self):
        self.assertGreater(config.MAX_TOKENS, 0)
        self.assertGreaterEqual(config.TEMPERATURE, 0.0)

    def test_lazy_api_key_loading(self):
        key = config.get_openai_api_key(required=False)
        self.assertTrue(key is None or isinstance(key, str))


if __name__ == "__main__":
    unittest.main()
