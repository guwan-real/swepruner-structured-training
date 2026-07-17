import tempfile
import unittest
from pathlib import Path

from swepruner_dataset_builder.api_cache import ApiCache
from swepruner_dataset_builder.api_provider import CachedLLMClient, FakeProvider, review_candidate
from swepruner_dataset_builder.config import load_config


ROOT = Path(__file__).parents[1]


class ApiCacheTests(unittest.TestCase):
    def test_cache_prevents_duplicate_request(self):
        config = load_config(ROOT / "config/default.toml")
        provider = FakeProvider([{"value": 1}])
        with tempfile.TemporaryDirectory() as directory:
            cache = ApiCache(Path(directory) / "cache.sqlite")
            client = CachedLLMClient(provider, cache, config)
            payload = {"issue_text": "x", "candidate_code": "y", "relation_metadata": {}}
            self.assertEqual(client.request("fake", "v1", payload), {"value": 1})
            self.assertEqual(client.request("fake", "v1", payload), {"value": 1})
            self.assertEqual(provider.calls, 1)
            cache.close()

    def test_primary_reviewer_agreement_and_conflict(self):
        config = load_config(ROOT / "config/default.toml")
        block = "a.py::f::Return::2::2"
        payload = {"block_id": block, "issue_text": "x", "candidate_code": "return x",
                   "relation_metadata": {"relation": "CALL"}}
        with tempfile.TemporaryDirectory() as directory:
            responses = [
                {"block_id": block, "role": "SUPPORT", "relation": "CALL", "confidence": 0.7, "reason": "call"},
                {"block_id": block, "role": "SUPPORT", "relation": "CALL", "confidence": 0.8, "reason": "call"},
            ]
            cache = ApiCache(Path(directory) / "cache.sqlite")
            decision = review_candidate(CachedLLMClient(FakeProvider(responses), cache, config), payload,
                                        "DROP", 0.7, "primary", "reviewer", config)
            self.assertTrue(decision["agreement"])
            self.assertEqual(decision["role"], "SUPPORT")
            cache.close()


if __name__ == "__main__":
    unittest.main()

