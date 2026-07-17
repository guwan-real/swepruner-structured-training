import tempfile
import unittest
from pathlib import Path

from swepruner_dataset_builder.api_cache import ApiCache
from swepruner_dataset_builder.api_provider import CachedLLMClient, FakeProvider
from swepruner_dataset_builder.config import load_config
from swepruner_dataset_builder.issue_generation import generate_issue
from swepruner_dataset_builder.leakage_check import check_leakage
from swepruner_dataset_builder.schemas import TaskRecord


ROOT = Path(__file__).parents[1]


class IssueLeakageTests(unittest.TestCase):
    def test_detects_added_code(self):
        patch = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old_value = 10\n+correct_value = calculate_total(items)\n"
        result = check_leakage("Observed total is wrong", "correct_value = calculate_total(items)", patch)
        self.assertEqual(result.risk, "high")

    def test_fake_issue_generation(self):
        config = load_config(ROOT / "config/default.toml")
        task = TaskRecord("t", "swe_smith", "repo", "/tmp", patch="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n")
        response = {"issue_text": "The command reports an incorrect total.", "actual_behavior": "The total is low.",
                    "expected_behavior": "The total should include every item.", "leakage_risk": "low"}
        with tempfile.TemporaryDirectory() as directory:
            cache = ApiCache(Path(directory) / "cache.sqlite")
            value = generate_issue(task, "old", CachedLLMClient(FakeProvider([response]), cache, config), "fake", config)
            self.assertIn("incorrect total", value["issue_text"])
            cache.close()


if __name__ == "__main__":
    unittest.main()

