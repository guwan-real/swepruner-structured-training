import tempfile
import unittest
from pathlib import Path

from swepruner_dataset_builder.config import load_config
from swepruner_dataset_builder.pipeline import build_dataset
from swepruner_dataset_builder.validation import validate_artifact_dir


ROOT = Path(__file__).parents[1]


class PipelineTests(unittest.TestCase):
    def test_fixture_build_is_complete_and_offline(self):
        config = load_config(ROOT / "config/default.toml")
        tasks = ROOT / "tests/fixtures/data_sources/swe_smith/tasks.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "swe_smith"
            result = build_dataset("swe_smith", tasks, output, config, seed=42, num_workers=2, offline=True)
            self.assertEqual(result["successful_tasks"], 7)
            self.assertGreaterEqual(result["samples"], 20)
            self.assertEqual(result["api"]["requests"], 0)
            self.assertTrue(validate_artifact_dir(output)["valid"])


if __name__ == "__main__":
    unittest.main()

