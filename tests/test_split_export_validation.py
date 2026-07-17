import json
import tempfile
import unittest
from pathlib import Path

from swepruner_dataset_builder.exporters import export_swepruner
from swepruner_dataset_builder.io_utils import read_jsonl, write_jsonl
from swepruner_dataset_builder.splitting import split_samples
from swepruner_dataset_builder.validation import validate_dataset
from swepruner_dataset_builder.task_adapters import normalize_swepruner_rows


class SplitExportValidationTests(unittest.TestCase):
    def sample(self, sample_id, repo, code="def f():\n    return 1"):
        return {
            "sample_id": sample_id, "task_id": sample_id, "dataset_source": "swe_smith",
            "query": "Return the expected value.", "repo_name": repo, "file_path": "a.py", "anchor_symbol": "f",
            "code": code, "line_numbers": [1, 2], "line_roles": ["SUPPORT", "CORE"],
            "line_keep_labels": [1, 1], "line_confidences": [0.85, 1.0],
            "line_provenance": [["function_signature"], ["patch_old_line"]],
            "line_relation_types": ["TYPE", "PATCH"], "document_label": 1,
            "metadata": {"input_revision": "buggy_base", "patch_mapping_verified": True,
                         "core_line_numbers": [2], "forbidden_added_hashes": [],
                         "block_spans": [{"start_line": 1, "end_line": 2}]},
        }

    def test_repo_split_and_near_duplicate_union(self):
        config = {"split": {"train_ratio": 0.8, "validation_ratio": 0.1, "test_ratio": 0.1}}
        samples = [self.sample("a", "r1"), self.sample("b", "r1", "def g():\n    return 2"),
                   self.sample("c", "r2"), self.sample("d", "r3", "def h():\n    return 3")]
        manifests = split_samples(samples, 42, config)
        locations = {repo: name for name, manifest in manifests.items() for repo in manifest["repositories"]}
        self.assertEqual(locations["r1"], locations["r2"])

    def test_export_and_validation(self):
        sample = self.sample("a", "r1")
        sample["line_roles"][0] = "DROP"
        sample["line_keep_labels"][0] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            mapping = root / "mapping.json"
            write_jsonl(source, [sample])
            mapping.write_text(json.dumps({"query": "query", "code": "document", "line_keep_labels": "labels"}))
            result = validate_dataset(source)
            self.assertTrue(result["valid"])
            export_swepruner(source, mapping, output)
            self.assertEqual(len(list(read_jsonl(output))), 1)

    def test_official_swepruner_kept_frags(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "official.jsonl"
            write_jsonl(source, [{"query": "Find the return.", "code": "def f():\n    return 1\n",
                                  "kept_frags": [2], "score": 0.9}])
            rows, failures = normalize_swepruner_rows(source, 1)
            self.assertFalse(failures)
            self.assertEqual(rows[0]["line_keep_labels"], [0, 1])
            self.assertEqual(rows[0]["document_label"], 1)
            self.assertEqual(rows[0]["metadata"]["original_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
