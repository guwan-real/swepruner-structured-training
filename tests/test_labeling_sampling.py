import unittest
from pathlib import Path

from swepruner_dataset_builder.config import load_config
from swepruner_dataset_builder.labeling import label_task
from swepruner_dataset_builder.leakage_check import hashes_present
from swepruner_dataset_builder.python_parser import build_repository_index
from swepruner_dataset_builder.sampling import build_task_outputs
from swepruner_dataset_builder.schemas import Role
from swepruner_dataset_builder.task_adapters import load_task_records


ROOT = Path(__file__).parents[1]
TASKS = ROOT / "tests/fixtures/data_sources/swe_smith/tasks.jsonl"


class LabelingSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config/default.toml")
        cls.tasks, _ = load_task_records("swe_smith", TASKS)
        cls.task = cls.tasks[0]
        cls.index = build_repository_index(cls.task.repo_path)
        cls.result = label_task(cls.task, cls.index, cls.config["confidence"])

    def test_patch_core_is_protected(self):
        self.assertTrue(self.result.core_block_ids)
        self.assertTrue(all(self.result.labels[item].role == Role.CORE for item in self.result.core_block_ids))
        self.assertTrue(self.result.patch_mapping_verified)

    def test_support_and_hard_negative(self):
        self.assertTrue(any(label.role == Role.SUPPORT for label in self.result.labels.values()))
        self.assertTrue(any(self.result.hard_negatives.values()))

    def test_sampling_has_aligned_line_labels(self):
        samples, relations, rankings = build_task_outputs(self.task, self.index, self.result, self.config, 42)
        self.assertTrue(samples)
        sample = samples[0]
        self.assertEqual(len(sample["code"].splitlines()), len(sample["line_roles"]))
        self.assertIn("CORE", sample["line_roles"])
        self.assertIn("DROP", sample["line_roles"])
        self.assertTrue(relations)
        self.assertTrue(rankings)

    def test_sampling_continues_after_leaking_hard_negative(self):
        for task in (self.tasks[1], self.tasks[3]):
            index = build_repository_index(task.repo_path)
            result = label_task(task, index, self.config["confidence"])
            samples, _, _ = build_task_outputs(task, index, result, self.config, 42)
            self.assertTrue(samples)
            self.assertTrue(all(not hashes_present(sample["code"], result.added_fragment_hashes) for sample in samples))


if __name__ == "__main__":
    unittest.main()
