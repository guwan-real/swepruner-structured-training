import unittest
from pathlib import Path

from swepruner_dataset_builder.python_parser import build_repository_index


FIXTURE = Path(__file__).parent / "fixtures/data_sources/swe_smith/repos/demo_repo"


class PythonAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = build_repository_index(FIXTURE)

    def test_extracts_symbols_and_blocks(self):
        names = {symbol.qualname for symbol in self.index.symbols.values()}
        types = {block.block_type for block in self.index.blocks.values()}
        self.assertIn("Ledger.add_entry", names)
        self.assertTrue({"IfHeader", "TryHeader", "Except", "Finally"}.issubset(types))

    def test_call_and_def_use_relations(self):
        relations = {str(item.relation) for item in self.index.relations}
        self.assertIn("CALL", relations)
        self.assertIn("DEF_USE", relations)


if __name__ == "__main__":
    unittest.main()

