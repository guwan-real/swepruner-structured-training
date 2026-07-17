import unittest
from swepruner_dataset_builder.real_data import parse_smith_repo

from swepruner_dataset_builder.patch_parser import parse_unified_diff


class PatchParserTests(unittest.TestCase):
    def test_deleted_and_added_lines(self):
        parsed = parse_unified_diff("--- a/a.py\n+++ b/a.py\n@@ -2,1 +2,1 @@\n-old\n+new\n")
        self.assertEqual(parsed.evidence[0].old_line, 2)
        self.assertEqual(parsed.evidence[0].old_text, "old")
        self.assertEqual([item.kind for item in parsed.evidence], ["deleted"])
        self.assertIn("new", parsed.added_lines)

    def test_pure_addition_has_context(self):
        parsed = parse_unified_diff("--- a/a.py\n+++ b/a.py\n@@ -2,0 +3,1 @@\n+new\n")
        self.assertEqual(parsed.evidence[0].kind, "addition_context")

    def test_smith_repo_identifier(self):
        self.assertEqual(parse_smith_repo("swesmith/oauthlib__oauthlib.1fd52536"),
                         ("oauthlib/oauthlib", "1fd52536"))


if __name__ == "__main__":
    unittest.main()
