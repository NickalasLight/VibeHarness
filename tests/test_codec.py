import unittest

from vibeharness.codec import (DecodeConstraint, ToolCallCodec, UnknownCodec,
                               get_codec)
from vibeharness.config import Config
from vibeharness.toolset import default_catalog


class GetCodecTest(unittest.TestCase):
    def test_resolves_json_codec(self):
        codec = get_codec("json")
        self.assertIsInstance(codec, ToolCallCodec)
        self.assertEqual(codec.name, "json")

    def test_unknown_codec_raises(self):
        with self.assertRaises(UnknownCodec):
            get_codec("does-not-exist")


class JSONCodecTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())
        self.codec = get_codec("json")

    def test_format_instructions_render_the_cap(self):
        text = self.codec.format_instructions(3)
        self.assertIn("JSON ARRAY", text)
        self.assertIn("at most 3 actions", text)

    def test_format_instructions_omit_cap_when_unbounded(self):
        self.assertNotIn("at most", self.codec.format_instructions(0))

    def test_constraint_is_a_json_schema_with_maxitems(self):
        c = self.codec.constraint(self.registry, 2)
        self.assertIsInstance(c, DecodeConstraint)
        self.assertIsNotNone(c.json_schema)
        self.assertEqual(c.json_schema.get("maxItems"), 2)
        self.assertIsNone(c.gbnf)

    def test_constraint_unbounded_has_no_maxitems(self):
        c = self.codec.constraint(self.registry, 0)
        self.assertNotIn("maxItems", c.json_schema)

    def test_parse_single_object_is_one_action(self):
        actions, err = self.codec.parse('{"tool": "list_directory", "args": {"path": "."}}')
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    def test_parse_array_of_actions(self):
        actions, err = self.codec.parse(
            '[{"tool": "a", "args": {}}, {"tool": "b", "args": {"x": 1}}]')
        self.assertIsNone(err)
        self.assertEqual(actions, [("a", {}), ("b", {"x": 1})])

    def test_parse_invalid_json_reports_error(self):
        actions, err = self.codec.parse("{ not json")
        self.assertIsNone(actions)
        self.assertIn("not valid JSON", err)

    def test_parse_missing_tool_field_reports_error(self):
        actions, err = self.codec.parse('[{"args": {}}]')
        self.assertIsNone(actions)
        self.assertIn("tool", err)

    def test_parse_non_object_args_reports_error(self):
        actions, err = self.codec.parse('[{"tool": "a", "args": 5}]')
        self.assertIsNone(actions)
        self.assertIn("'args'", err)


if __name__ == "__main__":
    unittest.main()
