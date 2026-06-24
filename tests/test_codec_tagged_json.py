import unittest

from vibeharness.codec import (DecodeConstraint, ToolCallCodec, get_codec)
from vibeharness.config import Config
from vibeharness.toolset import default_catalog


class GetCodecTaggedJsonTest(unittest.TestCase):
    def test_resolves_tagged_json_codec(self):
        codec = get_codec("tagged_json")
        self.assertIsInstance(codec, ToolCallCodec)
        self.assertEqual(codec.name, "tagged_json")

    def test_returns_module_codec_instance(self):
        from vibeharness.codecs import tagged_json_codec
        self.assertIs(get_codec("tagged_json"), tagged_json_codec.CODEC)


class TaggedJsonCodecTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())
        self.codec = get_codec("tagged_json")

    def test_format_instructions_render_the_cap(self):
        text = self.codec.format_instructions(3)
        self.assertIn("<local_toolcall></local_toolcall>", text)
        self.assertIn("at most 3 actions", text)

    def test_format_instructions_omit_cap_when_unbounded(self):
        self.assertNotIn("at most", self.codec.format_instructions(0))

    def test_turn_action_hint_mentions_tags(self):
        self.assertIn("<local_toolcall></local_toolcall>", self.codec.turn_action_hint())

    def test_constraint_is_unconstrained(self):
        c = self.codec.constraint(self.registry, 2)
        self.assertIsInstance(c, DecodeConstraint)
        self.assertIsNone(c.json_schema)
        self.assertIsNone(c.gbnf)

    def test_parse_single_block(self):
        actions, err = self.codec.parse(
            '<local_toolcall>{"tool": "list_directory", "args": {"path": "."}}</local_toolcall>')
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    def test_parse_multiple_blocks(self):
        raw = (
            '<local_toolcall>{"tool": "a", "args": {}}</local_toolcall>'
            '<local_toolcall>{"tool": "b", "args": {"x": 1}}</local_toolcall>'
        )
        actions, err = self.codec.parse(raw)
        self.assertIsNone(err)
        self.assertEqual(actions, [("a", {}), ("b", {"x": 1})])

    def test_parse_ignores_surrounding_prose(self):
        raw = (
            "Sure, here is what I'll do first:\n"
            '<local_toolcall>{"tool": "a", "args": {}}</local_toolcall>\n'
            "and then this:\n"
            '<local_toolcall>{"tool": "b", "args": {"x": 1}}</local_toolcall>\n'
            "Done."
        )
        actions, err = self.codec.parse(raw)
        self.assertIsNone(err)
        self.assertEqual(actions, [("a", {}), ("b", {"x": 1})])

    def test_parse_recovers_missing_final_closing_tag(self):
        raw = (
            '<local_toolcall>{"tool": "a", "args": {}}</local_toolcall>'
            '<local_toolcall>{"tool": "b", "args": {"x": 1}}'
        )
        actions, err = self.codec.parse(raw)
        self.assertIsNone(err)
        self.assertEqual(actions, [("a", {}), ("b", {"x": 1})])

    def test_parse_default_args_when_omitted(self):
        actions, err = self.codec.parse('<local_toolcall>{"tool": "a"}</local_toolcall>')
        self.assertIsNone(err)
        self.assertEqual(actions, [("a", {})])

    def test_parse_invalid_inner_json_reports_error(self):
        actions, err = self.codec.parse('<local_toolcall>{ not json</local_toolcall>')
        self.assertIsNone(actions)
        self.assertIn("invalid JSON", err)

    def test_parse_missing_tool_field_reports_error(self):
        actions, err = self.codec.parse('<local_toolcall>{"args": {}}</local_toolcall>')
        self.assertIsNone(actions)
        self.assertIn("tool", err)

    def test_parse_non_object_args_reports_error(self):
        actions, err = self.codec.parse('<local_toolcall>{"tool": "a", "args": 5}</local_toolcall>')
        self.assertIsNone(actions)
        self.assertIn("'args'", err)

    def test_parse_no_tags_reports_error(self):
        actions, err = self.codec.parse('{"tool": "a", "args": {}}')
        self.assertIsNone(actions)
        self.assertIn("no <local_toolcall>", err)


if __name__ == "__main__":
    unittest.main()
