import unittest

from vibeharness.codec import DecodeConstraint, ToolCallCodec, get_codec
from vibeharness.config import Config
from vibeharness.toolset import default_catalog


class GetGbnfCodecTest(unittest.TestCase):
    def test_get_codec_returns_codec(self):
        codec = get_codec("gbnf")
        self.assertIsInstance(codec, ToolCallCodec)
        self.assertEqual(codec.name, "gbnf")
        from vibeharness.codecs.gbnf_codec import CODEC
        self.assertIs(codec, CODEC)


class GbnfCodecTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())
        self.codec = get_codec("gbnf")

    def test_format_instructions_render_the_cap(self):
        text = self.codec.format_instructions(3)
        self.assertIn("JSON ARRAY", text)
        self.assertIn("at most 3 actions", text)

    def test_format_instructions_omit_cap_when_unbounded(self):
        self.assertNotIn("at most", self.codec.format_instructions(0))

    def test_turn_action_hint_mentions_json(self):
        self.assertIn("JSON array", self.codec.turn_action_hint())

    def test_constraint_sets_gbnf_and_no_json_schema(self):
        c = self.codec.constraint(self.registry, 2)
        self.assertIsInstance(c, DecodeConstraint)
        self.assertIsNone(c.json_schema)
        self.assertIsInstance(c.gbnf, str)
        self.assertTrue(c.gbnf.strip())

    def test_grammar_mentions_every_tool_name(self):
        grammar = self.codec.constraint(self.registry, 0).gbnf
        self.assertTrue(self.registry.all())
        for tool in self.registry.all():
            # tool names appear as escaped JSON string literals, e.g. \"read_file\"
            self.assertIn(tool.name, grammar)
            self.assertIn(f'\\"{tool.name}\\"', grammar)

    def test_grammar_has_core_rules(self):
        grammar = self.codec.constraint(self.registry, 0).gbnf
        for rule in ("root", "action", "toolname", "object", "string", "ws"):
            self.assertIn(rule, grammar)

    def test_capped_grammar_bounds_action_repetitions(self):
        # With max_actions=2 there is exactly one optional ", action" group;
        # unbounded uses the "*" repetition instead.
        capped = self.codec.constraint(self.registry, 2).gbnf
        uncapped = self.codec.constraint(self.registry, 0).gbnf
        self.assertEqual(capped.count('( "," ws action )?'), 1)
        self.assertIn('( "," ws action )*', uncapped)
        self.assertNotIn('( "," ws action )?', uncapped)

    def test_capped_grammar_repetition_count_matches_cap(self):
        g4 = self.codec.constraint(self.registry, 4).gbnf
        self.assertEqual(g4.count('( "," ws action )?'), 3)

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
