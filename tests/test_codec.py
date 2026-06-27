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

    def test_format_instructions_render_cap_and_format(self):
        # The instructions name the JSON-array format and render the cap when one
        # is given, and omit it when unbounded.
        capped = self.codec.format_instructions(3)
        self.assertIn("JSON ARRAY", capped)
        self.assertIn("at most 3 actions", capped)
        self.assertIn("JSON array", self.codec.turn_action_hint())
        self.assertNotIn("at most", self.codec.format_instructions(0))

    def test_constraint_delegates_to_registry_action_schema(self):
        # The codec owns turning a registry + cap into a DecodeConstraint; the
        # detailed schema SHAPE (oneOf/maxItems/required) is owned and asserted by
        # test_registry_schema.py. Here we only prove the codec delegates: the
        # constraint carries the registry's capped action schema as json_schema
        # (and no gbnf for the json codec).
        c = self.codec.constraint(self.registry, 2)
        self.assertIsInstance(c, DecodeConstraint)
        self.assertEqual(c.json_schema, self.registry.action_schema(max_items=2))
        self.assertIsNone(c.gbnf)

    def test_parse_single_object_is_one_action(self):
        actions, err = self.codec.parse('{"tool": "list_directory", "args": {"path": "."}}')
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    def test_parse_array_of_actions(self):
        actions, err = self.codec.parse(
            '[{"tool": "a", "args": {}}, {"tool": "b", "args": {"x": 1}}]')
        self.assertIsNone(err)
        self.assertEqual(actions, [("a", {}), ("b", {"x": 1})])

    def test_parse_error_cases_report_a_reason(self):
        # Each malformed payload yields no actions and an error naming the problem.
        cases = [
            ("{ not json", "not valid JSON"),       # unparseable JSON
            ('[{"args": {}}]', "tool"),              # missing tool field
            ('[{"tool": "a", "args": 5}]', "'args'"),  # non-object args
        ]
        for payload, fragment in cases:
            with self.subTest(payload=payload):
                actions, err = self.codec.parse(payload)
                self.assertIsNone(actions)
                self.assertIn(fragment, err)


if __name__ == "__main__":
    unittest.main()
