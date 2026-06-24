import json
import unittest

from vibeharness.codec import DecodeConstraint, ToolCallCodec, get_codec
from vibeharness.config import Config
from vibeharness.toolset import default_catalog


class GetHermesCodecTest(unittest.TestCase):
    def test_resolves_hermes_codec(self):
        codec = get_codec("hermes")
        self.assertIsInstance(codec, ToolCallCodec)
        self.assertEqual(codec.name, "hermes")

    def test_get_codec_returns_the_module_codec(self):
        from vibeharness.codecs.hermes_codec import CODEC
        self.assertIs(get_codec("hermes"), CODEC)

    def test_listed_in_available_codecs(self):
        from vibeharness.codec import available_codecs
        self.assertIn("hermes", available_codecs())


class HermesCodecTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())
        self.codec = get_codec("hermes")

    # ---- format_instructions emit the native convention ----
    def test_format_instructions_state_the_convention(self):
        text = self.codec.format_instructions(4)
        self.assertIn("<tool_call>", text)
        self.assertIn("</tool_call>", text)
        self.assertIn('"name"', text)
        self.assertIn('"arguments"', text)
        self.assertIn("at most 4 tool calls", text)

    def test_format_instructions_omit_cap_when_unbounded(self):
        self.assertNotIn("at most", self.codec.format_instructions(0))

    def test_turn_action_hint_mentions_tool_call(self):
        hint = self.codec.turn_action_hint()
        self.assertIn("<tool_call>", hint)
        self.assertIn("name", hint)

    # ---- the <tools> block has function-schema shape ----
    def test_tool_definitions_is_hermes_tools_block(self):
        block = self.codec.tool_definitions(self.registry)
        self.assertIsNotNone(block)
        self.assertTrue(block.startswith("<tools>"))
        self.assertTrue(block.rstrip().endswith("</tools>"))
        # every tool appears
        for name in self.registry.names():
            self.assertIn(name, block)

    def test_tools_block_lines_are_openai_function_schemas(self):
        block = self.registry.tools_block(style="hermes")
        inner = block.splitlines()[1:-1]  # strip <tools>/</tools>
        self.assertEqual(len(inner), len(self.registry.all()))
        for line in inner:
            obj = json.loads(line)
            self.assertEqual(obj["type"], "function")
            fn = obj["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertEqual(fn["parameters"]["type"], "object")
            self.assertIn("properties", fn["parameters"])

    def test_tools_block_parameters_match_args_schema(self):
        block = self.registry.tools_block(style="hermes")
        by_name = {json.loads(l)["function"]["name"]: json.loads(l)
                   for l in block.splitlines()[1:-1]}
        for tool in self.registry.all():
            self.assertEqual(by_name[tool.name]["function"]["parameters"],
                             tool._args_schema())

    def test_unknown_tools_block_style_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.tools_block(style="nope")

    # ---- constraint is unconstrained ----
    def test_constraint_is_unconstrained(self):
        c = self.codec.constraint(self.registry, 2)
        self.assertIsInstance(c, DecodeConstraint)
        self.assertIsNone(c.json_schema)
        self.assertIsNone(c.gbnf)

    # ---- parse: single call ----
    def test_parse_single_call(self):
        actions, err = self.codec.parse(
            '<tool_call>\n{"name": "list_directory", "arguments": {"path": "."}}\n</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    # ---- parse: multiple consecutive blocks, in order ----
    def test_parse_multiple_calls_in_order(self):
        actions, err = self.codec.parse(
            '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.txt", '
            '"content": "hi"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions, [
            ("write_file", {"path": "a.txt", "content": "hi"}),
            ("read_file", {"path": "a.txt"}),
        ])

    def test_parse_tolerates_surrounding_prose_and_think(self):
        actions, err = self.codec.parse(
            "<think>I should list the dir</think>\n"
            'Sure: <tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    def test_parse_missing_final_close_tag_recovered(self):
        actions, err = self.codec.parse(
            '<tool_call>\n{"name": "list_directory", "arguments": {"path": "."}}'
        )
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    def test_parse_absent_arguments_coerced_to_empty(self):
        actions, err = self.codec.parse('<tool_call>{"name": "finish"}</tool_call>')
        self.assertIsNone(err)
        self.assertEqual(actions, [("finish", {})])

    def test_parse_null_arguments_coerced_to_empty(self):
        actions, err = self.codec.parse(
            '<tool_call>{"name": "finish", "arguments": null}</tool_call>')
        self.assertIsNone(err)
        self.assertEqual(actions, [("finish", {})])

    # ---- parse: malformed input ----
    def test_parse_no_blocks_reports_error(self):
        actions, err = self.codec.parse("just some prose, no tags")
        self.assertIsNone(actions)
        self.assertIn("no <tool_call>", err)

    def test_parse_invalid_json_reports_error(self):
        actions, err = self.codec.parse('<tool_call>{not json}</tool_call>')
        self.assertIsNone(actions)
        self.assertIn("invalid JSON", err)

    def test_parse_missing_name_reports_error(self):
        actions, err = self.codec.parse(
            '<tool_call>{"arguments": {"path": "."}}</tool_call>')
        self.assertIsNone(actions)
        self.assertIn("name", err)

    def test_parse_non_object_arguments_reports_error(self):
        actions, err = self.codec.parse(
            '<tool_call>{"name": "read_file", "arguments": "oops"}</tool_call>')
        self.assertIsNone(actions)
        self.assertIn("arguments", err)

    def test_parse_empty_name_reports_error(self):
        actions, err = self.codec.parse(
            '<tool_call>{"name": "", "arguments": {}}</tool_call>')
        self.assertIsNone(actions)
        self.assertIn("name", err)

    def test_parse_handles_none_input(self):
        actions, err = self.codec.parse(None)
        self.assertIsNone(actions)
        self.assertIn("no <tool_call>", err)


if __name__ == "__main__":
    unittest.main()
