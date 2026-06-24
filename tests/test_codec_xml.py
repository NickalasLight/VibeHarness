import unittest

from vibeharness.codec import (DecodeConstraint, ToolCallCodec, get_codec)
from vibeharness.config import Config
from vibeharness.toolset import default_catalog


class GetXmlCodecTest(unittest.TestCase):
    def test_resolves_xml_codec(self):
        codec = get_codec("xml")
        self.assertIsInstance(codec, ToolCallCodec)
        self.assertEqual(codec.name, "xml")

    def test_get_codec_returns_the_module_codec(self):
        from vibeharness.codecs.xml_codec import CODEC
        self.assertIs(get_codec("xml"), CODEC)


class XmlCodecTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())
        self.codec = get_codec("xml")

    def test_format_instructions_show_the_format_and_cap(self):
        text = self.codec.format_instructions(3)
        self.assertIn("<tool_call name=", text)
        self.assertIn("<arg name=", text)
        self.assertIn("at most 3 actions", text)
        self.assertIn("raw string", text)

    def test_format_instructions_omit_cap_when_unbounded(self):
        self.assertNotIn("at most", self.codec.format_instructions(0))

    def test_turn_action_hint_mentions_tool_call(self):
        self.assertIn("<tool_call name=", self.codec.turn_action_hint())

    def test_constraint_is_unconstrained(self):
        c = self.codec.constraint(self.registry, 2)
        self.assertIsInstance(c, DecodeConstraint)
        self.assertIsNone(c.json_schema)
        self.assertIsNone(c.gbnf)

    def test_parse_single_call(self):
        actions, err = self.codec.parse(
            '<tool_call name="list_directory">'
            '<arg name="path">.</arg>'
            '</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    def test_parse_multiple_calls_in_order(self):
        actions, err = self.codec.parse(
            '<tool_call name="write_file">'
            '<arg name="path">a.txt</arg>'
            '<arg name="content">hi</arg>'
            '</tool_call>\n'
            '<tool_call name="read_file">'
            '<arg name="path">a.txt</arg>'
            '</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions, [
            ("write_file", {"path": "a.txt", "content": "hi"}),
            ("read_file", {"path": "a.txt"}),
        ])

    def test_parse_no_arg_call(self):
        actions, err = self.codec.parse('<tool_call name="finish"></tool_call>')
        self.assertIsNone(err)
        self.assertEqual(actions, [("finish", {})])

    def test_parse_coerces_int(self):
        actions, err = self.codec.parse(
            '<tool_call name="read_file">'
            '<arg name="path">a.txt</arg>'
            '<arg name="page">2</arg>'
            '</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions, [("read_file", {"path": "a.txt", "page": 2})])
        self.assertIsInstance(actions[0][1]["page"], int)

    def test_parse_coerces_float(self):
        actions, err = self.codec.parse(
            '<tool_call name="t"><arg name="x">1.5</arg></tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions[0][1]["x"], 1.5)
        self.assertIsInstance(actions[0][1]["x"], float)

    def test_parse_coerces_bool_case_insensitive(self):
        actions, err = self.codec.parse(
            '<tool_call name="t">'
            '<arg name="a">true</arg>'
            '<arg name="b">FALSE</arg>'
            '</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions[0][1], {"a": True, "b": False})

    def test_parse_keeps_plain_string(self):
        actions, err = self.codec.parse(
            '<tool_call name="t"><arg name="x">hello world</arg></tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions[0][1]["x"], "hello world")

    def test_parse_unescapes_entities(self):
        actions, err = self.codec.parse(
            '<tool_call name="t">'
            '<arg name="x">a &amp; b &lt;c&gt; &quot;d&quot;</arg>'
            '</tool_call>'
        )
        self.assertIsNone(err)
        self.assertEqual(actions[0][1]["x"], 'a & b <c> "d"')

    def test_parse_no_blocks_reports_error(self):
        actions, err = self.codec.parse("just some prose, no tags")
        self.assertIsNone(actions)
        self.assertIn("no <tool_call>", err)

    def test_parse_missing_name_reports_error(self):
        actions, err = self.codec.parse(
            '<tool_call><arg name="x">1</arg></tool_call>'
        )
        self.assertIsNone(actions)
        self.assertIn("name", err)


if __name__ == "__main__":
    unittest.main()
