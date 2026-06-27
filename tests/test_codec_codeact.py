import unittest

from vibeharness.codec import (DecodeConstraint, ToolCallCodec, get_codec)
from vibeharness.config import Config
from vibeharness.toolset import default_catalog


class GetCodecCodeActTest(unittest.TestCase):
    def test_resolves_codeact_codec(self):
        codec = get_codec("codeact")
        self.assertIsInstance(codec, ToolCallCodec)
        self.assertEqual(codec.name, "codeact")


class CodeActCodecTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())
        self.codec = get_codec("codeact")

    def test_format_instructions_render_the_cap(self):
        text = self.codec.format_instructions(3)
        self.assertIn("python", text)
        self.assertIn("at most 3", text)

    def test_format_instructions_omit_cap_when_unbounded(self):
        self.assertNotIn("at most", self.codec.format_instructions(0))

    def test_turn_action_hint_mentions_python(self):
        self.assertIn("python", self.codec.turn_action_hint())

    def test_constraint_is_unconstrained(self):
        c = self.codec.constraint(self.registry, 2)
        self.assertIsInstance(c, DecodeConstraint)
        self.assertIsNone(c.json_schema)
        self.assertIsNone(c.gbnf)

    def test_parse_single_call(self):
        actions, err = self.codec.parse(
            '```python\nlist_directory(path=".")\n```')
        self.assertIsNone(err)
        self.assertEqual(actions, [("list_directory", {"path": "."})])

    def test_parse_multiple_calls(self):
        actions, err = self.codec.parse(
            '```python\n'
            'create_file(path="notes/todo.txt", content="buy milk")\n'
            'read_file(path="notes/todo.txt")\n'
            '```')
        self.assertIsNone(err)
        self.assertEqual(actions, [
            ("create_file", {"path": "notes/todo.txt", "content": "buy milk"}),
            ("read_file", {"path": "notes/todo.txt"}),
        ])

    def test_parse_fenced_block_with_surrounding_prose(self):
        raw = (
            "Sure, here is my plan:\n\n"
            "```python\n"
            'read_file(path="a.txt")\n'
            "```\n\n"
            "That should do it."
        )
        actions, err = self.codec.parse(raw)
        self.assertIsNone(err)
        self.assertEqual(actions, [("read_file", {"path": "a.txt"})])

    def test_parse_without_fence_falls_back_to_whole_text(self):
        actions, err = self.codec.parse('read_file(path="a.txt")')
        self.assertIsNone(err)
        self.assertEqual(actions, [("read_file", {"path": "a.txt"})])

    def test_parse_coerces_int_bool_and_list_literals(self):
        actions, err = self.codec.parse(
            '```python\n'
            'tool(count=5, flag=True, items=["a", "b"], opts={"k": 1})\n'
            '```')
        self.assertIsNone(err)
        self.assertEqual(actions, [
            ("tool", {"count": 5, "flag": True, "items": ["a", "b"], "opts": {"k": 1}}),
        ])

    def test_parse_rejects_positional_args(self):
        actions, err = self.codec.parse('```python\nread_file("a.txt")\n```')
        self.assertIsNone(actions)
        self.assertIn("keyword", err)

    def test_parse_rejects_non_literal_value_name(self):
        actions, err = self.codec.parse('```python\nread_file(path=somevar)\n```')
        self.assertIsNone(actions)
        self.assertIn("literal", err)

    def test_parse_rejects_non_literal_value_call(self):
        actions, err = self.codec.parse('```python\nread_file(path=open("x"))\n```')
        self.assertIsNone(actions)
        self.assertIn("literal", err)

    def test_parse_syntax_error_reports_error(self):
        actions, err = self.codec.parse('```python\nread_file(path=\n```')
        self.assertIsNone(actions)
        self.assertIn("not valid Python", err)

    def test_parse_non_call_statement_reports_error(self):
        actions, err = self.codec.parse('```python\nx = 1\n```')
        self.assertIsNone(actions)
        self.assertIn("tool call", err)

    def test_parse_empty_block_reports_error(self):
        actions, err = self.codec.parse('```python\n\n```')
        self.assertIsNone(actions)
        self.assertIn("no tool calls", err)


if __name__ == "__main__":
    unittest.main()
