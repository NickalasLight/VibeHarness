import unittest

from vibeharness.config import Config
from vibeharness.memory import NarrativeMemory
from vibeharness.prompt import SystemPromptBuilder, build_turn_prompt
from vibeharness.toolset import default_catalog


class NarrativeMemoryTest(unittest.TestCase):
    def test_empty(self):
        self.assertIn("not taken any actions", NarrativeMemory().render())

    def test_first_then_connectors(self):
        m = NarrativeMemory()
        m.record("you wrote a file")
        m.record("you read it back")
        rendered = m.render()
        self.assertEqual(rendered, "First, you wrote a file\nThen, you read it back")
        self.assertEqual(len(m), 2)


class PromptTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())

    def test_system_prompt_lists_tools(self):
        sp = SystemPromptBuilder(self.registry).build()
        self.assertIn("write_file", sp)
        self.assertIn("validate", sp)

    def test_system_prompt_renders_every_tool_name_and_description(self):
        sp = SystemPromptBuilder(self.registry).build()
        for tool in self.registry.all():
            self.assertIn(tool.name, sp, f"tool name {tool.name!r} missing from system prompt")
            self.assertIn(tool.description, sp,
                          f"description of {tool.name!r} missing from system prompt")

    def test_system_prompt_omits_redundant_json_schema(self):
        # The action schema is enforced by the decoder's `format` grammar, so it is
        # deliberately not re-printed in the prompt.
        sp = SystemPromptBuilder(self.registry).build()
        self.assertNotIn("Action schema", sp)
        self.assertNotIn("oneOf", sp)

    def test_system_prompt_anchors_task_at_front(self):
        sp = SystemPromptBuilder(self.registry).build("DO THE THING")
        self.assertIn("DO THE THING", sp)
        self.assertIn("YOUR ASSIGNED TASK", sp)
        self.assertLess(sp.index("DO THE THING"), sp.index("# Tools"))  # before the docs

    def test_system_prompt_without_task_is_generic(self):
        sp = SystemPromptBuilder(self.registry).build()
        self.assertNotIn("YOUR ASSIGNED TASK", sp)

    def test_turn_prompt_reminds_task_at_the_end(self):
        prompt = build_turn_prompt("make a file", "First, you did a thing")
        self.assertIn("make a file", prompt)
        self.assertIn("First, you did a thing", prompt)
        # the task reminder sits AFTER the history (recency zone), not before it
        self.assertGreater(prompt.index("make a file"), prompt.index("First, you did a thing"))
        self.assertIn("next action", prompt.lower())


if __name__ == "__main__":
    unittest.main()
