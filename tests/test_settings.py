import os
import tempfile
import unittest

from vibeharness.config import Config
from vibeharness.settings import Settings


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("VIBEHARNESS_HOME")
        os.environ["VIBEHARNESS_HOME"] = self.tmp.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("VIBEHARNESS_HOME", None)
        else:
            os.environ["VIBEHARNESS_HOME"] = self._prev
        self.tmp.cleanup()

    def test_defaults_when_no_file(self):
        self.assertEqual(Settings.load(), {})
        self.assertEqual(Settings.apply(Config()).temperature, Config().temperature)

    def test_set_and_apply(self):
        field, value = Settings.set("temp", "0.5")
        self.assertEqual((field, value), ("temperature", 0.5))
        self.assertEqual(Settings.load(), {"temperature": 0.5})
        self.assertEqual(Settings.apply(Config()).temperature, 0.5)

    def test_set_friendly_keys(self):
        Settings.set("max-steps", "25")
        self.assertEqual(Settings.apply(Config()).max_steps, 25)

    def test_set_max_actions_per_turn_roundtrips(self):
        field, value = Settings.set("max-actions-per-turn", "3")
        self.assertEqual((field, value), ("max_actions_per_turn", 3))
        self.assertEqual(Settings.apply(Config()).max_actions_per_turn, 3)
        # underscore form maps to the same field
        Settings.set("max_actions_per_turn", "6")
        self.assertEqual(Settings.apply(Config()).max_actions_per_turn, 6)

    def test_set_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            Settings.set("frobnicate", "1")

    def test_set_bad_value_raises(self):
        with self.assertRaises(ValueError):
            Settings.set("temp", "not-a-number")

    def test_cli_override_layering(self):
        # saved settings are the base; an explicit override wins
        Settings.set("temp", "0.5")
        from dataclasses import replace
        cfg = replace(Settings.apply(Config()), temperature=1.0)
        self.assertEqual(cfg.temperature, 1.0)

    def test_reset(self):
        Settings.set("temp", "0.9")
        self.assertTrue(Settings.reset())
        self.assertEqual(Settings.load(), {})
        self.assertFalse(Settings.reset())  # nothing left to remove

    def test_ignores_unknown_persisted_fields(self):
        Settings.save({"temperature": 0.4, "bogus_field": 123})
        cfg = Settings.apply(Config())
        self.assertEqual(cfg.temperature, 0.4)  # known field applied, bogus ignored


if __name__ == "__main__":
    unittest.main()
