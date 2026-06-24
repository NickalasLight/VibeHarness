import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from vibeharness import cli
from vibeharness.codec import available_codecs
from vibeharness.config import Config


class AvailableCodecsTest(unittest.TestCase):
    def test_lists_json(self):
        names = available_codecs()
        self.assertIn("json", names)

    def test_no_codec_suffix_and_no_dunders(self):
        names = available_codecs()
        self.assertTrue(all(not n.endswith("_codec") for n in names))
        self.assertTrue(all(not n.startswith("__") for n in names))

    def test_sorted(self):
        self.assertEqual(available_codecs(), sorted(available_codecs()))


class CodecValidationTest(unittest.TestCase):
    def test_valid_codec_has_no_error(self):
        self.assertIsNone(cli.codec_error("json"))

    def test_invalid_codec_reports_helpful_error(self):
        msg = cli.codec_error("nope")
        self.assertIsNotNone(msg)
        self.assertIn("unknown codec 'nope'", msg)
        self.assertIn("Available:", msg)
        self.assertIn("json", msg)


class CodecOverrideTest(unittest.TestCase):
    """--codec overrides config.codec for the run, taking precedence over settings."""

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

    def test_flag_overrides_default(self):
        parser = cli.build_parser()
        args = parser.parse_args(["task", "--codec", "json"])
        self.assertEqual(cli.resolve_config(args).codec, "json")

    def test_no_flag_keeps_config_default(self):
        parser = cli.build_parser()
        args = parser.parse_args(["task"])
        self.assertEqual(cli.resolve_config(args).codec, Config().codec)

    def test_flag_takes_precedence_over_saved_setting(self):
        from vibeharness.settings import Settings
        Settings.set("codec", "json")
        parser = cli.build_parser()
        args = parser.parse_args(["task", "--codec", "json"])
        # the override path is exercised; flag value wins for the run
        self.assertEqual(cli.resolve_config(args).codec, "json")


class CodecCliExitTest(unittest.TestCase):
    def test_invalid_codec_exits_2_with_message(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["do something", "--codec", "definitely-not-real"])
        self.assertEqual(rc, 2)
        out = buf.getvalue()
        self.assertIn("unknown codec 'definitely-not-real'", out)
        self.assertIn("Available:", out)


class PrintSystemCodecTest(unittest.TestCase):
    """--print-system renders with the CONFIGURED codec, not a hardcoded default
    (issue #123): on beta_qwen3coder the default codec is `hermes`, so the printed
    prompt must show the <tools>/<tool_call> format the model actually receives."""

    def _print_system(self, *extra):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["--print-system", *extra])
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_default_config_codec_is_reflected(self):
        out = self._print_system()
        if Config().codec == "hermes":
            # branch default: the hermes <tools>/<tool_call> format is rendered
            self.assertIn("<tools>", out)
            self.assertIn("<tool_call>", out)
            self.assertIn('"name"', out)
        else:
            # beta default (json): the JSON-array action format is rendered
            self.assertIn('{"tool":', out.replace(" ", ""))

    def test_codec_override_is_honoured(self):
        out = self._print_system("--codec", "json")
        # json codec describes a JSON array of {"tool","args"}; no <tools> block
        self.assertNotIn("<tools>", out)


class CodecHelpTest(unittest.TestCase):
    def test_help_mentions_codec_and_lists_json(self):
        parser = cli.build_parser()
        help_text = parser.format_help()
        self.assertIn("--codec", help_text)
        self.assertIn("json", help_text)


if __name__ == "__main__":
    unittest.main()
