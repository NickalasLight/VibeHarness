"""Drift guard for the ``vibe`` CLI parser + ``--help`` (issue #175).

The point of these tests is to FAIL LOUDLY if a documented flag is removed from
the argparse parser or silently renamed, so ``vibe --help`` can never quietly fall
behind the real arguments again (the exact failure mode #175 was filed for: a stale
install whose ``--help`` lacked current flags).

We assert two layers:
  1. every documented flag string is a real option on the parser (option-string check),
  2. the destinations the run path reads (``args.<dest>``) exist with the right type,
and that ``--version`` reports BOTH the package version and the on-disk source path.

No model / Ollama required — we only build the parser and parse args.
"""
import io
import unittest
from contextlib import redirect_stdout

from vibeharness import cli
from vibeharness.settings import settable_keys


# Every option string `vibe --help` is expected to document. Keep this in lockstep
# with build_parser(); a removed/renamed flag breaks the matching test below.
DOCUMENTED_OPTIONS = [
    "--task-file", "--workdir",
    "--model", "--temp", "--top-p", "--top_k",
    "--codec", "--agent", "--toolset",
    "--max-steps", "--max-actions-per-turn",
    "--num-ctx",
    "--headless", "--web-snapshot-prose", "--advisor",
    "--no-color",
    "--set", "--show-config", "--reset-config",
    "--list-toolsets", "--list-agents", "--print-system",
    "--version",
]


def _all_option_strings(parser) -> set[str]:
    opts: set[str] = set()
    for action in parser._actions:
        opts.update(action.option_strings)
    return opts


class ParserExposesDocumentedFlagsTest(unittest.TestCase):
    def setUp(self):
        self.parser = cli.build_parser()
        self.options = _all_option_strings(self.parser)

    def test_every_documented_flag_is_a_real_option(self):
        missing = [o for o in DOCUMENTED_OPTIONS if o not in self.options]
        self.assertEqual(missing, [], f"documented flags missing from the parser: {missing}")

    def test_help_text_mentions_every_documented_flag(self):
        help_text = self.parser.format_help()
        missing = [o for o in DOCUMENTED_OPTIONS if o not in help_text]
        self.assertEqual(missing, [], f"flags absent from --help text: {missing}")

    def test_positional_task_is_optional_variadic(self):
        # `vibe` with no task must still parse (it prints help); the task is nargs="*".
        args = self.parser.parse_args([])
        self.assertEqual(args.task, [])

    def test_help_epilog_lists_settable_keys(self):
        help_text = self.parser.format_help()
        for key in settable_keys():
            self.assertIn(key, help_text, f"settable key {key!r} not shown in --help")


class SamplingAndContextOverridesTest(unittest.TestCase):
    """The per-run sampling/context flags must flow into the resolved Config (#175)."""

    def test_top_p_top_k_num_ctx_override_config(self):
        args = cli.build_parser().parse_args(
            ["task", "--top-p", "0.5", "--top_k", "7", "--num-ctx", "4096"])
        cfg = cli.resolve_config(args)
        self.assertAlmostEqual(cfg.top_p, 0.5)
        self.assertEqual(cfg.top_k, 7)
        self.assertEqual(cfg.num_ctx, 4096)

    def test_defaults_unset_when_flags_absent(self):
        args = cli.build_parser().parse_args(["task"])
        self.assertIsNone(args.top_p)
        self.assertIsNone(args.top_k)
        self.assertIsNone(args.num_ctx)


class VersionReportsSourcePathTest(unittest.TestCase):
    """`vibe --version` must show the version AND the absolute running source path."""

    def test_source_report_has_version_and_path(self):
        import vibeharness
        from pathlib import Path

        report = cli.build_source_report()
        self.assertIn(vibeharness.__version__, report)
        self.assertIn("build", report)
        self.assertIn("source:", report)
        pkg_dir = Path(vibeharness.__file__).resolve().parent
        self.assertIn(str(pkg_dir), report)

    def test_version_command_prints_source(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["--version"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("source:", out)


if __name__ == "__main__":
    unittest.main()
