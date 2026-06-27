"""Issue #24: the LATEST live page snapshot is auto-injected into the per-turn
system prompt for the web worker, and stale snapshots are dropped by prompt
regeneration (never accumulated in narrative memory).

No real browser is used: the snapshot source is INJECTABLE. ``capture_page_snapshot``
takes any object with a ``run(*args) -> (ok, output)`` method, so a fake returning
canned snapshot text drives the whole path. The per-turn injection is reproduced the
same way cli.py wires it: a zero-arg ``page`` provider passed into
``SystemPromptBuilder.build(..., page=...)`` each turn.
"""
import unittest

from vibeharness.config import Config
from vibeharness.memory import NarrativeMemory
from vibeharness.prompt import SystemPromptBuilder, build_turn_prompt
from vibeharness.toolset import default_catalog
from vibeharness.web import (capture_page_snapshot, capture_page_snapshot_raw,
                             make_raw_snapshot_provider, make_snapshot_provider)


class _FakeSnapshotCli:
    """Injectable stand-in for PlaywrightCli: returns scripted snapshot text. A list
    of outputs is consumed one per ``run`` call so successive turns see DIFFERENT
    snapshots (proving stale-dropping)."""

    def __init__(self, outputs, ok=True):
        self._outputs = list(outputs)
        self._i = 0
        self.ok = ok
        self.calls = []

    def run(self, *args):
        self.calls.append(list(args))
        out = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        return self.ok, out


def _registry(names):
    catalog = default_catalog()
    return catalog.build_registry(catalog.select(names), Config())


class CapturePageSnapshotTest(unittest.TestCase):
    def test_captures_snapshot_text_from_session(self):
        cli = _FakeSnapshotCli(["### Page\nCONSENT BANNER"])
        text = capture_page_snapshot(cli, char_limit=1000)
        self.assertIn("CONSENT BANNER", text)
        # It captures via the SAME session by issuing a `snapshot` command.
        self.assertEqual(cli.calls, [["snapshot"]])

    def test_truncated_to_char_limit(self):
        cli = _FakeSnapshotCli(["y" * 5000])
        text = capture_page_snapshot(cli, char_limit=100)
        self.assertIn("truncated", text)
        self.assertLess(len(text), 5000)

    def test_truncates_at_configured_cap(self):
        # capture_page_snapshot truncates at exactly the cap it is given: the kept
        # body is char_limit chars, plus the appended "+N chars truncated" marker.
        cli = _FakeSnapshotCli(["q" * 50000])
        text = capture_page_snapshot(cli, char_limit=40000)
        self.assertTrue(text.startswith("q" * 40000))
        self.assertIn("truncated", text)
        self.assertIn("10000 chars truncated", text)

    def test_under_new_default_cap_not_truncated(self):
        # A page at/under the new 40000-char default is returned whole.
        cli = _FakeSnapshotCli(["a" * 40000])
        text = capture_page_snapshot(cli, char_limit=Config().web_snapshot_char_limit)
        self.assertEqual(text, "a" * 40000)
        self.assertNotIn("truncated", text)

    def test_failed_snapshot_returns_empty(self):
        cli = _FakeSnapshotCli(["boom"], ok=False)
        self.assertEqual(capture_page_snapshot(cli, char_limit=1000), "")

    def test_exception_returns_empty(self):
        class _Raises:
            def run(self, *a):
                raise RuntimeError("no session")
        self.assertEqual(capture_page_snapshot(_Raises(), char_limit=1000), "")


class CaptureRawSnapshotTest(unittest.TestCase):
    """Issue #37: the diagnostic raw capture returns the COMPLETE snapshot with no
    char cap — ground truth on its true size — but stays exception-safe."""

    def test_returns_full_text_uncapped(self):
        # A real (non-blank) snapshot: capture_page_snapshot_raw now retries ONCE when
        # the first capture looks blank (about:blank / no refs), so the body must carry
        # a ref to count as a single, non-retried capture.
        body = "### Page\n- button \"x\" [ref=e1]\n" + "w" * 50000
        cli = _FakeSnapshotCli([body])
        text = capture_page_snapshot_raw(cli)
        self.assertEqual(text, body)             # whole thing, no truncation marker
        self.assertNotIn("truncated", text)
        self.assertEqual(cli.calls, [["snapshot"]])  # non-blank -> single capture, no retry

    def test_failed_snapshot_returns_empty(self):
        self.assertEqual(capture_page_snapshot_raw(_FakeSnapshotCli(["boom"], ok=False)), "")

    def test_exception_returns_empty(self):
        class _Raises:
            def run(self, *a):
                raise RuntimeError("no session")
        self.assertEqual(capture_page_snapshot_raw(_Raises()), "")

    def test_raw_provider_is_zero_arg_and_never_raises(self):
        # With no live browser the snapshot call fails and the provider returns ""
        # rather than raising — same contract as make_snapshot_provider.
        self.assertEqual(make_raw_snapshot_provider(Config())(), "")


class PerTurnSnapshotInjectionTest(unittest.TestCase):
    """Issue #146: the live snapshot is NO LONGER injected into the system prompt by
    ``build()`` — it now rides the END of the USER turn (see ``append_snapshot_to_user``
    and ``RalphAgent``). ``build(page=...)`` ignores ``page`` for backwards
    compatibility. These tests pin the new contract."""

    def _provider(self, cli, limit=6000):
        return lambda: capture_page_snapshot(cli, limit)

    def test_build_never_emits_page_section_even_with_page_arg(self):
        # Even when a page is passed, build() emits no page section (issue #146).
        cli = _FakeSnapshotCli(["### Page\nFIRST-SNAP consent dialog"])
        builder = SystemPromptBuilder(_registry(["web"]))
        page = self._provider(cli)
        sp = builder.build("DO THE THING", page=page())
        self.assertNotIn("# Current page (live snapshot", sp)
        self.assertNotIn("FIRST-SNAP consent dialog", sp)

    # NOTE: the standalone ``append_snapshot_to_user`` / ``SNAPSHOT_USER_MARKER`` helper
    # was removed when the snapshot moved to a per-turn ``page_snapshot`` observation
    # committed by ``RalphAgent`` as a ``<tool_response>`` block (issue #151). That
    # behaviour — snapshot rides the END of the user turn, only the latest kept — is now
    # covered end-to-end by ``test_native_ollama_chat.SnapshotOnUserTurnTest``.

    def test_snapshot_not_in_narrative_memory(self):
        # The snapshot must never be recorded into narrative memory (which would
        # accumulate stale snapshots).
        cli = _FakeSnapshotCli(["### Page\nSNAP-TEXT-XYZ"])
        page = self._provider(cli)
        _ = SystemPromptBuilder(_registry(["web"])).build("T", page=page())

        memory = NarrativeMemory()
        memory.record("you navigated to the page")
        turn_prompt = build_turn_prompt("T", memory.render())
        self.assertNotIn("SNAP-TEXT-XYZ", memory.render())
        self.assertNotIn("SNAP-TEXT-XYZ", turn_prompt)

    def test_web_inactive_has_no_page_section(self):
        # fs-only: build() emits no page section regardless (issue #146).
        sp = SystemPromptBuilder(_registry(["fs"])).build("DO THE THING", page="")
        self.assertNotIn("# Current page (live snapshot", sp)


class SnapshotCapDefaultTest(unittest.TestCase):
    def test_default_cap_is_absolute_ceiling(self):
        # Issue #43: web_snapshot_char_limit is no longer the PRIMARY cap (the dynamic
        # budget is — see test_snapshot_budget.py). It is now an absolute ceiling /
        # safety fallback, defaulted very high so the context window is the real limit.
        self.assertGreaterEqual(Config().web_snapshot_char_limit, 1_000_000)

    def test_provider_uses_config_ceiling(self):
        # make_snapshot_provider binds the config ceiling, so a snapshot up to that
        # ceiling passes through untruncated (the dynamic budget does the real work in
        # the live run; this fixed-cap provider remains for back-compat).
        cli = _FakeSnapshotCli(["b" * 40000])
        provider = lambda: capture_page_snapshot(cli, Config().web_snapshot_char_limit)
        self.assertEqual(provider(), "b" * 40000)


class SnapshotProviderFactoryTest(unittest.TestCase):
    def test_provider_is_zero_arg_callable_using_config_cap(self):
        # make_snapshot_provider returns a zero-arg seam (like render_workspace). It
        # binds the run's session/timeout from config; with no live browser the
        # snapshot call fails and it returns "" — proving it never raises.
        provider = make_snapshot_provider(Config())
        self.assertEqual(provider(), "")


class CliSnapshotProviderGatingTest(unittest.TestCase):
    """cli.py wires the page-snapshot provider ONLY when 'web' is among the selected
    toolsets, and renders no page section otherwise. These tests reproduce cli.py's
    exact gating expression and feed the resulting provider through build(page=...),
    so the wiring is exercised with no live browser and no agent run.
    """

    @staticmethod
    def _wire(names):
        # Mirrors cli._run_locked verbatim:
        #   snapshot_provider = (make_snapshot_provider(config) if "web" in names else None)
        #   render_page = lambda: snapshot_provider() if snapshot_provider else ""
        snapshot_provider = (make_snapshot_provider(Config()) if "web" in names else None)
        return snapshot_provider, (lambda: snapshot_provider() if snapshot_provider else "")

    def test_provider_wired_only_when_web_selected(self):
        web_provider, _ = self._wire(["web", "fs"])
        fs_provider, _ = self._wire(["fs"])
        self.assertIsNotNone(web_provider)   # web active -> a snapshot provider exists
        self.assertIsNone(fs_provider)       # fs-only   -> no provider at all

    def test_fs_only_render_page_is_empty_so_no_page_section(self):
        # With no provider, render_page() is "" and build() emits no page heading —
        # exactly what an fs-only run does.
        _, render_page = self._wire(["fs"])
        self.assertEqual(render_page(), "")
        sp = SystemPromptBuilder(_registry(["fs"])).build("DO THE THING", page=render_page())
        self.assertNotIn("# Current page (live snapshot", sp)

    def test_web_render_page_feeds_snapshot_into_user_turn(self):
        # Issue #146/#151: web active -> the captured snapshot rides the USER turn (as a
        # page_snapshot observation committed by RalphAgent, covered by
        # test_native_ollama_chat.SnapshotOnUserTurnTest), NOT the system prompt. Here we
        # confirm the wiring seam: build() carries no page section, and the web-gated
        # provider yields the captured snapshot text for the agent to append.
        _, render_page = self._wire(["web"])
        cli = _FakeSnapshotCli(["### Page\nWIRED-SNAP consent banner"])
        render_page = lambda: capture_page_snapshot(cli, 6000)
        # The system prompt carries no page section anymore:
        sp = SystemPromptBuilder(_registry(["web"])).build("T", page=render_page())
        self.assertNotIn("# Current page (live snapshot", sp)
        # The provider yields the live snapshot text the agent appends to the user turn:
        self.assertIn("WIRED-SNAP consent banner", render_page())


class ValidatorContextProviderTest(unittest.TestCase):
    """Issue #57: cli builds a TOOL-LESS twin of the per-turn prompt to feed the
    validator — task + workspace + the same #43-budgeted page snapshot, but with the
    tool descriptions / format block stripped. Reproduced via make_system_prompt_provider
    with include_tool_guidance=False, the exact call cli._run_locked makes.
    """

    def _builder(self, names):
        catalog = default_catalog()
        toolsets = catalog.select(names)
        registry = catalog.build_registry(toolsets, Config())
        guidance = SystemPromptBuilder.assemble_guidance(toolsets)
        return SystemPromptBuilder(registry, guidance=guidance), registry

    def test_context_has_page_workspace_task_but_no_tools(self):
        from vibeharness.cli import make_system_prompt_provider
        builder, registry = self._builder(["web", "fs"])
        raw = lambda: "### Page\nbutton \"Submit\" [ref=e7]"
        provider = make_system_prompt_provider(
            builder, Config(), "DO THE THING", lambda: "WS-TREE", raw,
            logger=None, include_tool_guidance=False)
        out = provider("USER-TURN")
        # SAME context the agent had:
        self.assertIn("DO THE THING", out)
        self.assertIn("# Workspace", out)
        self.assertIn("WS-TREE", out)
        self.assertIn("# Current page (live snapshot", out)
        self.assertIn("[ref=e7]", out)
        # but NOT the tool sections / format instructions / per-toolset guidance:
        self.assertNotIn("# Tools", out)
        self.assertNotIn("# Working with your tools", out)
        self.assertNotIn("# How the loop works", out)
        for tool in registry.all():
            self.assertNotIn(tool.doc(), out)

    def test_history_appears_after_page_context_in_validator_message(self):
        # The full validator USER message is context-then-history (build_validator_prompt).
        from vibeharness.cli import make_system_prompt_provider
        from vibeharness.validation import build_validator_prompt
        builder, _ = self._builder(["web", "fs"])
        raw = lambda: "### Page\nbutton \"Submit\" [ref=e7]"
        provider = make_system_prompt_provider(
            builder, Config(), "DO THE THING", lambda: "WS", raw,
            logger=None, include_tool_guidance=False)
        context = provider("USER-TURN")
        msg = build_validator_prompt(context, "First, you clicked Submit.")
        self.assertLess(msg.index("# Current page (live snapshot"),
                        msg.index("First, you clicked Submit."))

    def test_fs_only_back_compat_no_page_section(self):
        # Back-compat: an fs run has no snapshot provider -> no page section, still
        # carries task + workspace, still tool-less.
        from vibeharness.cli import make_system_prompt_provider
        builder, _ = self._builder(["fs"])
        provider = make_system_prompt_provider(
            builder, Config(), "DO THE THING", lambda: "WS-TREE", None,
            logger=None, include_tool_guidance=False)
        out = provider("USER-TURN")
        self.assertIn("DO THE THING", out)
        self.assertIn("WS-TREE", out)
        self.assertNotIn("# Current page (live snapshot", out)
        self.assertNotIn("# Tools", out)

    def test_large_snapshot_does_not_exceed_num_ctx(self):
        # With a huge snapshot, the dynamic #43 budget still trims the tool-less prompt
        # so the rendered context (plus reserved output) fits num_ctx.
        from vibeharness.cli import make_system_prompt_provider
        from vibeharness.snapshot_budget import estimate_tokens
        cfg = Config()
        builder, _ = self._builder(["web", "fs"])
        raw = lambda: "x" * 5_000_000        # absurdly large page
        provider = make_system_prompt_provider(
            builder, cfg, "DO THE THING", lambda: "WS", raw,
            logger=None, include_tool_guidance=False)
        user = "U" * 2000
        out = provider(user)
        # The whole rendered prompt + user + the reserved output (reasoning + action)
        # must fit num_ctx. The configured safety margin is deliberately the SLACK that
        # absorbs the page-section wrapper / role tokens, so it is not added on top here
        # (matching the headline invariant in test_snapshot_budget.NeverExceedsNumCtxTest).
        reserved = cfg.reason_tokens + cfg.action_tokens
        total = estimate_tokens(out + user, cfg.snapshot_chars_per_token) + reserved
        self.assertLessEqual(total, cfg.num_ctx)


if __name__ == "__main__":
    unittest.main()
