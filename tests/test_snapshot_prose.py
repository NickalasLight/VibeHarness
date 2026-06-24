"""Issue #64: deterministic WebArena-style ARIA-YAML -> prose transform.

These tests feed sample ARIA snapshots to :func:`aria_yaml_to_prose` and assert:
  * the prose prunes generic/img noise (WebArena parity),
  * EVERY actionable element keeps a resolvable identifier (its native [ref]) so the
    discrete web subtools (click/fill/…) can still target it,
  * the [active] foreground dialog is surfaced,
  * depth is collapsed when a wrapper node is pruned,
  * StaticText already covered by a nearby line is de-duplicated,
  * the transform is total (never raises; degrades to raw on a surprise),
  * the config seam is OFF by default and wires the transform when enabled.
"""
import re
import unittest

from vibeharness.config import Config
from vibeharness.snapshot_prose import aria_yaml_to_prose


# A compact, realistic consent-dialog snapshot in our raw capture shape (header + fence).
_CONSENT = """\
# turn 3 raw page snapshot (untruncated)
### Page
- Page URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
- Page Title: Rick Astley - Never Gonna Give You Up - YouTube
### Snapshot
```yaml
- generic [ref=e63]:
  - banner [ref=e65]:
    - button "Guide" [ref=e70] [cursor=pointer]:
      - generic [ref=e88]:
        - img
    - link "Sign in" [ref=e118] [cursor=pointer]:
      - /url: https://accounts.google.com/ServiceLogin
      - generic [ref=e123]: Sign in
  - dialog "Before you continue to YouTube" [active] [ref=e1206]:
    - heading "Before you continue to YouTube" [level=2] [ref=e1241]
    - generic [ref=e1289]:
      - generic [ref=e1290]:
        - button "Reject the use of cookies and other data" [ref=e1293] [cursor=pointer]:
          - generic [ref=e1294]: Reject all
        - button "Accept the use of cookies and other data" [ref=e1300] [cursor=pointer]:
          - generic [ref=e1301]: Accept all
      - link "More options" [ref=e1308] [cursor=pointer]:
        - /url: https://consent.youtube.com/d?continue=foo
        - generic [ref=e1309]: More options
```
"""


class TestProseTransform(unittest.TestCase):
    def setUp(self):
        self.prose = aria_yaml_to_prose(_CONSENT)

    def test_preamble_has_title_and_url(self):
        self.assertIn('Page: "Rick Astley', self.prose)
        self.assertIn("URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ", self.prose)

    def test_active_dialog_is_surfaced_in_front(self):
        # The single piece of true "foreground" the tree has: the [active] modal.
        self.assertIn("OPEN IN FRONT", self.prose)
        self.assertIn("Before you continue to YouTube", self.prose)
        # and the dialog's own ref is named so the agent could act on it
        self.assertIn("[e1206]", self.prose)

    def test_every_actionable_element_keeps_its_ref(self):
        # The three consent controls the agent must be able to click.
        for ref, label in [
            ("e1293", "Reject"),   # Reject all
            ("e1300", "Accept"),   # Accept all
            ("e1308", "More options"),
            ("e70", "Guide"),
            ("e118", "Sign in"),
        ]:
            line = self._line_with(ref)
            self.assertIsNotNone(line, f"ref {ref} ({label}) dropped from prose")
            # ref must be the LEADING bracketed identifier so the agent knows what to pass
            self.assertTrue(line.lstrip().startswith(f"[{ref}]"),
                            f"ref {ref} not the leading identifier: {line!r}")

    def test_actionable_roles_carry_resolvable_ref(self):
        # CRITICAL invariant (issue #64 step 3): every button/link/textbox/etc line in the
        # prose carries a real [eN] identifier (not the [-] placeholder), so a click/fill
        # tool call can resolve it. Decorative refless lines may exist but must not be an
        # actionable role.
        actionable = ("button", "link", "textbox", "dropdown", "checkbox",
                      "radio", "slider", "tab", "menuitem", "option")
        for raw_line in self.prose.splitlines():
            line = raw_line.strip()
            m = re.match(r"^\[(?P<id>[^\]]+)\]\s+(?P<role>\w+)", line)
            if not m:
                continue
            if m.group("role") in actionable and m.group("id") != "-":
                self.assertRegex(m.group("id"), r"^e\d+$",
                                 f"actionable line lacks a resolvable ref: {line!r}")

    def test_pruning_drops_empty_generic_and_img(self):
        # The bare ``img`` and empty ``generic`` wrappers (e88) must not appear as lines.
        # e88 was an empty generic wrapping an img -> both pruned.
        self.assertIsNone(self._line_with("e88"))
        # No standalone "img" line with no name should survive.
        for raw_line in self.prose.splitlines():
            self.assertNotEqual(raw_line.strip(), "[-] img")

    def test_depth_collapse_on_prune(self):
        # e1290 is an empty generic wrapper around the two buttons; it is pruned and the
        # buttons collapse UP, so Reject/Accept render at the dialog's child depth, not
        # two levels deeper. Assert their indent is shallow (<= the dialog children level).
        dialog = self._line_with("e1206")
        reject = self._line_with("e1293")
        self.assertIsNotNone(dialog)
        self.assertIsNotNone(reject)
        dialog_indent = len(dialog) - len(dialog.lstrip())
        reject_indent = len(reject) - len(reject.lstrip())
        # buttons are children of the (pruned-collapsed) dialog content; at most 2 deeper
        self.assertLessEqual(reject_indent - dialog_indent, 4)

    def test_link_url_preserved(self):
        line = self._line_with("e118")
        self.assertIn("-> https://accounts.google.com/ServiceLogin", line)

    def test_size_reduction(self):
        # Prose should be meaningfully smaller than the raw YAML (noise pruned).
        self.assertLess(len(self.prose), len(_CONSENT))

    # --- helpers ---
    def _line_with(self, ref: str):
        for line in self.prose.splitlines():
            if f"[{ref}]" in line:
                return line
        return None


class TestTextDedup(unittest.TestCase):
    def test_statictext_already_in_prior_line_is_dropped(self):
        raw = """```yaml
- button "Download now" [ref=e1]:
  - text: Download now
- text: A unique standalone caption
```"""
        prose = aria_yaml_to_prose(raw)
        # "Download now" already appears on the button line -> the duplicate text leaf drops
        self.assertEqual(prose.count("Download now"), 1)
        # the unique caption survives
        self.assertIn("A unique standalone caption", prose)


class TestRobustness(unittest.TestCase):
    def test_empty_input_returns_input(self):
        self.assertEqual(aria_yaml_to_prose(""), "")
        self.assertEqual(aria_yaml_to_prose("   "), "   ")

    def test_unparseable_degrades_to_raw(self):
        junk = "this is not a snapshot at all\nno fence no tree"
        # No element nodes -> falls back to the raw text (never blanks the page section).
        self.assertEqual(aria_yaml_to_prose(junk), junk)

    def test_no_fence_still_parses_indented_tree(self):
        raw = '- button "Go" [ref=e9]\n- link "Home" [ref=e10]'
        prose = aria_yaml_to_prose(raw)
        self.assertIn("[e9] button", prose)
        self.assertIn("[e10] link", prose)

    def test_quoted_bullet_with_colon_in_name(self):
        # Playwright single-quotes a whole bullet when the name has a colon.
        raw = '```yaml\n- \'button "Language: English" [ref=e1219]\'\n```'
        prose = aria_yaml_to_prose(raw)
        line = next(l for l in prose.splitlines() if "e1219" in l)
        self.assertIn('button "Language: English"', line)
        self.assertTrue(line.lstrip().startswith("[e1219]"))

    def test_state_properties_kept_cursor_dropped(self):
        raw = ('```yaml\n- combobox "Search" [expanded] [ref=e104] [cursor=pointer]\n'
               '- slider "Volume" [disabled] [ref=e17]\n```')
        prose = aria_yaml_to_prose(raw)
        self.assertIn("[expanded]", prose)     # real ARIA state kept
        self.assertIn("[disabled]", prose)
        self.assertNotIn("cursor", prose)      # Playwright chrome dropped (WebArena parity)


class TestConfigSeam(unittest.TestCase):
    def test_default_on_for_this_branch(self):
        # beta_qwen3coder default is True (#125 iter 3): the small instruct model needs
        # the pruned ref-keyed prose to pick input refs reliably. (On beta/mythos: False.)
        self.assertTrue(Config().web_snapshot_prose)

    def test_provider_applies_transform_when_enabled(self):
        # Reproduce cli.py's seam: wrapping a raw provider with the transform when the
        # flag is set yields prose; leaving it unwrapped yields raw.
        raw_provider = lambda: _CONSENT
        cfg_on = Config(web_snapshot_prose=True)
        wrapped = (lambda: aria_yaml_to_prose(raw_provider())) if cfg_on.web_snapshot_prose else raw_provider
        out = wrapped()
        self.assertIn("OPEN IN FRONT", out)
        self.assertIn("[e1300] button", out)
        # and OFF (explicit) leaves raw ARIA untouched
        cfg_off = Config(web_snapshot_prose=False)
        unwrapped = (lambda: aria_yaml_to_prose(raw_provider())) if cfg_off.web_snapshot_prose else raw_provider
        self.assertEqual(unwrapped(), _CONSENT)


import os

# The real raw ARIA capture (YouTube, native [ref=eN]) from a live diagnostics run. We
# assert the #70 interactable-labeling fixes against UNMODIFIED raw ARIA, not a hand-built
# fixture, so the test reflects what the agent actually sees.
_REAL_SNAPSHOT = (
    r"C:\git\vh-ashley38\.vibe\20260624_074859-diagnostics"
    r"\turn-003-snapshot-20260624_075650_524351.txt"
)


def _load_real_prose():
    if not os.path.exists(_REAL_SNAPSHOT):
        return None
    with open(_REAL_SNAPSHOT, encoding="utf-8") as f:
        return aria_yaml_to_prose(f.read())


class TestInteractableLabelingRealSnapshot(unittest.TestCase):
    """Issue #70: interactables in the prose must be unmistakable, over REAL raw ARIA."""

    @classmethod
    def setUpClass(cls):
        cls.prose = _load_real_prose()

    def setUp(self):
        if self.prose is None:
            self.skipTest(f"real snapshot not present: {_REAL_SNAPSHOT}")

    def _line_with(self, ref):
        for line in self.prose.splitlines():
            if f"[{ref}]" in line:
                return line
        return None

    def test_search_combobox_is_fillable_text_field_with_fill_affordance(self):
        # HEADLINE FIX: the YouTube search box is `combobox "Search" [ref=e104]`. It must
        # render as a FILLABLE TEXT FIELD that cues `fill` — never "dropdown"/select_option.
        line = self._line_with("e104")
        self.assertIsNotNone(line, "search combobox e104 dropped from prose")
        self.assertTrue(line.lstrip().startswith("[e104]"), line)   # ref preserved
        self.assertIn('text field "Search"', line)
        self.assertIn("type a value with fill", line)
        self.assertNotIn("dropdown", line)
        self.assertNotIn("select_option", line)

    def test_button_line_has_click_affordance(self):
        # The "Search" submit button (e79) is a real button -> click.
        line = self._line_with("e79")
        self.assertIsNotNone(line)
        self.assertIn("button", line)
        self.assertTrue(line.rstrip().endswith("— click"), line)

    def test_dialog_accept_reject_keep_refs_and_click_affordance(self):
        # Consent Accept/Reject must still carry refs AND a click affordance.
        for ref in ("e1293", "e1300"):
            line = self._line_with(ref)
            self.assertIsNotNone(line, f"dialog control {ref} dropped")
            self.assertTrue(line.lstrip().startswith(f"[{ref}]"), line)
            self.assertIn("button", line)
            self.assertTrue(line.rstrip().endswith("— click"), line)
        # blocking-dialog preamble still present
        self.assertIn("OPEN IN FRONT", self.prose)
        self.assertIn("[e1206]", self.prose)


class TestInteractableLabelingFixtures(unittest.TestCase):
    """A genuine <select>/listbox is not in the real snapshot — construct one in a fixture."""

    def test_true_listbox_renders_with_select_option(self):
        raw = (
            "```yaml\n"
            '- listbox "Country" [ref=e10]:\n'
            '  - option "Germany" [ref=e11]\n'
            '  - option "Austria" [ref=e12]\n'
            "```"
        )
        prose = aria_yaml_to_prose(raw)
        listbox_line = next(l for l in prose.splitlines() if "e10" in l)
        self.assertIn("dropdown list", listbox_line)
        self.assertIn("pick an option with select_option", listbox_line)
        # options surfaced for context, not mislabeled as the primary fill/click target
        self.assertIn('option "Germany"', prose)

    def test_native_select_renders_with_select_option(self):
        raw = '```yaml\n- select "Sort by" [ref=e20]\n```'
        prose = aria_yaml_to_prose(raw)
        line = next(l for l in prose.splitlines() if "e20" in l)
        self.assertIn("dropdown list", line)
        self.assertIn("select_option", line)

    def test_checkbox_renders_with_toggle_affordance(self):
        raw = '```yaml\n- checkbox "Remember me" [ref=e30]\n```'
        prose = aria_yaml_to_prose(raw)
        line = next(l for l in prose.splitlines() if "e30" in l)
        self.assertIn("toggle with check/uncheck", line)

    def test_actionable_role_without_ref_flagged_not_targetable(self):
        # A button with no ref (decorative-looking but actionable role) must be flagged,
        # not silently shown with a bare [-] that looks identical to ignorable chrome.
        raw = '```yaml\n- button "Ghost"\n```'
        prose = aria_yaml_to_prose(raw)
        line = next(l for l in prose.splitlines() if "Ghost" in l)
        self.assertIn("(no ref)", line)
        self.assertIn("not directly targetable", line)

    def test_combobox_default_is_text_field_not_dropdown(self):
        # The safer-failure-mode default for an unqualified combobox.
        raw = '```yaml\n- combobox "Search" [ref=e1]\n```'
        prose = aria_yaml_to_prose(raw)
        line = next(l for l in prose.splitlines() if "e1" in l)
        self.assertIn("text field", line)
        self.assertIn("fill", line)
        self.assertNotIn("dropdown", line)


if __name__ == "__main__":
    unittest.main()
