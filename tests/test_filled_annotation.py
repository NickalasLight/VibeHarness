"""Dedicated test suite for fill annotation (issues #205 and #227).

Issue #205 ground-truth: live_control_values reads ACTUAL current DOM values from the ARIA
snapshot — never a cache of intended action args. This remains the source of truth for what
the PAGE holds; it is no longer used for the snapshot annotation.

Issue #227 ground-truth: annotate_filled_snapshot now consumes a {ref: turn_index} fill_map
(agent-committed fills) rather than live DOM values. Only refs the agent ITSELF successfully
set with a non-empty value carry the '[already filled on turn X]' annotation. Prefilled/
placeholder DOM values are never annotated. The empty-value guard is enforced at recording
time: '' values never enter the map, never produce an annotation.

Each test below either:
  * constructs a DOM/ARIA snapshot state and asserts live_control_values against it (issue #205
    contract — DOM read is accurate); OR
  * constructs a fill_map and asserts annotate_filled_snapshot output (issue #227 contract —
    annotation reflects committed agent fills, not DOM values).
"""
import unittest

from vibeharness.web import (annotate_filled_snapshot, live_control_values)
from vibeharness.snapshot_prose import aria_yaml_to_prose


def _yaml(*lines: str) -> str:
    """Wrap ARIA-YAML body lines in the fenced block the raw capture emits."""
    return "```yaml\n" + "\n".join(lines) + "\n```"


def _prose(raw: str) -> str:
    """Convert raw ARIA YAML to prose (the display view the model sees)."""
    return aria_yaml_to_prose(raw)


# ---------------------------------------------------------------------------
# Issue #205 — live_control_values reads ACTUAL DOM values
# ---------------------------------------------------------------------------

class LiveControlValuesTest(unittest.TestCase):
    """``live_control_values`` reads the ACTUAL current value from the ARIA snapshot."""

    def test_empty_text_input_is_not_filled(self):
        raw = _yaml('- textbox "First name" [ref=e1]')
        self.assertEqual(live_control_values(raw), {})

    def test_committed_text_input_shows_real_value(self):
        raw = _yaml('- textbox "First name" [ref=e1]: Jason')
        self.assertEqual(live_control_values(raw), {"e1": "Jason"})

    def test_opened_popup_without_commit_stays_empty(self):
        # select_option(e689, "May 2016") opened a date-picker calendar but committed no
        # day -> the date field is still empty in the DOM. The OLD intended-args cache
        # falsely stamped it filled; the live read must show it empty.
        raw = _yaml(
            '- textbox "Date of birth" [ref=e689]',
            '- dialog "calendar" [ref=e700]:',
            '  - generic [ref=e701]: May 2016',
            '  - button "2016-05-01" [ref=e702]: "1"',
        )
        self.assertNotIn("e689", live_control_values(raw))

    def test_page_reformatted_value_uses_dom_value_not_intended_arg(self):
        # The page committed/reformatted the agent's intended "May 2016" to an ISO date.
        # live_control_values reflects the DOM value, not the arg.
        raw = _yaml('- textbox "Date of birth" [ref=e689]: 2016-05-01')
        self.assertEqual(live_control_values(raw), {"e689": "2016-05-01"})

    def test_native_select_committed_value(self):
        raw = _yaml('- combobox "State" [ref=e65]: TX')
        self.assertEqual(live_control_values(raw), {"e65": "TX"})

    def test_checkbox_checked_and_unchecked(self):
        checked = _yaml('- checkbox "I agree" [ref=e9] [checked]')
        unchecked = _yaml('- checkbox "Send me spam" [ref=e10]')
        self.assertEqual(live_control_values(checked), {"e9": "(checked)"})
        self.assertEqual(live_control_values(unchecked), {})

    def test_checkbox_explicit_checked_false_is_not_filled(self):
        raw = _yaml('- checkbox "I agree" [ref=e9] [checked=false]')
        self.assertEqual(live_control_values(raw), {})

    def test_radio_checked_and_unchecked(self):
        checked = _yaml('- radio "Yes" [ref=e11] [checked]')
        unchecked = _yaml('- radio "No" [ref=e12]')
        self.assertEqual(live_control_values(checked), {"e11": "(checked)"})
        self.assertEqual(live_control_values(unchecked), {})

    def test_switch_checked(self):
        raw = _yaml('- switch "Notifications" [ref=e13] [checked]')
        self.assertEqual(live_control_values(raw), {"e13": "(checked)"})

    def test_multi_select_joins_selected_options(self):
        raw = _yaml(
            '- listbox "Skills" [ref=e30]:',
            '  - option "Python" [selected]',
            '  - option "Go" [selected]',
            '  - option "Rust"',
        )
        self.assertEqual(live_control_values(raw), {"e30": "Python, Go"})

    def test_listbox_with_no_selection_is_empty(self):
        raw = _yaml(
            '- listbox "Skills" [ref=e30]:',
            '  - option "Python"',
            '  - option "Go"',
        )
        self.assertEqual(live_control_values(raw), {})

    def test_readonly_with_value_is_filled(self):
        # A readonly field still reflects a real DOM value -> live_control_values sees it.
        raw = _yaml('- textbox "Reference" [ref=e40] [readonly]: REF-123')
        self.assertEqual(live_control_values(raw), {"e40": "REF-123"})

    def test_disabled_empty_is_not_filled(self):
        raw = _yaml('- textbox "Locked" [ref=e41] [disabled]')
        self.assertEqual(live_control_values(raw), {})

    def test_cleared_value_flips_back_to_empty(self):
        # Same ref, two successive snapshots: filled then cleared. The annotator is
        # stateless per-snapshot, so the cleared snapshot shows empty.
        filled = _yaml('- textbox "City" [ref=e2]: Dallas')
        cleared = _yaml('- textbox "City" [ref=e2]')
        self.assertEqual(live_control_values(filled), {"e2": "Dallas"})
        self.assertEqual(live_control_values(cleared), {})

    def test_multiple_controls_only_filled_annotated(self):
        raw = _yaml(
            '- textbox "First name" [ref=e1]: Jason',
            '- textbox "Last name" [ref=e2]',
            '- textbox "Email" [ref=e3]: jason@example.com',
            '- checkbox "Agree" [ref=e4]',
        )
        self.assertEqual(live_control_values(raw),
                         {"e1": "Jason", "e3": "jason@example.com"})

    def test_stale_ref_absent_from_dom_gets_no_annotation(self):
        # A ref that an old cache "knew" but is GONE from the snapshot yields nothing.
        raw = _yaml('- button "Continue" [ref=e81]')
        self.assertNotIn("e689", live_control_values(raw))

    def test_changed_ref_does_not_leak_old_value(self):
        # The field re-rendered under a NEW ref; the old ref's stale value must not appear.
        raw = _yaml('- textbox "City" [ref=e99]')
        self.assertEqual(live_control_values(raw), {})

    def test_empty_and_blank_snapshot_safe(self):
        self.assertEqual(live_control_values(""), {})
        self.assertEqual(live_control_values("   "), {})

    def test_value_with_inner_colon_preserved(self):
        raw = _yaml('- textbox "Time" [ref=e5]: 10:30 AM')
        self.assertEqual(live_control_values(raw), {"e5": "10:30 AM"})

    def test_spinbutton_value(self):
        raw = _yaml('- spinbutton "Quantity" [ref=e6]: 7')
        self.assertEqual(live_control_values(raw), {"e6": "7"})

    def test_button_with_label_value_is_not_a_filled_control(self):
        # A nav button whose label contains a colon must not be mistaken for a filled value.
        raw = _yaml('- button "Language: English" [ref=e1219]')
        self.assertEqual(live_control_values(raw), {})


# ---------------------------------------------------------------------------
# Issue #227 — annotate_filled_snapshot uses agent fill_map, not DOM values
# ---------------------------------------------------------------------------

class AnnotateFillMapTest(unittest.TestCase):
    """``annotate_filled_snapshot`` stamps ``[already filled on turn X]`` for tracked refs only.

    The source of truth is the agent's ``_fill_map: {ref: turn_index}`` — NOT live DOM values.
    Prefilled/placeholder DOM values are never in the map and are never annotated.
    """

    def _snap(self, *lines: str) -> str:
        """Build a prose snapshot with minimal ref-bearing lines."""
        # produce a simple plain-text snapshot (mimics the display view the model sees)
        return "\n".join(lines)

    def test_tracked_ref_gets_already_filled_annotation(self):
        snap = "[e1] textbox First name"
        result = annotate_filled_snapshot(snap, {"e1": 3})
        self.assertIn("[already filled on turn 3]", result)

    def test_turn_index_is_correct(self):
        snap = "[e5] textbox Email"
        result = annotate_filled_snapshot(snap, {"e5": 7})
        self.assertIn("[already filled on turn 7]", result)

    def test_untracked_ref_has_no_annotation(self):
        # e2 has a live DOM value but is NOT in the fill_map — must not be annotated.
        snap = "[e2] textbox Last name"
        result = annotate_filled_snapshot(snap, {"e1": 1})
        self.assertNotIn("already filled", result)

    def test_placeholder_dom_value_not_annotated(self):
        # A snapshot line for a control that holds a placeholder (e.g. "75201") but was
        # NEVER filled by the agent — not in fill_map — must not carry the annotation.
        snap = "[e10] textbox ZIP"  # DOM may have "75201" but agent never set it
        result = annotate_filled_snapshot(snap, {})  # empty fill_map
        self.assertNotIn("already filled", result)

    def test_multiple_refs_only_tracked_ones_annotated(self):
        snap = "\n".join([
            "[e1] textbox First name",
            "[e2] textbox Last name",
            "[e3] textbox Email",
        ])
        fill_map = {"e1": 2, "e3": 4}
        result = annotate_filled_snapshot(snap, fill_map)
        lines = result.splitlines()
        self.assertIn("[already filled on turn 2]", lines[0])
        self.assertNotIn("already filled", lines[1])
        self.assertIn("[already filled on turn 4]", lines[2])

    def test_empty_fill_map_returns_snapshot_unchanged(self):
        snap = "[e1] textbox Name"
        self.assertEqual(annotate_filled_snapshot(snap, {}), snap)

    def test_empty_snapshot_returns_empty(self):
        self.assertEqual(annotate_filled_snapshot("", {"e1": 1}), "")

    def test_wording_is_exact_no_do_not_fill_again(self):
        snap = "[e1] textbox Name"
        result = annotate_filled_snapshot(snap, {"e1": 1})
        self.assertNotIn("DO NOT FILL AGAIN", result)
        self.assertNotIn("ALREADY FILLED WITH", result)
        self.assertIn("[already filled on turn 1]", result)

    def test_ref_not_in_snapshot_but_in_fill_map_no_crash(self):
        # A ref in fill_map that is absent from the snapshot (stale ref) — graceful no-op.
        snap = "[e99] button Continue"
        result = annotate_filled_snapshot(snap, {"e1": 1})
        self.assertNotIn("already filled", result)

    def test_empty_string_snapshot_with_nonempty_fill_map(self):
        self.assertEqual(annotate_filled_snapshot("", {"e1": 2}), "")

    def test_fill_map_turn_zero_renders_correctly(self):
        # Turn 0 is unusual but must not be filtered out (the guard is on empty VALUES,
        # not on turn index 0).
        snap = "[e1] textbox Name"
        result = annotate_filled_snapshot(snap, {"e1": 0})
        self.assertIn("[already filled on turn 0]", result)


# ---------------------------------------------------------------------------
# Issue #227 — empty-value guard: '' values never annotated
# ---------------------------------------------------------------------------

class EmptyValueGuardTest(unittest.TestCase):
    """The empty-value guard prevents '' fills from being annotated.

    Enforced at recording time in RalphAgent._record_fill_if_needed: a fill/type/select_option
    with an empty value ('') never enters _fill_map, so it can never produce an annotation.
    These tests verify the annotate_filled_snapshot layer is also safe even if someone
    somehow constructed a fill_map manually — the annotation only fires when the ref IS in
    fill_map (the real guard is upstream, at the agent level).
    """

    def test_annotate_filled_snapshot_empty_map_no_annotation(self):
        snap = "[e77] textbox Honeypot"
        result = annotate_filled_snapshot(snap, {})
        self.assertNotIn("already filled", result)

    def test_annotate_filled_snapshot_none_equivalent_no_annotation(self):
        # A ref with DOM value '' would NOT appear in fill_map (agent never records it).
        # Simulate by not putting it in the map.
        snap = "[e79] textbox AnotherHoneypot"
        result = annotate_filled_snapshot(snap, {})
        self.assertNotIn("already filled", result)


# ---------------------------------------------------------------------------
# Issue #227 — RalphAgent._fill_map recording behaviour (unit tests)
# ---------------------------------------------------------------------------

class FillMapRecordingTest(unittest.TestCase):
    """Verify ``RalphAgent._record_fill_if_needed`` populates ``_fill_map`` correctly.

    We test the helper method directly to avoid a full agent/LLM stack.
    """

    def _make_agent(self):
        """Build a minimal RalphAgent with no client/registry dependencies."""
        from unittest.mock import MagicMock
        from vibeharness.agent import RalphAgent
        from vibeharness.config import Config

        client = MagicMock()
        client.supports_native_tools.return_value = False
        client.supports_structured_history.return_value = False
        registry = MagicMock()
        registry.names.return_value = []
        config = Config()
        validator = MagicMock()
        return RalphAgent(client, registry, "sys", config, validator)

    def test_fill_records_ref_on_success_with_nonempty_value(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("fill", {"target": "e1", "text": "Jason"}, turn_idx=3, ok=True)
        self.assertEqual(agent._fill_map, {"e1": 3})

    def test_type_records_ref_on_success_with_nonempty_value(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("type", {"target": "e2", "text": "hello"}, turn_idx=5, ok=True)
        self.assertEqual(agent._fill_map, {"e2": 5})

    def test_select_option_records_on_success_with_nonempty_value(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("select_option", {"target": "e3", "value": "TX"}, turn_idx=2, ok=True)
        self.assertEqual(agent._fill_map, {"e3": 2})

    def test_check_records_on_success_no_value_arg(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("check", {"target": "e4"}, turn_idx=1, ok=True)
        self.assertEqual(agent._fill_map, {"e4": 1})

    def test_failed_fill_not_recorded(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("fill", {"target": "e1", "text": "Jason"}, turn_idx=3, ok=False)
        self.assertEqual(agent._fill_map, {})

    def test_empty_value_fill_not_recorded(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("fill", {"target": "e1", "text": ""}, turn_idx=1, ok=True)
        self.assertEqual(agent._fill_map, {})

    def test_empty_value_select_option_not_recorded(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("select_option", {"target": "e5", "value": ""}, turn_idx=2, ok=True)
        self.assertEqual(agent._fill_map, {})

    def test_empty_value_type_not_recorded(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("type", {"target": "e6", "text": ""}, turn_idx=1, ok=True)
        self.assertEqual(agent._fill_map, {})

    def test_non_fill_tool_not_recorded(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("click", {"target": "e7"}, turn_idx=1, ok=True)
        self.assertEqual(agent._fill_map, {})

    def test_fill_with_missing_target_not_recorded(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("fill", {"text": "hello"}, turn_idx=1, ok=True)
        self.assertEqual(agent._fill_map, {})

    def test_multiple_fills_accumulate(self):
        agent = self._make_agent()
        agent._record_fill_if_needed("fill", {"target": "e1", "text": "Jason"}, turn_idx=1, ok=True)
        agent._record_fill_if_needed("fill", {"target": "e2", "text": "Smith"}, turn_idx=2, ok=True)
        self.assertEqual(agent._fill_map, {"e1": 1, "e2": 2})

    def test_fill_map_shared_with_caller(self):
        """fill_map kwarg IS the dict — updates are visible to the caller."""
        from unittest.mock import MagicMock
        from vibeharness.agent import RalphAgent
        from vibeharness.config import Config

        client = MagicMock()
        client.supports_native_tools.return_value = False
        client.supports_structured_history.return_value = False
        registry = MagicMock()
        registry.names.return_value = []
        shared: dict[str, int] = {}
        agent = RalphAgent(client, registry, "sys", Config(), MagicMock(),
                           fill_map=shared)
        agent._record_fill_if_needed("fill", {"target": "e1", "text": "hi"}, turn_idx=4, ok=True)
        # The SAME dict is updated — the snapshot provider closure sees the change.
        self.assertIs(agent._fill_map, shared)
        self.assertEqual(shared, {"e1": 4})


# ---------------------------------------------------------------------------
# Issue #205 (kept) — calendar header surfaced in prose
# ---------------------------------------------------------------------------

class CalendarHeaderSurfacedTest(unittest.TestCase):
    """Issue #205 Defect 2: an open calendar's month/year must reach the model."""

    CAL = _yaml(
        '- dialog "Earliest available start date calendar" [ref=e189]:',
        '  - button "Previous year" [ref=e191]: «',
        '  - generic [ref=e193]: July 2026',
        '  - button "Next month" [ref=e194]: ›',
        '  - button "2026-07-21" [ref=e221]: "21"',
        '  - button "2026-07-22" [ref=e222]: "22"',
    )

    def test_calendar_open_snapshot_contains_month_year(self):
        prose = aria_yaml_to_prose(self.CAL)
        self.assertIn("Calendar showing: July 2026", prose)

    def test_no_false_calendar_line_on_ordinary_text(self):
        # A stray "May 2016" in page prose (no day grid) must NOT trigger the header.
        raw = _yaml(
            '- text: We launched in May 2016 and have grown since.',
            '- button "Submit" [ref=e1]',
        )
        self.assertNotIn("Calendar showing", aria_yaml_to_prose(raw))


if __name__ == "__main__":
    unittest.main()
