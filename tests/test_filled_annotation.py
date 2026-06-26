"""Dedicated test suite for the filled/empty snapshot annotator (issue #205).

The annotator MUST ALWAYS reflect the page's LIVE DOM control values, derived from the
raw Playwright ARIA snapshot captured every turn — NEVER a cache of intended action args.
The original bug: an optimistic ``filled_controls`` map recorded what the agent *tried* to
set (``cli.py`` checkpoint), so a ``select_option`` that merely OPENED a calendar (returning
ok=True without committing) left the field empty yet stamped
``[ALREADY FILLED WITH 'May 2016']`` on it forever, convincing every model the field was set.

Each test below constructs a DOM/ARIA snapshot STATE and asserts the annotation against that
state — never against past actions. We exercise:
  * empty control -> never "ALREADY FILLED";
  * text/fill committed -> the REAL committed value;
  * select_option/combobox that only opened a popup (no commit) -> empty;
  * a value the PAGE changed/reformatted (e.g. "May 2016" -> "2016-05-01") -> the DOM value;
  * checkbox/radio checked & unchecked;
  * multi-select -> joined selected option names;
  * readonly (with value) -> value; disabled (empty) -> empty;
  * a value cleared after being set -> flips back to empty;
  * multiple controls -> only the truly-filled ones annotated;
  * a ref present in an old cache but absent/changed in the DOM -> no stale annotation;
  * a calendar-open snapshot surfaces its current month/year to the model.
"""
import unittest

from vibeharness.web import (annotate_filled_snapshot, live_control_values)
from vibeharness.snapshot_prose import aria_yaml_to_prose


def _yaml(*lines: str) -> str:
    """Wrap ARIA-YAML body lines in the fenced block the raw capture emits."""
    return "```yaml\n" + "\n".join(lines) + "\n```"


def _annotated(raw: str) -> str:
    """Full pipeline as the run wires it (#205): derive live values from the raw ARIA,
    then annotate the prose view the model actually sees."""
    return annotate_filled_snapshot(aria_yaml_to_prose(raw), live_control_values(raw))


class LiveControlValuesTest(unittest.TestCase):
    """``live_control_values`` reads the ACTUAL current value from the ARIA snapshot."""

    def test_empty_text_input_is_not_filled(self):
        raw = _yaml('- textbox "First name" [ref=e1]')
        self.assertEqual(live_control_values(raw), {})
        self.assertNotIn("ALREADY FILLED", _annotated(raw))

    def test_committed_text_input_shows_real_value(self):
        raw = _yaml('- textbox "First name" [ref=e1]: Jason')
        self.assertEqual(live_control_values(raw), {"e1": "Jason"})
        self.assertIn("ALREADY FILLED WITH 'Jason'", _annotated(raw))

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
        self.assertNotIn("ALREADY FILLED", _annotated(raw).split("\n")[0])
        # specifically the date field line carries no filled marker
        line = next(l for l in _annotated(raw).splitlines() if "[e689]" in l)
        self.assertNotIn("ALREADY FILLED", line)

    def test_page_reformatted_value_uses_dom_value_not_intended_arg(self):
        # The agent tried to set "May 2016"; the page committed/reformatted it to an ISO
        # date. The annotation MUST show the DOM value, not the intended arg.
        raw = _yaml('- textbox "Date of birth" [ref=e689]: 2016-05-01')
        self.assertEqual(live_control_values(raw), {"e689": "2016-05-01"})
        ann = _annotated(raw)
        self.assertIn("ALREADY FILLED WITH '2016-05-01'", ann)
        self.assertNotIn("May 2016", ann)

    def test_native_select_committed_value(self):
        raw = _yaml('- combobox "State" [ref=e65]: TX')
        self.assertEqual(live_control_values(raw), {"e65": "TX"})

    def test_checkbox_checked_and_unchecked(self):
        checked = _yaml('- checkbox "I agree" [ref=e9] [checked]')
        unchecked = _yaml('- checkbox "Send me spam" [ref=e10]')
        self.assertEqual(live_control_values(checked), {"e9": "(checked)"})
        self.assertEqual(live_control_values(unchecked), {})
        self.assertIn("ALREADY FILLED WITH '(checked)'", _annotated(checked))
        self.assertNotIn("ALREADY FILLED", _annotated(unchecked))

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
        # A readonly field still reflects a real DOM value -> annotate it.
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
        ann = _annotated(raw)
        self.assertIn("ALREADY FILLED WITH 'Jason'", ann)
        self.assertIn("ALREADY FILLED WITH 'jason@example.com'", ann)
        for line in ann.splitlines():
            if "[e2]" in line or "[e4]" in line:
                self.assertNotIn("ALREADY FILLED", line)

    def test_stale_ref_absent_from_dom_gets_no_annotation(self):
        # A ref that an old optimistic cache "knew" but is GONE from the current snapshot
        # must never be annotated — the live read only ever sees what's on the page now.
        raw = _yaml('- button "Continue" [ref=e81]')
        self.assertNotIn("e689", live_control_values(raw))
        # annotate_filled_snapshot with a stale map only marks refs that are present.
        self.assertNotIn("ALREADY FILLED",
                         annotate_filled_snapshot(aria_yaml_to_prose(raw),
                                                  {"e689": "May 2016"}))

    def test_changed_ref_does_not_leak_old_value(self):
        # The field re-rendered under a NEW ref; the old ref's stale value must not appear.
        raw = _yaml('- textbox "City" [ref=e99]')
        self.assertEqual(live_control_values(raw), {})

    def test_empty_and_blank_snapshot_safe(self):
        self.assertEqual(live_control_values(""), {})
        self.assertEqual(live_control_values("   "), {})
        self.assertEqual(annotate_filled_snapshot("", {"e1": "x"}), "")

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
