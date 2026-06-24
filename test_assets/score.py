"""Scoring layer for the FlashTec job-application benchmark.

Given the values an agent actually entered into the application form (``filled``)
and an answer key (``score_evaluation.json``), produce a per-field pass/fail
report and an overall score.  This turns the job-app tester into a measurable,
repeatable benchmark.

Quick start
-----------
    import json, pathlib
    from test_assets.score import score, load_answer_key

    answer_key = load_answer_key()                 # reads score_evaluation.json
    filled = {...}                                 # see "Capturing a run" below
    report = score(filled, answer_key)
    print(report["score_pct"], report["correct_count"], "/", report["total"])

Match types (declared per field in the answer key)
--------------------------------------------------
* ``exact``        - case-insensitive, trimmed string/equality.
* ``one_of``       - actual equals any value in the ``expected`` list.
* ``numeric``      - numbers compared within ``tolerance`` (``$``/``,`` stripped).
* ``contains``     - free text; >= ``min_keywords`` of ``keywords`` present.
* ``contains_all`` - multi-select/array; >= ``min_keywords`` keywords present
                     across the joined actual values.

Fields with ``"scored": false`` are excluded from the denominator.

Capturing a run's entered values to feed ``score()``
----------------------------------------------------
``filled`` is a flat ``dict`` of ``{field_key: value}`` using the STABLE field
keys from ``shared/applicationSchema.ts`` (firstName, lastName, ...).  Two ways
a benchmark runner can obtain it:

1. Server audit log (cleanest, recommended).
   On a successful submit the FlashTec server writes a per-application audit log
   whose path is returned in the POST /api/applications response as ``logFile``.
   That file contains a ``FINAL SUBMITTED VALUES (validated)`` section with every
   field on its own ``  key : <json-value>`` line.  ``parse_audit_log()`` below
   parses that section into the ``filled`` dict.  (Note: the EEO fields
   gender/ethnicity/veteranStatus/disabilityStatus and the signature image are
   written as ``[masked]`` - which is fine because those fields are non-scored.)

2. DOM scrape.  After the agent finishes the wizard, read each rendered control's
   value via the driver (e.g. Playwright ``input_value`` / checkbox ``checked`` /
   multi-select state) and assemble the same ``{field_key: value}`` dict.  The DOM
   ids are randomized per page load, so map controls back to stable keys by label
   or by the ``data-field``/name attribute the form exposes.

Recommendation for the job_app_benchmark repo (separate repo - do NOT change it
here): the cleanest capture would be a small read-back endpoint, e.g.
``GET /api/applications/:reference/data`` (or extending the existing
``GET /api/applications/:reference`` response) to return the full validated
``data`` JSON that the server already persists in the ``Application.data``
column.  A runner could then fetch the exact submitted object directly instead of
parsing the human-readable audit log.  Until that exists, ``parse_audit_log()``
covers the gap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

DEFAULT_ANSWER_KEY = Path(__file__).with_name("score_evaluation.json")


def load_answer_key(path: str | Path = DEFAULT_ANSWER_KEY) -> dict:
    """Load the answer-key JSON (defaults to the sibling score_evaluation.json)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #


def _norm_str(value: Any) -> str:
    """Lower-cased, whitespace-collapsed string form of any value."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _to_number(value: Any) -> float | None:
    """Best-effort numeric parse: strips $, commas, spaces, trailing text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    cleaned = re.sub(r"[,$\s]", "", str(value))
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(m.group(0)) if m else None


def _iter_actual_parts(actual: Any) -> Iterable[str]:
    """Yield string parts of an actual value (handles list/multi-select)."""
    if isinstance(actual, (list, tuple)):
        for v in actual:
            yield str(v)
    else:
        yield str(actual)


# --------------------------------------------------------------------------- #
# Match implementations
# --------------------------------------------------------------------------- #


def _match_exact(spec: dict, actual: Any) -> bool:
    return _norm_str(actual) == _norm_str(spec.get("expected"))


def _match_one_of(spec: dict, actual: Any) -> bool:
    expected = spec.get("expected")
    options = expected if isinstance(expected, (list, tuple)) else [expected]
    a = _norm_str(actual)
    return any(a == _norm_str(opt) for opt in options)


def _match_numeric(spec: dict, actual: Any) -> bool:
    exp = _to_number(spec.get("expected"))
    act = _to_number(actual)
    if exp is None or act is None:
        return False
    tol = float(spec.get("tolerance", 0))
    return abs(exp - act) <= tol


def _match_contains(spec: dict, actual: Any) -> bool:
    keywords = spec.get("keywords", []) or []
    min_kw = int(spec.get("min_keywords", 1 if keywords else 0))
    haystack = _norm_str(actual)
    if not keywords:
        # No keywords declared: treat as "must be non-empty".
        return bool(haystack)
    hits = sum(1 for kw in keywords if _norm_str(kw) in haystack)
    return hits >= min_kw


def _match_contains_all(spec: dict, actual: Any) -> bool:
    keywords = spec.get("keywords", []) or []
    min_kw = int(spec.get("min_keywords", len(keywords)))
    haystack = _norm_str(" ".join(_iter_actual_parts(actual)))
    hits = sum(1 for kw in keywords if _norm_str(kw) in haystack)
    return hits >= min_kw


_MATCHERS = {
    "exact": _match_exact,
    "one_of": _match_one_of,
    "numeric": _match_numeric,
    "contains": _match_contains,
    "contains_all": _match_contains_all,
}


def match_field(spec: dict, actual: Any) -> bool:
    """Evaluate one field's actual value against its answer-key spec."""
    matcher = _MATCHERS.get(spec.get("match"))
    if matcher is None:
        raise ValueError(f"Unknown match type: {spec.get('match')!r}")
    return matcher(spec, actual)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_MISSING = object()


def score(filled: dict, answer_key: dict) -> dict:
    """Score ``filled`` against ``answer_key``.

    Returns::

        {
          "per_field": [
            {"field", "expected", "actual", "correct", "match", "scored"},
            ...
          ],
          "correct_count": int,   # scored fields that passed
          "total": int,           # scored fields
          "score_pct": float,     # 0..100, 0 when total == 0
        }

    Non-scored fields are still reported (for visibility) but excluded from
    ``correct_count`` / ``total`` / ``score_pct``.
    """
    fields = answer_key.get("fields", answer_key)
    per_field: list[dict] = []
    correct_count = 0
    total = 0

    for field, spec in fields.items():
        scored = bool(spec.get("scored", True))
        actual = filled.get(field, _MISSING)
        present = actual is not _MISSING
        actual_val = None if not present else actual

        correct = present and match_field(spec, actual_val)

        per_field.append(
            {
                "field": field,
                "expected": spec.get("expected"),
                "actual": actual_val,
                "correct": bool(correct),
                "match": spec.get("match"),
                "scored": scored,
            }
        )

        if scored:
            total += 1
            if correct:
                correct_count += 1

    score_pct = round(100.0 * correct_count / total, 2) if total else 0.0
    return {
        "per_field": per_field,
        "correct_count": correct_count,
        "total": total,
        "score_pct": score_pct,
    }


# --------------------------------------------------------------------------- #
# Capture helper: parse the server's per-application audit log
# --------------------------------------------------------------------------- #

_AUDIT_HEADER = "FINAL SUBMITTED VALUES"


def parse_audit_log(path: str | Path) -> dict:
    """Parse a FlashTec per-application audit log into a ``filled`` dict.

    Reads the ``FINAL SUBMITTED VALUES (validated)`` section, where each line is
    ``  <key> : <json-value>``.  Masked EEO/signature values come back as the
    literal string ``[masked]`` (those fields are non-scored).
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()

    # Find the start of the final-values block.
    start = None
    for i, line in enumerate(lines):
        if _AUDIT_HEADER in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"No '{_AUDIT_HEADER}' section found in {path}")

    filled: dict[str, Any] = {}
    line_re = re.compile(r"^\s{2,}(\w+)\s*:\s*(.*)$")
    for line in lines[start:]:
        if set(line.strip()) <= {"="} and line.strip():
            break  # hit the closing separator
        if line.strip().startswith("-"):
            continue  # the dashed underline
        m = line_re.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        try:
            filled[key] = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            filled[key] = raw  # e.g. the bare token [masked]
    return filled
