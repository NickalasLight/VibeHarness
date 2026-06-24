"""Tests for the job-application scoring layer (test_assets/score.py).

Covers:
* A correctly-filled form scores 100% (all scored fields PASS).
* A deliberately-wrong fill (firstName="Statham") fails THAT field, with
  expected="Jason", and lowers the total.
* Each match type behaves: exact mismatch fails; one_of accepts an alternative;
  numeric within tolerance passes (and outside fails); contains matches a keyword.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Import test_assets/score.py by path (test_assets is not a package).
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = REPO_ROOT / "test_assets" / "score.py"
FIXTURE_PATH = REPO_ROOT / "test_assets" / "jason_statham.json"
ANSWER_KEY_PATH = REPO_ROOT / "test_assets" / "score_evaluation.json"

_spec = importlib.util.spec_from_file_location("jobscore", SCORE_PATH)
score_mod = importlib.util.module_from_spec(_spec)
sys.modules["jobscore"] = score_mod
_spec.loader.exec_module(score_mod)

score = score_mod.score
load_answer_key = score_mod.load_answer_key
match_field = score_mod.match_field
parse_audit_log = score_mod.parse_audit_log


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def answer_key():
    return load_answer_key(ANSWER_KEY_PATH)


@pytest.fixture(scope="module")
def jason():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _perfect_fill():
    """A flat {field: value} dict that should satisfy every scored field.

    Values are chosen to be canonical form values where the fixture wording and
    the form option diverge (degree, heardAboutUs, skills checkboxes).
    """
    return {
        "firstName": "Jason",
        "lastName": "Statham",
        "email": "jason.statham.dev@example.com",
        "phone": "(214) 555-0199",
        "addressLine1": "1820 McKinney Avenue, Apt 14C",
        "city": "Dallas",
        "state": "TX",
        "postalCode": "75201",
        "country": "United States",
        "workAuthorization": "U.S. Citizen",
        "requireSponsorship": "No",
        "willingToRelocate": "No, I am not willing to relocate",
        "availableStartDate": "2026-07-21",
        "workPreference": "Hybrid",
        "currentTitle": "Senior Software Engineer",
        "currentCompany": "Lone Star Payments Inc.",
        "totalExperienceYears": 9,
        "dotnetExperienceYears": 9,
        "skills": ["C#", ".NET / .NET Core", "ASP.NET Core", "SQL Server", "Azure",
                   "Domain-Driven Design"],
        "linkedinUrl": "https://www.linkedin.com/in/jason-statham-dotnet",
        "portfolioUrl": "https://jstatham.dev",
        "highestDegree": "Bachelor's Degree",
        "school": "The University of Texas at Dallas",
        "fieldOfStudy": "Computer Science",
        "graduationDate": "2016-05",
        "salaryExpectation": "$165,000",
        "whyInterested": ("FlashTec's bet on .NET, Azure, and event-driven architecture "
                          "to move real money at five-nines reliability is exactly the "
                          "payments and ledger problem space I have worked in."),
        "heardAboutUs": "Company Website",
        "signatureName": "Jason Statham",
    }


# --------------------------------------------------------------------------- #
# 100% case
# --------------------------------------------------------------------------- #


def test_perfect_fill_scores_100(answer_key):
    report = score(_perfect_fill(), answer_key)
    assert report["score_pct"] == 100.0
    assert report["correct_count"] == report["total"]
    # every scored field is correct
    for row in report["per_field"]:
        if row["scored"]:
            assert row["correct"], f"expected PASS for {row['field']}: {row}"


def test_total_counts_only_scored_fields(answer_key):
    report = score(_perfect_fill(), answer_key)
    scored_fields = [k for k, v in answer_key["fields"].items() if v.get("scored", True)]
    assert report["total"] == len(scored_fields)
    # there are non-scored fields in the key, so total < number of fields
    assert report["total"] < len(answer_key["fields"])


# --------------------------------------------------------------------------- #
# Deliberately wrong fill
# --------------------------------------------------------------------------- #


def test_wrong_firstname_fails_that_field_and_lowers_total(answer_key):
    perfect = _perfect_fill()
    baseline = score(perfect, answer_key)

    wrong = dict(perfect)
    wrong["firstName"] = "Statham"  # wrong value
    report = score(wrong, answer_key)

    row = next(r for r in report["per_field"] if r["field"] == "firstName")
    assert row["correct"] is False
    assert row["expected"] == "Jason"
    assert row["actual"] == "Statham"

    assert report["correct_count"] == baseline["correct_count"] - 1
    assert report["score_pct"] < baseline["score_pct"]


def test_missing_field_is_marked_incorrect(answer_key):
    perfect = _perfect_fill()
    del perfect["email"]
    report = score(perfect, answer_key)
    row = next(r for r in report["per_field"] if r["field"] == "email")
    assert row["correct"] is False
    assert row["actual"] is None


# --------------------------------------------------------------------------- #
# Per-match-type behavior
# --------------------------------------------------------------------------- #


def test_exact_match():
    spec = {"match": "exact", "expected": "Jason"}
    assert match_field(spec, "Jason") is True
    assert match_field(spec, "  jason ") is True   # trimmed + case-insensitive
    assert match_field(spec, "Statham") is False


def test_one_of_accepts_alternative():
    spec = {"match": "one_of", "expected": ["TX", "Texas"]}
    assert match_field(spec, "Texas") is True       # alternative accepted
    assert match_field(spec, "tx") is True
    assert match_field(spec, "California") is False


def test_numeric_within_tolerance():
    spec = {"match": "numeric", "expected": 165000, "tolerance": 1000}
    assert match_field(spec, "$165,000") is True    # strips $ and ,
    assert match_field(spec, 165500) is True         # within tolerance
    assert match_field(spec, 170000) is False        # outside tolerance


def test_numeric_zero_tolerance():
    spec = {"match": "numeric", "expected": 9, "tolerance": 0}
    assert match_field(spec, "9 years") is True
    assert match_field(spec, 8) is False


def test_contains_matches_keyword():
    spec = {"match": "contains", "keywords": ["payments", "azure", "ledger"],
            "min_keywords": 2}
    assert match_field(spec, "Built payment ledger systems on Azure.") is True
    assert match_field(spec, "I like azure skies.") is False   # only 1 keyword


def test_contains_all_multiselect():
    spec = {"match": "contains_all",
            "keywords": ["C#", ".NET", "SQL Server", "Azure"], "min_keywords": 3}
    assert match_field(spec, ["C#", ".NET / .NET Core", "Azure", "SQL Server"]) is True
    assert match_field(spec, ["C#", "Python"]) is False


def test_unknown_match_type_raises():
    with pytest.raises(ValueError):
        match_field({"match": "frobnicate", "expected": 1}, 1)


# --------------------------------------------------------------------------- #
# Capture helper: parse_audit_log
# --------------------------------------------------------------------------- #


def test_parse_audit_log_roundtrip(tmp_path):
    log = tmp_path / "FT-APP-20260624-ABC123.log"
    log.write_text(
        "\n".join(
            [
                "=" * 78,
                "FLASHTEC JOB APPLICATION — AUDIT LOG",
                "Reference:      FT-APP-20260624-ABC123",
                "=" * 78,
                "",
                "FINAL SUBMITTED VALUES (validated)",
                "-" * 78,
                '  firstName              : "Jason"',
                '  lastName               : "Statham"',
                "  totalExperienceYears   : 9",
                '  salaryExpectation      : 165000',
                "  gender                 : [masked]",
                "",
                "=" * 78,
                "END OF LOG — FT-APP-20260624-ABC123",
                "=" * 78,
            ]
        ),
        encoding="utf-8",
    )
    filled = parse_audit_log(log)
    assert filled["firstName"] == "Jason"
    assert filled["lastName"] == "Statham"
    assert filled["totalExperienceYears"] == 9
    assert filled["salaryExpectation"] == 165000
    assert filled["gender"] == "[masked]"


def test_parse_audit_log_then_score(tmp_path, answer_key):
    # Build a minimal log from the perfect fill and confirm it scores 100%.
    perfect = _perfect_fill()
    body = ["FINAL SUBMITTED VALUES (validated)", "-" * 78]
    for k, v in perfect.items():
        body.append(f"  {k.ljust(22)} : {json.dumps(v)}")
    body += ["", "=" * 78]
    log = tmp_path / "ref.log"
    log.write_text("\n".join(body), encoding="utf-8")

    filled = parse_audit_log(log)
    report = score(filled, answer_key)
    assert report["score_pct"] == 100.0
