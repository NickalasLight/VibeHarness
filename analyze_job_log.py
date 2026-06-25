"""Parse a vibe run log and extract karriere.at job metrics.

Usage: python analyze_job_log.py <log_file>
Output: JSON with escalations, validator_saves, estimated_cost_usd
"""
import json
import re
import sys


def analyze(log_path: str) -> dict:
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return {"error": f"log not found: {log_path}"}

    # Count escalation events
    escalations = len(re.findall(r"\[ESCALATION\]", text))

    # Count validator saves (validation PASSED lines)
    validator_saves = len(re.findall(r"validation PASSED|verdict.*passed.*true|PASSED.*verdict", text, re.I))

    # Extract API usage from [API_USAGE] line (last occurrence wins)
    tokens_in = 0
    tokens_out = 0
    estimated_cost_usd = 0.0
    for m in re.finditer(
        r"\[API_USAGE\]\s+tokens_in=(\d+)\s+tokens_out=(\d+)\s+estimated_cost_usd=([\d.]+)",
        text,
    ):
        tokens_in = int(m.group(1))
        tokens_out = int(m.group(2))
        estimated_cost_usd = float(m.group(3))

    # Detect success (confirmation message)
    success_patterns = [
        r"vielen dank", r"thank you", r"application.*submitted", r"bewerbung.*eingereicht",
        r"danke f.r", r"erfolgreich.*eingereicht", r"application was sent",
    ]
    completed = any(re.search(p, text, re.I) for p in success_patterns)

    # Detect blocked (phone verification, login failure, captcha)
    blocked_patterns = [
        r"phone.*verif", r"sms.*code", r"captcha", r"login.*fail", r"kann nicht fortfahren",
    ]
    blocked = any(re.search(p, text, re.I) for p in blocked_patterns)

    # Count total turns (number of "Turn N" lines)
    turns = len(re.findall(r"^Turn \d+", text, re.M))

    return {
        "escalations": escalations,
        "validator_saves": validator_saves,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "estimated_cost_usd": estimated_cost_usd,
        "completed": completed,
        "blocked": blocked,
        "turns": turns,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_job_log.py <log_file>")
        sys.exit(1)
    result = analyze(sys.argv[1])
    print(json.dumps(result, indent=2))
