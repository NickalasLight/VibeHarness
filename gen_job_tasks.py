"""Generate per-job task files from task_karriere_base.txt.

Usage: python gen_job_tasks.py
Creates task_karriere_job1.txt ... task_karriere_job9.txt
"""
import pathlib

JOBS = [
    "https://www.karriere.at/jobs/10019804",  # SCIO Automation — Software Inbetriebnahmetechniker C#
    "https://www.karriere.at/jobs/10017465",  # Fill GmbH — Senior Softwareentwickler C#/.NET
    "https://www.karriere.at/jobs/7814933",   # TGW Logistics — Senior C# Entwickler
    "https://www.karriere.at/jobs/7780757",   # KNAPP — Softwareentwickler C#
    "https://www.karriere.at/jobs/7831088",   # FERCHAU Austria — Senior Software Entwickler C#/.NET
    "https://www.karriere.at/jobs/7712261",   # EBCONT — Software-Developer Microsoft C#/.net
    "https://www.karriere.at/jobs/7746332",   # epunkt GmbH — Software Developer C#/.Net (on-site)
    "https://www.karriere.at/jobs/7824701",   # Fronius — Software Engineer (C#, Azure, AI)
    "https://www.karriere.at/jobs/7824695",   # Fronius — Software Engineer (C#/.NET)
]

base = pathlib.Path("task_karriere_base.txt").read_text(encoding="utf-8")

for i, url in enumerate(JOBS, 1):
    content = base.replace("{JOB_URL}", url)
    out = pathlib.Path(f"task_karriere_job{i}.txt")
    out.write_text(content, encoding="utf-8")
    print(f"  Wrote {out} -> {url}")

print("Done.")
