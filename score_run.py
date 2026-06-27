"""Ad-hoc scorer for a vibe job-application run log. Not part of the package."""
import json, sys, glob, os

EXPECTED = {
    "Jason": "firstName", "Statham": "lastName",
    "jason.statham.dev@example.com": "email",
    "+1 (214) 555-0199": "phone",
    "1820 McKinney Avenue": "addressLine1", "Apt 14C": "addressLine2",
    "Dallas": "city", "TX": "state", "75201": "postalCode",
    "United States": "country",
    "U.S. Citizen": "workAuth", "Senior Software Engineer": "currentTitle",
    "Lone Star Payments Inc.": "currentCompany",
    "https://www.linkedin.com/in/jason-statham-dotnet": "linkedin",
    "https://jstatham.dev": "portfolio",
    "The University of Texas at Dallas": "school",
    "Computer Science": "fieldOfStudy", "165000": "salary",
}

def main(path):
    d = json.load(open(path, encoding="utf-8"))
    turns = d.get("turns", [])
    fields_set = {}          # unique field-value pairs successfully set
    page_advances = 0
    combobox_selects = 0
    fail_count = 0
    ok_count = 0
    for t in turns:
        for a in t.get("actions", []):
            tool = a.get("tool"); ok = a.get("ok"); obs = (a.get("observation") or "")
            args = a.get("args", {})
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            if not ok:
                continue
            if tool in ("fill", "type"):
                val = args.get("text") or args.get("value") or ""
                if val:
                    fields_set[(args.get("target",""), val)] = val
            if tool == "select_option" and ("selected" in obs and "combobox" in obs):
                combobox_selects += 1
                fields_set[(args.get("target",""), args.get("value",""))] = args.get("value","")
            if tool in ("check",):
                fields_set[(args.get("target",""), "checked")] = "checked"
            if tool == "click" and ("PAGE CHANGED" in obs or "Continue" in obs
                                     or "Next" in obs or "Submit" in obs or "Review" in obs):
                page_advances += 1
    # Robust page count: highest "Step N of 8" reached anywhere in the log (obs text).
    import re as _re
    steps_seen = set()
    for t in turns:
        for a in t.get("actions", []):
            for m in _re.finditer(r"Step (\d+) of 8", a.get("observation") or ""):
                steps_seen.add(int(m.group(1)))
    max_step = max(steps_seen) if steps_seen else 1
    page_advances = max(page_advances, max_step - 1)
    unique_fields = len(fields_set)
    print(f"max step reached (of 8): {max_step}")
    print(f"file: {os.path.basename(path)}")
    print(f"finished: {d.get('finished')}")
    print(f"turns: {len(turns)}  ok_actions: {ok_count}  failed_actions: {fail_count}")
    print(f"unique field/value pairs set: {unique_fields}")
    print(f"combobox selects (succeeded): {combobox_selects}")
    print(f"page advances (Continue/Next/Submit clicked w/ nav result): {page_advances}")
    score = unique_fields + 10 * page_advances
    print(f"SCORE = {unique_fields} fields + {10*page_advances} page-bonus = {score}")

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob(r"C:/git/vibethinkharnessProto1/.vibe/2026*.json"), key=os.path.getmtime)
    main(p)
