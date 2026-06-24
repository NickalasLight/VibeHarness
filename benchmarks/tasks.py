"""The benchmark task ladder.

Ten file-operation tasks of strictly increasing complexity. Each is a small,
immutable :class:`Task` value with:

  - ``id``      — a short stable identifier (also defines display order),
  - ``number``  — its 1-based rung on the ladder,
  - ``prompt``  — the natural-language instruction handed to the agent verbatim,
  - ``setup``   — optional ``(workdir) -> None`` that seeds any pre-existing files
                  so the check is fully self-contained,
  - ``check``   — ``(workdir) -> (passed, detail)`` that inspects the resulting
                  working directory deterministically.

Every check reads only the files under ``workdir`` and returns a boolean plus a
human-readable detail string; none of them touch the network, a clock, or random
state, so a given filesystem state always yields the same verdict.

The prompts pin down exact paths and exact content where possible, so that a
correct run is unambiguous and an LLM judge (the real validator) and the
mechanical ``check`` agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# A check returns (passed, human-readable detail).
CheckResult = tuple[bool, str]


@dataclass(frozen=True)
class Task:
    """One benchmark task: an instruction plus a deterministic grader."""
    id: str
    number: int
    prompt: str
    check: Callable[[Path], CheckResult]
    setup: Optional[Callable[[Path], None]] = None

    def run_setup(self, workdir: Path) -> None:
        if self.setup is not None:
            self.setup(workdir)

    def run_check(self, workdir: Path) -> CheckResult:
        try:
            return self.check(workdir)
        except Exception as e:  # a check must never explode the runner
            return False, f"check raised {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Small helpers shared by the checks. Kept private and side-effect-free.
# --------------------------------------------------------------------------- #
def _read(workdir: Path, rel: str) -> Optional[str]:
    p = workdir / rel
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _norm(text: str) -> str:
    """Compare content forgivingly: normalise newlines and trailing whitespace
    so a model adding a trailing newline isn't penalised, while the substantive
    content must still match exactly."""
    return text.replace("\r\n", "\n").strip()


# --------------------------------------------------------------------------- #
# 1. Create a single file with exact content.
# --------------------------------------------------------------------------- #
def _check_1(workdir: Path) -> CheckResult:
    content = _read(workdir, "greeting.txt")
    if content is None:
        return False, "greeting.txt was not created"
    if _norm(content) != "Hello, world!":
        return False, f"greeting.txt content was {content!r}, expected 'Hello, world!'"
    return True, "greeting.txt contains the exact text"


TASK_1 = Task(
    id="create_file",
    number=1,
    prompt=(
        "Create a file named 'greeting.txt' in the current working directory whose "
        "entire contents are exactly the text: Hello, world!"
    ),
    check=_check_1,
)


# --------------------------------------------------------------------------- #
# 2. Read a file and create a derived file.
# --------------------------------------------------------------------------- #
def _setup_2(workdir: Path) -> None:
    (workdir / "source.txt").write_text("apple\nbanana\ncherry\n", encoding="utf-8")


def _check_2(workdir: Path) -> CheckResult:
    out = _read(workdir, "line_count.txt")
    if out is None:
        return False, "line_count.txt was not created"
    if _norm(out) != "3":
        return False, f"line_count.txt was {out!r}, expected '3'"
    return True, "line_count.txt correctly reports 3 lines"


TASK_2 = Task(
    id="read_derive",
    number=2,
    prompt=(
        "There is a file named 'source.txt' in the current working directory. Read it, "
        "count how many lines of text it contains, and create a new file named "
        "'line_count.txt' whose entire contents are just that number (and nothing else)."
    ),
    setup=_setup_2,
    check=_check_2,
)


# --------------------------------------------------------------------------- #
# 3. Create a small directory tree.
# --------------------------------------------------------------------------- #
def _check_3(workdir: Path) -> CheckResult:
    main = workdir / "project" / "src" / "main.txt"
    readme = workdir / "project" / "README.txt"
    if not (workdir / "project").is_dir():
        return False, "directory 'project' was not created"
    if not (workdir / "project" / "src").is_dir():
        return False, "directory 'project/src' was not created"
    if not main.is_file():
        return False, "file 'project/src/main.txt' was not created"
    if not readme.is_file():
        return False, "file 'project/README.txt' was not created"
    return True, "project/, project/src/, project/src/main.txt and project/README.txt all exist"


TASK_3 = Task(
    id="dir_tree",
    number=3,
    prompt=(
        "In the current working directory, create a directory named 'project'. Inside "
        "'project' create a subdirectory named 'src'. Then create two files: "
        "'project/README.txt' and 'project/src/main.txt'. Each file may contain any text."
    ),
    check=_check_3,
)


# --------------------------------------------------------------------------- #
# 4. Append AND prepend to an existing file.
# --------------------------------------------------------------------------- #
def _setup_4(workdir: Path) -> None:
    (workdir / "log.txt").write_text("MIDDLE\n", encoding="utf-8")


def _check_4(workdir: Path) -> CheckResult:
    content = _read(workdir, "log.txt")
    if content is None:
        return False, "log.txt is missing"
    lines = [ln for ln in _norm(content).split("\n")]
    if lines != ["HEADER", "MIDDLE", "FOOTER"]:
        return False, f"log.txt lines were {lines!r}, expected ['HEADER', 'MIDDLE', 'FOOTER']"
    return True, "log.txt is HEADER / MIDDLE / FOOTER in order"


TASK_4 = Task(
    id="append_prepend",
    number=4,
    prompt=(
        "There is a file named 'log.txt' in the current working directory containing a "
        "single line: MIDDLE. Without deleting that line, modify the file so that a new "
        "line reading exactly HEADER comes before it and a new line reading exactly "
        "FOOTER comes after it. The final file must read, line by line: HEADER, then "
        "MIDDLE, then FOOTER."
    ),
    setup=_setup_4,
    check=_check_4,
)


# --------------------------------------------------------------------------- #
# 5. Copy a file (original must remain).
# --------------------------------------------------------------------------- #
def _setup_5(workdir: Path) -> None:
    (workdir / "config.ini").write_text("[settings]\nmode=fast\n", encoding="utf-8")


def _check_5(workdir: Path) -> CheckResult:
    original = _read(workdir, "config.ini")
    backup = _read(workdir, "config.ini.bak")
    if original is None:
        return False, "original config.ini no longer exists (copy must not move it)"
    if backup is None:
        return False, "config.ini.bak was not created"
    if _norm(original) != _norm(backup):
        return False, "config.ini.bak content does not match config.ini"
    return True, "config.ini.bak is an exact copy and config.ini still exists"


TASK_5 = Task(
    id="copy_file",
    number=5,
    prompt=(
        "There is a file named 'config.ini' in the current working directory. Make a "
        "copy of it named 'config.ini.bak' with identical contents. The original "
        "'config.ini' must still exist unchanged afterwards."
    ),
    setup=_setup_5,
    check=_check_5,
)


# --------------------------------------------------------------------------- #
# 6. Move / rename a file (original name must be gone).
# --------------------------------------------------------------------------- #
def _setup_6(workdir: Path) -> None:
    (workdir / "draft.txt").write_text("version one\n", encoding="utf-8")


def _check_6(workdir: Path) -> CheckResult:
    if (workdir / "draft.txt").exists():
        return False, "draft.txt still exists; it should have been renamed (moved), not copied"
    final = _read(workdir, "final.txt")
    if final is None:
        return False, "final.txt was not created"
    if _norm(final) != "version one":
        return False, f"final.txt content was {final!r}, expected 'version one'"
    return True, "draft.txt was renamed to final.txt with its content preserved"


TASK_6 = Task(
    id="move_rename",
    number=6,
    prompt=(
        "There is a file named 'draft.txt' in the current working directory. Rename it "
        "to 'final.txt', preserving its contents. After you are done there must be no "
        "file named 'draft.txt' — only 'final.txt' holding the original text."
    ),
    setup=_setup_6,
    check=_check_6,
)


# --------------------------------------------------------------------------- #
# 7. Search-and-report into a results file.
# --------------------------------------------------------------------------- #
def _setup_7(workdir: Path) -> None:
    (workdir / "a.txt").write_text("the quick brown fox\n", encoding="utf-8")
    (workdir / "b.txt").write_text("lazy dog sleeping\n", encoding="utf-8")
    (workdir / "c.txt").write_text("another quick note\n", encoding="utf-8")


def _check_7(workdir: Path) -> CheckResult:
    report = _read(workdir, "matches.txt")
    if report is None:
        return False, "matches.txt was not created"
    body = _norm(report)
    # Expect both files that contain 'quick' to be named; b.txt must not appear.
    has_a = "a.txt" in body
    has_c = "c.txt" in body
    has_b = "b.txt" in body
    if not (has_a and has_c):
        return False, f"matches.txt should name a.txt and c.txt; got: {body!r}"
    if has_b:
        return False, f"matches.txt wrongly names b.txt (it does not contain 'quick'): {body!r}"
    return True, "matches.txt names exactly the files containing 'quick' (a.txt, c.txt)"


TASK_7 = Task(
    id="search_report",
    number=7,
    prompt=(
        "The current working directory contains several .txt files. Find every file "
        "whose contents include the word 'quick'. Create a file named 'matches.txt' "
        "that lists the file names of exactly those matching files (one name per line). "
        "Do not list files that do not contain the word 'quick'."
    ),
    setup=_setup_7,
    check=_check_7,
)


# --------------------------------------------------------------------------- #
# 8. Multi-file creation with a manifest.
# --------------------------------------------------------------------------- #
def _check_8(workdir: Path) -> CheckResult:
    expected = {"one.txt": "1", "two.txt": "2", "three.txt": "3"}
    for name, want in expected.items():
        got = _read(workdir, name)
        if got is None:
            return False, f"{name} was not created"
        if _norm(got) != want:
            return False, f"{name} content was {got!r}, expected {want!r}"
    manifest = _read(workdir, "manifest.txt")
    if manifest is None:
        return False, "manifest.txt was not created"
    body = _norm(manifest)
    for name in expected:
        if name not in body:
            return False, f"manifest.txt does not list {name}: {body!r}"
    return True, "one/two/three.txt created with correct contents and all listed in manifest.txt"


TASK_8 = Task(
    id="multi_manifest",
    number=8,
    prompt=(
        "In the current working directory, create three files: 'one.txt' containing "
        "exactly 1, 'two.txt' containing exactly 2, and 'three.txt' containing exactly 3. "
        "Then create a file named 'manifest.txt' that lists the names of all three files "
        "you created, one file name per line."
    ),
    check=_check_8,
)


# --------------------------------------------------------------------------- #
# 9. Transform content across several files.
# --------------------------------------------------------------------------- #
def _setup_9(workdir: Path) -> None:
    (workdir / "n1.txt").write_text("hello\n", encoding="utf-8")
    (workdir / "n2.txt").write_text("world\n", encoding="utf-8")
    (workdir / "n3.txt").write_text("agent\n", encoding="utf-8")


def _check_9(workdir: Path) -> CheckResult:
    expected = {"n1.txt": "HELLO", "n2.txt": "WORLD", "n3.txt": "AGENT"}
    for name, want in expected.items():
        got = _read(workdir, name)
        if got is None:
            return False, f"{name} is missing"
        if _norm(got) != want:
            return False, f"{name} content was {got!r}, expected uppercased {want!r}"
    return True, "n1/n2/n3.txt were each transformed to uppercase in place"


TASK_9 = Task(
    id="transform_many",
    number=9,
    prompt=(
        "The current working directory contains three files: 'n1.txt', 'n2.txt' and "
        "'n3.txt', each holding a single lowercase word. For each of these three files, "
        "replace its contents with the UPPERCASE version of the word it currently holds. "
        "For example a file containing 'hello' must end up containing 'HELLO'. Modify all "
        "three files."
    ),
    setup=_setup_9,
    check=_check_9,
)


# --------------------------------------------------------------------------- #
# 10. A small multi-step pipeline: create + read + edit + verify.
# --------------------------------------------------------------------------- #
def _setup_10(workdir: Path) -> None:
    (workdir / "inventory.csv").write_text(
        "item,qty\napples,3\nbananas,5\ncherries,2\n", encoding="utf-8"
    )


def _check_10(workdir: Path) -> CheckResult:
    # The pipeline: read inventory.csv, sum the qty column, write the total into
    # summary.txt as "total=<N>", then append a final line "DONE" to the same file.
    summary = _read(workdir, "summary.txt")
    if summary is None:
        return False, "summary.txt was not created"
    lines = [ln for ln in _norm(summary).split("\n") if ln.strip()]
    if "total=10" not in lines:
        return False, f"summary.txt should contain a line 'total=10'; got {lines!r}"
    if not lines or lines[-1].strip() != "DONE":
        return False, f"summary.txt should end with a line 'DONE'; got {lines!r}"
    # The source inventory must be left intact.
    inv = _read(workdir, "inventory.csv")
    if inv is None or "apples,3" not in inv:
        return False, "inventory.csv was altered or removed; it must be left intact"
    return True, "summary.txt has total=10 then a final DONE line; inventory.csv intact"


TASK_10 = Task(
    id="pipeline",
    number=10,
    prompt=(
        "There is a CSV file named 'inventory.csv' in the current working directory with "
        "a header row 'item,qty' followed by rows of an item name and an integer quantity. "
        "Do the following in order: (1) read inventory.csv; (2) add up all the quantities "
        "in the 'qty' column; (3) create a file named 'summary.txt' containing a line that "
        "reads exactly total=<SUM> where <SUM> is the total you computed; (4) append a final "
        "line reading exactly DONE to summary.txt. Leave inventory.csv unchanged."
    ),
    setup=_setup_10,
    check=_check_10,
)


# The ordered ladder. Order here is the canonical display/run order.
TASKS: list[Task] = [
    TASK_1, TASK_2, TASK_3, TASK_4, TASK_5,
    TASK_6, TASK_7, TASK_8, TASK_9, TASK_10,
]


def get_tasks(numbers: "list[int] | None" = None) -> list[Task]:
    """Return the task ladder, optionally subset to the given 1-based ``numbers``
    (preserving ladder order). Unknown numbers raise ``ValueError``."""
    if not numbers:
        return list(TASKS)
    by_number = {t.number: t for t in TASKS}
    unknown = [n for n in numbers if n not in by_number]
    if unknown:
        raise ValueError(f"no such task number(s): {unknown}; valid: 1..{len(TASKS)}")
    wanted = set(numbers)
    return [t for t in TASKS if t.number in wanted]
