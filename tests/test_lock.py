"""Unit tests for the machine-global single-instance lock.

All tests point the lock at a temp path (never the real ~/.vibeharness) and use
no live model. We simulate a "live" foreign holder by writing a lockfile whose
pid is our own process (which is, by definition, alive) but distinct from the
acquiring lock's notion of ownership where needed; for the typed-error path we
write a foreign-but-live pid and monkeypatch the liveness check so the test is
deterministic and independent of real OS pids.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vibeharness import lock as lock_mod
from vibeharness.lock import SingleInstanceLock, VibeAlreadyRunning


@pytest.fixture
def lock_path(tmp_path) -> Path:
    """A lockfile path inside a temp dir (parent dir intentionally not yet made,
    so we also exercise the mkdir-on-acquire behaviour)."""
    return tmp_path / ".vibeharness" / "vibe.lock"


def _write_lock(path: Path, *, pid: int, workdir="/w", log_path="/w/.vibe/x.json",
                started="2026-06-24T00:00:00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pid": pid, "workdir": workdir, "log_path": log_path, "started": started,
    }), encoding="utf-8")


def test_acquire_release_round_trip(lock_path):
    lock = SingleInstanceLock(lock_path)
    assert not lock_path.exists()

    lock.acquire("/work/dir", "/work/dir/.vibe/log.json")
    assert lock_path.exists()
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["workdir"] == "/work/dir"
    assert data["log_path"] == "/work/dir/.vibe/log.json"
    assert data["started"]  # ISO timestamp present

    lock.release()
    assert not lock_path.exists()


def test_live_lock_raises_with_identifying_details(lock_path, monkeypatch):
    # A foreign, *live* holder. Use a pid different from ours and force the
    # liveness check to report it alive so the test is deterministic.
    foreign_pid = os.getpid() + 100000
    _write_lock(lock_path, pid=foreign_pid, workdir="/other/work",
                log_path="/other/work/.vibe/run.json", started="2026-06-24T12:34:56")
    monkeypatch.setattr(lock_mod, "_pid_alive", lambda pid: True)

    lock = SingleInstanceLock(lock_path)
    with pytest.raises(VibeAlreadyRunning) as exc:
        lock.acquire("/my/work", "/my/work/.vibe/run.json")

    err = exc.value
    assert err.pid == foreign_pid
    assert err.workdir == "/other/work"
    assert err.log_path == "/other/work/.vibe/run.json"
    assert err.started == "2026-06-24T12:34:56"
    # The foreign lockfile must be left untouched.
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["pid"] == foreign_pid


def test_stale_lock_is_reclaimed(lock_path, monkeypatch):
    # A lock owned by a pid that is definitely not alive must be overwritten.
    dead_pid = os.getpid() + 100000
    _write_lock(lock_path, pid=dead_pid, workdir="/dead/work")
    monkeypatch.setattr(lock_mod, "_pid_alive", lambda pid: False)

    lock = SingleInstanceLock(lock_path)
    lock.acquire("/fresh/work", "/fresh/work/.vibe/log.json")

    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["workdir"] == "/fresh/work"


def test_corrupt_lock_is_treated_as_stale(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{not valid json", encoding="utf-8")

    lock = SingleInstanceLock(lock_path)
    lock.acquire("/fresh/work", "/fresh/work/.vibe/log.json")  # must not raise
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()


def test_release_only_removes_our_own_lock(lock_path, monkeypatch):
    # A lock owned by another (live) pid must NOT be removed by our release().
    foreign_pid = os.getpid() + 100000
    _write_lock(lock_path, pid=foreign_pid)
    monkeypatch.setattr(lock_mod, "_pid_alive", lambda pid: True)

    lock = SingleInstanceLock(lock_path)
    lock.release()  # we never acquired; the foreign lock must survive
    assert lock_path.exists()
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["pid"] == foreign_pid


def test_release_on_exception_still_clears_our_lock(lock_path):
    # Simulate the cli finally-path: acquire, raise inside the body, release in
    # finally. The lock must be gone afterwards.
    lock = SingleInstanceLock(lock_path)
    with pytest.raises(RuntimeError):
        lock.acquire("/work", "/work/.vibe/log.json")
        try:
            raise RuntimeError("boom from the run body")
        finally:
            lock.release()
    assert not lock_path.exists()


def test_release_is_safe_when_already_gone(lock_path):
    lock = SingleInstanceLock(lock_path)
    lock.acquire("/work", "/work/.vibe/log.json")
    lock_path.unlink()
    lock.release()  # must not raise
    assert not lock_path.exists()


def test_context_manager_holds_then_releases(lock_path):
    lock = SingleInstanceLock(lock_path)
    with lock.hold("/work", "/work/.vibe/log.json"):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_pid_alive_for_current_process_is_true():
    assert lock_mod._pid_alive(os.getpid()) is True


def test_pid_alive_for_impossible_pid_is_false():
    # A never-used very-high pid should not be alive.
    assert lock_mod._pid_alive(os.getpid() + 100000) is False
    assert lock_mod._pid_alive(0) is False
    assert lock_mod._pid_alive(-1) is False
