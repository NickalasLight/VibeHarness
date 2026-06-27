"""Machine-global single-instance lock for ``vibe`` runs.

Only one model stream is supported at a time, so the CLI acquires a global
lock at startup. The lock lives at a machine-global path (default
``~/.vibeharness/vibe.lock``) and stores JSON identifying the active run::

    {"pid": 1234, "workdir": "...", "log_path": "...", "started": "<ISO>"}

Acquisition policy:
  - No lockfile, or an unreadable/corrupt lockfile, or a lockfile whose ``pid``
    is not alive (a crashed prior run) => the lock is stale and we reclaim it.
  - A lockfile whose ``pid`` IS alive => raise :class:`VibeAlreadyRunning`,
    carrying the existing run's identifying details so the caller can print a
    helpful message and refuse to start.

Release removes the lockfile only if it is *ours* (pid matches), so a process
never deletes another run's lock. Release is safe if the file is already gone.

stdlib only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def default_lock_path() -> Path:
    """Machine-global lockfile path: ``~/.vibeharness/vibe.lock``."""
    return Path.home() / ".vibeharness" / "vibe.lock"


def _pid_alive(pid: int) -> bool:
    """Best-effort, stdlib-only, cross-platform 'is this pid alive?' check.

    Any uncertainty is reported conservatively. On POSIX, ``os.kill(pid, 0)``
    raises ``ProcessLookupError`` for a dead pid and ``PermissionError`` for a
    live one we don't own (which still means alive). On Windows, ``os.kill``
    with signal 0 likewise succeeds for a live pid and raises for a dead one.
    """
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but we lack permission to signal it => alive.
        return True
    except OSError:
        # On Windows a dead pid surfaces as OSError; treat as not alive.
        return False
    except Exception:
        # Unknown failure: be conservative and assume not alive so a genuinely
        # stale lock can always be reclaimed.
        return False
    return True


class VibeAlreadyRunning(RuntimeError):
    """Raised when a live ``vibe`` run already holds the single-instance lock.

    Carries the active run's identifying details for a helpful message.
    """

    def __init__(self, pid: int, workdir: str, log_path: str, started: str):
        self.pid = pid
        self.workdir = workdir
        self.log_path = log_path
        self.started = started
        super().__init__(
            f"another vibe run is already active (pid {pid}, workdir {workdir})"
        )


class SingleInstanceLock:
    """A machine-global single-instance lock (context manager + acquire/release).

    Usage::

        lock = SingleInstanceLock()
        lock.acquire(workdir, log_path)
        try:
            ...  # do the work
        finally:
            lock.release()

    or as a context manager::

        with SingleInstanceLock().hold(workdir, log_path):
            ...
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_lock_path()
        self._held = False

    # -- introspection ----------------------------------------------------- #
    def _read(self) -> dict | None:
        """Return the parsed lock contents, or None if missing/corrupt."""
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:
            # Corrupt / unreadable lockfile -> treat as no usable lock (stale).
            return None

    # -- acquire / release ------------------------------------------------- #
    def acquire(self, workdir: str | os.PathLike, log_path: str | os.PathLike) -> None:
        """Acquire the lock for this run.

        If no lock exists, or the existing lock is stale (corrupt, or its pid is
        not alive), write our lock and succeed. If a live lock exists, raise
        :class:`VibeAlreadyRunning`.
        """
        existing = self._read()
        if existing is not None:
            pid = existing.get("pid")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None
            if pid is not None and pid != os.getpid() and _pid_alive(pid):
                raise VibeAlreadyRunning(
                    pid=pid,
                    workdir=str(existing.get("workdir", "")),
                    log_path=str(existing.get("log_path", "")),
                    started=str(existing.get("started", "")),
                )
            # else: stale (corrupt, dead pid, or our own) -> reclaim below.

        payload = {
            "pid": os.getpid(),
            "workdir": str(workdir),
            "log_path": str(log_path),
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._held = True

    def release(self) -> None:
        """Remove the lockfile, but only if it is ours. Safe if already gone."""
        existing = self._read()
        if existing is not None:
            pid = existing.get("pid")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None
            if pid != os.getpid():
                # Not our lock (another run reclaimed it, or corrupt) -> leave it.
                self._held = False
                return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        self._held = False

    # -- context manager --------------------------------------------------- #
    def hold(self, workdir: str | os.PathLike, log_path: str | os.PathLike) -> "SingleInstanceLock":
        """Return self after acquiring, for use as a context manager."""
        self.acquire(workdir, log_path)
        return self

    def __enter__(self) -> "SingleInstanceLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
