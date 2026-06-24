"""Shared pytest configuration: markers, one-time resource probes, and the
deterministic/real-environment fixtures the integration tests build on.

This replaces the per-file copy-pasted `_ollama_up` / `_server_up` /
`_cli_available` helpers (one each was hand-rolled in tests/integration/) with a
single cached probe per resource, and registers the `needs_ollama` / `needs_web`
/ `slow` markers so `pytest` emits no unknown-marker warnings.

Default test selection is intentionally unchanged: no `addopts` filter is set
here (the markers live in pyproject.toml only as registrations), so the normal
`python -m pytest tests/ -q --ignore=tests/integration` still collects and runs
exactly the same unit tests as before.
"""
from __future__ import annotations

import functools
import os
import shutil
import urllib.request

import pytest

from vibeharness.config import Config


# --------------------------------------------------------------------------- #
# Marker registration (also declared in pyproject.toml; registered here too so
# the markers resolve even if pytest is invoked without the ini).
# --------------------------------------------------------------------------- #
def pytest_configure(config):
    config.addinivalue_line("markers", "needs_ollama: requires a live Ollama server")
    config.addinivalue_line("markers", "needs_web: requires playwright-cli + the demo web app")
    config.addinivalue_line("markers", "slow: slow / real-model test (not part of the fast unit lane)")


# --------------------------------------------------------------------------- #
# One-time resource probes (cached for the whole session).
# --------------------------------------------------------------------------- #
WEB_BASE = "http://localhost:3000"


@functools.lru_cache(maxsize=1)
def ollama_is_up() -> bool:
    """True iff a live Ollama answers /api/version. Probed once, then cached."""
    try:
        with urllib.request.urlopen(Config().ollama_url + "/api/version", timeout=2):
            return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def web_server_is_up() -> bool:
    """True iff the demo web app at :3000 answers. Probed once, then cached."""
    try:
        with urllib.request.urlopen(WEB_BASE, timeout=2):
            return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def playwright_cli_available() -> bool:
    """True iff the `playwright-cli` executable is on PATH."""
    return shutil.which("playwright-cli") is not None


# --------------------------------------------------------------------------- #
# Auto-skip: a test marked needs_ollama / needs_web skips cleanly (never errors)
# when its resource is down.
# --------------------------------------------------------------------------- #
def pytest_runtest_setup(item):
    if item.get_closest_marker("needs_ollama") and not ollama_is_up():
        pytest.skip("Ollama not reachable - start it with `ollama serve`")
    if item.get_closest_marker("needs_web") and not (
        playwright_cli_available() and web_server_is_up()
    ):
        pytest.skip("needs playwright-cli + the demo app at localhost:3000")


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
@pytest.fixture
def greedy_config() -> Config:
    """A deterministic, greedy Config for repeatable (esp. live) runs: temperature
    0 in both phases and small token budgets so runs are fast and repeatable."""
    from dataclasses import replace

    return replace(
        Config(),
        temperature=0.0,
        action_temperature=0.0,
        reason_tokens=256,
        action_tokens=512,
    )


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """A fresh real temp directory that the test chdirs into (and is restored from
    afterwards). Replaces the per-file setUp/tearDown TemporaryDirectory + chdir
    boilerplate. Yields the workspace Path."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path
