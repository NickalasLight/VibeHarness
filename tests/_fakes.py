"""Shared test doubles for the VibeHarness suite.

These collapse the several near-identical fake clients/validators that used to be
re-declared privately in test_agent.py, test_streaming_log.py, test_validation.py
and test_web.py into one canonical place. Behaviour is intentionally a superset of
all the old copies so each consuming test keeps its exact semantics.

Fakes are the EXCEPTION, reserved for determinism-critical units and error paths;
the real-environment integration tests live under tests/integration/.
"""
from __future__ import annotations

import json

from vibeharness.llm import Decision, LLMClient
from vibeharness.validation import Validator, Verdict


class FakeLLMClient(LLMClient):
    """Returns scripted actions instead of calling a model.

    Each scripted action may be a dict/list (json-encoded for you) or a raw
    string (passed through verbatim, e.g. to feed deliberately invalid JSON).
    When the script is exhausted it repeats the LAST action, so a loop can run on
    to its step budget. Optionally drives on_reason/on_action streaming callbacks.
    """

    def __init__(self, actions, *, reasoning="", stream=False):
        self._actions = actions
        self._i = 0
        self._reasoning = reasoning
        self._stream = stream

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        action = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        payload = action if isinstance(action, str) else json.dumps(action)
        if self._stream:
            if on_reason and self._reasoning:
                on_reason(self._reasoning)
            if on_action:
                on_action(payload)
        return Decision(reasoning=self._reasoning, action_json=payload)


class RecordingLLMClient(FakeLLMClient):
    """Like FakeLLMClient but records every `system`/`user` prompt and the exact
    streaming callbacks it was handed (used to prove pass-through wiring)."""

    def __init__(self, actions, **kwargs):
        super().__init__(actions, **kwargs)
        self.systems = []
        self.users = []
        self.last_system = None
        self.last_user = None
        self.last_on_reason = None
        self.last_on_action = None

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        self.systems.append(system)
        self.users.append(user)
        self.last_system, self.last_user = system, user
        self.last_on_reason, self.last_on_action = on_reason, on_action
        return super().decide(system, user, action_schema, on_reason, on_action)


class FakeValidator(Validator):
    """A scripted Validator. Records each call and optionally streams a little so a
    reporter observing the validator can be exercised."""

    def __init__(self, passed=True, reason="looks complete", *, stream=True):
        self._passed, self._reason = passed, reason
        self._stream = stream
        self.calls = []

    def validate(self, task, history, claim, on_reason=None, on_action=None):
        self.calls.append({"task": task, "history": history, "claim": claim})
        if self._stream:
            if on_reason:
                on_reason("judging")
            if on_action:
                on_action('{"verdict":"pass"}')
        return Verdict(self._passed, self._reason)


class FakeCli:
    """Stand-in for PlaywrightCli: records calls, returns a scripted result. Used
    only for the pure-unit error paths a live browser can't cheaply force; the real
    arg-mapping is proven end-to-end in tests/integration/test_web_live.py."""

    def __init__(self, ok=True, output="### Page\nok"):
        self.ok, self.output = ok, output
        self.calls = []

    def run(self, *args):
        self.calls.append(list(args))
        return self.ok, self.output


class ScriptedVerdictClient(LLMClient):
    """An LLMClient that returns a fixed verdict-JSON string for the LLMValidator,
    while recording the prompt and streaming through the supplied callbacks the way
    a real streaming client would."""

    def __init__(self, verdict_json: str):
        self._verdict = verdict_json
        self.last_system = None
        self.last_user = None
        self.last_on_reason = None
        self.last_on_action = None

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        self.last_system, self.last_user = system, user
        self.last_on_reason, self.last_on_action = on_reason, on_action
        if on_reason:
            on_reason("judging")
        if on_action:
            on_action(self._verdict)
        return Decision(reasoning="<think>judging</think>", action_json=self._verdict)
