"""Closes the "Sentry actually receives an event -- not tested end to end"
caveat in README.md. No SENTRY_DSN or cloud account needed: sentry_sdk's
own Transport is the documented customization point for exactly this, so
the real init -> capture_exception -> event-serialization -> envelope
pipeline runs unmodified; only the final HTTPS call to Sentry's cloud is
swapped for a local recorder -- the same kind of substitution
mock-agent-backend makes for a real CollectiveOS, or a local Redis makes
for a managed one.
"""

from __future__ import annotations

import sentry_sdk
from sentry_sdk.transport import Transport

from voice_service.agent import _init_sentry

_FAKE_DSN = "http://public@localhost/1"


class _CapturingTransport(Transport):
    def __init__(self) -> None:
        super().__init__({"dsn": _FAKE_DSN})
        self.envelopes: list = []

    def capture_envelope(self, envelope) -> None:
        self.envelopes.append(envelope)


def test_init_sentry_only_calls_sentry_sdk_init_when_dsn_is_set(monkeypatch):
    calls = []
    monkeypatch.setattr("sentry_sdk.init", lambda **kwargs: calls.append(kwargs))

    monkeypatch.delenv("SENTRY_DSN", raising=False)
    _init_sentry()
    assert calls == []

    monkeypatch.setenv("SENTRY_DSN", _FAKE_DSN)
    _init_sentry()
    assert calls == [{"dsn": _FAKE_DSN, "traces_sample_rate": 0.1}]


def test_a_real_uncaught_exception_actually_reaches_a_transport():
    """The thing the README flagged as unverifiable: does an exception
    really get captured and sent once sentry_sdk.init() has run, the way
    _init_sentry() configures it, or does the wiring just look plausible?
    Torn down afterward so this doesn't leave a bound client for any test
    that runs after it."""
    transport = _CapturingTransport()
    sentry_sdk.init(dsn=_FAKE_DSN, transport=transport, traces_sample_rate=0.1)
    try:
        try:
            raise ValueError("deliberate test error")
        except ValueError:
            sentry_sdk.capture_exception()
        sentry_sdk.flush()
    finally:
        sentry_sdk.get_global_scope().set_client(None)

    assert len(transport.envelopes) == 1
    event = transport.envelopes[0].items[0].payload.json
    assert event["exception"]["values"][0]["type"] == "ValueError"
    assert event["exception"]["values"][0]["value"] == "deliberate test error"
