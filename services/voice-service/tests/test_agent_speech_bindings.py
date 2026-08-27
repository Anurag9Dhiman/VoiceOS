"""_make_speech_composer binds SpeechComposer's raw_speak/raw_interrupt to
AgentSession primitives. session.say() is unavailable for Gemini Live
(RealtimeCapabilities.supports_say is False -- see
test_realtime_model_capabilities.py), so raw_speak goes through
session.generate_reply(instructions=...) instead. These tests use a
hand-rolled fake AgentSession -- no real LiveKit session needed -- to prove
the binding itself, independent of SpeechComposer's own already-tested
priority logic (test_speech_composer.py).
"""

import asyncio

from voice_service.agent import ScriptedSpeechGuard, _make_speech_composer
from voice_service.turn_manager import UndeliveredTracker


class _FakeHandle:
    def __init__(self):
        self.interrupted = False
        self._callback = None

    def add_done_callback(self, callback):
        self._callback = callback

    def finish(self, *, interrupted: bool) -> None:
        self.interrupted = interrupted
        if self._callback:
            self._callback(self)


class _FakeSession:
    def __init__(self):
        self.generate_reply_calls: list[dict] = []
        self.interrupt_calls = 0
        self.user_state = "listening"

    def generate_reply(self, **kwargs):
        self.generate_reply_calls.append(kwargs)
        return _FakeHandle()

    async def interrupt(self):
        self.interrupt_calls += 1


def _composer():
    session = _FakeSession()
    tracker = UndeliveredTracker()
    guard = ScriptedSpeechGuard()
    return _make_speech_composer(session, tracker, guard), session, tracker, guard


def test_raw_speak_calls_generate_reply_with_the_text_verbatim_in_instructions():
    composer, session, _, _ = _composer()

    asyncio.run(composer.speak("Checking your calendar now.", "low"))

    assert len(session.generate_reply_calls) == 1
    call = session.generate_reply_calls[0]
    assert "Checking your calendar now." in call["instructions"]
    assert call["tool_choice"] == "none"


def test_raw_speak_flips_the_guard_true_then_false_around_the_call():
    composer, session, _, guard = _composer()
    observed_during_call = {}

    original_generate_reply = session.generate_reply

    def spy(**kwargs):
        observed_during_call["speaking"] = guard.speaking
        return original_generate_reply(**kwargs)

    session.generate_reply = spy

    asyncio.run(composer.speak("Found three meetings.", "low"))

    assert observed_during_call["speaking"] is True
    assert guard.speaking is False


def test_undelivered_tracker_marks_an_interrupted_reply_as_undelivered():
    composer, _, tracker, _ = _composer()

    handle = asyncio.run(composer.speak("Moved the 9 o'clock to Thursday.", "low"))
    handle.finish(interrupted=True)

    assert tracker.drain() == ["Moved the 9 o'clock to Thursday."]


def test_high_priority_speech_calls_raw_interrupt_bound_to_session_interrupt():
    """raw_interrupt's actual binding (await session.interrupt()) was
    never exercised -- SpeechComposer only calls it internally for
    high-priority speech, and no test drove that path."""
    composer, session, _, _ = _composer()

    asyncio.run(composer.speak("Shall I go ahead?", "high"))

    assert session.interrupt_calls == 1
