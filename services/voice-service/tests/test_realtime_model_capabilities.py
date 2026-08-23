"""Decisive, zero-network check of the two assumptions the Gemini Live
migration is built on. Constructing a RealtimeModel doesn't connect to
anything -- these assertions would have caught a wrong design before any
of agent.py was written around them, and guard against a future
livekit-plugins-google upgrade silently changing either behavior.
"""

from livekit.plugins import google


def test_realtime_model_does_not_support_say():
    """raw_speak in agent.py deliberately uses session.generate_reply(...)
    instead of session.say(text) because this is False -- if a future
    plugin version flips it to True, raw_speak should be simplified back
    to session.say(), not left as an unnecessary workaround."""
    model = google.realtime.RealtimeModel(api_key="placeholder")

    assert model.capabilities.supports_say is False


def test_realtime_model_does_not_honor_per_response_tool_choice():
    """classify_utterance's "always classify first" instruction is
    unenforceable for this reason -- ScriptedSpeechGuard and the reactive
    session.interrupt() in RoutingAgent.classify_utterance exist because of
    this, not as belt-and-suspenders."""
    model = google.realtime.RealtimeModel(api_key="placeholder")

    assert model.capabilities.per_response_tool_choice is False
