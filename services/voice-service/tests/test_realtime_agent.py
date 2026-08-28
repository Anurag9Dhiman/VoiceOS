"""RoutingAgent.classify_utterance is the only reliable per-turn hook into
a Gemini Live session (llm_node, the old hook, is never invoked for a
RealtimeModel session). These tests call the decorated tool directly --
no real LiveKit session, room, or network needed, since @function_tool's
descriptor binding makes it callable like a normal bound async method.
"""

import asyncio
from types import SimpleNamespace

import pytest
from voice_contract import RouterClass
from voice_service.agent import (
    CLASSIFY_UTTERANCE_SCHEMA,
    RoutingAgent,
    ScriptedSpeechGuard,
    WakeState,
    _AgentRef,
    _after_phrase,
    _context_instructions,
    _LatestUserTranscript,
)

_WAKE_WORD = "hey voiceos"
_SLEEP_WORD = "voiceos go to sleep"


class _FakeController:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    async def handle_utterance(self, text, router_class=None):
        self.calls.append((text, router_class))


class _FakeSession:
    def __init__(self):
        self.interrupted = False

    async def interrupt(self):
        self.interrupted = True


def _agent(*, awake: bool = True):
    """Defaults to already-awake: these tests are about classify_utterance's
    post-wake behavior. The wake gate itself gets its own dedicated tests
    below."""
    controller = _FakeController()
    guard = ScriptedSpeechGuard()
    latest = _LatestUserTranscript()
    wake_state = WakeState(awake=awake)
    agent = RoutingAgent(controller, guard, latest, wake_state, _WAKE_WORD, _SLEEP_WORD)
    ctx = SimpleNamespace(session=_FakeSession())
    return agent, controller, guard, ctx


def test_schema_enum_matches_the_contracts_router_class_exactly():
    from typing import get_args

    schema_values = set(CLASSIFY_UTTERANCE_SCHEMA["parameters"]["properties"]["router_class"]["enum"])
    assert schema_values == set(get_args(RouterClass))


@pytest.mark.parametrize("router_class", ["small_talk", "simple_lookup"])
def test_local_answer_classes_are_told_to_answer_and_not_interrupted(router_class):
    agent, controller, _, ctx = _agent()

    result = asyncio.run(
        agent.classify_utterance({"router_class": router_class, "transcript": "hey"}, ctx)
    )

    assert "answer" in result.lower()
    assert ctx.session.interrupted is False
    assert controller.calls == [("hey", router_class)]


@pytest.mark.parametrize(
    "router_class",
    ["new_intent", "modify_inflight", "confirmation_reply", "session_query"],
)
def test_forwarding_classes_interrupt_and_stay_silent(router_class):
    agent, controller, _, ctx = _agent()

    result = asyncio.run(
        agent.classify_utterance(
            {"router_class": router_class, "transcript": "clear my morning"}, ctx
        )
    )

    assert "silent" in result.lower()
    assert ctx.session.interrupted is True
    assert controller.calls == [("clear my morning", router_class)]


def test_scripted_speech_guard_short_circuits_without_calling_the_controller():
    agent, controller, guard, ctx = _agent()
    guard.speaking = True

    result = asyncio.run(
        agent.classify_utterance({"router_class": "new_intent", "transcript": "ignore me"}, ctx)
    )

    assert "ignore" in result.lower()
    assert controller.calls == []
    assert ctx.session.interrupted is False


def test_context_instructions_reflect_no_active_task():
    text = _context_instructions(None, None)

    assert "no in-flight task" in text


def test_context_instructions_reflect_pending_confirmation():
    text = _context_instructions("t1", "user_confirm")

    assert "awaiting the user's confirmation reply" in text


def test_context_instructions_reflect_pending_clarification():
    text = _context_instructions("t1", "user_clarify")

    assert "awaiting the user's clarification" in text


def test_context_instructions_reflect_a_running_task_not_yet_waiting():
    text = _context_instructions("t1", None)

    assert "running right now, not yet awaiting a reply" in text


def test_agent_ref_holder_starts_empty_and_is_set_after_construction():
    ref = _AgentRef()
    assert ref.agent is None

    controller = _FakeController()
    agent = RoutingAgent(
        controller,
        ScriptedSpeechGuard(),
        _LatestUserTranscript(),
        WakeState(awake=True),
        _WAKE_WORD,
        _SLEEP_WORD,
    )
    ref.agent = agent

    assert ref.agent is agent


def test_latest_captured_transcript_overrides_the_model_supplied_one():
    agent, controller, _, ctx = _agent()
    agent._latest_transcript.text = "the actual STT transcript"

    asyncio.run(
        agent.classify_utterance(
            {"router_class": "new_intent", "transcript": "the model's paraphrase"}, ctx
        )
    )

    assert controller.calls == [("the actual STT transcript", "new_intent")]


def test_after_phrase_returns_none_when_absent():
    assert _after_phrase("just talking normally", _WAKE_WORD) is None


def test_after_phrase_returns_the_remainder_in_one_breath():
    assert _after_phrase("hey voiceos, clear my morning", _WAKE_WORD) == "clear my morning"


def test_after_phrase_is_case_insensitive():
    assert _after_phrase("HEY VOICEOS clear my morning", _WAKE_WORD) == "clear my morning"


def test_after_phrase_returns_empty_string_for_the_phrase_alone():
    assert _after_phrase("hey voiceos", _WAKE_WORD) == ""


def test_after_phrase_rejects_a_longer_word_containing_the_phrase():
    assert _after_phrase("hey voiceostron activate", _WAKE_WORD) is None


def test_after_phrase_sleep_word_rejects_normal_conversation_containing_similar_words():
    assert _after_phrase("I need to go to sleep early tonight", _SLEEP_WORD) is None


def test_after_phrase_tolerates_a_comma_inserted_between_words():
    """The exact failure found live: Gemini Live's own transcription
    naturally inserts a comma when a multi-word phrase is spoken as a
    direct address ("voiceos, go to sleep") -- a literal-substring match
    would silently reject this and the sleep word would just never work."""
    assert _after_phrase("voiceos, go to sleep now.", _SLEEP_WORD) == "now."
    assert _after_phrase("hey, voiceos, clear my morning", _WAKE_WORD) == "clear my morning"


def test_asleep_with_no_wake_word_stays_silent_and_never_calls_the_controller():
    agent, controller, _, ctx = _agent(awake=False)

    result = asyncio.run(
        agent.classify_utterance({"router_class": "new_intent", "transcript": "clear my morning"}, ctx)
    )

    assert "silent" in result.lower() or "wake" in result.lower()
    assert controller.calls == []
    assert agent._wake_state.awake is False


def test_asleep_with_wake_word_and_command_wakes_and_processes_the_remainder():
    agent, controller, _, ctx = _agent(awake=False)

    result = asyncio.run(
        agent.classify_utterance(
            {"router_class": "new_intent", "transcript": "hey voiceos, clear my morning"}, ctx
        )
    )

    assert agent._wake_state.awake is True
    assert controller.calls == [("clear my morning", "new_intent")]
    assert "silent" in result.lower()  # new_intent still tells the model to stay silent


def test_asleep_with_wake_word_alone_wakes_without_calling_the_controller():
    agent, controller, _, ctx = _agent(awake=False)

    result = asyncio.run(
        agent.classify_utterance({"router_class": "small_talk", "transcript": "hey voiceos"}, ctx)
    )

    assert agent._wake_state.awake is True
    assert controller.calls == []
    assert "listening" in result.lower() or "woke" in result.lower()


def test_already_awake_is_unaffected_by_the_wake_gate():
    """Once awake, later turns don't need to repeat the wake word --
    matches every other test in this file, which all default to awake."""
    agent, controller, _, ctx = _agent(awake=True)

    asyncio.run(
        agent.classify_utterance({"router_class": "new_intent", "transcript": "clear my morning"}, ctx)
    )

    assert controller.calls == [("clear my morning", "new_intent")]


def test_awake_and_sleep_word_heard_goes_back_to_sleep_without_calling_the_controller():
    agent, controller, _, ctx = _agent(awake=True)

    result = asyncio.run(
        agent.classify_utterance(
            {"router_class": "small_talk", "transcript": "voiceos go to sleep"}, ctx
        )
    )

    assert agent._wake_state.awake is False
    assert controller.calls == []
    assert "sleep" in result.lower() or "silent" in result.lower()


def test_after_going_back_to_sleep_a_plain_utterance_is_ignored_again():
    agent, controller, _, ctx = _agent(awake=True)

    asyncio.run(
        agent.classify_utterance(
            {"router_class": "small_talk", "transcript": "voiceos go to sleep"}, ctx
        )
    )
    assert agent._wake_state.awake is False

    result = asyncio.run(
        agent.classify_utterance({"router_class": "new_intent", "transcript": "clear my morning"}, ctx)
    )

    assert "silent" in result.lower() or "wake" in result.lower()
    assert controller.calls == []


def test_asleep_ignores_the_sleep_word_itself_since_it_only_applies_while_awake():
    """The sleep word only matters once awake -- while asleep, only the
    wake word matters, and "voiceos go to sleep" doesn't happen to contain
    the wake word "hey voiceos"."""
    agent, controller, _, ctx = _agent(awake=False)

    result = asyncio.run(
        agent.classify_utterance(
            {"router_class": "small_talk", "transcript": "voiceos go to sleep"}, ctx
        )
    )

    assert agent._wake_state.awake is False
    assert controller.calls == []
    assert "silent" in result.lower() or "wake" in result.lower()
