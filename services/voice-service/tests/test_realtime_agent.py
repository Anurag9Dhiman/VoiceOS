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
    _AgentRef,
    _context_instructions,
    _LatestUserTranscript,
)


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


def _agent():
    controller = _FakeController()
    guard = ScriptedSpeechGuard()
    latest = _LatestUserTranscript()
    agent = RoutingAgent(controller, guard, latest)
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
    agent = RoutingAgent(controller, ScriptedSpeechGuard(), _LatestUserTranscript())
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
