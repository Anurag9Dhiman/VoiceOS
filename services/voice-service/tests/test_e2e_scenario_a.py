"""End-to-end proof for the weeks 5-6 deliverable: "Scenario A works against
the mock backend, including the mid-confirmation edit" -- run for real
against a live mock-agent-backend over an actual WebSocket, with the Haiku
router bypassed (router_class passed directly) since there's no live
Anthropic key in this environment. Everything downstream of routing --
CollectiveOSClient, ConversationController's event relay, the interrupt and
confirmation-response handling -- runs unmocked, over a real socket.

Confirmation is risk-tiered, not blanket (see conversation.py's module
docstring): new_intent/modify_inflight/session_query forward immediately;
CollectiveOS's own confirmation_request/risk_class:write is what actually
pauses for a write, exactly as scripted in mock_agent_backend/scenarios.py.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from mock_agent_backend.app import app as mock_app
from voice_service.collectiveos_client import CollectiveOSClient
from voice_service.conversation import ConversationController


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@asynccontextmanager
async def _running_mock_backend() -> AsyncIterator[str]:
    port = _free_port()
    config = uvicorn.Config(mock_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        yield f"ws://127.0.0.1:{port}/v1/ws"
    finally:
        server.should_exit = True
        await task


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    async def _poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


def test_scenario_a_end_to_end_including_mid_confirmation_edit():
    async def scenario():
        async with _running_mock_backend() as ws_url:
            speak_calls: list[tuple[str, str]] = []

            async def speak(text: str, priority: str) -> None:
                speak_calls.append((text, priority))

            controller = ConversationController(client=CollectiveOSClient(ws_url), speak=speak)
            await controller.start(session_id=str(uuid4()), user_id=str(uuid4()))

            # 1. new_intent -> ack, progress, first confirmation_request.
            await controller.handle_utterance(
                "clear my morning, I'm sick", router_class="new_intent"
            )
            await _wait_until(lambda: any("Shall I go ahead" in t for t, _ in speak_calls))
            assert controller.current_task_id is not None
            assert controller.waiting_reason == "user_confirm"

            # 2. mid-confirmation follow-up -> interrupt -> modified confirmation_request.
            await controller.handle_utterance("wait, keep the 10am", router_class="modify_inflight")
            await _wait_until(
                lambda: any("keep the 10am" in t and "Shall I go ahead" in t for t, _ in speak_calls)
            )

            # 3. approve -> confirmation_response -> running -> done.
            await controller.handle_utterance("yes, go ahead", router_class="confirmation_reply")
            await _wait_until(lambda: any("Done" in t for t, _ in speak_calls))

            assert controller.current_task_id is None
            assert controller.waiting_reason is None

            spoken_texts = [t for t, _ in speak_calls]
            assert any("10am stays" in t for t in spoken_texts), spoken_texts
            assert any(p == "high" for _, p in speak_calls if "Shall I go ahead" in _)

            await controller.stop()

    asyncio.run(scenario())


def test_scenario_c_partial_failure_end_to_end():
    """Same real-socket proof for scenario C's shape (batch + mid-flight
    interrupt + partial failure) -- confirms the controller's protocol
    handling isn't scenario-A-specific."""

    async def scenario():
        async with _running_mock_backend() as ws_url:
            speak_calls: list[tuple[str, str]] = []

            async def speak(text: str, priority: str) -> None:
                speak_calls.append((text, priority))

            controller = ConversationController(client=CollectiveOSClient(ws_url), speak=speak)
            await controller.start(session_id=str(uuid4()), user_id=str(uuid4()))

            await controller.handle_utterance(
                "cancel the subscriptions I'm not using", router_class="new_intent"
            )
            await _wait_until(lambda: controller.waiting_reason == "user_confirm")

            await controller.handle_utterance("yes, cancel them", router_class="confirmation_reply")
            # mock backend holds the batch open ~0.3s for a mid-flight interrupt
            await asyncio.sleep(0.05)
            await controller.handle_utterance("wait, keep Spotify", router_class="modify_inflight")

            await _wait_until(lambda: any("Done" in t for t, _ in speak_calls), timeout=3)

            spoken_texts = [t for t, _ in speak_calls]
            assert any("kept Spotify" in t for t in spoken_texts), spoken_texts
            assert any("still active" in t for t in spoken_texts), spoken_texts

            await controller.stop()

    asyncio.run(scenario())
