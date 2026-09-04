"""End-to-end proof that the local fallback engages when CollectiveOS is
genuinely, permanently unreachable -- not just simulated at the
ReconnectingCollectiveOSClient level (test_resilient_client.py already
covers that in isolation with a fake transport). Here the mock backend is
a real process that gets actually stopped mid-test, so every reconnect
attempt against it really does fail, over a real socket.

Uses a fake local agent, not a real agentkit/Gemini-backed one: this repo's
e2e tests deliberately avoid needing a live LLM key to run (see
test_e2e_scenario_a.py's own docstring) -- LocalFallbackAgent's actual
Gemini-backed safety framing was verified separately, live, against the
real model (see local_agent.py's module docstring and README.md).
"""

from __future__ import annotations

import asyncio
import socket
from uuid import uuid4

import uvicorn
from mock_agent_backend.app import app as mock_app
from voice_service.conversation import ConversationController
from voice_service.resilient_client import ReconnectingCollectiveOSClient

from .test_e2e_scenario_a import _wait_until


class _FakeLocalAgent:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def respond(self, text: str) -> str:
        self.prompts.append(text)
        return "I can't reach the main system right now, but I've noted what you asked."

    async def aclose(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_falls_back_locally_once_collectiveos_is_actually_dead():
    async def scenario():
        port = _free_port()
        config = uvicorn.Config(mock_app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        ws_url = f"ws://127.0.0.1:{port}/v1/ws"

        speech: list[tuple[str, str]] = []

        async def speak(text: str, priority: str) -> None:
            speech.append((text, priority))

        local_agent = _FakeLocalAgent()
        client = ReconnectingCollectiveOSClient(ws_url, base_delay=0.05, max_retries=2)
        send_calls = 0
        real_send = client.send

        async def counting_send(event):
            nonlocal send_calls
            send_calls += 1
            await real_send(event)

        client.send = counting_send  # type: ignore[method-assign]
        controller = ConversationController(client=client, speak=speak, local_agent=local_agent)

        await controller.start(session_id=str(uuid4()), user_id=str(uuid4()), resume=False)
        assert client.state == "connected"

        # Baseline: CollectiveOS is genuinely up, this reaches it for real.
        await controller.handle_utterance(
            "help me prep for the board meeting Friday", router_class="new_intent"
        )
        await _wait_until(lambda: controller.waiting_reason == "external")
        assert local_agent.prompts == []  # never engaged while CollectiveOS is healthy
        assert send_calls == 1

        # Actually kill the server -- not a simulated drop. Every reconnect
        # attempt against this port will genuinely fail from here on. A
        # brief pause gives the client's own connection machinery a moment
        # to notice the close frame before the next send() is attempted.
        server.should_exit = True
        await server_task
        await asyncio.sleep(0.1)

        await controller.handle_utterance("what's the status", router_class="session_query")

        assert local_agent.prompts == ["what's the status"]
        assert send_calls == 2  # this turn's send() attempt genuinely failed
        assert speech[-1] == (
            "I can't reach the main system right now, but I've noted what you asked.",
            "low",
        )

        # Latched: a further turn doesn't even attempt send() again -- not
        # a real connection to retry, since the server is genuinely dead.
        await controller.handle_utterance("one more thing", router_class="new_intent")
        assert local_agent.prompts == ["what's the status", "one more thing"]
        assert send_calls == 2

        await controller.stop()

    asyncio.run(scenario())
