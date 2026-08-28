"""End-to-end proof that ReconnectingCollectiveOSClient actually recovers a
real dropped connection, not just a simulated one (test_resilient_client.py
already covers the wrapper's logic in isolation with a fake transport).

Here the drop is real: the underlying websocket is force-closed directly
(bypassing our own close(), so `_closing` stays False and the wrapper reads
it exactly like a genuine network failure), against a live mock-agent-backend
process, and the *same* ConversationController -- not a freshly constructed
one -- has to notice, reconnect with resume=True, and let the conversation
continue. This is the piece test_e2e_scenario_b.py doesn't cover: there,
the "hang up and resume" was a deliberate, explicit action (controller.stop()
followed by a brand new controller). This test is the involuntary case.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from voice_service.conversation import ConversationController
from voice_service.resilient_client import ReconnectingCollectiveOSClient

from .test_e2e_scenario_a import _running_mock_backend, _wait_until


def test_unexpected_drop_mid_task_recovers_and_completes():
    async def scenario():
        async with _running_mock_backend() as ws_url:
            speech: list[tuple[str, str]] = []

            async def speak(text: str, priority: str) -> None:
                speech.append((text, priority))

            client = ReconnectingCollectiveOSClient(ws_url, base_delay=0.05, max_retries=5)
            controller = ConversationController(client=client, speak=speak)

            await controller.start(session_id=str(uuid4()), user_id=str(uuid4()), resume=False)
            assert client.state == "connected"

            await controller.handle_utterance(
                "help me prep for the board meeting Friday", router_class="new_intent"
            )
            await _wait_until(lambda: controller.waiting_reason == "external")
            task_id_before_drop = controller.current_task_id

            # Simulate a real network drop: close the underlying socket
            # directly, bypassing CollectiveOSClient.close() and therefore
            # ReconnectingCollectiveOSClient._closing -- indistinguishable,
            # from the wrapper's point of view, from the wifi actually dying.
            raw_ws = client._client._ws  # noqa: SLF001 -- intentional white-box drop
            await raw_ws.close()

            def _reconnected() -> bool:
                new_ws = getattr(client._client, "_ws", None)  # noqa: SLF001
                return client.state == "connected" and new_ws is not None and new_ws is not raw_ws

            await _wait_until(_reconnected, timeout=5)

            # The conversation continues on the SAME controller -- no new
            # ConversationController or client was constructed.
            assert controller.current_task_id == task_id_before_drop

            await controller.handle_utterance("where were we", router_class="session_query")
            await _wait_until(lambda: controller.current_task_id is None, timeout=5)

            spoken = [t for t, _ in speech]
            assert any("ready for Friday" in t for t in spoken), spoken

            await controller.stop()

    asyncio.run(scenario())
