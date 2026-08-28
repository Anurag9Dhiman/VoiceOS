"""End-to-end proof for the weeks 7-8 deliverable: "Scenario B works
against the mock backend, including hang-up mid-task and resume next day."
Two separate ConversationController instances (one per WebSocket
connection, exactly like two separate phone calls) sharing one
SessionStore -- the same thing Redis would do across restarts in
production. Everything downstream of routing runs unmocked, over a real
socket, same as test_e2e_scenario_a.py.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from voice_service.collectiveos_client import CollectiveOSClient
from voice_service.conversation import ConversationController
from voice_service.session_store import InMemorySessionStore

from .test_e2e_scenario_a import _running_mock_backend, _wait_until


def test_scenario_b_hangup_and_resume_next_session():
    async def scenario():
        async with _running_mock_backend() as ws_url:
            user_id = str(uuid4())
            shared_store = InMemorySessionStore()

            # --- Session 1: opens the task, gets blocked externally, hangs up. ---
            first_call_speech: list[tuple[str, str]] = []

            async def speak1(text: str, priority: str) -> None:
                first_call_speech.append((text, priority))

            controller1 = ConversationController(
                client=CollectiveOSClient(ws_url), speak=speak1, session_store=shared_store
            )
            await controller1.start(session_id=str(uuid4()), user_id=user_id, resume=False)

            await controller1.handle_utterance(
                "help me prep for the board meeting Friday", router_class="new_intent"
            )
            await _wait_until(lambda: controller1.waiting_reason == "external")
            assert controller1.active_task_ids

            await controller1.stop()  # simulates hanging up mid-task

            snapshot = await shared_store.load(user_id)
            assert snapshot is not None
            assert snapshot.active_task_ids == controller1.active_task_ids
            assert snapshot.ended_at is not None

            # --- Session 2: a new call, next day, resumes. ---
            second_call_speech: list[tuple[str, str]] = []

            async def speak2(text: str, priority: str) -> None:
                second_call_speech.append((text, priority))

            controller2 = ConversationController(
                client=CollectiveOSClient(ws_url), speak=speak2, session_store=shared_store
            )
            await controller2.start(session_id=str(uuid4()), user_id=user_id, resume=True)

            # Task state carried over from the snapshot before any event
            # arrives on this new connection.
            assert controller2.active_task_ids == controller1.active_task_ids

            await controller2.handle_utterance("where were we", router_class="session_query")
            await _wait_until(lambda: controller2.current_task_id is None)

            spoken = [t for t, _ in second_call_speech]
            assert any("Yesterday" in t for t in spoken), spoken
            assert any("ready for Friday" in t for t in spoken), spoken

            await controller2.stop()

    asyncio.run(scenario())
