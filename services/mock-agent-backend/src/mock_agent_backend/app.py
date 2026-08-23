"""Mock CollectiveOS agent backend.

Implements the voice-layer <-> CollectiveOS event contract (v1) with the
three reference scenarios scripted, so the voice layer can be built and
tested end to end before CollectiveOS integration happens for real. See
../../../../contract/README.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import voice_contract as c
from . import scenarios as sc
from .session import STORE, TaskRecord

logger = logging.getLogger("mock_agent_backend")

app = FastAPI(title="Mock CollectiveOS agent backend")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/v1/ws")
async def voice_session(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        first_raw = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    try:
        start = c.parse_voice_to_agent(first_raw)
    except Exception:
        logger.exception("first message was not a valid event")
        await websocket.close(code=4400)
        return

    if not isinstance(start, c.SessionStart):
        logger.warning("expected session_start, got %r", start.type)
        await websocket.close(code=4400)
        return

    incoming: asyncio.Queue[c.VoiceToAgentEvent] = asyncio.Queue()

    async def reader() -> None:
        while True:
            try:
                raw = await websocket.receive_json()
            except WebSocketDisconnect:
                return
            try:
                event = c.parse_voice_to_agent(raw)
            except Exception:
                logger.exception("dropping malformed event: %r", raw)
                continue
            await incoming.put(event)
            if isinstance(event, c.SessionEnd):
                return

    reader_task = asyncio.create_task(reader())

    async def send(event: c.AgentToVoiceEvent) -> None:
        await websocket.send_json(c.dump_agent_event(event))

    try:
        if start.resume:
            await _handle_resume(start, send, incoming)
        else:
            await _handle_fresh_session(start, send, incoming)
    finally:
        await reader_task
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()


async def _handle_fresh_session(
    start: c.SessionStart,
    send: sc.Sender,
    incoming: asyncio.Queue[c.VoiceToAgentEvent],
) -> None:
    first = await incoming.get()
    if not (isinstance(first, c.UserUtterance) and first.router_class == "new_intent"):
        return

    task_id = str(uuid4())
    if sc.matches_scenario_a(first.text):
        STORE.put_task(TaskRecord(task_id, start.user_id, "a", status="running"))
        await sc.run_scenario_a(task_id, send, incoming)
        STORE.tasks_by_user[start.user_id][task_id].status = "completed"
    elif sc.matches_scenario_b(first.text):
        STORE.put_task(TaskRecord(task_id, start.user_id, "b", status="blocked", waiting_reason="external"))
        await sc.run_scenario_b_start(task_id, send)
    elif sc.matches_scenario_c(first.text):
        STORE.put_task(TaskRecord(task_id, start.user_id, "c", status="running"))
        await sc.run_scenario_c(task_id, send, incoming)
        STORE.tasks_by_user[start.user_id][task_id].status = "completed"
    else:
        logger.info("no scripted scenario matches utterance: %r", first.text)


async def _handle_resume(
    start: c.SessionStart,
    send: sc.Sender,
    incoming: asyncio.Queue[c.VoiceToAgentEvent],
) -> None:
    pending = STORE.non_terminal_tasks_for(start.user_id)
    first = await incoming.get()
    if not (isinstance(first, c.SessionQuery) and pending):
        return

    # MVP: one in-flight task per user across the reference scenarios.
    task = pending[0]
    if task.scenario == "b":
        await sc.run_scenario_b_resume(task.task_id, send)
        task.status = "completed"
        task.waiting_reason = None
