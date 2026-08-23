"""Framework-agnostic conversation orchestration: router decision in,
contract events out, CollectiveOS events back out as speech (via a
SpeechComposer). No LiveKit import here on purpose -- this and
speech_composer.py are what weeks 5-8 actually add, and both are testable
against a real mock-agent-backend over a real socket without any audio
hardware. agent.py is the thin LiveKit-specific binding on top.

Send and receive are decoupled deliberately: `handle_utterance` only
decides what to *send* based on the router's classification; a separate
receive loop speaks whatever CollectiveOS sends back, whenever it arrives.
That mirrors how the real duplex connection behaves -- an ack can arrive
while the user is still mid-sentence on their next utterance.

Multi-task + entity stack + resume (weeks 7-8): active tasks are tracked in
arrival order, not as a single slot, so more than one task can be in flight
at once (plan: "multi-task sessions"). `current_task_id` picks whichever
task is actually waiting on the user, falling back to the most recent --
that's the one an unqualified "wait, keep the 10am" should target. Session
state (active task ids + the entity stack) is snapshotted to a SessionStore
keyed by user_id, so a resumed connection can restore both.

Universal local confirmation gate: every classified action -- small_talk,
simple_lookup, new_intent, modify_inflight, session_query -- is proposed
and held as a `_PendingAction` rather than acted on immediately; the next
utterance is parsed as approve/reject via the same `_parse_decision` used
for `confirmation_reply`, and only on approval does `_execute_action` run
the original logic. `confirmation_reply` itself is exempt -- it's already
the user's confirmation of a CollectiveOS-originated
`confirmation_request`, so gating it again would be circular.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from voice_contract import (
    Ack,
    AgentToVoiceEvent,
    ClarificationRequest,
    ConfirmationRequest,
    ConfirmationResponse,
    Decision,
    Done,
    Error,
    Interrupt,
    Priority,
    Progress,
    SessionQuery,
    Speak as SpeakEvent,
    TaskUpdate,
    UserUtterance,
)

from .ack import AckGenerator
from .collectiveos_client import CollectiveOSTransport
from .entity_stack import EntityStack
from .rate_limiter import InMemoryRateLimiter, RateLimiter
from .router import HaikuRouter
from .session_store import InMemorySessionStore, SessionSnapshot, SessionStore

logger = logging.getLogger("voice_service.conversation")

Speak = Callable[[str, Priority], Awaitable[object]]

_APPROVE_WORDS = re.compile(r"\b(yes|yeah|yep|approve|go ahead|do it|sure|correct|confirm)\b", re.I)
# Deliberately narrow to unambiguous negations. Words like "cancel"/"stop"/
# "don't" are NOT included: "cancel the 9am" or "don't do the 11, just the
# 9" name a specific change, not a blanket no -- those need to reach
# CollectiveOS as a modification with the full text, not get silently
# downgraded to reject.
_REJECT_WORDS = re.compile(r"\b(no|nope|nah|negative|reject)\b", re.I)


def _parse_decision(text: str) -> tuple[Decision, str | None]:
    """Maps a spoken reply to a confirmation onto approve/reject/modify.
    Anything that isn't a clean yes/no is treated as a modification carrying
    the utterance itself -- CollectiveOS decides what to do with it."""
    if _APPROVE_WORDS.search(text) and not _REJECT_WORDS.search(text):
        return "approve", None
    if _REJECT_WORDS.search(text) and not _APPROVE_WORDS.search(text):
        return "reject", None
    return "modify", text


# confirmation_reply is deliberately absent: it's already the user's
# confirmation of something CollectiveOS asked, so it's handled directly
# in handle_utterance rather than gated a second time here.
_LOCALLY_GATED_CLASSES = ("small_talk", "simple_lookup", "new_intent", "modify_inflight", "session_query")


@dataclass
class _PendingAction:
    kind: str  # one of _LOCALLY_GATED_CLASSES
    text: str
    # Captured at proposal time for modify_inflight, so the eventual send
    # targets the task that was actually current when this was proposed,
    # not whatever happens to be current by the time the user approves.
    target_task_id: str | None = None


def _describe_action(router_class: str, text: str) -> str:
    # Deliberately NOT "Shall I go ahead?" -- that's CollectiveOS's own
    # write-confirmation phrasing (see mock_agent_backend/scenarios.py). A
    # real user hearing the same question twice in a row, once for this
    # local gate and again for CollectiveOS's own confirmation, wouldn't
    # know which one they were answering.
    if router_class in ("small_talk", "simple_lookup"):
        return "I'm ready to reply. OK to proceed?"
    if router_class == "new_intent":
        return f"I'll ask CollectiveOS to handle: {text!r}. OK to proceed?"
    if router_class == "modify_inflight":
        return f"I'll send this as an update to the current task: {text!r}. OK to proceed?"
    if router_class == "session_query":
        return "I'll check on your session status. OK to proceed?"
    return "I'm ready to proceed. OK to proceed?"


class ConversationController:
    def __init__(
        self,
        *,
        client: CollectiveOSTransport,
        speak: Speak,
        router: HaikuRouter | None = None,
        ack: AckGenerator | None = None,
        entities: EntityStack | None = None,
        session_store: SessionStore | None = None,
        rate_limiter: RateLimiter | None = None,
        on_task_state_changed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._client = client
        self._speak = speak
        self._router = router or HaikuRouter()
        self._ack = ack or AckGenerator()
        self._entities = entities or EntityStack()
        self._session_store = session_store or InMemorySessionStore()
        self._rate_limiter = rate_limiter or InMemoryRateLimiter()
        # Optional: notified whenever current_task_id/waiting_reason may have
        # changed, so a caller with no other way to observe this state (e.g.
        # a Gemini Live session, which has no per-call has_active_task
        # argument the way HaikuRouter.classify does) can push it in as
        # updated instructions instead.
        self._on_task_state_changed = on_task_state_changed
        # Set by handle_utterance whenever a locally-gated action (see
        # _LOCALLY_GATED_CLASSES) is proposed but not yet approved; the
        # *next* utterance is consumed as the answer to it rather than
        # being classified normally.
        self._pending_action: _PendingAction | None = None

        self._session_id: str | None = None
        self._user_id: str | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._started_at: float | None = None

        # Active tasks in arrival/touch order; last = most recently active.
        self._task_order: list[str] = []
        self._task_waiting_reason: dict[str, str | None] = {}

    @property
    def active_task_ids(self) -> list[str]:
        return list(self._task_order)

    @property
    def current_task_id(self) -> str | None:
        """Whichever task is actually waiting on the user, else the most
        recently active one -- the natural target for an unqualified
        follow-up like "wait, keep the 10am"."""
        for task_id in reversed(self._task_order):
            if self._task_waiting_reason.get(task_id):
                return task_id
        return self._task_order[-1] if self._task_order else None

    @property
    def waiting_reason(self) -> str | None:
        task_id = self.current_task_id
        return self._task_waiting_reason.get(task_id) if task_id else None

    async def start(self, *, session_id: str, user_id: str, resume: bool = False) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._started_at = time.time()

        if resume:
            snapshot = await self._session_store.load(user_id)
            if snapshot is not None:
                self._entities.restore(snapshot.entity_stack)
                for task_id in snapshot.active_task_ids:
                    self._touch_task(task_id)
                if snapshot.active_task_ids and self._on_task_state_changed is not None:
                    await self._on_task_state_changed()

        await self._client.connect(session_id=session_id, user_id=user_id, resume=resume)
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def stop(self) -> None:
        await self._persist_session(ended=True)
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._client.close()

    async def handle_utterance(self, text: str, router_class: str | None = None) -> None:
        """router_class is normally decided by the Haiku router; tests may
        pass it directly to exercise the send-side logic without a live
        Anthropic key. If a _pending_action is outstanding, router_class is
        ignored entirely -- this utterance is consumed as the approve/reject
        answer to it instead of being classified."""
        if self._user_id is not None and not await self._rate_limiter.allow(self._user_id):
            logger.warning("rate limit exceeded for user %s, dropping: %r", self._user_id, text)
            await self._speak("Let's slow down a moment.", "low")
            return

        if self._pending_action is not None:
            await self._resolve_pending_action(text)
            return

        router_class = router_class or await self._router.classify(
            text, has_active_task=self.current_task_id is not None
        )

        if router_class == "confirmation_reply":
            if self.current_task_id is None:
                logger.warning("confirmation_reply with no active task, dropping: %r", text)
                return
            decision, modification = _parse_decision(text)
            await self._client.send(
                ConfirmationResponse(
                    session_id=self._session_id,
                    task_id=self.current_task_id,
                    decision=decision,
                    modification=modification,
                )
            )
            return

        if router_class == "modify_inflight" and self.current_task_id is None:
            logger.warning("modify_inflight with no active task, dropping: %r", text)
            return

        if router_class in _LOCALLY_GATED_CLASSES:
            target_task_id = self.current_task_id if router_class == "modify_inflight" else None
            self._pending_action = _PendingAction(kind=router_class, text=text, target_task_id=target_task_id)
            await self._speak(_describe_action(router_class, text), "high")
            return

        logger.warning("unhandled router_class %r for utterance %r", router_class, text)

    async def _resolve_pending_action(self, text: str) -> None:
        pending = self._pending_action
        self._pending_action = None
        assert pending is not None  # only called when set, just cleared above

        decision, _ = _parse_decision(text)
        if decision != "approve":
            await self._speak("Okay, cancelled.", "low")
            return

        await self._execute_action(pending.kind, pending.text, target_task_id=pending.target_task_id)

    async def _execute_action(self, router_class: str, text: str, *, target_task_id: str | None = None) -> None:
        if router_class in ("small_talk", "simple_lookup"):
            reply = await self._ack.instant_ack(router_class, text)
            await self._speak(reply, "low")
            return

        if router_class == "new_intent":
            entity_refs = self._entities.resolve(text)
            await self._client.send(
                UserUtterance(
                    session_id=self._session_id,
                    text=text,
                    router_class="new_intent",
                    entity_refs=entity_refs,
                    ts=datetime.now(UTC).isoformat(),
                )
            )
            return

        if router_class == "modify_inflight":
            await self._client.send(
                Interrupt(session_id=self._session_id, target_task_id=target_task_id, text=text)
            )
            return

        if router_class == "session_query":
            await self._client.send(SessionQuery(session_id=self._session_id, query=text))
            return

    def _touch_task(self, task_id: str) -> None:
        if task_id in self._task_order:
            self._task_order.remove(task_id)
        self._task_order.append(task_id)
        self._task_waiting_reason.setdefault(task_id, None)

    def _drop_task(self, task_id: str) -> None:
        if task_id in self._task_order:
            self._task_order.remove(task_id)
        self._task_waiting_reason.pop(task_id, None)

    async def _persist_session(self, *, ended: bool = False) -> None:
        if self._user_id is None:
            return
        await self._session_store.save(
            SessionSnapshot(
                user_id=self._user_id,
                active_task_ids=self.active_task_ids,
                entity_stack=self._entities.snapshot(),
                started_at=self._started_at or time.time(),
                ended_at=time.time() if ended else None,
            )
        )

    async def _receive_loop(self) -> None:
        async for event in self._client:
            await self._handle_agent_event(event)

    async def _handle_agent_event(self, event: AgentToVoiceEvent) -> None:
        if isinstance(event, Ack):
            self._touch_task(event.task_id)
            self._entities.observe_agent_speech(event.text)
            await self._speak(event.text, "low")
        elif isinstance(event, (Progress, SpeakEvent)):
            self._touch_task(event.task_id)
            self._entities.observe_agent_speech(event.text)
            await self._speak(event.text, event.priority)
        elif isinstance(event, ConfirmationRequest):
            self._touch_task(event.task_id)
            self._task_waiting_reason[event.task_id] = "user_confirm"
            self._entities.observe_agent_speech(event.speak)
            await self._speak(event.speak, "high")
        elif isinstance(event, ClarificationRequest):
            self._touch_task(event.task_id)
            self._task_waiting_reason[event.task_id] = "user_clarify"
            self._entities.observe_agent_speech(event.speak)
            await self._speak(event.speak, "high")
        elif isinstance(event, TaskUpdate):
            self._touch_task(event.task_id)
            self._task_waiting_reason[event.task_id] = event.waiting_reason
        elif isinstance(event, Done):
            self._entities.observe_agent_speech(event.summary_speak)
            await self._speak(event.summary_speak, "high")
            self._drop_task(event.task_id)
        elif isinstance(event, Error):
            await self._speak(event.speak, "high")
            if not event.recoverable:
                self._drop_task(event.task_id)
        else:
            logger.warning("unhandled agent event type %r", type(event).__name__)

        await self._persist_session()
        if self._on_task_state_changed is not None:
            await self._on_task_state_changed()
