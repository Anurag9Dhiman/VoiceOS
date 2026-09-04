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

Confirmation is risk-tiered, not blanket: every category acts immediately
on classification. `small_talk`/`simple_lookup` never leave this process,
so there's nothing to confirm. `new_intent`/`modify_inflight`/
`session_query` forward straight to CollectiveOS, which is the system
that actually knows the resulting risk_class -- its own
`confirmation_request` (`risk_class: write`) already pauses for exactly
the actions that need a human checkpoint, and never does for reads. A
local pre-send gate here would either be redundant with that (for writes)
or needless friction (for reads) -- an earlier revision of this file
added exactly that blanket gate and it was deliberately removed once this
became clear.

Local fallback (alongside CollectiveOS, not a replacement for it): if
CollectiveOS is unreachable -- `start()`'s connect() fails, or a later
send() does -- and a `local_agent` was configured (see local_agent.py;
optional, off unless the `local-agent` extra is installed and wired in
agent.py), every category that would otherwise forward gets answered by
it instead, framed so it can never claim to have executed a real task
(voice-service holds zero connector credentials by design). With no
`local_agent` configured, this is exactly today's behavior: the
exception propagates.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
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
from .local_agent import LocalFallbackAgent
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
        local_agent: LocalFallbackAgent | None = None,
        on_task_state_changed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._client = client
        self._speak = speak
        self._router = router or HaikuRouter()
        self._ack = ack or AckGenerator()
        self._entities = entities or EntityStack()
        self._session_store = session_store or InMemorySessionStore()
        self._rate_limiter = rate_limiter or InMemoryRateLimiter()
        # Optional: see module docstring's "Local fallback" section. None
        # (the default) means CollectiveOS being unreachable behaves
        # exactly as it always has -- an exception propagates.
        self._local_agent = local_agent
        self._degraded = False
        # Optional: notified whenever current_task_id/waiting_reason may have
        # changed, so a caller with no other way to observe this state (e.g.
        # a Gemini Live session, which has no per-call has_active_task
        # argument the way HaikuRouter.classify does) can push it in as
        # updated instructions instead.
        self._on_task_state_changed = on_task_state_changed

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

        try:
            await self._client.connect(session_id=session_id, user_id=user_id, resume=resume)
        except Exception:
            if self._local_agent is None:
                raise
            logger.warning(
                "CollectiveOS unreachable at session start (session %s) -- falling back to the local agent",
                session_id,
            )
            self._degraded = True
            return
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _send_or_fallback(self, event: object, text: str) -> None:
        """CollectiveOS being unreachable is not a crash: falls back to
        the local agent (if one is configured) instead of forwarding --
        either because start() already found CollectiveOS unreachable
        (self._degraded) or because this send() call itself fails, which
        also latches self._degraded so later turns this call don't
        re-attempt a doomed send(). With no local agent configured, this
        is exactly today's behavior: the exception propagates."""
        if not self._degraded:
            try:
                await self._client.send(event)  # type: ignore[arg-type]
                return
            except Exception:
                if self._local_agent is None:
                    raise
                logger.warning("send() failed, CollectiveOS unreachable -- falling back locally: %r", text)
                self._degraded = True

        reply = await self._local_agent.respond(text)  # type: ignore[union-attr]
        await self._speak(reply, "low")

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
        Anthropic key."""
        if self._user_id is not None and not await self._rate_limiter.allow(self._user_id):
            logger.warning("rate limit exceeded for user %s, dropping: %r", self._user_id, text)
            await self._speak("Let's slow down a moment.", "low")
            return

        router_class = router_class or await self._router.classify(
            text, has_active_task=self.current_task_id is not None
        )

        if router_class == "confirmation_reply":
            if self.current_task_id is None:
                logger.warning("confirmation_reply with no active task, dropping: %r", text)
                return
            decision, modification = _parse_decision(text)
            await self._send_or_fallback(
                ConfirmationResponse(
                    session_id=self._session_id,
                    task_id=self.current_task_id,
                    decision=decision,
                    modification=modification,
                ),
                text,
            )
            return

        if router_class == "modify_inflight":
            if self.current_task_id is None:
                logger.warning("modify_inflight with no active task, dropping: %r", text)
                return
            await self._send_or_fallback(
                Interrupt(session_id=self._session_id, target_task_id=self.current_task_id, text=text),
                text,
            )
            return

        if router_class in ("small_talk", "simple_lookup"):
            reply = await self._ack.instant_ack(router_class, text)
            await self._speak(reply, "low")
            return

        if router_class == "new_intent":
            entity_refs = self._entities.resolve(text)
            await self._send_or_fallback(
                UserUtterance(
                    session_id=self._session_id,
                    text=text,
                    router_class="new_intent",
                    entity_refs=entity_refs,
                    ts=datetime.now(UTC).isoformat(),
                ),
                text,
            )
            return

        if router_class == "session_query":
            await self._send_or_fallback(SessionQuery(session_id=self._session_id, query=text), text)
            return

        logger.warning("unhandled router_class %r for utterance %r", router_class, text)

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
