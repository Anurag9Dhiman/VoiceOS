"""LiveKit Agents entrypoint. Weeks 3-4 proved the raw STT/TTS round trip
(EchoAgent, now retired); weeks 5-6 added the turn manager, the Haiku
router, and the CollectiveOS bridge on top of it. This revision replaces
the cascaded Deepgram+Cartesia+text-router pipeline with Gemini Live
(google.realtime.RealtimeModel) -- one full-duplex audio session standing
in for STT+LLM+TTS all at once.

This module is deliberately thin: routing decisions, contract event
forwarding, and confirmation/interrupt handling all live in conversation.py
(framework-agnostic, unit-tested against a real mock-agent-backend socket).
RoutingAgent's job is now a single tool, classify_utterance, that the live
model is instructed to call on every turn -- the only reliable hook into a
RealtimeModel-backed turn (llm_node, the old hook, is never invoked for a
realtime session; LiveKit's own turn-completion logic short-circuits it).

Two things this design has to work around, both confirmed against the
installed livekit-agents/livekit-plugins-google source rather than assumed:

1. Forced tool-calling (tool_choice) is not honored for Gemini Live through
   LiveKit (per_response_tool_choice=False in RealtimeCapabilities) -- so
   "always call classify_utterance first" is an instruction, not a
   guarantee. ScriptedSpeechGuard + a reactive session.interrupt() are the
   safety net, not a hard gate.
2. session.say() requires RealtimeCapabilities.supports_say, which is False
   for this plugin -- so raw_speak drives session.generate_reply(instructions=...)
   instead, which (usefully) returns the same SpeechHandle shape say() does,
   so turn_manager.py's undelivered-utterance tracking needed no changes.

The classify_utterance tool asks the model for its own rendition of the
utterance too (a `transcript` argument), since a RealtimeModel session has
no discrete finalized-transcript step of its own for code to read the way
the old STT pipeline did. Where possible this is overridden by a higher-
fidelity source: Gemini Live's own input transcription, surfaced as
AgentSession's "conversation_item_added" events, is captured into
_latest_user_transcript and preferred over the model-authored one whenever
it's available for the same turn.

Wake word: the session stays silent and takes no action on anything until
Settings.wake_word is heard (checked in code against the real transcript,
not left to the model's own judgment -- the same lesson as
ScriptedSpeechGuard/the reactive interrupt() above: instructions alone
aren't enforced for this session type). WakeState.awake flips true the
first time the phrase is found and stays true until Settings.sleep_word is
heard, which flips it back -- also a code-level check, for the same
reason. Neither phrase needs repeating for ordinary follow-up turns; only
saying "go back to sleep" (or whatever sleep_word is configured to) does.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
)
from livekit.plugins import google

from .ack import AckGenerator
from .config import Settings
from .conversation import ConversationController
from .entity_stack import EntityStack
from .latency import LatencyAggregator
from .llm_provider import make_llm_client
from .rate_limiter import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from .resilient_client import ReconnectingCollectiveOSClient
from .router import _ROUTER_CLASSES, _SYSTEM_PROMPT
from .session_store import InMemorySessionStore, RedisSessionStore, SessionStore
from .speech_composer import SpeechComposer
from .turn_manager import TurnManagerSettings, UndeliveredTracker

logger = logging.getLogger("voice_service.agent")

TURN_MANAGER_SETTINGS = TurnManagerSettings()

_INSTRUCTIONS = (
    _SYSTEM_PROMPT
    + "\n\nAlso include a `transcript` field: your best rendition of exactly "
    "what the user said. After classifying: for small_talk and simple_lookup, "
    "answer the user yourself, briefly. For every other category, say "
    "nothing -- a separate system will respond on CollectiveOS's behalf."
)

CLASSIFY_UTTERANCE_SCHEMA = {
    "name": "classify_utterance",
    "description": "Record the router classification and transcript for the utterance.",
    "parameters": {
        "type": "object",
        "properties": {
            "router_class": {"type": "string", "enum": list(_ROUTER_CLASSES)},
            "transcript": {
                "type": "string",
                "description": "Your best rendition of exactly what the user said.",
            },
        },
        "required": ["router_class", "transcript"],
    },
}

_LOCAL_ANSWER_CLASSES = ("small_talk", "simple_lookup")


@dataclass
class ScriptedSpeechGuard:
    """Set around every CollectiveOS-driven raw_speak() call. A scripted
    line is narration, not a real user turn -- if the model tries to run
    classify_utterance against its own scripted speech (plausible, since
    tool_choice isn't enforced), this short-circuits it instead of
    forwarding a phantom utterance to ConversationController."""

    speaking: bool = False


@dataclass
class WakeState:
    """True once the wake word has been heard; flips back to False once the
    sleep word is heard. See module docstring for why this is a
    code-level check against the real transcript, not an instruction the
    model follows."""

    awake: bool = False


def _after_phrase(text: str, phrase: str) -> str | None:
    """None if the phrase isn't present; otherwise whatever follows it, so
    "hey voiceos, clear my morning" wakes and hands off the remainder as
    this turn's utterance in one breath, matching normal wake-word UX.
    Used for both the wake word and the sleep word -- same matching rule
    either way.

    Matches word-by-word with optional punctuation/whitespace between each
    word, not the phrase as one literal substring -- confirmed live that
    Gemini Live's own transcription naturally inserts a comma when a
    multi-word phrase is spoken as a direct address ("voiceos, go to
    sleep"), which a literal-substring match would silently reject."""
    words = phrase.split()
    pattern = r"\b" + r"\b[\s,]*\b".join(re.escape(word) for word in words) + r"\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return None
    # Strip the separator punctuation people naturally say after a wake
    # phrase ("hey voiceos, clear my morning") along with whitespace.
    return text[match.end() :].strip(" ,:;-").strip()


@dataclass
class _LatestUserTranscript:
    """Gemini Live's own input transcription, captured from AgentSession's
    conversation_item_added events -- higher fidelity than the transcript
    the model self-reports as a classify_utterance argument, since it's the
    provider's own speech recognition rather than the model's paraphrase.
    Used when available; the tool argument is the fallback."""

    text: str | None = None


@dataclass
class _AgentRef:
    """Breaks a construction-order cycle: ConversationController needs an
    on_task_state_changed callback at construction time, but that callback
    needs to call RoutingAgent.update_instructions(), and RoutingAgent needs
    the controller passed into its own constructor. Set once RoutingAgent
    is actually built; the callback no-ops before then (the receive loop
    can start, and therefore call it, before session.start() runs)."""

    agent: RoutingAgent | None = None


def _context_instructions(current_task_id: str | None, waiting_reason: str | None) -> str:
    """The old HaikuRouter.classify() got has_active_task fresh on every
    call; a live session's instructions are static unless something pushes
    updates, so this is that something -- without it, confirmation_reply
    ("yes go ahead") has no signal that a confirmation is even pending and
    the model has nothing to go on."""
    if current_task_id is None:
        context = "There is no in-flight task right now."
    elif waiting_reason == "user_confirm":
        context = "There is an in-flight task awaiting the user's confirmation reply right now."
    elif waiting_reason == "user_clarify":
        context = "There is an in-flight task awaiting the user's clarification right now."
    else:
        context = "There is an in-flight task running right now, not yet awaiting a reply."
    return _INSTRUCTIONS + "\n\n" + context


class RoutingAgent(Agent):
    """No text LLM configured -- Gemini Live is the entire audio pipeline.
    classify_utterance is the only reliable per-turn hook into a
    RealtimeModel session; everything downstream of a classification is a
    lookup table (ack.py) or a protocol-following forward (conversation.py),
    exactly as before."""

    def __init__(
        self,
        controller: ConversationController,
        guard: ScriptedSpeechGuard,
        latest_transcript: _LatestUserTranscript,
        wake_state: WakeState,
        wake_word: str,
        sleep_word: str,
    ) -> None:
        super().__init__(instructions=_INSTRUCTIONS)
        self._controller = controller
        self._guard = guard
        self._latest_transcript = latest_transcript
        self._wake_state = wake_state
        self._wake_word = wake_word
        self._sleep_word = sleep_word

    @function_tool(raw_schema=CLASSIFY_UTTERANCE_SCHEMA)
    async def classify_utterance(self, raw_arguments: dict[str, object], ctx: RunContext) -> str:
        if self._guard.speaking:
            return "That was your own scripted line, not a user turn. Ignore it."

        router_class = raw_arguments.get("router_class")
        transcript = str(self._latest_transcript.text or raw_arguments.get("transcript") or "")

        if not self._wake_state.awake:
            remainder = _after_phrase(transcript, self._wake_word)
            if remainder is None:
                return "The wake word wasn't heard. Stay completely silent."
            self._wake_state.awake = True
            transcript = remainder
            if not transcript:
                return "Just woke up. Acknowledge briefly that you're listening now."
        elif _after_phrase(transcript, self._sleep_word) is not None:
            self._wake_state.awake = False
            return "Going back to sleep. Acknowledge briefly, then stay completely silent until woken again."

        await self._controller.handle_utterance(transcript, router_class=router_class)  # type: ignore[arg-type]

        if router_class in _LOCAL_ANSWER_CLASSES:
            return "Go ahead and answer now, briefly."

        await ctx.session.interrupt()
        return "Stay silent. A separate system will respond on CollectiveOS's behalf."


def _record_metric(aggregator: LatencyAggregator, metric: object) -> None:
    # STTMetrics/EOUMetrics/TTSMetrics never fire for a RealtimeModel
    # session (no separate STT/TTS stages to emit them from) -- left in
    # place rather than guessed-at, pending live verification of whatever
    # metric types (if any) RealtimeModel sessions actually emit.
    if isinstance(metric, metrics.STTMetrics):
        aggregator.record("stt.duration", metric.duration)
    elif isinstance(metric, metrics.EOUMetrics):
        aggregator.record("eou.end_of_utterance_delay", metric.end_of_utterance_delay)
    elif isinstance(metric, metrics.TTSMetrics):
        aggregator.record("tts.ttfb", metric.ttfb)


def _make_speech_composer(
    session: AgentSession, tracker: UndeliveredTracker, guard: ScriptedSpeechGuard
) -> SpeechComposer:
    async def raw_speak(text: str) -> object:
        guard.speaking = True
        try:
            handle = session.generate_reply(
                instructions=(
                    "Say exactly the following to the user, verbatim, with no "
                    f"additions, no rephrasing, no extra commentary: {text!r}"
                ),
                tool_choice="none",
                allow_interruptions=TURN_MANAGER_SETTINGS.allow_interruptions,
            )
        finally:
            guard.speaking = False
        tracker.track(handle, text)
        return handle

    async def raw_interrupt() -> None:
        await session.interrupt()

    return SpeechComposer(
        raw_speak=raw_speak,
        raw_interrupt=raw_interrupt,
        is_user_speaking=lambda: session.user_state == "speaking",
    )


def _build_stores(settings: Settings) -> tuple[SessionStore, RateLimiter]:
    """Split out of entrypoint() so this branch -- the entire reason
    REDIS_URL exists -- is unit-testable on its own. RedisRateLimiter was
    once implemented but never actually reached this branch (only the
    session store did); this split makes that specific regression easy to
    catch again."""
    if settings.redis_url:
        return (
            RedisSessionStore.from_url(settings.redis_url),
            RedisRateLimiter.from_url(settings.redis_url),
        )
    return InMemorySessionStore(), InMemoryRateLimiter()


async def entrypoint(ctx: JobContext) -> None:
    settings = Settings()
    aggregator = LatencyAggregator()
    tracker = UndeliveredTracker()
    guard = ScriptedSpeechGuard()
    latest_transcript = _LatestUserTranscript()
    wake_state = WakeState()

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            api_key=settings.google_api_key,
            model=settings.gemini_live_model,
            **TURN_MANAGER_SETTINGS.as_realtime_model_kwargs(),
        ),
    )

    llm_client = make_llm_client(settings)
    composer = _make_speech_composer(session, tracker, guard)
    agent_ref = _AgentRef()

    async def _on_task_state_changed() -> None:
        if agent_ref.agent is None:
            return
        await agent_ref.agent.update_instructions(
            _context_instructions(controller.current_task_id, controller.waiting_reason)
        )

    session_store, rate_limiter = _build_stores(settings)

    controller = ConversationController(
        client=ReconnectingCollectiveOSClient(settings.collectiveos_ws_url),
        speak=composer.speak,
        ack=AckGenerator(client=llm_client),
        entities=EntityStack(),
        session_store=session_store,
        rate_limiter=rate_limiter,
        on_task_state_changed=_on_task_state_changed,
    )

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        _record_metric(aggregator, ev.metrics)
        logger.info(aggregator.summary_line())

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        if ev.item.role == "user":
            text = ev.item.text_content
            if text:
                latest_transcript.text = text

    await ctx.connect()
    # resume=False: real resume-detection needs a signal from the call setup
    # (room metadata / participant attributes carrying "this user has a
    # pending task") that doesn't exist yet -- that's an integration-time
    # decision for whatever places the call, not something inferable here.
    await controller.start(session_id=ctx.room.name, user_id=ctx.job.id, resume=False)

    async def _stop_controller() -> None:
        await controller.stop()

    ctx.add_shutdown_callback(_stop_controller)

    routing_agent = RoutingAgent(
        controller, guard, latest_transcript, wake_state, settings.wake_word, settings.sleep_word
    )
    agent_ref.agent = routing_agent
    await session.start(agent=routing_agent, room=ctx.room)


def _init_sentry() -> None:
    """Read SENTRY_DSN straight from the environment rather than through
    Settings(): Settings requires GOOGLE_API_KEY too, and eagerly
    constructing it here would reintroduce the exact bug already fixed
    once -- `voice-service --help` failing with no keys set.

    Split out of main() so this branch is unit-testable without also
    invoking cli.run_app()."""
    sentry_dsn = os.environ.get("SENTRY_DSN")
    if sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)


def main() -> None:
    _init_sentry()

    # ws_url/api_key/api_secret are left unset here: WorkerOptions already
    # falls back to the LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET env
    # vars itself, and only needs them once a job actually connects -- not
    # at CLI parse time, which is what eagerly building Settings() here
    # would otherwise block (e.g. `voice-service --help`).
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
