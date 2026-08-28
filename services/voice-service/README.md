# voice-service

Weeks 3-10 of the MVP plan, now on **Gemini Live** (`google.realtime.RealtimeModel`
via LiveKit's `livekit-agents[google]` plugin): one full-duplex audio-in/
audio-out session standing in for the STT+LLM+TTS pipeline this used to be
(Deepgram, Cartesia, and a text-based router LLM). An instant-ack fast
path, a turn manager, and a CollectiveOS bridge over the frozen event
contract (shared as the [`voice-contract`](../../libs/voice-contract)
package) sit on top — with multi-task session tracking, an entity stack,
session persistence/resume, and reconnection resilience.

The ack generator's `simple_lookup` answer and the offline router-eval
harness still run on **either Anthropic (Claude Haiku, designed default) or
Gemini** — `gemini_client.py` adapts Gemini's function-calling API to the
exact client shape `router.py`/`ack.py` already expect from Anthropic, so
neither module (or their tests) knows or cares which provider is actually
behind them. Only `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` needs to exist,
not both — see `config.py`'s `resolved_llm_provider`. This is a separate
concern from the live audio path, which needs its own `GOOGLE_API_KEY`.

## How a turn flows

0. The session stays completely silent until `Settings.wake_word` (default
   `"hey voiceos"`) is heard — checked in code against the real transcript
   (`agent.py`'s `WakeState`/`_after_phrase`), not left to the model's
   own judgment, for the same reason as step 1's reactive safety net below:
   instructions alone aren't enforced for this session type. Once awake,
   stays awake until `Settings.sleep_word` (default `"voiceos go to
   sleep"` — deliberately branded and comma-free, since a bare generic
   phrase risks matching normal conversation, and `_after_phrase` matches
   word-by-word with optional punctuation between words, not the phrase
   as one literal string, since Gemini Live's own transcription naturally
   inserts a comma for a direct address like "voiceos, go to sleep").

   Step 0's gate above decides whether to *act* on what's heard — but the
   realtime Gemini Live session itself still opens and streams continuously
   for the whole call regardless, the moment the call connects. `wake_gate.py`
   is the other half: a `WakeWordDetector` Protocol plus a real
   `LiveKitAudioFrameSource` adapter (confirmed against the installed
   `livekit-rtc` API — `AudioStream.from_participant`,
   `TrackSource.SOURCE_MICROPHONE`) for gating *when the expensive session
   even opens* on a cheap local detector instead, the way an always-on
   device like Alexa/Google Home works. `entrypoint()` already calls this
   gate before constructing the realtime session (see
   `_build_wake_detector()`) — it's just wired to `None` today, since no
   on-device engine license or microphone hardware exists in this
   environment to build and verify a real one against. Plugging in a real
   detector (Porcupine or similar) is a one-line change to
   `_build_wake_detector()`; nothing else in `entrypoint()` needs to
   change. The gating logic itself (`wait_for_wake_word`) is fully unit
   tested against a scripted frame source — only `LiveKitAudioFrameSource`
   needs a real room and real audio to verify live.
1. Gemini Live's own turn detection decides when the user's turn ends, then
   the model is instructed to call `classify_utterance` — the only reliable
   per-turn hook into a `RealtimeModel` session (`agent.py`'s
   `RoutingAgent`; the old `llm_node` hook is never invoked for a realtime
   session at all). The tool call carries the model's own rendition of the
   utterance, but this is overridden whenever available by Gemini Live's
   own input transcription, captured off `AgentSession`'s
   `conversation_item_added` events — higher fidelity than the model's
   self-reported paraphrase.
2. The tool handler calls `ConversationController.handle_utterance`
   (`conversation.py`) directly with that transcript and the model's
   chosen category — one of the six from the contract.
3. Confirmation is **risk-tiered, not blanket**. `small_talk`/`simple_lookup`
   are answered immediately — the tool handler tells the model to answer
   briefly itself; `ack.py`'s `simple_lookup` path makes one real
   text-completion call, never crosses the wire to CollectiveOS, so there's
   nothing to confirm. `new_intent`/`modify_inflight`/`session_query`
   forward immediately too — the tool handler calls `session.interrupt()`
   (forced tool-choice isn't honored for Gemini Live through LiveKit, so
   this is a *reactive* safety net, not a guarantee the model stayed
   silent) and sends to CollectiveOS over `ReconnectingCollectiveOSClient`
   (`resilient_client.py`, wraps the raw `collectiveos_client.py`; points
   at `mock-agent-backend` by default) as `user_utterance` / `interrupt` /
   `session_query`. `new_intent` utterances get pronouns resolved against
   `entity_stack.py` first (attached as `entity_refs`). An unexpected drop
   is retried with exponential backoff and resumes the same session
   automatically — `handle_utterance`'s send side and `_receive_loop`'s
   receive side don't need to know a reconnect happened.

   **The actual gate lives downstream, at CollectiveOS**: every write step
   requires its own `confirmation_request` (`risk_class: write`) before it
   executes; reads never do. An earlier revision of this file added a
   *local*, VoiceOS-side pre-send gate in front of every category
   regardless of risk — that was deliberately removed once it became clear
   it either duplicated CollectiveOS's own gate (for the categories that
   turn out to be writes) or added pure friction (for the categories, like
   `small_talk`, that can never be risky at all). `confirmation_reply` is
   how the user answers CollectiveOS's real gate, and always forwards
   immediately via `_parse_decision`.

   (This also retires the "richer confirmation UX — batched approval for a
   string of low-risk actions" item from earlier planning: it assumed the
   local blanket gate this section describes removing, so there's no
   longer a local approval step left to batch.)
4. Whatever CollectiveOS sends back is spoken via `speech_composer.py`
   (priority preemption, one-breath logging) → `agent.py`'s `speak`
   binding, which drives `session.generate_reply(instructions=...)`
   rather than `session.say()` (`RealtimeCapabilities.supports_say` is
   `False` for this plugin), and tracks utterances a barge-in cut off as
   undelivered (`turn_manager.py`, riding `SpeechHandle.interrupted`, which
   `generate_reply()` returns just as `say()` did).
5. Task state (which tasks are active, which is waiting on the user) and
   the entity stack are snapshotted to a `SessionStore` (`session_store.py`
   — in-memory by default, Redis in production) keyed by user_id, so a
   session that resumes hours or days later picks up where it left off.

Every call into `handle_utterance` is metered first, before the router or
CollectiveOS ever sees it: `rate_limiter.py`'s per-user token bucket
(`InMemoryRateLimiter` by default, `RedisRateLimiter` in production) caps
bursts and degrades a flood to "let's slow down a moment" instead of an
unbounded Anthropic bill or a runaway loop.

`conversation.py` and `speech_composer.py` have no LiveKit import at all —
they're what weeks 5-8 actually add, proven against a **real, live
mock-agent-backend over an actual socket**:
- `tests/test_e2e_scenario_a.py` — new intent → confirmation → mid-confirmation
  edit → modified confirmation → done
- `tests/test_e2e_scenario_b.py` — new intent → externally blocked → hang up
  → **new session, resumed** → status summary → done
- Scenario C's shape (batch + mid-flight interrupt + partial failure) also
  covered in `test_e2e_scenario_a.py`
- `tests/test_e2e_reconnect.py` — the underlying socket is force-closed
  mid-task (bypassing our own `close()`, so it's indistinguishable from a
  real network drop), and the **same** `ConversationController` — not a
  freshly constructed one — reconnects with backoff, resumes, and finishes
  the task

The router is bypassed in all three via an explicit `router_class` argument
(no live LLM key of either kind in this environment) — everything else,
including the resume flow across two separate `ConversationController`
instances sharing one `SessionStore`, runs unmocked.

## What's here vs. what needs live credentials or infrastructure to prove out

Unit- and integration-tested (155 tests, all green — 93% statement
coverage; every remaining uncovered line is `entrypoint()`/`main()`/the
router-eval CLI or `wake_gate.py`'s real audio adapter, all of which need
live credentials or a real room to exercise, not more unit tests): the
two load-bearing
`RealtimeCapabilities` assumptions the Gemini Live design depends on
(`supports_say`/`per_response_tool_choice`, both `False`, checked directly
against the installed `livekit-plugins-google`, zero network),
`classify_utterance`'s routing/interrupt/silence decisions and its
`ScriptedSpeechGuard` recursion guard (`RoutingAgent` called directly, no
real LiveKit session needed), the wake/sleep gate (`_after_phrase`'s
one-breath "wake word + command" parsing, case-insensitivity, tolerating a
comma inserted between words, and rejecting a longer word that merely
contains the phrase; full asleep→awake→asleep round trip through
`classify_utterance`), the `raw_speak`→`generate_reply`
binding, router tool-call parsing, ack templates, the full multi-task
`ConversationController` state machine (including which task an
unqualified follow-up targets when more than one is active), the entity
stack's pronoun resolution and its deliberate refusal to treat
sentence-initial capitalized words as entities, session snapshot/restore,
the speech composer's priority and one-breath logic, the router eval
harness's scoring/reporting (not the model's actual judgment — see below),
the reconnect wrapper's backoff/give-up logic against a fake transport, the
rate limiter's token-bucket math, `RedisSessionStore`/`RedisRateLimiter`
against a real local Redis instance (round-trip, overwrite, cross-instance
sharing, and — for the rate limiter — real TTL-driven window expiry, not a
mocked clock; skipped automatically if no Redis is reachable),
`_build_stores`'s REDIS_URL branch (the regression that once left
`RedisRateLimiter` unwired even with Redis configured), the Sentry
wiring's real capture pipeline against a substituted transport,
`TurnManagerSettings.as_realtime_model_kwargs` (`min_interruption_duration`
was dead config until wired to `RealtimeModel`'s `realtime_input_config`
— confirmed both via `google-genai`'s own field docs and by opening a
real Gemini Live session with it set, no rejection), the Gemini
adapter's request/response
translation against real `google-genai` types (constructed directly, no
network) *and* that `HaikuRouter`/`AckGenerator` work against it completely
unmodified, and all three reference scenarios *plus* an unexpected mid-task
connection drop, end to end over a real socket.

**Not verifiable in this environment** (in priority order for the Gemini
Live migration specifically):
- Whether real speech reliably contains the wake/sleep phrase cleanly
  enough for `_after_phrase`'s regex to catch it (accents, background
  noise) — the punctuation-tolerance fix was itself found live (Gemini
  Live inserts a comma for "voiceos, go to sleep"), but the underlying
  logic is otherwise only tested against text, not real audio
- Whether the model actually calls `classify_utterance` on ~every turn,
  and whether audio leaks before `session.interrupt()` lands for the four
  CollectiveOS-forwarding classes — forced tool-choice isn't honored for
  Gemini Live through LiveKit, so the "always classify first" instruction
  is best-effort, not enforced (confirmed via
  `RealtimeCapabilities.per_response_tool_choice is False`, see
  `tests/test_realtime_model_capabilities.py`)
- The model-authored `transcript` fallback's fidelity when Gemini Live's
  own input-transcription event doesn't arrive in time — check against
  `entity_stack.py`'s pronoun resolution and CollectiveOS's own logs
- Whether `session.generate_reply(instructions=...)` reproduces short
  CollectiveOS lines (especially `ConfirmationRequest.speak`) verbatim
  closely enough — `session.say()` isn't available for this plugin
  (`RealtimeCapabilities.supports_say is False`)
- `session.interrupt()`/`SpeechHandle.interrupted` correctness under
  `RealtimeModel` — more load-bearing now than under the old cascaded
  pipeline, since the reactive safety net above depends on it firing fast
- `session.user_state == "speaking"` correctness under `RealtimeModel`
- That `gemini_live_model`'s default
  (`gemini-2.5-flash-native-audio-preview-12-2025`, the only
  Gemini-Developer-API model this plugin version validates against that
  keeps mid-session instructions mutable) actually supports function
  calling reliably in practice — recheck against Google's current model
  catalog, since native-audio preview variants have had function-calling
  regressions before
- The router's actual classification *judgment* in practice — its
  *plumbing* is tested; scoring it for real needs a live
  `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`
  (`uv run python -m voice_service.router_eval`) — this eval is now a
  text-only proxy, decoupled from the live `classify_utterance` tool-call
  path, since there's no way to eval the live model's spoken-utterance
  judgment offline without real audio
- That a real Sentry.io project actually receives an event — the SDK-level
  pipeline itself (`sentry_sdk.init()` configured exactly as `_init_sentry()`
  does, a real uncaught exception, real event serialization, a real
  envelope) is now verified live by substituting sentry_sdk's own
  `Transport` hook for a local recorder (see `tests/test_sentry_wiring.py`)
  — only the final HTTPS call to Sentry's actual cloud endpoint is
  unverified, which needs a real `SENTRY_DSN`
- Real-world audio, latency numbers, and audio edge cases (noise, silence,
  crosstalk) — genuinely need hardware this environment doesn't have.
  Dropped-connection recovery specifically *is* now covered (see above) —
  that gap is closed, not just narrowed.
- `wake_gate.py`'s `LiveKitAudioFrameSource` — the gating *decision*
  (`wait_for_wake_word`) is fully unit tested; reading real microphone
  frames from a real room via `rtc.AudioStream.from_participant` needs a
  real room with a real participant publishing real audio to verify live.
  No real `WakeWordDetector` is wired in either (`_build_wake_detector()`
  returns `None`) — needs an on-device engine license (Porcupine or
  similar) this environment doesn't have.

**Out of reach from this repo entirely** (see
[`/collectiveos-integration`](../../collectiveos-integration) at the repo
root): the real CollectiveOS WebSocket endpoint and its DB migration live
in CollectiveOS's own repo, which doesn't exist in this workspace. That
directory is the prepared handoff — migration SQL plus a structural guide
— per the plan's own "open the integration branch early" sequencing.

## Setup

```sh
cp .env.example .env   # GOOGLE_API_KEY, + one of ANTHROPIC_API_KEY/GEMINI_API_KEY
uv run pytest            # all unit + e2e tests, no live services required

# in one terminal: the mock backend this service talks to by default
cd ../mock-agent-backend && uv run mock-agent-backend

# in another: the voice service, console mode (no LiveKit account needed)
uv run voice-service console

# score the router against the labeled eval set (needs a live LLM key)
uv run python -m voice_service.router_eval
```

`config.py` requires `GOOGLE_API_KEY` unconditionally (it authenticates the
Gemini Live `RealtimeModel` that's now the entire audio pipeline), plus
**at least one** of `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` for the
ack/eval text-completion path (one combined error either way if something's
missing). If both LLM keys are set, Anthropic wins unless `LLM_PROVIDER=gemini`
says otherwise — `resolved_llm_provider` is the single source of truth every
call site defers to. `GEMINI_LIVE_MODEL` (the realtime audio model) defaults
to `gemini-2.5-flash-native-audio-preview-12-2025`; `GEMINI_MODEL` (the
separate plain-text model used by `ack.py`'s `simple_lookup` answer and
`router_eval.py`) defaults to `gemini-flash-latest` — both independently
overridable. `GEMINI_MODEL` deliberately stays on the `-latest` alias
rather than a dated string: unlike the Live API, the plain
`generate_content` API supports it, so it never needs a manual bump when
Google deprecates a specific dated model. (Checked live 2026-08-27: a
proposed `gemini-3.6-flash-live-001` for `GEMINI_LIVE_MODEL` doesn't
exist — the real API returns a 1008 policy-violation error — while plain
`gemini-3.6-flash` does exist and works; `GEMINI_LIVE_MODEL` stays on the
`2.5` model until a real, live-verified `3.x` Live model name is
confirmed. See `config.py`'s comment on `gemini_live_model`.) `COLLECTIVEOS_WS_URL`
defaults to `ws://localhost:8000/v1/ws` — mock-agent-backend's own default
port. `REDIS_URL` is optional; unset means session state and rate-limit
counters are in-process only (lost on restart) rather than persisted to
Redis. `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` are only needed
for `voice-service dev`/`start` against a real room, not `console`.
`SENTRY_DSN` is optional and read directly in `main()`, not through
`Settings()` — same reasoning as `LIVEKIT_URL`: `--help` works with zero
env vars set, and that has to keep being true as things get added, not
just be true today. `WAKE_WORD` defaults to `"hey voiceos"` — the session
takes no action on anything until it's heard; see "How a turn flows" step 0.

`LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY` are optional and enable
LangSmith tracing on `router.py`'s `classify` and `ack.py`'s
`_simple_lookup_answer` (the two actual LLM-call boundaries, `@traceable`
on both) — read directly by the `langsmith` package itself from the
environment, not through `Settings()` or any init call in this codebase.
Confirmed live that an invalid `LANGCHAIN_API_KEY` only logs a warning on
the failed trace submission and doesn't affect the decorated call's
actual return value — same fail-open shape as everything else optional
here.

**A version pin worth knowing about:** `google-genai` (every published
version) requires `websockets<17.0`, which conflicted with the
`websockets>=17.0.1` this project had pinned for `resilient_client.py`.
Resolved by relaxing to `websockets>=14.0,<17.0` — the `websockets.asyncio`
module structure both `collectiveos_client.py` and `resilient_client.py`
depend on has existed since 13.0, well within that range. Full suite
re-verified green at `websockets==16.1.1` before this was considered safe,
not assumed.
