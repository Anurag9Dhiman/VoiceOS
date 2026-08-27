# Voice ↔ CollectiveOS event contract — v1 (frozen)

This is the seam from `voice-platform-mvp-plan-v2.md` §7, expanded to the level a
mock backend and a real integration can both be built against. One JSON Schema
per direction lives in [`schemas/`](schemas/); the three reference scenario
traces live in [`scenarios/`](scenarios/) and are the source of truth the mock
backend's tests replay against.

Changes to this contract after v1 must be **additive only** (new optional
fields, new enum values) to avoid breaking whichever side didn't change —
consistent with the plan's "strictly additive" schema rule for the DB side.

## Relationship to MCP

CollectiveOS uses MCP (Model Context Protocol) internally to call its own
tool backends (calendar, tasks, connectors) as part of its reasoning loop —
that's the standard, intended use of MCP: agent-to-tool-server, not
inter-service routing. This is entirely on CollectiveOS's side of the seam
and invisible from here: this contract only ever carries the events listed
below, regardless of what CollectiveOS uses internally to actually execute
a task. Nothing about MCP changes this contract, and this contract's own
"additive only" rule is exactly what keeps it that way as CollectiveOS's
internals evolve.

## Transport

One JSON-over-WebSocket connection per voice session. Every message is a
single JSON object with a required `"type"` field identifying which event it
is. Unknown fields on a known event type should be ignored, not rejected
(forward compatibility). An unknown `"type"` should be logged and ignored by
the receiver, not treated as a fatal error.

## IDs

- `session_id` — UUID, minted by the voice layer at `session_start`, stable
  for the life of one call (including a resumed call — see below).
- `user_id` — UUID, stable per user across sessions.
- `task_id` — UUID, minted by CollectiveOS when it opens a task in response
  to a `new_intent` utterance. All events about that task carry it.
- Entity references (`meeting_id_123`, etc.) are resolved to concrete
  connector-native IDs by the voice layer's entity stack **before** sending.
  CollectiveOS never receives a bare pronoun.

## Resume semantics

`session_start` with `"resume": true` means: the voice layer is opening a new
WebSocket for a user who has at least one task in a non-terminal state
(`running`, or waiting on `user_confirm` / `user_clarify` / `external`) from a
previous session that ended (via `session_end`, a dropped connection, or a
hangup) before the task reached `done`. CollectiveOS should reply with
`progress` or `speak` events summarizing where each such task stands before
resuming normal flow. A resumed session gets a **new** `session_id` —
continuity is tracked via `user_id` + the task's own `task_id`, not by
reusing the old session_id.

## Priority and preemption

`ack`, `progress`, and `speak` carry `"priority": "low"` unless stated
otherwise; `confirmation_request`, `clarification_request`, and `error` are
always `"priority": "high"`. High-priority events preempt whatever the voice
layer is currently speaking; low-priority events are droppable if the user
is mid-utterance. This is a voice-layer behavior, not something CollectiveOS
needs to implement — documented here so scripted scenarios in the mock
backend set priority correctly.

## Risk class

Every `task_update` step and every `confirmation_request` carries
`risk_class`: `"read"` or `"write"`. Per the plan's CollectiveOS rules, `write`
always requires a `confirmation_request` before the step executes; `read`
never does. The voice layer re-derives risk class from its own classifier and
the stricter of the two wins — the mock backend always sends the class it
intends to enforce, so tests can assert on it directly.

## Events: voice layer → CollectiveOS

See [`schemas/voice_to_agent.schema.json`](schemas/voice_to_agent.schema.json).

| type | when | required fields |
|---|---|---|
| `session_start` | opening the WebSocket | `session_id`, `user_id`, `resume` |
| `user_utterance` | a finalized (non-partial) transcript | `session_id`, `text`, `router_class`, `entity_refs`, `ts` |
| `interrupt` | barge-in with new content mid-task | `session_id`, `target_task_id` (nullable), `text` |
| `confirmation_response` | reply to a `confirmation_request` | `session_id`, `task_id`, `decision` |
| `session_query` | "where were we" / status check | `session_id`, `query` |
| `session_end` | call ends (hangup, timeout, explicit close) | `session_id` |

`router_class` is one of: `small_talk`, `simple_lookup`, `new_intent`,
`modify_inflight`, `confirmation_reply`, `session_query`. Only `new_intent`,
`modify_inflight`, `confirmation_reply`, and `session_query` are ever
forwarded to CollectiveOS as `user_utterance` — `small_talk` and
`simple_lookup` are handled locally by the voice layer and never cross the
wire (documented here so the mock backend never has to special-case them).

`confirmation_response.decision` is one of: `approve`, `reject`, `modify`.
`modification` is required when `decision == "modify"` and omitted otherwise.

## Events: CollectiveOS → voice layer

See [`schemas/agent_to_voice.schema.json`](schemas/agent_to_voice.schema.json).

| type | when | required fields |
|---|---|---|
| `ack` | immediately on accepting a `new_intent` | `task_id`, `text` |
| `progress` | non-terminal update, safe to drop | `task_id`, `text`, `priority` |
| `confirmation_request` | before any `write`/device-control step | `task_id`, `priority`, `speak`, `options`, `risk_class` |
| `clarification_request` | model confidence is low | `task_id`, `priority`, `speak` |
| `task_update` | step-level status change | `task_id`, `status`, `waiting_reason`, `step` |
| `speak` | narration that isn't ack/progress/confirm/clarify/done | `task_id`, `text`, `priority` |
| `done` | task reaches a terminal state | `task_id`, `outcome`, `summary_speak` |
| `error` | unrecoverable or recoverable failure | `task_id`, `recoverable`, `speak` |

`task_update.status` is one of: `pending`, `planning`, `running`,
`waiting`, `blocked`, `cancelled`, `completed`, `failed` — the task state
machine referenced in the architecture diagrams. `waiting_reason` is one of
`user_confirm`, `user_clarify`, `external`, or `null`; it is non-null iff
`status` is `waiting` (`user_confirm`/`user_clarify`) or `blocked`
(`external`). This is the field the voice layer reads to decide whether to
speak-and-listen (`waiting`) or merely narrate (`blocked`).

`done.outcome` is one of: `completed`, `partial`, `failed`.

Every `confirmation_request` and `clarification_request` must be answerable
by a one word reply, and every `speak`/`summary_speak`/`ack`/`progress` text
field obeys the plan's one-breath rule (short enough to say in one breath —
enforced by convention and scenario authoring here, not by schema length
limits, since the right limit is prosodic, not a character count).

## Reference scenarios

Three fixtures under [`scenarios/`](scenarios/), each a JSON array of
`{ "direction": "voice_to_agent" | "agent_to_voice", "event": {...} }` in
wire order. These are what `services/mock-agent-backend`'s tests replay and
assert against, and what the mock backend's scripted behavior is authored
from.

1. **`scenario_a_follow_up_message.json`** — new intent → confirmation
   request → user follow-up (`interrupt`) that modifies the pending action
   before it's approved → modified confirmation → completion. Exercises the
   mid-confirmation edit path.
2. **`scenario_b_multi_day_plan_resume.json`** — new intent opens a
   multi-step task → session ends (hangup) while the task is still `running`
   → a session the next day opens with `resume: true` → CollectiveOS
   summarizes where the task stands → task continues to `done`. Exercises
   session resume and the `voice_sessions`/`tasks.waiting_reason` contract
   the DB migration exists for.
3. **`scenario_c_batch_partial_failure.json`** — new intent opens a batch
   action across multiple items → confirmation → approved → mid-flight
   `interrupt` removes one item from the batch → remaining items execute,
   one fails → `done` with `outcome: "partial"`. Exercises partial failure
   and mid-flight interruption together.
