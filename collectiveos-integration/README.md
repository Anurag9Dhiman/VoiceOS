# CollectiveOS integration handoff

Per the plan's own sequencing (sec. 10: *"Open the CollectiveOS integration
branch early with just the additive migration ... even though integration
itself comes last"*), this directory is what that branch needs, prepared
ahead of time since **the CollectiveOS repo doesn't exist in this
workspace** — voice-platform is deliberately scoped as a front-end that
knows nothing about CollectiveOS's internals beyond the frozen contract
(see the project's own architecture memory: CollectiveOS is a separate
system, this repo only talks to it over `/contract`). Nothing here runs or
is tested from this repo; it's meant to be copied into the CollectiveOS
repo when that integration actually happens.

## What's here

- **`migrations/0001_voice_sessions.sql`** — the two additive schema
  changes from plan sec. 6: a new `voice_sessions` table and one new
  `tasks.waiting_reason` column. Nothing existing changes shape.

## What CollectiveOS needs to build (not included here — CollectiveOS's own code)

A FastAPI WebSocket endpoint implementing `/contract` (`session_start`,
`user_utterance`, `interrupt`, `confirmation_response`, `session_query`,
`session_end` in; `ack`, `progress`, `confirmation_request`,
`clarification_request`, `task_update`, `speak`, `done`, `error` out),
backed by real task state in Postgres instead of the in-memory `STORE` in
`services/mock-agent-backend/src/mock_agent_backend/session.py`.

**`services/mock-agent-backend`'s `app.py` is the structural reference** for
what that endpoint's shape should look like — same routes, same connection
lifecycle (accept → read `session_start` → dispatch on the first
substantive event → relay until `session_end`), same event models
(`libs/voice-contract`, already a real installable package this endpoint
could depend on directly). What differs is everything `scenarios.py`
currently *scripts*: CollectiveOS replaces the scripted event sequences
with its actual agent output — internally calling its own tool backends
over MCP, per its own architecture — and `session.py`'s in-memory `STORE`
becomes real reads/writes against `tasks`/`task_steps` and the new
`voice_sessions` table. MCP is entirely internal to that agent loop; it
never appears on `/contract` (see `contract/README.md`'s "Relationship to
MCP") — this endpoint's job is exactly the same shape whether the tool
call behind an `ack`/`progress`/`done` was implemented via MCP, a direct
SDK call, or anything else.

## Integration-day checklist

1. Apply `migrations/0001_voice_sessions.sql`.
2. Build the WebSocket endpoint per the shape above.
3. Point `voice-service`'s `COLLECTIVEOS_WS_URL` at it instead of
   `ws://localhost:8000/v1/ws` — no other voice-service code changes, since
   `CollectiveOSClient` only depends on the contract, not on which process
   is speaking it.
4. Replay `contract/scenarios/*.json` against the real endpoint the same
   way `services/mock-agent-backend/tests/` do, to confirm it honors the
   contract before switching voice-service over for real.
