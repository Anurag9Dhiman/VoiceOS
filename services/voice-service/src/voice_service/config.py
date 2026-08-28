"""Env-driven settings for the realtime voice model and the ack/eval LLM
provider. Fails fast with one combined error listing every missing key,
rather than letting each plugin raise its own ValueError one at a time as
it happens to be constructed.

Deliberately doesn't include LIVEKIT_URL/API_KEY/API_SECRET: WorkerOptions
already reads those itself with its own env fallback and only needs them
once a job connects to a real room, not for `console` mode (which runs a
simulated local job and needs no LiveKit account at all). Requiring them
here would block the fastest test path for no reason -- it did, in fact,
until this was noticed.

GOOGLE_API_KEY is the one hard requirement: it authenticates the Gemini
Live RealtimeModel that now is the entire audio pipeline (STT+LLM+TTS in
one full-duplex session -- see agent.py). GEMINI_LIVE_MODEL is separate
from GEMINI_MODEL below on purpose: the former is the realtime audio model,
the latter is the plain text-completion model AckGenerator's simple_lookup
answer still uses, and they're independently overridable.

ANTHROPIC_API_KEY/GEMINI_API_KEY are NOT the live audio path -- they feed
AckGenerator's simple_lookup text completion and the offline router_eval
harness only (see ack.py, gemini_client.py, router_eval.py). Each is
optional individually; at least one of the two must be present.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(validation_alias="GOOGLE_API_KEY")
    # One of the two Gemini-Developer-API (non-Vertex) live models the
    # installed livekit-plugins-google actually validates against --
    # deliberately NOT gemini-3.1-flash-live-preview, whose mid-session
    # instructions/tool updates don't apply until the next session (the
    # plugin sets mutable_instructions=False for any "3.1" model), which
    # would break generate_reply(instructions=...)-driven speech entirely.
    gemini_live_model: str = Field(
        default="gemini-3.6-flash-live-001",
        validation_alias="GEMINI_LIVE_MODEL",
    )

    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.6-flash", validation_alias="GEMINI_MODEL")

    # Explicit override for when both keys happen to be set. Unset means
    # "pick whichever one key is actually present"; if both are present
    # with no override, Anthropic wins -- it's the designed default,
    # Gemini is the fallback, not the other way around.
    llm_provider: Literal["anthropic", "gemini"] | None = Field(
        default=None, validation_alias="LLM_PROVIDER"
    )

    # Points at mock-agent-backend by default (its own default port) --
    # override to point at real CollectiveOS once it exists.
    collectiveos_ws_url: str = Field(
        default="ws://localhost:8000/v1/ws", validation_alias="COLLECTIVEOS_WS_URL"
    )

    # None -> session state (active tasks, entity stack) lives in-process
    # only, lost on restart. Set to use Redis (plan sec. 6); RedisSessionStore
    # is structurally complete but unverified -- no Redis instance exists in
    # this environment.
    redis_url: str | None = Field(default=None, validation_alias="REDIS_URL")

    # The session stays silent and takes no action at all until this phrase
    # is heard (case-insensitive, whole-phrase match) -- see agent.py's
    # WakeState/_after_wake_word. Once heard, the session stays awake for
    # the rest of the call; there's no re-sleep phrase.
    wake_word: str = Field(default="hey voiceos", validation_alias="WAKE_WORD")

    @model_validator(mode="after")
    def _require_at_least_one_llm_key(self) -> Settings:
        if not self.anthropic_api_key and not self.gemini_api_key:
            raise ValueError(
                "at least one of ANTHROPIC_API_KEY or GEMINI_API_KEY is required"
            )
        return self

    @property
    def resolved_llm_provider(self) -> Literal["anthropic", "gemini"]:
        if self.llm_provider:
            return self.llm_provider
        if self.gemini_api_key and not self.anthropic_api_key:
            return "gemini"
        if self.anthropic_api_key and not self.gemini_api_key:
            return "anthropic"
        return "anthropic"  # both set, no explicit choice -> designed default
