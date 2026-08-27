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
    #
    # There is no "-latest" alias for the Live API (unlike gemini_model
    # below), so this has to be a literal dated string -- checked
    # 2026-08-27 against a proposed "gemini-3.6-flash-live-001": that
    # model does not exist (real API returns a 1008 policy-violation
    # error, "not found ... or not supported for bidiGenerateContent").
    # Don't swap this without opening a real session against it first --
    # the installed plugin's mutable_instructions check is just
    # `"3.1" not in model`, not real capability detection, so it can't
    # tell you a new model is safe; only a live session can.
    gemini_live_model: str = Field(
        default="gemini-2.5-flash-native-audio-preview-12-2025",
        validation_alias="GEMINI_LIVE_MODEL",
    )

    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    # "-latest" is an alias Google keeps pointed at their current
    # recommended flash model -- deliberately not pinned to a dated
    # string like gemini_live_model above has to be, so this one never
    # needs a manual bump when a specific dated model gets deprecated.
    gemini_model: str = Field(default="gemini-flash-latest", validation_alias="GEMINI_MODEL")

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

    # None -> session state (active tasks, entity stack) AND per-user rate
    # limiting both live in-process only, lost on restart / not shared
    # across workers. Set to back both with Redis instead (RedisSessionStore,
    # RedisRateLimiter -- see agent.py's entrypoint(), which switches both
    # on this one setting together; verified live against a real Redis
    # instance, see test_redis_session_store.py / test_redis_rate_limiter.py).
    redis_url: str | None = Field(default=None, validation_alias="REDIS_URL")

    # The session stays silent and takes no action at all until this phrase
    # is heard (case-insensitive, whole-phrase match) -- see agent.py's
    # WakeState/_after_phrase. Once heard, the session stays awake until
    # sleep_word is heard.
    wake_word: str = Field(default="hey voiceos", validation_alias="WAKE_WORD")

    # Puts the session back to sleep (requires wake_word again). Default
    # deliberately includes the brand name, not just "go to sleep" -- a
    # bare generic phrase risks matching normal conversation ("I need to
    # go to sleep early tonight, can you set a reminder") and putting the
    # session to sleep mid-request. No internal punctuation, deliberately
    # -- _after_phrase matches the literal phrase, and a transcript may or
    # may not render a comma where a human would pause.
    sleep_word: str = Field(default="voiceos go to sleep", validation_alias="SLEEP_WORD")

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
