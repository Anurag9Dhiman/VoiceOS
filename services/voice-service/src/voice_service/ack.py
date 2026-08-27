"""Ack generator (plan sec. 4a): handles `small_talk` and `simple_lookup`
entirely locally -- these never cross the wire to CollectiveOS. small_talk
is template-only (cheap, deterministic); simple_lookup gets a single fast
Haiku call since answering "what did you just say" actually requires
looking at something, not just picking a canned line.
"""

from __future__ import annotations

import re

from anthropic import AsyncAnthropic
from langsmith import traceable
from voice_contract import RouterClass

from .router import DEFAULT_MODEL, MessagesClient

_SMALL_TALK_TEMPLATES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(thanks|thank you|appreciate it)\b", re.I), "You're welcome."),
    (re.compile(r"\b(hi|hello|hey)\b", re.I), "Hey, what can I help with?"),
    (re.compile(r"\bhow are you\b", re.I), "Doing well, thanks. What do you need?"),
    (re.compile(r"\b(bye|goodbye|see ya)\b", re.I), "Talk soon."),
)
_SMALL_TALK_FALLBACK = "Got it."

_SIMPLE_LOOKUP_SYSTEM = (
    "You are a voice assistant answering a quick factual question from the "
    "conversation so far. Answer in one short breath -- a sentence or less, "
    "no lists, nothing that takes more than a couple seconds to say aloud."
)


class AckGenerator:
    def __init__(self, client: MessagesClient | None = None, model: str = DEFAULT_MODEL) -> None:
        self._client = client or AsyncAnthropic().messages
        self._model = model

    async def instant_ack(self, router_class: RouterClass, text: str) -> str:
        if router_class == "small_talk":
            return self._small_talk_template(text)
        if router_class == "simple_lookup":
            return await self._simple_lookup_answer(text)
        raise ValueError(f"instant_ack only handles small_talk/simple_lookup, got {router_class!r}")

    def _small_talk_template(self, text: str) -> str:
        for pattern, reply in _SMALL_TALK_TEMPLATES:
            if pattern.search(text):
                return reply
        return _SMALL_TALK_FALLBACK

    @traceable(name="simple_lookup_answer", run_type="llm")
    async def _simple_lookup_answer(self, text: str) -> str:
        response = await self._client.create(
            model=self._model,
            max_tokens=64,
            system=_SIMPLE_LOOKUP_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts).strip() or "Not sure, sorry."
