"""The Haiku router: classifies each finalized utterance into one of the
six categories from the contract (voice-platform-mvp-plan-v2.md sec. 1 and
../../../contract/README.md). This is the *only* place an LLM call decides
control flow in the voice layer -- everything downstream of a classification
is a lookup table (ack.py) or a protocol-following forward (conversation.py).

Uses a forced tool call rather than free-text + parsing: the model can only
return one of the six literal strings, so there's no classification output
to parse or validate beyond "did the field come back at all."
"""

from __future__ import annotations

from typing import Protocol

from anthropic import AsyncAnthropic
from langsmith import traceable
from voice_contract import RouterClass

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_ROUTER_CLASSES: tuple[RouterClass, ...] = (
    "small_talk",
    "simple_lookup",
    "new_intent",
    "modify_inflight",
    "confirmation_reply",
    "session_query",
)

_SYSTEM_PROMPT = """You are the router for a voice assistant's turn-taking layer. \
Classify the user's utterance into exactly one category:

- small_talk: greetings, thanks, chit-chat with no task content ("hey", "thanks", "how are you").
- simple_lookup: a question answerable from context already in hand, no new task ("what did you just say").
- new_intent: the user wants something done that isn't already in flight.
- modify_inflight: the user is editing or interrupting a task that's currently running or awaiting confirmation \
("wait, keep the 10am", "actually make it 3pm instead").
- confirmation_reply: a direct yes/no/edit reply to a pending confirmation question.
- session_query: the user is asking about session/task status, not asking for new work \
("where were we", "what's the status", "did that finish").

Call classify_utterance exactly once with your answer."""

_CLASSIFY_TOOL = {
    "name": "classify_utterance",
    "description": "Record the router classification for the utterance.",
    "input_schema": {
        "type": "object",
        "properties": {
            "router_class": {
                "type": "string",
                "enum": list(_ROUTER_CLASSES),
            }
        },
        "required": ["router_class"],
    },
}


class MessagesClient(Protocol):
    """The one method router.py needs from an Anthropic client -- narrow on
    purpose so tests can inject a fake without depending on the real SDK's
    response types."""

    async def create(self, **kwargs: object) -> object: ...


class RouterError(RuntimeError):
    """Raised when the model doesn't return a usable classification (no
    tool call, or an isolated/unparseable one) -- callers decide the
    fallback (e.g. treat as new_intent, or surface a clarification)."""


class HaikuRouter:
    def __init__(self, client: MessagesClient | None = None, model: str = DEFAULT_MODEL) -> None:
        self._client = client or AsyncAnthropic().messages
        self._model = model

    @traceable(name="router_classify", run_type="llm")
    async def classify(self, text: str, *, has_active_task: bool = False) -> RouterClass:
        context = (
            "There is currently an in-flight task awaiting the user's input."
            if has_active_task
            else "There is no in-flight task right now."
        )
        response = await self._client.create(
            model=self._model,
            max_tokens=64,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{context}\n\nUtterance: {text!r}"}],
            tools=[_CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_utterance"},
        )

        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "tool_use":
                router_class = getattr(block, "input", {}).get("router_class")
                if router_class in _ROUTER_CLASSES:
                    return router_class  # type: ignore[return-value]

        raise RouterError(f"no usable classify_utterance tool call in response: {response!r}")
