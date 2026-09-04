"""Local fallback for when CollectiveOS is unreachable -- see
conversation.py's module docstring for when this actually engages
(never when CollectiveOS is reachable; this is "alongside", not a
replacement for, the frozen contract).

Built on agentkit (a separate framework, `../../../agentkit` from this
file, an optional `local-agent` extra -- see pyproject.toml) rather than
a second reasoning loop hand-rolled here. The import is lazy, inside
build_local_agent(), specifically so importing this module -- and
everything that imports it (conversation.py, agent.py) -- never fails
just because the extra isn't installed; build_local_agent() returns None
in that case instead.

Safety constraint, not optional: voice-service holds zero connector
credentials by design (the project's blast-radius rule). This agent
cannot execute a real task no matter what it's asked, so every call is
framed to make that explicit -- the underlying model must never
free-text-claim to have "done" something it structurally cannot do.
Confirmed live (not just by reading this prompt) that the framing holds;
see tests/test_local_agent.py and the README.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("voice_service.local_agent")

_GOAL_TEMPLATE = (
    "The main system (CollectiveOS) is currently unreachable. You have NO "
    "ability to execute real tasks, connectors, or device actions right "
    "now -- none at all, regardless of what's asked. You may only: answer "
    "from your own knowledge or this conversation's memory, or acknowledge "
    "the request so it can be handled once the connection is restored. "
    "Never say or imply that you completed, started, or scheduled a real "
    "action -- that would be false. Keep it to one short breath, speakable "
    "aloud.\n\nUser said: {utterance!r}"
)


class LocalFallbackAgent:
    """Narrow wrapper around an agentkit.Agent -- the only two things
    voice-service needs from it."""

    def __init__(self, agent: object) -> None:
        self._agent = agent

    async def respond(self, utterance: str) -> str:
        return await self._agent.run(_GOAL_TEMPLATE.format(utterance=utterance))  # type: ignore[attr-defined]

    async def aclose(self) -> None:
        await self._agent.aclose()  # type: ignore[attr-defined]


def build_local_agent(*, api_key: str, db_path: str = "voice_local_agent.db") -> LocalFallbackAgent | None:
    """None if the `local-agent` extra isn't installed -- the one place
    the agentkit import is attempted."""
    try:
        import agentkit
        from agentkit.reasoning_backends.gemini import GeminiReasoner
    except ImportError:
        logger.info("agentkit not installed -- no local fallback agent (uv sync --extra local-agent to enable)")
        return None

    agent = agentkit.Agent(
        memory=agentkit.SQLiteMemory(db_path),
        reasoning=GeminiReasoner(api_key=api_key),
        reflect=False,  # a degraded-mode fallback has nothing to self-improve toward
    )
    return LocalFallbackAgent(agent)
