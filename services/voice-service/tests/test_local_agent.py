"""LocalFallbackAgent's wrapper logic and safety framing -- tested against
a fake agentkit.Agent, no real Gemini call or even agentkit installed
needed for most of this. build_local_agent()'s "agentkit not installed"
path is exercised by actually hiding the import (see
test_build_local_agent_returns_none_when_agentkit_is_not_importable)
rather than relying on this dev environment's real install state.
"""

from __future__ import annotations

import asyncio
import sys

from voice_service.local_agent import LocalFallbackAgent, build_local_agent


class _FakeAgentkitAgent:
    def __init__(self) -> None:
        self.goals: list[str] = []
        self.closed = False

    async def run(self, goal: str) -> str:
        self.goals.append(goal)
        return "I can't do that right now, but I've noted it."

    async def aclose(self) -> None:
        self.closed = True


def test_respond_returns_the_underlying_agents_answer():
    fake_agent = _FakeAgentkitAgent()
    local_agent = LocalFallbackAgent(fake_agent)

    reply = asyncio.run(local_agent.respond("clear my morning"))

    assert reply == "I can't do that right now, but I've noted it."


def test_respond_frames_the_goal_with_the_safety_constraint_and_the_utterance():
    """The one thing that actually matters here: the underlying model
    must be told plainly it cannot execute real actions, on every call --
    not just once at construction time."""
    fake_agent = _FakeAgentkitAgent()
    local_agent = LocalFallbackAgent(fake_agent)

    asyncio.run(local_agent.respond("clear my morning, I'm sick"))

    goal = fake_agent.goals[0]
    assert "clear my morning, I'm sick" in goal
    assert "no ability" in goal.lower() or "cannot" in goal.lower()
    assert "never" in goal.lower()  # "never say or imply that you completed..."


def test_aclose_closes_the_underlying_agent():
    fake_agent = _FakeAgentkitAgent()
    local_agent = LocalFallbackAgent(fake_agent)

    asyncio.run(local_agent.aclose())

    assert fake_agent.closed is True


def test_build_local_agent_returns_none_when_agentkit_is_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "agentkit", None)

    assert build_local_agent(api_key="unused") is None
