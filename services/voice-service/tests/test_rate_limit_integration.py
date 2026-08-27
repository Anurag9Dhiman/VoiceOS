"""Proves the rate limiter is actually wired into ConversationController,
not just correct in isolation (test_rate_limiter.py) -- a flood of
utterances from one user gets throttled without ever reaching the router
or CollectiveOS.
"""

import asyncio

from voice_contract import UserUtterance
from voice_service.conversation import ConversationController
from voice_service.rate_limiter import InMemoryRateLimiter

from .test_conversation import FakeClient


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CountingRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def classify(self, text, *, has_active_task=False):
        self.calls += 1
        return "new_intent"


def test_flooding_one_user_gets_throttled_before_reaching_the_router():
    async def scenario():
        client = FakeClient()
        router = CountingRouter()
        speak_calls: list[tuple[str, str]] = []

        async def speak(text, priority):
            speak_calls.append((text, priority))

        limiter = InMemoryRateLimiter(capacity=2, refill_rate=0.0, clock=_FakeClock())
        controller = ConversationController(
            client=client, speak=speak, router=router, rate_limiter=limiter
        )
        await controller.start(session_id="s1", user_id="flooder")

        for i in range(5):
            await controller.handle_utterance(f"utterance {i}")

        # Only the first 2 (the bucket's capacity) actually reached the
        # router and got forwarded; the rest were throttled.
        assert router.calls == 2
        assert sum(1 for e in client.sent if isinstance(e, UserUtterance)) == 2
        assert speak_calls.count(("Let's slow down a moment.", "low")) == 3

        await controller.stop()

    asyncio.run(scenario())


def test_different_users_have_independent_limits():
    async def scenario():
        client = FakeClient()
        router = CountingRouter()

        async def speak(text, priority):
            pass

        limiter = InMemoryRateLimiter(capacity=1, refill_rate=0.0, clock=_FakeClock())
        controller = ConversationController(
            client=client, speak=speak, router=router, rate_limiter=limiter
        )

        await controller.start(session_id="s1", user_id="alice")
        await controller.handle_utterance("hi")
        await controller.handle_utterance("hi again")  # throttled for alice
        await controller.stop()

        client2 = FakeClient()
        controller2 = ConversationController(
            client=client2, speak=speak, router=router, rate_limiter=limiter
        )
        await controller2.start(session_id="s2", user_id="bob")
        await controller2.handle_utterance("hi from bob")  # bob has his own bucket
        await controller2.stop()

        assert router.calls == 2  # alice's first + bob's first, not alice's second

    asyncio.run(scenario())
