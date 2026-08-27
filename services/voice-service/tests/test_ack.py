import asyncio
from types import SimpleNamespace

from voice_service.ack import AckGenerator


class _FakeMessages:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_small_talk_matches_template_without_calling_the_model():
    fake = _FakeMessages(response=None)
    ack = AckGenerator(client=fake)

    reply = asyncio.run(ack.instant_ack("small_talk", "hey there"))

    assert reply == "Hey, what can I help with?"
    assert fake.calls == []  # template path never touches the model


def test_small_talk_falls_back_when_no_template_matches():
    ack = AckGenerator(client=_FakeMessages(response=None))

    reply = asyncio.run(ack.instant_ack("small_talk", "lovely weather today"))

    assert reply == "Got it."


def test_simple_lookup_uses_a_fast_model_call():
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="It's Tuesday.")])
    fake = _FakeMessages(response)
    ack = AckGenerator(client=fake)

    reply = asyncio.run(ack.instant_ack("simple_lookup", "what day is it"))

    assert reply == "It's Tuesday."
    assert fake.calls[0]["messages"][0]["content"] == "what day is it"


def test_instant_ack_rejects_a_router_class_it_does_not_handle():
    """new_intent/modify_inflight/etc. never reach AckGenerator in
    practice (conversation.py only calls this for small_talk/simple_lookup)
    -- this guard is what would catch a future caller getting that wrong."""
    ack = AckGenerator(client=_FakeMessages(response=None))

    try:
        asyncio.run(ack.instant_ack("new_intent", "clear my morning"))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "new_intent" in str(exc)
