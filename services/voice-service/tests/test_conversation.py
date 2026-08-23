import asyncio

import pytest
from voice_contract import (
    Ack,
    ConfirmationRequest,
    Done,
    Interrupt,
    Progress,
    SessionQuery,
    TaskUpdate,
    UserUtterance,
)
from voice_service.conversation import ConversationController
from voice_service.session_store import InMemorySessionStore


class FakeClient:
    """Stands in for CollectiveOSClient: records what's sent, and lets a
    test push agent_to_voice events for the controller's receive loop to
    consume, without touching a real socket."""

    def __init__(self) -> None:
        self.sent: list[object] = []
        self.connected = False
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def connect(self, *, session_id, user_id, resume=False):
        self.connected = True
        self.session_id = session_id

    async def send(self, event):
        self.sent.append(event)

    def push(self, event) -> None:
        self._queue.put_nowait(event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._queue.get()

    async def close(self):
        self.closed = True


class FakeRouter:
    def __init__(self, router_class: str) -> None:
        self.router_class = router_class

    async def classify(self, text, *, has_active_task=False):
        return self.router_class


class FakeAck:
    async def instant_ack(self, router_class, text):
        return f"ack:{router_class}"


async def _settle() -> None:
    """Let the controller's background receive-loop task process whatever
    was just pushed onto the fake client's queue."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _controller(client: FakeClient, router_class: str | None = None):
    speak_calls: list[tuple[str, str]] = []

    async def speak(text, priority):
        speak_calls.append((text, priority))

    controller = ConversationController(
        client=client,
        speak=speak,
        router=FakeRouter(router_class) if router_class else None,
        ack=FakeAck(),
    )
    return controller, speak_calls


def test_small_talk_is_proposed_then_spoken_locally_on_approval():
    async def scenario():
        client = FakeClient()
        controller, speak_calls = _controller(client, "small_talk")
        await controller.start(session_id="s1", user_id="u1")

        await controller.handle_utterance("hey")
        assert speak_calls[-1] == ("I'm ready to reply. OK to proceed?", "high")
        assert client.sent == []

        await controller.handle_utterance("yes go ahead")
        assert speak_calls[-1] == ("ack:small_talk", "low")
        assert client.sent == []
        await controller.stop()

    asyncio.run(scenario())


def test_small_talk_proposal_is_dropped_on_rejection():
    async def scenario():
        client = FakeClient()
        controller, speak_calls = _controller(client, "small_talk")
        await controller.start(session_id="s1", user_id="u1")

        await controller.handle_utterance("hey")
        await controller.handle_utterance("no, don't")

        assert speak_calls[-1] == ("Okay, cancelled.", "low")
        assert client.sent == []
        await controller.stop()

    asyncio.run(scenario())


def test_new_intent_forwards_as_user_utterance_once_approved():
    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client, "new_intent")
        await controller.start(session_id="s1", user_id="u1")

        await controller.handle_utterance("clear my morning, I'm sick")
        assert client.sent == []  # proposed, not sent yet

        await controller.handle_utterance("yes go ahead")

        assert len(client.sent) == 1
        sent = client.sent[0]
        assert isinstance(sent, UserUtterance)
        assert sent.session_id == "s1"
        assert sent.router_class == "new_intent"
        assert sent.text == "clear my morning, I'm sick"
        await controller.stop()

    asyncio.run(scenario())


def test_ack_event_sets_current_task_and_speaks_it():
    async def scenario():
        client = FakeClient()
        controller, speak_calls = _controller(client)
        await controller.start(session_id="s1", user_id="u1")

        client.push(Ack(task_id="t1", text="Checking your calendar now."))
        await _settle()

        assert controller.current_task_id == "t1"
        assert speak_calls == [("Checking your calendar now.", "low")]
        await controller.stop()

    asyncio.run(scenario())


def test_modify_inflight_sends_interrupt_targeting_current_task():
    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client, "modify_inflight")
        await controller.start(session_id="s1", user_id="u1")
        client.push(Ack(task_id="t1", text="Checking your calendar now."))
        await _settle()

        await controller.handle_utterance("wait, keep the 10am")
        assert client.sent == []  # proposed, not sent yet

        await controller.handle_utterance("yes go ahead")

        interrupt = client.sent[-1]
        assert isinstance(interrupt, Interrupt)
        assert interrupt.target_task_id == "t1"
        assert interrupt.text == "wait, keep the 10am"
        await controller.stop()

    asyncio.run(scenario())


def test_modify_inflight_targets_the_task_current_at_proposal_time_not_approval_time():
    """The target task is captured when the action is proposed, so it can't
    drift if current_task_id changes while the proposal is awaiting the
    user's approval."""

    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client, "modify_inflight")
        await controller.start(session_id="s1", user_id="u1")
        client.push(Ack(task_id="t1", text="Checking your calendar now."))
        await _settle()

        await controller.handle_utterance("wait, keep the 10am")  # proposed while t1 is current

        # A second task becomes current before the user answers.
        client.push(Ack(task_id="t2", text="Starting a different task."))
        await _settle()
        assert controller.current_task_id == "t2"

        await controller.handle_utterance("yes go ahead")

        interrupt = client.sent[-1]
        assert isinstance(interrupt, Interrupt)
        assert interrupt.target_task_id == "t1"  # still t1, not the now-current t2
        await controller.stop()

    asyncio.run(scenario())


def test_modify_inflight_with_no_active_task_is_dropped_not_crashed():
    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client, "modify_inflight")
        await controller.start(session_id="s1", user_id="u1")

        await controller.handle_utterance("wait, keep the 10am")  # no active task yet

        assert client.sent == []
        await controller.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "text,expected_decision,expected_modification",
    [
        ("yes, go ahead", "approve", None),
        ("no, don't do that", "reject", None),
        ("actually just cancel the 9am", "modify", "actually just cancel the 9am"),
    ],
)
def test_confirmation_reply_maps_to_decision(text, expected_decision, expected_modification):
    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client, "confirmation_reply")
        await controller.start(session_id="s1", user_id="u1")
        client.push(ConfirmationRequest(task_id="t1", speak="Shall I go ahead?", options=["approve", "reject", "modify"], risk_class="write"))
        await _settle()

        await controller.handle_utterance(text)

        response = client.sent[-1]
        assert response.task_id == "t1"
        assert response.decision == expected_decision
        assert response.modification == expected_modification
        await controller.stop()

    asyncio.run(scenario())


def test_session_query_forwards_the_query_once_approved():
    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client, "session_query")
        await controller.start(session_id="s1", user_id="u1")

        await controller.handle_utterance("where were we")
        assert client.sent == []  # proposed, not sent yet

        await controller.handle_utterance("yes go ahead")

        sent = client.sent[-1]
        assert isinstance(sent, SessionQuery)
        assert sent.query == "where were we"
        await controller.stop()

    asyncio.run(scenario())


def test_done_clears_active_task_and_speaks_summary_at_high_priority():
    async def scenario():
        client = FakeClient()
        controller, speak_calls = _controller(client)
        await controller.start(session_id="s1", user_id="u1")
        client.push(Ack(task_id="t1", text="Checking your calendar now."))
        await _settle()

        client.push(Done(task_id="t1", outcome="completed", summary_speak="All done."))
        await _settle()

        assert controller.current_task_id is None
        assert ("All done.", "high") in speak_calls
        await controller.stop()

    asyncio.run(scenario())


def test_multiple_active_tasks_are_all_tracked():
    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client)
        await controller.start(session_id="s1", user_id="u1")

        client.push(Ack(task_id="t1", text="Working on the first thing."))
        await _settle()
        client.push(Ack(task_id="t2", text="Working on the second thing."))
        await _settle()

        assert controller.active_task_ids == ["t1", "t2"]
        await controller.stop()

    asyncio.run(scenario())


def test_current_task_prefers_the_one_waiting_on_the_user():
    """With two tasks in flight, an unqualified follow-up should target
    whichever one is actually waiting on the user, not just whichever was
    touched most recently."""

    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client)
        await controller.start(session_id="s1", user_id="u1")

        client.push(Ack(task_id="t1", text="Starting task one."))
        await _settle()
        client.push(
            ConfirmationRequest(
                task_id="t1", speak="Shall I go ahead?", options=["approve", "reject", "modify"], risk_class="write"
            )
        )
        await _settle()
        # t2 becomes the most recently *touched* task, but it isn't the one
        # waiting on the user -- t1 is.
        client.push(Progress(task_id="t2", text="Still working on task two."))
        await _settle()

        assert controller.current_task_id == "t1"
        assert controller.waiting_reason == "user_confirm"
        await controller.stop()

    asyncio.run(scenario())


def test_new_intent_attaches_resolved_pronouns_as_entity_refs():
    async def scenario():
        client = FakeClient()
        controller, _ = _controller(client, "new_intent")
        await controller.start(session_id="s1", user_id="u1")

        client.push(Progress(task_id="t1", text="Found the 10am and the 3pm on your calendar."))
        await _settle()

        await controller.handle_utterance("actually, keep it")
        await controller.handle_utterance("yes go ahead")

        sent = client.sent[-1]
        assert isinstance(sent, UserUtterance)
        assert sent.entity_refs  # resolved against the most recent mention (3pm)

    asyncio.run(scenario())


def test_session_snapshot_persists_and_restores_on_resume():
    async def scenario():
        store = InMemorySessionStore()
        client = FakeClient()
        controller, _ = _controller(client)
        controller._session_store = store  # same store a real resumed session would share
        await controller.start(session_id="s1", user_id="u1")

        client.push(Ack(task_id="t1", text="Working on it."))
        await _settle()
        await controller.stop()

        snapshot = await store.load("u1")
        assert snapshot is not None
        assert snapshot.active_task_ids == ["t1"]
        assert snapshot.ended_at is not None

        client2 = FakeClient()
        controller2, _ = _controller(client2)
        controller2._session_store = store
        await controller2.start(session_id="s2", user_id="u1", resume=True)

        assert controller2.active_task_ids == ["t1"]
        await controller2.stop()

    asyncio.run(scenario())


def test_on_task_state_changed_fires_when_an_agent_event_touches_a_task():
    """A caller with no per-call has_active_task signal of its own (a live
    Gemini session, unlike HaikuRouter.classify) needs some other way to
    learn the state changed -- this is that hook."""

    async def scenario():
        client = FakeClient()
        calls = []

        async def on_task_state_changed():
            calls.append(True)

        controller = ConversationController(
            client=client,
            speak=lambda text, priority: asyncio.sleep(0),
            ack=FakeAck(),
            on_task_state_changed=on_task_state_changed,
        )
        await controller.start(session_id="s1", user_id="u1")

        client.push(Ack(task_id="t1", text="Checking your calendar now."))
        await _settle()

        assert len(calls) == 1
        await controller.stop()

    asyncio.run(scenario())


def test_on_task_state_changed_fires_on_resume_with_restored_tasks():
    async def scenario():
        store = InMemorySessionStore()
        client = FakeClient()
        controller, _ = _controller(client)
        controller._session_store = store
        await controller.start(session_id="s1", user_id="u1")
        client.push(Ack(task_id="t1", text="Working on it."))
        await _settle()
        await controller.stop()

        client2 = FakeClient()
        calls = []

        async def on_task_state_changed():
            calls.append(True)

        controller2 = ConversationController(
            client=client2,
            speak=lambda text, priority: asyncio.sleep(0),
            ack=FakeAck(),
            session_store=store,
            on_task_state_changed=on_task_state_changed,
        )
        await controller2.start(session_id="s2", user_id="u1", resume=True)

        assert calls == [True]
        await controller2.stop()

    asyncio.run(scenario())
