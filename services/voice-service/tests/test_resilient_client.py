import asyncio

import pytest
from voice_contract import Ack, Done, Progress, SessionQuery
from voice_service.resilient_client import ReconnectFailed, ReconnectingCollectiveOSClient


class _Fixture:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.fail_first_n_connects = 0
        self.instances: list["_FakeRawClient"] = []


class _FakeRawClient:
    """Stands in for CollectiveOSClient: `connect()` can be told to fail
    the first N times (simulating a down/unreachable server), and its
    event stream either ends naturally (simulating a drop) or hangs until
    `close()` is called (simulating a live, idle connection)."""

    def __init__(self, fixture: _Fixture, batch_events: list, hang: bool = False) -> None:
        self.fixture = fixture
        self.batch_events = batch_events
        self.hang = hang
        self.sent: list = []
        self.resume: bool | None = None
        self._closed_event = asyncio.Event()

    async def connect(self, *, session_id, user_id, resume=False):
        self.fixture.connect_calls += 1
        if self.fixture.connect_calls <= self.fixture.fail_first_n_connects:
            raise ConnectionRefusedError("simulated: server unreachable")
        self.resume = resume

    async def send(self, event) -> None:
        self.sent.append(event)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for event in self.batch_events:
            yield event
        if self.hang:
            await self._closed_event.wait()

    async def close(self) -> None:
        self._closed_event.set()


def _factory(fixture: _Fixture, batches: list, hang_last: bool = False):
    state = {"i": 0}

    def make(url: str) -> _FakeRawClient:
        i = state["i"]
        state["i"] += 1
        hang = hang_last and i == len(batches) - 1
        client = _FakeRawClient(fixture, batches[i], hang=hang)
        fixture.instances.append(client)
        return client

    return make


def test_connect_succeeds_on_first_try():
    async def scenario():
        fixture = _Fixture()
        client = ReconnectingCollectiveOSClient(
            "ws://x", client_factory=_factory(fixture, [[]]), base_delay=0.001
        )
        await client.connect(session_id="s1", user_id="u1")

        assert client.state == "connected"
        assert fixture.connect_calls == 1

    asyncio.run(scenario())


def test_connect_retries_then_succeeds():
    async def scenario():
        fixture = _Fixture()
        fixture.fail_first_n_connects = 2
        client = ReconnectingCollectiveOSClient(
            "ws://x",
            client_factory=_factory(fixture, [[], [], []]),
            base_delay=0.001,
            max_retries=5,
        )
        await client.connect(session_id="s1", user_id="u1")

        assert client.state == "connected"
        assert fixture.connect_calls == 3

    asyncio.run(scenario())


def test_connect_gives_up_after_max_retries():
    async def scenario():
        fixture = _Fixture()
        fixture.fail_first_n_connects = 100
        client = ReconnectingCollectiveOSClient(
            "ws://x",
            client_factory=_factory(fixture, [[]] * 10),
            base_delay=0.001,
            max_retries=3,
        )
        with pytest.raises(ReconnectFailed):
            await client.connect(session_id="s1", user_id="u1")

        assert client.state == "failed"
        assert fixture.connect_calls == 4  # initial attempt + 3 retries

    asyncio.run(scenario())


def test_receive_loop_reconnects_after_drop_and_resumes():
    async def scenario():
        fixture = _Fixture()
        batch1 = [Ack(task_id="t1", text="hello")]
        batch2 = [
            Progress(task_id="t1", text="still going"),
            Done(task_id="t1", outcome="completed", summary_speak="done"),
        ]
        client = ReconnectingCollectiveOSClient(
            "ws://x", client_factory=_factory(fixture, [batch1, batch2]), base_delay=0.001
        )
        await client.connect(session_id="s1", user_id="u1", resume=False)

        received = []
        async for event in client:
            received.append(event)
            if len(received) == 3:
                break

        assert [type(e).__name__ for e in received] == ["Ack", "Progress", "Done"]
        # first connection was a fresh session; the reconnect after the
        # drop must resume the same one.
        assert fixture.instances[0].resume is False
        assert fixture.instances[1].resume is True

    asyncio.run(scenario())


def test_clean_close_does_not_trigger_a_reconnect():
    async def scenario():
        fixture = _Fixture()
        batch1 = [Ack(task_id="t1", text="hi")]
        client = ReconnectingCollectiveOSClient(
            "ws://x", client_factory=_factory(fixture, [batch1], hang_last=True), base_delay=0.001
        )
        await client.connect(session_id="s1", user_id="u1")

        received = []

        async def consume():
            async for event in client:
                received.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)  # let it receive the one event and start hanging
        await client.close()
        await asyncio.wait_for(task, timeout=1)

        assert client.state == "idle"
        assert fixture.connect_calls == 1  # no reconnect attempted
        assert len(received) == 1

    asyncio.run(scenario())


def test_send_failures_are_not_retried():
    async def scenario():
        fixture = _Fixture()
        client = ReconnectingCollectiveOSClient(
            "ws://x", client_factory=_factory(fixture, [[]]), base_delay=0.001
        )
        await client.connect(session_id="s1", user_id="u1")

        async def failing_send(event):
            raise ConnectionResetError("simulated mid-send drop")

        client._client.send = failing_send  # type: ignore[method-assign]

        with pytest.raises(ConnectionResetError):
            await client.send(SessionQuery(session_id="s1", query="won't arrive"))

        assert fixture.connect_calls == 1  # send() never triggers a reconnect

    asyncio.run(scenario())


def test_send_before_connect_raises():
    async def scenario():
        client = ReconnectingCollectiveOSClient("ws://x", base_delay=0.001)

        with pytest.raises(RuntimeError, match="not connected"):
            await client.send(SessionQuery(session_id="s1", query="anything"))

    asyncio.run(scenario())


def test_receive_loop_gives_up_when_every_reconnect_attempt_after_a_drop_fails():
    """test_connect_gives_up_after_max_retries covers giving up on the
    very first connect() -- this is the other place ReconnectFailed can
    happen: a drop mid-stream where every subsequent reconnect attempt
    also fails. The receive loop must end cleanly (not raise) rather than
    retrying forever."""

    async def scenario():
        fixture = _Fixture()
        fixture_first_batch = [Ack(task_id="t1", text="hello")]
        # One batch for the initial successful connect, plus enough more
        # for every reconnect attempt the factory will be asked to build
        # a client for (even though each of those connect() calls fails).
        client = ReconnectingCollectiveOSClient(
            "ws://x",
            client_factory=_factory(fixture, [fixture_first_batch, [], [], []]),
            base_delay=0.001,
            max_retries=2,
        )
        await client.connect(session_id="s1", user_id="u1", resume=False)

        # Every reconnect attempt after the initial drop fails.
        fixture.fail_first_n_connects = 100

        received = []
        async for event in client:
            received.append(event)

        assert [type(e).__name__ for e in received] == ["Ack"]
        assert client.state == "failed"

    asyncio.run(scenario())
