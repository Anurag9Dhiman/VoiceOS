"""CollectiveOSClient itself had zero direct unit tests -- only exercised
indirectly through e2e tests that always connect before doing anything
else, so its defensive guards and edge-case branches were never hit."""

from __future__ import annotations

import asyncio

import pytest
import websockets.exceptions
from voice_contract import SessionEnd
from voice_service.collectiveos_client import CollectiveOSClient


def test_send_before_connect_raises():
    client = CollectiveOSClient("ws://unused")

    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(client.send(SessionEnd(session_id="s1")))


def test_receive_before_connect_raises():
    client = CollectiveOSClient("ws://unused")

    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(client.receive())


def test_close_before_ever_connecting_is_a_no_op():
    client = CollectiveOSClient("ws://unused")

    asyncio.run(client.close())  # must not raise


def test_close_swallows_a_connection_already_closed_by_the_far_end():
    """close() tries to send SessionEnd as a courtesy -- if the far end
    already dropped the socket, that send failing must not stop close()
    from finishing and releasing the connection."""

    class _AlreadyClosedWs:
        async def send(self, data):
            raise websockets.exceptions.ConnectionClosed(None, None)

        async def close(self):
            self.closed = True

    async def scenario():
        client = CollectiveOSClient("ws://unused")
        client._ws = _AlreadyClosedWs()
        client._session_id = "s1"

        await client.close()  # must not raise despite the send failing

        assert client._ws is None

    asyncio.run(scenario())
