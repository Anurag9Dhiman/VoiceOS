"""Live verification for RedisSessionStore against a real Redis instance --
closes the "structurally complete but unverified live" caveat repeated in
session_store.py/config.py/README.md for most of this project's history.
Uses db 15 (redis's conventional scratch db) rather than the production
default (db 0), and flushes it before/after, so this never collides with a
real voice-service instance pointed at the same Redis. Skipped, not failed,
when no Redis is reachable -- unlike mock-agent-backend (runs in-process),
Redis is a real external dependency this environment may or may not have.
"""

from __future__ import annotations

import asyncio

import pytest
import redis as redis_sync

from voice_service.session_store import RedisSessionStore, SessionSnapshot

_TEST_REDIS_URL = "redis://localhost:6379/15"


def _redis_reachable() -> bool:
    try:
        redis_sync.from_url(_TEST_REDIS_URL, socket_connect_timeout=0.5).ping()
        return True
    except redis_sync.RedisError:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_reachable(), reason="no Redis reachable at localhost:6379"
)


@pytest.fixture(autouse=True)
def _clean_test_db():
    redis_sync.from_url(_TEST_REDIS_URL).flushdb()
    yield
    redis_sync.from_url(_TEST_REDIS_URL).flushdb()


def test_load_missing_user_returns_none():
    store = RedisSessionStore.from_url(_TEST_REDIS_URL)

    result = asyncio.run(store.load("nobody"))

    assert result is None


def test_save_then_load_round_trips_through_real_redis():
    store = RedisSessionStore.from_url(_TEST_REDIS_URL)
    snapshot = SessionSnapshot(
        user_id="u1",
        active_task_ids=["t1", "t2"],
        entity_stack=[{"label": "Acme Gym", "entity_id": "mention_acme_gym"}],
    )

    async def scenario():
        await store.save(snapshot)
        return await store.load("u1")

    loaded = asyncio.run(scenario())

    assert loaded == snapshot


def test_save_overwrites_previous_snapshot_for_same_user():
    store = RedisSessionStore.from_url(_TEST_REDIS_URL)

    async def scenario():
        await store.save(SessionSnapshot(user_id="u1", active_task_ids=["t1"]))
        await store.save(SessionSnapshot(user_id="u1", active_task_ids=["t2"]))
        return await store.load("u1")

    loaded = asyncio.run(scenario())

    assert loaded.active_task_ids == ["t2"]


def test_a_second_store_instance_sees_what_the_first_one_saved():
    """The actual point of Redis over InMemorySessionStore: state survives a
    process restart / is shared across workers. Two independent client
    connections stand in for that here."""
    writer = RedisSessionStore.from_url(_TEST_REDIS_URL)
    reader = RedisSessionStore.from_url(_TEST_REDIS_URL)

    async def scenario():
        await writer.save(SessionSnapshot(user_id="u1", active_task_ids=["t1"]))
        return await reader.load("u1")

    loaded = asyncio.run(scenario())

    assert loaded.active_task_ids == ["t1"]


def test_ended_at_none_round_trips_correctly():
    """json.dumps/loads round-tripping None specifically -- easy to get
    wrong (e.g. by accidentally coercing to a string or dropping the key)."""
    store = RedisSessionStore.from_url(_TEST_REDIS_URL)
    snapshot = SessionSnapshot(user_id="u1", ended_at=None)

    async def scenario():
        await store.save(snapshot)
        return await store.load("u1")

    loaded = asyncio.run(scenario())

    assert loaded.ended_at is None
