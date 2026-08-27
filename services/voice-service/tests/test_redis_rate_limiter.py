"""Live verification for RedisRateLimiter against a real Redis instance --
same caveat as test_redis_session_store.py, now closed. Uses db 15 (redis's
conventional scratch db), flushed before/after, so this never collides with
a real voice-service instance pointed at the same Redis. Skipped, not
failed, when no Redis is reachable.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import redis as redis_sync

from voice_service.rate_limiter import RedisRateLimiter

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


def test_allows_up_to_the_limit_within_the_window():
    limiter = RedisRateLimiter.from_url(_TEST_REDIS_URL, limit=3, window_seconds=60)

    async def scenario():
        return [await limiter.allow("u1") for _ in range(3)]

    assert asyncio.run(scenario()) == [True, True, True]


def test_denies_once_the_limit_is_exceeded_within_the_window():
    limiter = RedisRateLimiter.from_url(_TEST_REDIS_URL, limit=2, window_seconds=60)

    async def scenario():
        return [await limiter.allow("u1") for _ in range(3)]

    assert asyncio.run(scenario()) == [True, True, False]


def test_buckets_are_independent_per_key():
    limiter = RedisRateLimiter.from_url(_TEST_REDIS_URL, limit=1, window_seconds=60)

    async def scenario():
        u1_first = await limiter.allow("u1")
        u1_second = await limiter.allow("u1")
        u2_first = await limiter.allow("u2")
        return u1_first, u1_second, u2_first

    assert asyncio.run(scenario()) == (True, False, True)


def test_window_expires_and_the_count_resets():
    """The actual behavior only a real Redis EXPIRE proves: past the window,
    the fixed-window counter really does reset, via TTL, not just a mocked
    clock. 1-second window, so this test costs ~1s of real wall time."""
    limiter = RedisRateLimiter.from_url(_TEST_REDIS_URL, limit=1, window_seconds=1)

    async def scenario():
        first = await limiter.allow("u1")
        immediately_after = await limiter.allow("u1")
        time.sleep(1.2)
        after_window_expires = await limiter.allow("u1")
        return first, immediately_after, after_window_expires

    assert asyncio.run(scenario()) == (True, False, True)


def test_a_second_limiter_instance_shares_state_with_the_first():
    """The actual point of Redis over InMemoryRateLimiter: the count is
    shared across processes/workers, not per-process. Two independent
    client connections stand in for that here."""
    first = RedisRateLimiter.from_url(_TEST_REDIS_URL, limit=1, window_seconds=60)
    second = RedisRateLimiter.from_url(_TEST_REDIS_URL, limit=1, window_seconds=60)

    async def scenario():
        return await first.allow("u1"), await second.allow("u1")

    assert asyncio.run(scenario()) == (True, False)
