"""Per-user rate limiting (plan sec. 8; flagged as a gap in this project's
own architecture blueprint: "no per-user throttling on router calls exists
yet"). Token bucket, one bucket per user_id: a burst up to `capacity` is
allowed, refilling at `refill_rate` tokens/sec after that. Every routed
utterance costs one token, so a user hammering the router -- or a bug
looping utterances -- degrades to "please slow down" instead of an
unbounded Anthropic bill or a runaway loop.

Same shape as session_store.py on purpose: an in-memory default (fully
tested, used unless REDIS_URL is set) and a Redis-backed variant for
production, verified live against a real Redis instance -- see
tests/test_redis_rate_limiter.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol


class RateLimiter(Protocol):
    async def allow(self, key: str) -> bool: ...


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class InMemoryRateLimiter:
    def __init__(
        self,
        *,
        capacity: int = 20,
        refill_rate: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    async def allow(self, key: str) -> bool:
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(self._capacity), last_refill=now)
            self._buckets[key] = bucket

        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
        bucket.last_refill = now

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return True
        return False


class RedisRateLimiter:
    """Fixed-window counter over Redis (INCR + EXPIRE) rather than a true
    token bucket -- atomic without a Lua script, and window-edge burst
    slop is an acceptable tradeoff for what this protects against (a
    runaway loop, not a precise API quota).
    """

    def __init__(
        self,
        redis_client: object,
        *,
        limit: int = 20,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit:",
    ) -> None:
        self._redis = redis_client
        self._limit = limit
        self._window_seconds = window_seconds
        self._key_prefix = key_prefix

    @classmethod
    def from_url(cls, url: str, **kwargs: object) -> RedisRateLimiter:
        import redis.asyncio as redis

        return cls(redis.from_url(url), **kwargs)

    async def allow(self, key: str) -> bool:
        redis_key = f"{self._key_prefix}{key}"
        count = await self._redis.incr(redis_key)  # type: ignore[attr-defined]
        if count == 1:
            await self._redis.expire(redis_key, self._window_seconds)  # type: ignore[attr-defined]
        return count <= self._limit
