"""_build_stores is the entire reason REDIS_URL exists: it's what
entrypoint() calls to decide in-memory vs Redis for both the session store
and the rate limiter together. RedisRateLimiter was once fully implemented
but never actually wired in here -- only the session store was -- so this
regression is exactly what a test on this function is for.
"""

from __future__ import annotations

from voice_service.agent import _build_stores
from voice_service.config import Settings
from voice_service.rate_limiter import InMemoryRateLimiter, RedisRateLimiter
from voice_service.session_store import InMemorySessionStore, RedisSessionStore


def _settings(*, redis_url: str | None) -> Settings:
    return Settings(
        _env_file=None,
        GOOGLE_API_KEY="gg_key",
        GEMINI_API_KEY="gm_key",
        REDIS_URL=redis_url,
    )


def test_no_redis_url_builds_in_memory_stores():
    session_store, rate_limiter = _build_stores(_settings(redis_url=None))

    assert isinstance(session_store, InMemorySessionStore)
    assert isinstance(rate_limiter, InMemoryRateLimiter)


def test_redis_url_builds_both_stores_as_redis_backed():
    """The regression this test exists for: REDIS_URL must switch BOTH the
    session store AND the rate limiter, not just one of them."""
    session_store, rate_limiter = _build_stores(
        _settings(redis_url="redis://localhost:6379/0")
    )

    assert isinstance(session_store, RedisSessionStore)
    assert isinstance(rate_limiter, RedisRateLimiter)
