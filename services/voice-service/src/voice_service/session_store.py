"""Session store (plan sec. 6): the voice_sessions bookkeeping -- active
task ids and the entity stack snapshot -- keyed by user_id so a resumed
call (new session_id, same user, hours or days later) can restore what a
prior session left off with. In-memory by default (tests, local dev);
Redis in production, behind the same interface so nothing above this layer
cares which one is backing it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SessionSnapshot:
    user_id: str
    active_task_ids: list[str] = field(default_factory=list)
    entity_stack: list[dict[str, str]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None


class SessionStore(Protocol):
    async def save(self, snapshot: SessionSnapshot) -> None: ...
    async def load(self, user_id: str) -> SessionSnapshot | None: ...


class InMemorySessionStore:
    """Sub-ms reads, matching the plan's rationale for Redis (session state
    doesn't belong in Postgres-speed storage) -- this is that same shape
    without a Redis dependency, for tests and single-process local dev."""

    def __init__(self) -> None:
        self._snapshots: dict[str, SessionSnapshot] = {}

    async def save(self, snapshot: SessionSnapshot) -> None:
        self._snapshots[snapshot.user_id] = snapshot

    async def load(self, user_id: str) -> SessionSnapshot | None:
        return self._snapshots.get(user_id)


class RedisSessionStore:
    """One JSON blob per user. Verified live against a real Redis instance --
    see tests/test_redis_session_store.py."""

    def __init__(self, redis_client: object, *, key_prefix: str = "voice_session:") -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix

    @classmethod
    def from_url(cls, url: str, **kwargs: object) -> RedisSessionStore:
        import redis.asyncio as redis

        return cls(redis.from_url(url), **kwargs)

    def _key(self, user_id: str) -> str:
        return f"{self._key_prefix}{user_id}"

    async def save(self, snapshot: SessionSnapshot) -> None:
        payload = json.dumps(
            {
                "user_id": snapshot.user_id,
                "active_task_ids": snapshot.active_task_ids,
                "entity_stack": snapshot.entity_stack,
                "started_at": snapshot.started_at,
                "ended_at": snapshot.ended_at,
            }
        )
        await self._redis.set(self._key(snapshot.user_id), payload)  # type: ignore[attr-defined]

    async def load(self, user_id: str) -> SessionSnapshot | None:
        raw = await self._redis.get(self._key(user_id))  # type: ignore[attr-defined]
        if raw is None:
            return None
        data = json.loads(raw)
        return SessionSnapshot(**data)
