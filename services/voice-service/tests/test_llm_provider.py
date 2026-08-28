"""make_llm_client had zero test coverage -- both branches (gemini,
anthropic) untested despite being on the construction path entrypoint()
and router_eval.py both depend on."""

from __future__ import annotations

from anthropic.resources.messages.messages import AsyncMessages
from voice_service.config import Settings
from voice_service.gemini_client import GeminiMessagesClient
from voice_service.llm_provider import make_llm_client


def _settings(**overrides) -> Settings:
    values = {
        "GOOGLE_API_KEY": "gg_key",
        "GEMINI_API_KEY": None,
        "ANTHROPIC_API_KEY": None,
        **overrides,
    }
    return Settings(_env_file=None, **values)


def test_gemini_provider_returns_a_gemini_messages_client():
    settings = _settings(GEMINI_API_KEY="gm_key", GEMINI_MODEL="gemini-2.5-flash")

    client = make_llm_client(settings)

    assert isinstance(client, GeminiMessagesClient)
    assert client._model == "gemini-2.5-flash"


def test_anthropic_provider_returns_anthropics_bound_messages_method():
    settings = _settings(ANTHROPIC_API_KEY="an_key")

    client = make_llm_client(settings)

    assert isinstance(client, AsyncMessages)
