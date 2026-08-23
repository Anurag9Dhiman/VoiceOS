import pytest
from pydantic import ValidationError
from voice_service.config import Settings


def _speech_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gg_key")


def test_settings_load_from_env_with_anthropic_key(monkeypatch):
    _speech_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "an_key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.google_api_key == "gg_key"
    assert settings.anthropic_api_key == "an_key"
    assert settings.gemini_api_key is None


def test_settings_load_from_env_with_only_gemini_key(monkeypatch):
    """The actual deployment shape this project runs in: no Anthropic
    account, Gemini only."""
    _speech_keys(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gm_key")

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key is None
    assert settings.gemini_api_key == "gm_key"


def test_collectiveos_ws_url_defaults_to_local_mock_backend(monkeypatch):
    _speech_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gm_key")
    monkeypatch.delenv("COLLECTIVEOS_WS_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.collectiveos_ws_url == "ws://localhost:8000/v1/ws"


def test_settings_fail_fast_when_google_api_key_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gm_key")  # satisfy the LLM-key requirement

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    # loc uses each field's validation_alias (the env var name), not the
    # Python attribute name.
    missing = {e["loc"][0] for e in exc_info.value.errors() if e["loc"]}
    assert missing == {"GOOGLE_API_KEY"}


def test_settings_require_at_least_one_llm_key(monkeypatch):
    _speech_keys(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "at least one of ANTHROPIC_API_KEY or GEMINI_API_KEY" in str(exc_info.value)


@pytest.mark.parametrize(
    "anthropic,gemini,override,expected",
    [
        ("an_key", None, None, "anthropic"),
        (None, "gm_key", None, "gemini"),
        ("an_key", "gm_key", None, "anthropic"),  # both set, no override -> designed default
        ("an_key", "gm_key", "gemini", "gemini"),  # explicit override wins either way
        (None, "gm_key", "gemini", "gemini"),
    ],
)
def test_resolved_llm_provider(monkeypatch, anthropic, gemini, override, expected):
    _speech_keys(monkeypatch)
    for key, value in [
        ("ANTHROPIC_API_KEY", anthropic),
        ("GEMINI_API_KEY", gemini),
        ("LLM_PROVIDER", override),
    ]:
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    settings = Settings(_env_file=None)

    assert settings.resolved_llm_provider == expected
