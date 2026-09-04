"""Adapts Gemini's function-calling API to the same narrow interface
router.py and ack.py already expect from an Anthropic Messages client --
`async def create(**kwargs) -> object` returning something with a
`.content` list of blocks shaped like Anthropic's (`type` + `input`/`text`).

This is the only place in the codebase that knows Gemini exists: router.py,
ack.py, and every existing test for them stay provider-agnostic and
unmodified -- only the client object constructed in agent.py/router_eval.py
changes based on which API key is configured (see config.py's
resolved_llm_provider). Deliberately not a rewrite of router.py/ack.py to
some new provider-neutral interface: Anthropic's Messages shape was already
the interface those modules were built against, and inventing a second
abstraction on top would be more code to maintain for the same outcome.

Gemini's `FunctionDeclaration.parameters_json_schema` accepts a raw JSON
Schema dict directly, so the existing Anthropic-style tool definitions
(`_CLASSIFY_TOOL` in router.py) pass through with zero translation --
that's what makes this adapter this small.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol

from google import genai
from google.genai import types

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


class GeminiModelsClient(Protocol):
    """The one method this needs from google-genai's client -- narrow on
    purpose, same reasoning as router.py's MessagesClient Protocol, so
    tests can inject a fake without a live Gemini key."""

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any: ...


class GeminiMessagesClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        models: GeminiModelsClient | None = None,
    ) -> None:
        self._models = models or genai.Client(api_key=api_key).aio.models
        self._model = model

    async def create(
        self,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        **_ignored: object,
    ) -> object:
        # `model` here is whatever router.py/ack.py's own DEFAULT_MODEL is
        # (an Anthropic model string) -- neither caller knows which
        # provider is actually behind `client`, so this adapter always
        # uses its own configured Gemini model instead of that value.
        config_kwargs: dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens

        if tools:
            config_kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            parameters_json_schema=tool["input_schema"],
                        )
                        for tool in tools
                    ]
                )
            ]
            if tool_choice and tool_choice.get("type") == "tool":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[tool_choice["name"]],
                    )
                )

        # This codebase only ever sends a single user turn (router.py and
        # ack.py both build `messages=[{"role": "user", "content": ...}]`)
        # -- no multi-turn history to translate.
        user_text = messages[0]["content"] if messages else ""

        response = await self._models.generate_content(
            model=self._model,
            contents=user_text,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        return _to_anthropic_shape(response)


def _to_anthropic_shape(response: Any) -> object:
    blocks: list[object] = []
    function_calls = getattr(response, "function_calls", None)
    if function_calls:
        for call in function_calls:
            blocks.append(SimpleNamespace(type="tool_use", name=call.name, input=dict(call.args or {})))
    elif getattr(response, "text", None):
        blocks.append(SimpleNamespace(type="text", text=response.text))
    return SimpleNamespace(content=blocks)
