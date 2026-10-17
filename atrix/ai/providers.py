"""Multi-provider LLM abstraction.

The bot is wired to one provider at a time, selected by ``config.LLM_PROVIDER``.
Most modern hosted LLMs expose an OpenAI-compatible HTTP API, so the
``OpenAICompatibleProvider`` covers DeepSeek, OpenAI, Groq, OpenRouter, Together,
and Mistral with one code path. Anthropic and Gemini use their own SDKs.

All providers expose the same ``chat`` interface and return a unified
``ProviderResponse`` (text + normalized tool calls) so the rest of the
codebase never has to care which model is active.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import atrix.config as config

logger = logging.getLogger("atrix.ai.providers")


@dataclass
class ProviderResponse:
    """The unified return shape every provider must produce."""
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Any = None


class LLMProvider:
    name: str = "abstract"

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.85,
        max_tokens: int = 350,
    ) -> ProviderResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI-compatible (DeepSeek, OpenAI, Groq, OpenRouter, Together, Mistral)
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(LLMProvider):
    """Any endpoint that speaks the OpenAI chat-completions protocol."""

    def __init__(self, api_key: str, base_url: str, model: str, name: str = "openai-compatible"):
        if not api_key:
            raise ValueError(f"{name} provider needs an API key. Check your .env file.")
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.name = name
        self._supports_tools = True

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.85,
        max_tokens: int = 350,
    ) -> ProviderResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and self._supports_tools:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        text = (msg.content or "").strip()

        tool_calls: List[Dict[str, Any]] = []
        for tc in (msg.tool_calls or []):
            func = getattr(tc, "function", None)
            if not func:
                continue
            try:
                args = json.loads(func.arguments or "{}")
            except Exception:
                args = {}
            tool_calls.append({"name": func.name, "arguments": args})

        return ProviderResponse(text=text, tool_calls=tool_calls, raw=response)


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("anthropic provider needs ANTHROPIC_API_KEY in .env")
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from exc
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.name = "anthropic"

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.get("function", {}).get("name", ""),
                "description": t.get("function", {}).get("description", ""),
                "input_schema": t.get("function", {}).get("parameters",
                                                          {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

    @staticmethod
    def _split_system(messages: List[Dict[str, Any]]):
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") != "system"]
        return "\n\n".join(sys_parts), chat_msgs

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]):
        out = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                out.append({"role": role, "content": content})
        return out

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.85,
        max_tokens: int = 350,
    ) -> ProviderResponse:
        system, chat_msgs = self._split_system(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(chat_msgs),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = await self.client.messages.create(**kwargs)

        text_parts, tool_calls = [], []
        for block in (response.content or []):
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "name": getattr(block, "name", ""),
                    "arguments": getattr(block, "input", {}) or {},
                })

        return ProviderResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("gemini provider needs GEMINI_API_KEY in .env")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini provider requires `google-genai`. "
                "Install with: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=api_key)
        self.model_name = model
        self.name = "gemini"

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]):
        """Map our OpenAI-style messages to the google-genai Contents format.

        The new SDK accepts a list of ``{"role": "user"|"model",
        "parts": [text]}`` dicts. System instructions go into a separate
        ``config`` field on the request.
        """
        system_parts = []
        contents = []
        for m in messages:
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                if isinstance(content, str) and content:
                    system_parts.append(content)
            elif role == "user":
                if isinstance(content, str) and content:
                    contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                if isinstance(content, str) and content:
                    contents.append({"role": "model", "parts": [{"text": content}]})
        return system_parts, contents

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.85,
        max_tokens: int = 350,
    ) -> ProviderResponse:
        system_parts, contents = self._convert_messages(messages)
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_parts:
            config.system_instruction = "\n\n".join(system_parts)

        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        text = (response.text or "").strip() if response else ""
        return ProviderResponse(text=text, tool_calls=[], raw=response)


# ---------------------------------------------------------------------------
# Presets & factory
# ---------------------------------------------------------------------------

_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "kind": "openai", "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY", "model_env": "DEEPSEEK_MODEL",
        "label": "DeepSeek",
    },
    "openai": {
        "kind": "openai", "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY", "model_env": "OPENAI_MODEL",
        "label": "OpenAI",
    },
    "groq": {
        "kind": "openai", "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
        "label": "Groq",
    },
    "openrouter": {
        "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "api_key_env": "OPENROUTER_API_KEY", "model_env": "OPENROUTER_MODEL",
        "label": "OpenRouter",
    },
    "together": {
        "kind": "openai", "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "api_key_env": "TOGETHER_API_KEY", "model_env": "TOGETHER_MODEL",
        "label": "Together AI",
    },
    "mistral": {
        "kind": "openai", "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY", "model_env": "MISTRAL_MODEL",
        "label": "Mistral",
    },
    "anthropic": {
        "kind": "anthropic",
        "default_model": "claude-3-5-haiku-latest",
        "api_key_env": "ANTHROPIC_API_KEY", "model_env": "ANTHROPIC_MODEL",
        "label": "Anthropic",
    },
    "gemini": {
        "kind": "gemini",
        "default_model": "gemini-3.8-flash",
        "api_key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
        "label": "Google Gemini",
    },
    "custom": {
        "kind": "openai", "base_url": "",
        "default_model": "",
        "api_key_env": "CUSTOM_API_KEY", "model_env": "CUSTOM_MODEL",
        "label": "Custom OpenAI-compatible",
    },
}


def list_providers() -> List[Dict[str, str]]:
    return [
        {
            "id": key,
            "label": p["label"],
            "default_model": p.get("default_model", ""),
            "key_env": p.get("api_key_env", ""),
        }
        for key, p in _PROVIDER_PRESETS.items()
    ]


def build_provider() -> LLMProvider:
    provider_key = (config.LLM_PROVIDER or "deepseek").lower()
    if provider_key not in _PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider_key}'. "
            f"Supported: {', '.join(_PROVIDER_PRESETS)}"
        )

    preset = _PROVIDER_PRESETS[provider_key]
    api_key = os.getenv(preset["api_key_env"], "").strip()
    model = os.getenv(preset["model_env"], preset.get("default_model", "")).strip()
    label = preset["label"]

    if not api_key:
        raise ValueError(f"{label} provider selected but {preset['api_key_env']} is empty in .env")

    if preset["kind"] == "openai":
        base_url = preset.get("base_url", "")
        if provider_key == "custom":
            base_url = os.getenv("CUSTOM_BASE_URL", "").strip()
            if not base_url:
                raise ValueError("custom provider needs CUSTOM_BASE_URL in .env")
        if not model:
            raise ValueError(f"{label} provider needs a model set via {preset['model_env']}")
        return OpenAICompatibleProvider(api_key=api_key, base_url=base_url, model=model, name=label)

    if preset["kind"] == "anthropic":
        if not model:
            model = preset["default_model"]
        return AnthropicProvider(api_key=api_key, model=model)

    if preset["kind"] == "gemini":
        if not model:
            model = preset["default_model"]
        return GeminiProvider(api_key=api_key, model=model)

    raise ValueError(f"Unhandled provider kind: {preset['kind']}")