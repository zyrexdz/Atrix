"""LLM-facing layer: providers, brain, persona, semantic recall, summarization, languages."""

from atrix.ai.providers import ProviderResponse, build_provider
from atrix.ai.brain import chat_response, generate_proactive_message, generate_self_correction
from atrix.ai.languages import detect_language, LANGUAGES, get_user_language, set_user_language

__all__ = [
    "ProviderResponse", "build_provider",
    "chat_response", "generate_proactive_message", "generate_self_correction",
    "detect_language", "LANGUAGES", "get_user_language", "set_user_language",
]