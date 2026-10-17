"""Runtime configuration loaded from environment / .env.

All env vars the bot cares about are declared here, validated once at import
time, and exported as module-level constants. Keep the surface small: new
options go here, not scattered through the codebase.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
import pytz
import tzlocal

load_dotenv()
logger = logging.getLogger("atrix.config")


def _env(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else default


# ----- Discord ---------------------------------------------------------------

DISCORD_TOKEN = _env("DISCORD_TOKEN")

_raw_allowed_user = _env("ALLOWED_USER_ID")
ALLOWED_USER_ID = _raw_allowed_user if _raw_allowed_user.isdigit() else ""

if _raw_allowed_user and not ALLOWED_USER_ID:
    logger.warning(
        "ALLOWED_USER_ID='%s' is not a valid Discord snowflake; ignoring it.", _raw_allowed_user,
    )


# ----- LLM provider ---------------------------------------------------------
#
# Set LLM_PROVIDER to one of: deepseek, openai, anthropic, gemini, groq,
# openrouter, together, mistral, custom. Each provider has its own API key
# env var and optional model override (see .env.example).

LLM_PROVIDER = _env("LLM_PROVIDER", "deepseek").lower()

DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-chat")

OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.8-flash")

GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

TOGETHER_API_KEY = _env("TOGETHER_API_KEY")
TOGETHER_MODEL = _env("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

MISTRAL_API_KEY = _env("MISTRAL_API_KEY")
MISTRAL_MODEL = _env("MISTRAL_MODEL", "mistral-small-latest")

CUSTOM_API_KEY = _env("CUSTOM_API_KEY")
CUSTOM_BASE_URL = _env("CUSTOM_BASE_URL")
CUSTOM_MODEL = _env("CUSTOM_MODEL")


# ----- Voice transcription (Whisper) ----------------------------------------
#
# Optional. Independent of LLM_PROVIDER. Leave empty to disable.

OPENAI_TRANSCRIBE_MODEL = _env("OPENAI_TRANSCRIBE_MODEL", "whisper-1")


# ----- Timezone --------------------------------------------------------------

_tz_setting = _env("TIMEZONE", "auto")
if not _tz_setting or _tz_setting.lower() == "auto":
    try:
        USER_TIMEZONE = pytz.timezone(str(tzlocal.get_localzone_name()))
    except Exception:
        USER_TIMEZONE = pytz.UTC
else:
    try:
        USER_TIMEZONE = pytz.timezone(_tz_setting)
    except Exception:
        USER_TIMEZONE = pytz.UTC


def get_current_time() -> datetime:
    return datetime.now(USER_TIMEZONE)


def format_current_time_str() -> str:
    return get_current_time().strftime("%A, %B %d, %Y - %I:%M %p")