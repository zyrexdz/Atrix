"""Voice message handling — *receive* side only.

Discord voice messages arrive as ``Message`` objects with either the ``voice``
flag set (discord.py 2.3+) or an attachment whose ``content_type`` starts
with ``audio/``. We download the bytes, hand them to OpenAI Whisper for
transcription, and return plain text. The bot never sends voice notes back.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import aiohttp
import discord
from openai import AsyncOpenAI

import atrix.config as config

logger = logging.getLogger("atrix.integrations.voice")

AUDIO_MIME_PREFIXES = (
    "audio/ogg", "audio/opus", "audio/webm", "audio/mpeg", "audio/mp4",
    "audio/mp3", "audio/wav", "audio/x-wav", "audio/flac", "audio/x-m4a",
)
AUDIO_FILENAME_HINTS = (".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm", ".flac", ".oga")


def _has_openai_key() -> bool:
    return bool(config.OPENAI_API_KEY)


def is_voice_message(message: discord.Message) -> bool:
    flags = getattr(message, "flags", None)
    if flags is not None and getattr(flags, "voice", False):
        return True
    for att in message.attachments:
        ct = (att.content_type or "").lower()
        if any(ct.startswith(p) for p in AUDIO_MIME_PREFIXES):
            return True
        if att.filename.lower().endswith(AUDIO_FILENAME_HINTS):
            return True
    return False


def pick_voice_attachment(message: discord.Message) -> Optional[discord.Attachment]:
    for att in message.attachments:
        ct = (att.content_type or "").lower()
        if any(ct.startswith(p) for p in AUDIO_MIME_PREFIXES):
            return att
        if att.filename.lower().endswith(AUDIO_FILENAME_HINTS):
            return att
    return None


async def download_attachment(attachment: discord.Attachment, *, timeout: float = 10.0) -> Optional[bytes]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()
    except Exception as exc:
        logger.warning("Voice download failed: %s", exc)
        return None


async def transcribe_bytes(audio: bytes, *, filename: str = "voice.ogg") -> Optional[str]:
    if not _has_openai_key():
        return None
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    try:
        file_obj = io.BytesIO(audio)
        file_obj.name = filename
        result = await client.audio.transcriptions.create(
            model=config.OPENAI_TRANSCRIBE_MODEL,
            file=file_obj,
            response_format="text",
        )
        text = (result or "").strip() if isinstance(result, str) else getattr(result, "text", "")
        return text or None
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return None


async def transcribe_message(message: discord.Message) -> Optional[str]:
    attachment = pick_voice_attachment(message)
    if attachment is None:
        return None
    audio = await download_attachment(attachment)
    if not audio:
        return None
    text = await transcribe_bytes(audio, filename=attachment.filename)
    if text:
        logger.info("Transcribed %d bytes from %s -> %d chars", len(audio), message.author, len(text))
    return text


def status_line() -> str:
    if _has_openai_key():
        return f"🎙️ Voice transcription: **enabled** (model `{config.OPENAI_TRANSCRIBE_MODEL}`)"
    return "🎙️ Voice transcription: **disabled** — set `OPENAI_API_KEY` in `.env` to enable."