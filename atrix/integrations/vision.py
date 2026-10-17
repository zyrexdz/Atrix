"""Image and video understanding via Google Gemini (google-genai SDK).

If a Gemini API key is configured, the bot can actually look at images and
short videos sent in Discord — describe them, react to memes, read chat
screenshots, transcribe the audio on a Reel, etc. If no Gemini key is
set, we fall back to OCR via EasyOCR for images, or basic metadata for
videos. The bot should keep working without it.

Uses the new ``google-genai`` SDK (``from google import genai``). The
older ``google-generativeai`` package was deprecated Nov 30, 2025.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import aiohttp

import atrix.config as config

logger = logging.getLogger("atrix.integrations.vision")

VIDEO_MIME_TYPES = (
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
    "video/x-msvideo", "video/ogg",
)

IMAGE_MIME_TYPES = (
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif",
)


def _get_client():
    """Return a configured google-genai Client, or None if no key or no SDK."""
    if not config.GEMINI_API_KEY:
        return None
    try:
        from google import genai
    except ImportError:
        return None
    try:
        return genai.Client(api_key=config.GEMINI_API_KEY)
    except Exception as exc:
        logger.warning("Failed to construct Gemini client: %s", exc)
        return None


async def _download_to_bytes(url: str, timeout: float = 10.0) -> Optional[bytes]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()
    except Exception as exc:
        logger.debug("Image download failed: %s", exc)
        return None


async def _describe_with_gemini(image_bytes: bytes, mime: str, user_hint: str = "") -> Optional[str]:
    """Ask Gemini 3.8 Flash (free tier) to describe an image via inline base64."""
    client = _get_client()
    if client is None:
        return None
    try:
        model = config.GEMINI_MODEL or "gemini-3.8-flash"
        prompt = (
            "You are looking at an image someone sent in a Discord DM. "
            "Briefly describe what you see in 1-3 sentences, casual tone, no preamble. "
            "If it's a meme, capture the joke. If it's a chat screenshot, transcribe it. "
            "If it has visible text, read it."
        )
        if user_hint:
            prompt += f"\nContext: {user_hint}"

        contents = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
                "mime_type": mime,
            },
        ]
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
        )
        text = (response.text or "").strip() if response else ""
        return text or None
    except Exception as exc:
        logger.warning("Gemini image understanding failed: %s", exc)
        return None


async def _ocr_with_easyocr(image_bytes: bytes) -> Optional[str]:
    """Last-resort OCR via EasyOCR if Gemini isn't available."""
    try:
        import easyocr  # type: ignore
    except ImportError:
        return None
    try:
        import asyncio
        reader = easyocr.Reader(["en"], gpu=False, download_enabled=True)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: reader.readtext(image_bytes, detail=0))
        text = " | ".join(t.strip() for t in results if t and t.strip())
        return text or None
    except Exception as exc:
        logger.debug("EasyOCR failed: %s", exc)
        return None


async def understand_image(url: str, filename: str, mime: str = "image/png",
                            user_hint: str = "") -> str:
    """Best-effort image understanding. Order: Gemini vision → OCR → dimensions."""
    data = await _download_to_bytes(url)
    if not data:
        return f"Image attachment: {filename} (could not download)"

    desc = await _describe_with_gemini(data, mime, user_hint)
    if desc:
        return f"Image: {filename}. {desc}"

    ocr = await _ocr_with_easyocr(data)
    if ocr:
        return f"Image: {filename}. Visible text: \"{ocr}\""

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        return f"Image: {filename} ({img.format or 'image'}, {img.size[0]}x{img.size[1]})"
    except Exception:
        return f"Image attachment: {filename}"


async def understand_video(url: str, filename: str, mime: str = "video/mp4",
                            user_hint: str = "") -> str:
    """Best-effort video understanding. Used for Reels / short videos sent
    directly as Discord attachments.
    """
    data = await _download_to_bytes(url, timeout=20.0)
    if not data:
        return f"Video attachment: {filename} (could not download)"

    client = _get_client()
    if client is None:
        return (f"Video attachment: {filename} ({len(data)} bytes) — "
                f"install google-genai and set GEMINI_API_KEY to enable video understanding")

    try:
        model = config.GEMINI_MODEL or "gemini-3.8-flash"
        prompt = (
            "You are watching a short video (Instagram Reel, TikTok, or similar) "
            "someone sent in a Discord DM. Describe what's actually happening in "
            "2-3 casual sentences. If it's a meme, capture the joke and cultural "
            "context. Transcribe any on-screen text. Note the audio/sound if it's "
            "a trending sound. Write casually like you're texting a friend about it."
        )
        if user_hint:
            prompt += f"\n\nContext: {user_hint}"

        contents = [
            {"type": "text", "text": prompt},
            {
                "type": "video",
                "data": base64.b64encode(data).decode("utf-8"),
                "mime_type": mime,
            },
        ]
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
        )
        text = (response.text or "").strip() if response else ""
        if text:
            return f"Video: {filename}. {text}"
        return f"Video: {filename} (downloaded, but couldn't analyze)"
    except Exception as exc:
        logger.warning("Gemini video understanding failed for %s: %s", filename, exc)
        return f"Video attachment: {filename} ({len(data)} bytes) — Gemini failed: {exc}"