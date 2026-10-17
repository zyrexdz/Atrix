"""Instagram Reel understanding.

Multiple strategies, tried in order until one works:

1. Several free public download services (different providers — when one
   is rate-limited or down, the others often still work).
2. yt-dlp as a robust fallback (pip install yt-dlp).
3. If all URL fetches fail, fall back to telling the user to send the
   video file directly — that path uses Gemini's native video input and
   is 100% reliable.

User experience: paste a Reel URL in the DM, the bot tries each service
in turn, watches the video via Gemini 3.8 Flash (free tier) which accepts
video input directly, and tells you what the Reel is about.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Optional

import aiohttp

import atrix.config as config

logger = logging.getLogger("atrix.integrations.reels")

# Match instagram.com / instagr.am Reel and post URLs.
_REEL_URL_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:reel|reels|p|tv)/[A-Za-z0-9_-]+/?",
    re.IGNORECASE,
)

# Multiple free services. Each takes a URL and returns either a direct
# video URL or a JSON containing one. We try them in order; first success wins.
_DOWNLOAD_SERVICES = [
    {
        "name": "saveig",
        "url": "https://api.saveig.app/api/v1/download",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "payload_builder": lambda u: {"url": u},
        "video_url_extractor": lambda data: _walk_for_video_url(data),
    },
    {
        "name": "snapinsta",
        "url": lambda u: f"https://api.snapinsta.app/v1/lookup?url={u}",
        "method": "GET",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        },
        "video_url_extractor": lambda data: _walk_for_video_url(data),
    },
    {
        "name": "sssinstagram",
        "url": lambda u: f"https://api.sssinstagram.com/v1/lookup?url={u}",
        "method": "GET",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        },
        "video_url_extractor": lambda data: _walk_for_video_url(data),
    },
    {
        "name": "igram",
        "url": lambda u: f"https://api.igram.world/api/v1/fetch?url={u}",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "payload_builder": lambda u: {"url": u, "platform": "instagram"},
        "video_url_extractor": lambda data: _walk_for_video_url(data),
    },
    {
        "name": "instagram-direct-api",
        "url_builder": lambda u: u.rstrip("/") + "/?__a=1&__d=dis",
        "method": "GET",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        },
        "video_url_extractor": lambda data: _extract_from_graphql(data),
    },
]


def _walk_for_video_url(obj) -> Optional[str]:
    """Recursively find the first video_url string in a JSON blob."""
    if isinstance(obj, dict):
        for key in ("url", "video_url", "play", "src", "download_url"):
            if key in obj and isinstance(obj[key], str) and obj[key].startswith("http"):
                if any(obj[key].lower().endswith(ext) or ext in obj[key].lower()
                       for ext in (".mp4", ".mov", ".webm")):
                    return obj[key]
        for v in obj.values():
            found = _walk_for_video_url(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _walk_for_video_url(item)
            if found:
                return found
    return None


def _extract_from_graphql(data) -> Optional[str]:
    """Pull a video URL out of Instagram's __a=1 JSON response."""
    try:
        items = data.get("items", [])
        if not items:
            return None
        item = items[0]
        candidates = []
        if "video_versions" in item:
            for v in item["video_versions"]:
                if v.get("url"):
                    candidates.append(v["url"])
        if "carousel_media" in item:
            for c in item["carousel_media"]:
                for v in c.get("video_versions", []):
                    if v.get("url"):
                        candidates.append(v["url"])
        return candidates[0] if candidates else None
    except Exception:
        return None


def is_reel_url(url: str) -> bool:
    return bool(_REEL_URL_RE.search(url or ""))


def extract_reel_url(text: str) -> Optional[str]:
    """Pull the first Instagram Reel/post URL from *text*."""
    m = _REEL_URL_RE.search(text or "")
    return m.group(0) if m else None


async def _try_service(session: aiohttp.ClientSession, service: dict, url: str) -> Optional[bytes]:
    """Attempt one download service. Returns video bytes or None."""
    name = service["name"]
    if "url_builder" in service:
        target = service["url_builder"](url)
    else:
        target = service["url"]
    headers = service.get("headers", {})
    method = service.get("method", "GET")

    try:
        if method == "POST":
            payload = service.get("payload_builder", lambda u: {"url": u})(url)
            async with session.post(target, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=12)) as resp:
                if resp.status != 200:
                    logger.debug("[reels] %s returned HTTP %s", name, resp.status)
                    return None
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    data = await resp.json(content_type=None)
                    video_url = service["video_url_extractor"](data)
                else:
                    text = await resp.text()
                    video_url = service["video_url_extractor"](text)
        else:
            async with session.get(target, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=12)) as resp:
                if resp.status != 200:
                    logger.debug("[reels] %s returned HTTP %s", name, resp.status)
                    return None
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    data = await resp.json(content_type=None)
                    video_url = service["video_url_extractor"](data)
                else:
                    text = await resp.text()
                    video_url = service["video_url_extractor"](text)

        if not video_url:
            return None

        # Now fetch the actual video bytes.
        async with session.get(video_url, timeout=aiohttp.ClientTimeout(total=30)) as vresp:
            if vresp.status != 200:
                return None
            data = await vresp.read()
            if len(data) < 1024:
                return None
            return data
    except Exception as exc:
        logger.debug("Reel service %s failed: %s", name, exc)
        return None


async def _try_ytdlp(url: str) -> Optional[bytes]:
    """Last-resort: try yt-dlp if installed."""
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        return None
    try:
        import asyncio
        loop = asyncio.get_event_loop()

        def _fetch():
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": False,
                "outtmpl": "-",
                "format": "best[ext=mp4]/best",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return None
                # Get the direct URL and download bytes.
                direct_url = info.get("url")
                if not direct_url:
                    return None
                # Use ydl's download.
                ydl_opts["outtmpl"] = "-"
                result = ydl.process_info(info)
                # yt-dlp writes to stdout/file; the simpler path is to fetch
                # the direct URL ourselves.
                return direct_url

        direct_url = await loop.run_in_executor(None, _fetch)
        if not direct_url:
            return None

        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(direct_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                if len(data) < 1024:
                    return None
                return data
    except Exception as exc:
        logger.debug("yt-dlp fallback failed: %s", exc)
        return None


async def download_reel(url: str) -> Optional[bytes]:
    """Try each service in order. Returns the first successful download.

    Falls back to yt-dlp as a final attempt. If everything fails, returns
    None and the user is asked to send the video file directly.
    """
    attempts: list[str] = []
    async with aiohttp.ClientSession() as session:
        for service in _DOWNLOAD_SERVICES:
            attempts.append(service["name"])
            data = await _try_service(session, service, url)
            if data:
                logger.info("Downloaded Reel via %s (%d bytes)", service["name"], len(data))
                return data

    # Final fallback: yt-dlp.
    data = await _try_ytdlp(url)
    if data:
        logger.info("Downloaded Reel via yt-dlp (%d bytes)", len(data))
        return data

    logger.warning("All Reel download attempts failed (%s) for %s", attempts, url)
    return None


REEL_UNDERSTANDING_PROMPT = """\
You are watching a short Instagram Reel someone sent in a Discord DM.

Describe what's actually happening in 2-3 casual sentences.

If it's a meme:
- Identify the joke, the cultural context, the format (POV, sound trend, edit, etc.)
- Transcribe any on-screen text
- Note the key audio/sound if it's a trending sound

If it's a tutorial / info / storytime:
- Give the gist in 1-2 sentences

If it's a vibe / aesthetic / outfit:
- Describe the visual in a way a friend would react to

Write casually, like you're texting a friend about it. No headers, no bullets.
"""


def _get_client():
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


async def understand_video_with_gemini(video_bytes: bytes, user_hint: str = "") -> Optional[str]:
    """Send video bytes to Gemini and get a description."""
    client = _get_client()
    if client is None:
        return None
    try:
        model = config.GEMINI_MODEL or "gemini-3.8-flash"
        prompt = REEL_UNDERSTANDING_PROMPT
        if user_hint:
            prompt += f"\n\nContext: {user_hint}"

        contents = [
            {"type": "text", "text": prompt},
            {
                "type": "video",
                "data": base64.b64encode(video_bytes).decode("utf-8"),
                "mime_type": "video/mp4",
            },
        ]
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
        )
        return (response.text or "").strip() if response else None
    except Exception as exc:
        logger.warning("Gemini video understanding failed: %s", exc)
        return None


async def extract_reel_context(url: str, user_hint: str = "") -> Optional[str]:
    """Top-level: given an Instagram Reel URL, return a context string for the LLM."""
    if not is_reel_url(url):
        return None
    video = await download_reel(url)
    if not video:
        return (
            f"[Instagram Reel: couldn't fetch the video from {url}. "
            f"If you want me to react to it, save the Reel and send it as a file in the DM — "
            f"that always works.]"
        )
    description = await understand_video_with_gemini(video, user_hint)
    if not description:
        return f"[Instagram Reel: video fetched but Gemini couldn't analyze it]"
    return f"[Instagram Reel]: {description}"