"""External integrations: voice, vision, link preview, web search, reels."""

from atrix.integrations.voice import is_voice_message, transcribe_message, status_line as voice_status
from atrix.integrations.vision import understand_image, understand_video
from atrix.integrations.link_preview import extract_link_info
from atrix.integrations.image_preview import inspect_image_attachment
from atrix.integrations.web_search import search_web
from atrix.integrations.reels import (
    is_reel_url, extract_reel_url, extract_reel_context,
)

__all__ = [
    "is_voice_message", "transcribe_message", "voice_status",
    "understand_image", "understand_video",
    "extract_link_info",
    "inspect_image_attachment",
    "search_web",
    "is_reel_url", "extract_reel_url", "extract_reel_context",
]