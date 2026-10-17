"""OCR fallback for image attachments. Kept for backward compat; the
default vision path now lives in ``vision.py`` and uses Gemini."""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger("atrix.integrations.image_preview")

_ocr_reader = None
_ocr_attempted = False


def _get_ocr_reader():
    global _ocr_reader, _ocr_attempted
    if _ocr_attempted:
        return _ocr_reader
    _ocr_attempted = True
    try:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=True, download_enabled=True)
    except Exception as exc:
        logger.debug("OCR reader not initialized: %s", exc)
        _ocr_reader = None
    return _ocr_reader


async def inspect_image_attachment(attachment_url: str, filename: str) -> str:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return f"Image attachment: {filename}"
                data = await resp.read()

        def _process():
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(data))
                width, height = img.size
                format_name = img.format or "image"
                extracted = []
                reader = _get_ocr_reader()
                if reader:
                    try:
                        results = reader.readtext(data, detail=0)
                        extracted = [t.strip() for t in results if t.strip()]
                    except Exception as exc:
                        logger.debug("OCR extraction error: %s", exc)
                if extracted:
                    joined = " | ".join(extracted)
                    return f"Image: {filename} ({format_name}, {width}x{height}). Visible text: \"{joined}\""
                return f"Image: {filename} ({format_name}, {width}x{height})"
            except Exception as exc:
                logger.debug("Error reading image %s: %s", filename, exc)
                return f"Image attachment: {filename}"

        return await asyncio.to_thread(_process)
    except Exception as exc:
        logger.debug("Could not inspect attachment %s: %s", filename, exc)
        return f"Image attachment: {filename}"