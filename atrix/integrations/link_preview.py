"""Link metadata extraction for YouTube, Twitter/X, Instagram, TikTok, etc."""
from __future__ import annotations

import html
import logging
import re
from typing import Optional

import aiohttp

logger = logging.getLogger("atrix.integrations.links")

URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')


async def extract_link_info(text: str) -> Optional[str]:
    matches = URL_REGEX.findall(text)
    if not matches:
        return None

    url = matches[0].strip()
    if not url.startswith("http"):
        url = "https://" + url

    try:
        if "youtube.com" in url or "youtu.be" in url:
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            async with aiohttp.ClientSession() as session:
                async with session.get(oembed_url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return f"YouTube video: \"{data.get('title', '')}\" by {data.get('author_name', '')}"

        if "twitter.com" in url or "x.com" in url:
            fx_url = url.replace("twitter.com", "api.fxtwitter.com").replace("x.com", "api.fxtwitter.com")
            async with aiohttp.ClientSession() as session:
                async with session.get(fx_url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tweet = data.get("tweet", {})
                        text_content = tweet.get("text", "")
                        author = tweet.get("author", {}).get("name", "")
                        if text_content:
                            return f"Tweet by {author}: \"{text_content[:200]}\""

        fetch_url = url
        if "instagram.com" in url:
            fetch_url = url.replace("instagram.com", "vxinstagram.com")

        if "tiktok.com" in url:
            oembed_url = f"https://www.tiktok.com/oembed?url={url}"
            async with aiohttp.ClientSession() as session:
                async with session.get(oembed_url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return f"TikTok by {data.get('author_name', '')}: \"{data.get('title', '')}\""

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(fetch_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                page = await resp.text(errors="ignore")

                og_title = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']', page, re.I)
                if not og_title:
                    og_title = re.search(r'<meta\s+content=["\'](.*?)["\']\s+property=["\']og:title["\']', page, re.I)
                title_tag = re.search(r'<title>(.*?)</title>', page, re.I | re.S)
                og_desc = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']', page, re.I)
                if not og_desc:
                    og_desc = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', page, re.I)

                title = ""
                if og_title:
                    title = html.unescape(og_title.group(1).strip())
                elif title_tag:
                    title = html.unescape(title_tag.group(1).strip())
                desc = html.unescape(og_desc.group(1).strip()) if og_desc else ""

                if title and desc:
                    return f"Link Content: \"{title}\" — {desc[:180]}"
                if title:
                    return f"Link Content: \"{title}\""
                if desc:
                    return f"Link Content: \"{desc[:180]}\""
    except Exception as exc:
        logger.debug("Could not extract preview for %s: %s", url, exc)

    return None