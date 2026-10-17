"""Topic affinity — track what the user actually talks about."""
from __future__ import annotations

import json
import logging
import re
from typing import List

import atrix.memory.store as store

logger = logging.getLogger("atrix.behavior.topics")

EXTRACTION_PROMPT = """You extract short topic tags from a Discord conversation.

Rules:
- Output ONLY a JSON array of 1-3 short lowercase phrases.
- Each phrase is 1-3 words max. Examples: "minecraft", "math test", "best friend", "valorant ranked", "school stress".
- Skip filler ("ok", "lol", "yeah", "bro") — only real topics.
- Skip Atrix (the bot). Topics are about the user and their life.

Conversation:
{conv}

Respond ONLY with valid JSON like: ["topic one", "topic two"]
"""


async def extract_and_bump(user_id: str, recent_messages: list[dict]) -> List[str]:
    if not recent_messages:
        return []
    conv = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages[-12:])
    # Lazy import to avoid a circular dependency with the providers module.
    from atrix.ai.providers import build_provider
    try:
        provider = build_provider()
        response = await provider.chat(
            [{"role": "system", "content": EXTRACTION_PROMPT.format(conv=conv)}],
            temperature=0.1,
            max_tokens=80,
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
        topics = json.loads(text)
        if not isinstance(topics, list):
            return []
        topics = [str(t).strip().lower() for t in topics if t and len(str(t)) < 60]
        for t in topics:
            store.bump_topic(user_id, t)
        return topics
    except Exception as exc:
        logger.debug("Topic extraction failed: %s", exc)
        return []


def format_for_prompt(user_id: str, top_n: int = 5) -> str:
    rows = store.get_top_topics(user_id, limit=top_n)
    if not rows:
        return ""
    items = ", ".join(f"{r['topic']} (×{r['count']})" for r in rows)
    return f"TOPICS THIS USER TALKS ABOUT MOST: {items}. Lean into these in conversation."