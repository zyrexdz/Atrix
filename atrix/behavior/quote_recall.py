"""Cross-session quote recall — pulls exact past quotes when relevant.

Layered on top of semantic memory: where ``semantic.py`` returns fuzzy
similarity hits, this module formats them as concrete past quotes that
the bot can reference in conversation.
"""
from __future__ import annotations

from typing import List, Tuple

import atrix.memory.store as store
from atrix.ai.semantic import find_relevant


async def find_relevant_past_quotes(user_id: str, query: str, top_k: int = 2) -> List[Tuple[str, str]]:
    """Return a list of (role, content) past messages relevant to *query*."""
    hits = await find_relevant(user_id, query, top_k=top_k, min_similarity=0.6)
    return [(role, content) for role, content, _ in hits]


def format_for_prompt(quotes: List[Tuple[str, str]]) -> str:
    if not quotes:
        return ""
    lines = []
    for role, content in quotes:
        if len(content) > 140:
            content = content[:137] + "…"
        speaker = "you said" if role == "assistant" else "they said"
        lines.append(f"  - {speaker}: \"{content}\"")
    return "PAST QUOTES THAT MIGHT BE RELEVANT (use sparingly, don't force):\n" + "\n".join(lines)