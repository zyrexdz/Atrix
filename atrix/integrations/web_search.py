"""Web search via DuckDuckGo. Free, no key, used by the brain for live facts.

Also exports an entity-lookup helper that scans user messages for
named entities (people, places, things) and runs targeted searches so
Atrix doesn't have to ask "who is that?" when the user mentions someone
she doesn't already know about.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Tuple

from ddgs import DDGS

logger = logging.getLogger("atrix.integrations.web_search")


async def search_web(query: str, max_results: int = 3) -> str:
    """Run a single web search and return a flat string of results."""
    def _search():
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return ""
                lines = []
                for r in results:
                    title = r.get("title", "").strip()
                    body = r.get("body", "").strip()
                    if title and body:
                        lines.append(f"- {title}: {body}")
                return "\n".join(lines)
        except Exception as exc:
            logger.warning("Web search error for '%s': %s", query, exc)
            return ""
    return await asyncio.to_thread(_search)


# ---------------------------------------------------------------------------
# Entity extraction + automatic lookup
# ---------------------------------------------------------------------------

# Common words to ignore when scanning for proper nouns.
_STOPWORDS = {
    "i", "im", "ive", "id", "ill", "you", "youre", "youve", "youll",
    "he", "she", "it", "we", "they", "them", "their", "your", "my",
    "this", "that", "these", "those", "what", "when", "where", "why",
    "how", "who", "which", "lol", "lmao", "idk", "tbh", "ngl",
    "yeah", "yea", "nah", "ok", "okay", "bro", "fr", "wait", "wtf",
    "the", "and", "but", "or", "so", "if", "then", "because",
    "today", "yesterday", "tomorrow", "now", "later",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "really", "actually", "literally", "definitely", "probably", "maybe",
    "school", "home", "work", "life", "day", "time", "way", "thing",
    "guy", "girl", "boy", "man", "woman", "person", "people",
}


def extract_entities(text: str, max_entities: int = 4) -> List[str]:
    """Pull likely-proper-noun phrases out of *text*.

    Heuristics:
    - Sequences of capitalized words (not at sentence start) are proper nouns.
    - Quoted phrases count too.
    - Single capitalized words after lowercase context are proper nouns.
    - Skip common stopwords and very short words.
    """
    if not text:
        return []

    found: List[str] = []
    seen = set()

    # 1. Quoted phrases: "Taylor Swift", 'Squid Game', etc.
    for m in re.finditer(r'["\']([A-Z][\w\s\.\-]{1,40}?)["\']', text):
        e = m.group(1).strip()
        if e and e.lower() not in seen and len(e) > 1:
            found.append(e)
            seen.add(e.lower())

    # 2. Capitalized word sequences (1-3 words, capitalized).
    for m in re.finditer(r'(?<![A-Za-z])([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', text):
        phrase = m.group(1).strip()
        words = phrase.split()
        if not words:
            continue
        if words[0].lower() in _STOPWORDS:
            continue
        if any(len(w) < 2 for w in words):
            continue
        if phrase.lower() in seen:
            continue
        found.append(phrase)
        seen.add(phrase.lower())

    # 3. Single capitalized words that look like names.
    for m in re.finditer(r'(?<![A-Za-z])([A-Z][a-z]{2,})', text):
        word = m.group(1)
        if word.lower() in _STOPWORDS or word.lower() in seen:
            continue
        if len(word) < 3:
            continue
        # Skip if it's the first word of a sentence (often just sentence start).
        # Approximation: if the previous character is `.` or `?` or `!` or start.
        start = m.start()
        if start == 0:
            continue
        prev = text[start - 1]
        if prev in ".?!" and text[start:start + 1].isupper():
            continue
        found.append(word)
        seen.add(word.lower())

    return found[:max_entities]


async def lookup_entities(entities: List[str]) -> List[Tuple[str, str]]:
    """Search the web for each entity. Returns ``(entity, snippet_or_empty)``."""
    if not entities:
        return []
    results: List[Tuple[str, str]] = []
    for entity in entities:
        snippet = await search_web(f"{entity} who is", max_results=2)
        if snippet:
            results.append((entity, snippet[:400]))
    return results


def format_for_context(lookups: List[Tuple[str, str]]) -> str:
    """Render entity lookups as a context block for the LLM."""
    if not lookups:
        return ""
    parts = [
        "LIVE LOOKUPS for things the user just mentioned "
        "(use this to know who/what they are — don't ask the user to explain):"
    ]
    for entity, snippet in lookups:
        parts.append(f"  • {entity}:\n{snippet}")
    return "\n".join(parts)