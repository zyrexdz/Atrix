"""Semantic memory — embed messages, find relevant past context.

When the user mentions "my brother", we want the bot to recall the thing they
said about their brother three weeks ago, even if it was tagged differently
or buried in long-ago messages. Embedding-based recall does this.

Embeddings are stored as BLOBs in each user's isolated SQLite. The cosine
similarity is computed in Python, so we don't need a vector DB.

Uses the new ``google-genai`` SDK.
"""
from __future__ import annotations

import logging
import struct
from typing import List, Optional, Tuple

import atrix.config as config
import atrix.memory.store as store

logger = logging.getLogger("atrix.ai.semantic")

# We request 768-dim embeddings to match what the old code stored.
# gemini-embedding-001 supports custom output_dimensionality.
EMBEDDING_DIM = 768
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_CACHE_LIMIT = 400


def _bytes_to_vec(b: bytes) -> List[float]:
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))


def _vec_to_bytes(v: List[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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


async def embed_text(text: str) -> Optional[List[float]]:
    """Get an embedding for *text* via Gemini.

    Returns None if Gemini isn't configured or the call fails. The bot should
    keep working without semantic recall — this layer is best-effort.
    """
    if not text or not text.strip():
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        from google.genai import types
        response = await client.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        if not response or not response.embeddings:
            return None
        return list(response.embeddings[0].values)
    except Exception as exc:
        logger.debug("Embedding call failed: %s", exc)
        return None


async def store_embedding(user_id: str, message_id: int, text: str) -> None:
    """Compute and persist the embedding for a single message."""
    emb = await embed_text(text)
    if emb is None:
        return
    store.store_message_embedding(user_id, message_id, _vec_to_bytes(emb))


async def find_relevant(
    user_id: str, query: str, top_k: int = 3, min_similarity: float = 0.55,
) -> List[Tuple[str, str, float]]:
    """Find the most semantically similar past messages to *query*.

    Returns a list of (role, content, similarity) tuples, most-similar first.
    Skips very-low-similarity hits to keep noise out of the prompt.
    """
    q_emb = await embed_text(query)
    if q_emb is None:
        return []
    rows = store.get_recent_message_embeddings(user_id, limit=EMBEDDING_CACHE_LIMIT)
    if not rows:
        return []
    scored: List[Tuple[str, str, float]] = []
    for r in rows:
        try:
            v = _bytes_to_vec(bytes(r["embedding"]))
        except Exception:
            continue
        sim = _cosine(q_emb, v)
        if sim >= min_similarity:
            scored.append((r["role"], r["content"], sim))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def format_for_prompt(quotes: List[Tuple[str, str, float]]) -> str:
    """Render the semantic-recall block for the system prompt."""
    if not quotes:
        return ""
    lines = []
    for role, content, sim in quotes:
        if len(content) > 140:
            content = content[:137] + "…"
        speaker = "you (Atrix)" if role == "assistant" else "they (the user)"
        lines.append(f"  - {speaker}: \"{content}\" (relevance {sim:.2f})")
    return (
        "PAST MESSAGES THAT MIGHT BE RELEVANT TO RIGHT NOW "
        "(don't quote them verbatim, but let them color your response):\n"
        + "\n".join(lines)
    )