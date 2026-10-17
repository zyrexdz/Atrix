"""Conversation summarization for long chats.

Once a conversation runs past ~24 messages, the LLM context gets noisy. We
keep a rolling summary that gets refreshed every N messages; the system
prompt includes the latest summary + only the most recent messages.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

import atrix.memory.store as store
from atrix.ai.providers import build_provider

logger = logging.getLogger("atrix.ai.summary")

SUMMARIZE_EVERY_N_MESSAGES = 24
KEEP_RECENT_MESSAGES = 16

SUMMARY_PROMPT = """You are writing a private rolling summary of a Discord DM conversation between a teenage girl bot and a real human friend.

Capture:
- The arc of the conversation (what was discussed, in what order).
- Anything important the user revealed (vulnerabilities, news, plans, preferences).
- Anything the bot said that should be remembered.
- The current emotional vibe (venting, joking, planning, drifting).

Write 3-5 short paragraphs in casual third person. Don't be clinical. Don't use bullet points.

Conversation to summarize:
{conv}
"""


def needs_summary(user_id: str) -> bool:
    """Return True if the user has enough messages to warrant a new summary."""
    recent = store.get_recent_messages(user_id, limit=10_000)
    last = store.get_latest_conversation_summary(user_id)
    last_count = last["message_count"] if last else 0
    return len(recent) - last_count >= SUMMARIZE_EVERY_N_MESSAGES


async def refresh_summary(user_id: str) -> str:
    """Generate a new summary from the recent message history and persist it."""
    recent = store.get_recent_messages(user_id, limit=80)
    if not recent:
        return ""
    conv = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    try:
        provider = build_provider()
        response = await provider.chat(
            [
                {"role": "system", "content": SUMMARY_PROMPT.format(conv=conv)},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        text = (response.text or "").strip()
        if text:
            store.save_conversation_summary(user_id, text, message_count=len(recent))
        return text
    except Exception as exc:
        logger.debug("Summarization failed: %s", exc)
        return store.get_latest_conversation_summary(user_id)["summary_text"] if store.get_latest_conversation_summary(user_id) else ""


def get_recent_for_prompt(user_id: str) -> List[Dict[str, str]]:
    """Return the most recent KEEP_RECENT_MESSAGES messages for the LLM context."""
    return store.get_recent_messages(user_id, limit=KEEP_RECENT_MESSAGES)


def get_latest_summary(user_id: str) -> Optional[Dict[str, Any]]:
    """Alias for store.get_latest_conversation_summary, kept here for clarity."""
    return store.get_latest_conversation_summary(user_id)