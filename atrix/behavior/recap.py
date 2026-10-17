"""End-of-day recap — one soft line at the user's usual bedtime."""
from __future__ import annotations

import logging
from typing import Optional

import atrix.memory.store as store
from atrix.ai.providers import build_provider

logger = logging.getLogger("atrix.behavior.recap")

RECAP_PROMPT = """You are Atrix sending your friend a one-line end-of-day check-in.

Write ONE short, casual line (max 12 words). No "hope you had a good day!"
cliches. Reference what they actually went through today, or just acknowledge
that the day was a lot.

Examples of good recaps:
- "today was a lot huh, get some rest"
- "you cooked today ngl, see you tomorrow"
- "rough day, sleep it off"
- "you were actually kind of funny today 💀 gn"

If nothing notable happened today, just say "gn, see you tomorrow" or similar.
NEVER use exclamation marks. NEVER use periods at the end. NEVER be motivational.
"""


async def generate_recap(user_id: str) -> Optional[str]:
    today_key = store.get_user_current_time(user_id).strftime("%Y-%m-%d")
    state = store.get_user_state(user_id)
    last_user = state.get("last_user_msg_time") or ""
    if not last_user.startswith(today_key):
        return None

    msgs = store.get_recent_messages(user_id, limit=200)
    conv = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[-30:])
    try:
        provider = build_provider()
        response = await provider.chat(
            [
                {"role": "system", "content": RECAP_PROMPT},
                {"role": "user", "content": f"Today's conversation:\n{conv}"},
            ],
            temperature=0.85,
            max_tokens=80,
        )
        text = (response.text or "").strip().strip('"').strip("'").strip()
        text = text.splitlines()[0].strip() if text else ""
        if text:
            store.save_daily_summary(user_id, today_key, text, message_count=len(msgs))
        return text or None
    except Exception as exc:
        logger.debug("Recap generation failed: %s", exc)
        return None