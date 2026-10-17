"""Personality drift — small style adjustments based on engagement."""
from __future__ import annotations

import atrix.memory.store as store

LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.6


def hint_for_prompt(user_id: str) -> str:
    rate = store.get_engagement_rate(user_id)
    if rate is None:
        return ""
    if rate < LOW_THRESHOLD:
        return (
            "ENGAGEMENT: recent replies aren't landing as well. Try mixing things up — "
            "ask a real question, bring up something specific from memory, change the rhythm."
        )
    if rate > HIGH_THRESHOLD:
        return (
            "ENGAGEMENT: recent replies are landing well. Keep doing what you're doing — "
            "this user's taste matches the current style."
        )
    return ""