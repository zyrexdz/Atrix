"""Smart timing — learn when the user is actually around."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import atrix.memory.store as store

DEFAULT_ACTIVE_HOURS = 12
RECENT_ACTIVITY_MINUTES = 30
FALLBACK_HOURS = list(range(11, 24))


def get_learned_active_hours(user_id: str) -> List[int]:
    hist = store.get_active_hours(user_id)
    if not hist:
        return list(FALLBACK_HOURS)
    pairs = sorted(enumerate(hist), key=lambda x: x[1], reverse=True)
    active = sorted([h for h, c in pairs[:DEFAULT_ACTIVE_HOURS] if c > 0])
    return active or list(FALLBACK_HOURS)


def is_within_active_window(user_id: str, now: Optional[datetime] = None) -> bool:
    if now is None:
        now = store.get_user_current_time(user_id)
    return now.hour in get_learned_active_hours(user_id)


def should_skip_for_recent_activity(user_id: str) -> bool:
    return store.is_user_currently_active(user_id, window_minutes=RECENT_ACTIVITY_MINUTES)


def _fmt_hour(h: int) -> str:
    if h == 0:  return "12am"
    if h == 12: return "12pm"
    if h < 12:  return f"{h}am"
    return f"{h - 12}pm"


def format_for_prompt(user_id: str) -> str:
    hours = get_learned_active_hours(user_id)
    if not hours:
        return ""
    if len(hours) == 24:
        return "USER SCHEDULE: usually active all day"
    runs = []
    start = prev = hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        runs.append((start, prev))
        start = prev = h
    runs.append((start, prev))
    parts = [f"{_fmt_hour(s)}-{_fmt_hour((e + 1) % 24)}" for s, e in runs]
    return (f"USER SCHEDULE: usually active {', '.join(parts)} (user's local time). "
            f"Proactive pings should fall inside this window.")