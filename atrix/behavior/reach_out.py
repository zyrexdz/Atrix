"""Smart outreach — Atrix reaches out when the user is quiet, but with
real reasoning, not a fixed timer.

When the user hasn't messaged in a while, instead of a hardcoded "are you
alive?" ping, this module asks the LLM to decide:

- whether to reach out at all (vs stay quiet)
- what kind of message (casual, concerned, callback, funny, just sharing)
- what to say given the last topic, the relationship state, and how long
  the user has been quiet

No hardcoded rules. The LLM judges everything. We just give it the
context — typical activity pattern, current quiet time, last topic,
relationship state, what's been happening in the bot's life — and let
it decide.

We also track outreach history so the bot doesn't spam: if the user
doesn't respond, the LLM sees the unanswered outreach and decides
whether to try again or back off.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import atrix.memory.store as store
from atrix.ai.persona import format_for_prompt as _format_persona
from atrix.ai.providers import build_provider

logger = logging.getLogger("atrix.behavior.reach_out")

# How recent the user's typical activity pattern is for the LLM judge.
TYPICAL_ACTIVITY_LOOKBACK = 50
# How many recent outreach attempts to surface to the LLM.
RECENT_OUTREACH_LOOKBACK = 5

# The LLM judge for whether to reach out and how.
DECISION_PROMPT = """You are Atrix, a teenage girl Discord friend. You're deciding whether to send your friend a proactive message because they've been quiet.

CONTEXT:
- How long since their last message: {quiet_hours:.1f} hours
- Their typical activity pattern (recent): {typical_pattern}
- Last thing they talked about: "{last_topic}"
- How many hours you usually text: they tend to text every {typical_gap_hours:.1f} hours when active
- Current time for them: {time_of_day}
- Your persona (your life details): {persona}
- How you currently feel about them: {state_description}
- Your recent outreach attempts: {recent_outreach}

DECIDE:
- should_reach_out: bool. Send a proactive message right now?
- reason: short string explaining why
- style: one of:
  - "casual" — just a casual "yo" or "wyd" type check-in
  - "share" — share something from your own life (a thought, a meme, a complaint)
  - "callback" — reference something specific from your past conversations
  - "concerned" — gently express you've been thinking about them being quiet
  - "back_off" — they've been non-responsive, you should give them space
- message: the actual message to send. 1-3 short lines, casual Discord voice, your persona.

RULES:
- "back_off" if they've been pinged multiple times recently and never responded. Don't be annoying.
- "concerned" only if they've been unusually quiet for their pattern AND the relationship is close.
- "callback" only if you have a specific recent thing to reference. Don't force it.
- "casual" or "share" are the default — most reach-outs should feel like normal texting.
- "share" is good when you have a current obsession/mood from your persona that's been in your life.
- If they're quiet but not abnormally so (e.g. they usually take breaks), don't reach out. Real friends don't panic over normal gaps.
- Match the energy to how long they've been gone. A few hours = casual "wyd". A few days = something more personal. A week+ = "hey been thinking about you".

Return ONLY valid JSON.
"""


def _typical_pattern(events: List[Dict[str, Any]], user_id: str) -> str:
    """Render the user's recent activity pattern as a human-readable string."""
    if not events:
        return "no data yet"
    # Average gap between user messages (we don't have per-message events
    # for non-rude messages, so we approximate by event count vs elapsed time).
    return f"{len(events)} recent interactions logged"


def _typical_gap_hours(user_id: str) -> float:
    """Estimate the user's typical gap between messages (in hours)."""
    hist = store.get_recent_messages(user_id, limit=80)
    user_msgs = [m for m in hist if m.get("role") == "user"]
    if len(user_msgs) < 3:
        return 24.0  # assume once a day as default
    # Compute time deltas between consecutive user messages
    deltas: List[float] = []
    # We have role/content but no timestamps here; use message_embeddings
    # table which has created_at. Or just use the relational events log.
    events = store.get_recent_relationship_events(user_id, limit=30)
    if len(events) < 2:
        return 24.0
    from datetime import datetime as _dt
    times: List[_dt] = []
    for e in events:
        try:
            times.append(_dt.fromisoformat(e["created_at"].replace("Z", "+00:00")))
        except Exception:
            continue
    if len(times) < 2:
        return 24.0
    times.sort()
    deltas_hours = [
        (times[i + 1] - times[i]).total_seconds() / 3600
        for i in range(len(times) - 1)
    ]
    if not deltas_hours:
        return 24.0
    # Use median for robustness against outliers.
    deltas_hours.sort()
    return deltas_hours[len(deltas_hours) // 2]


def _format_time_of_day(hour: int) -> str:
    if 0 <= hour < 5:  return "very late night / past midnight"
    if 5 <= hour < 9:  return "early morning, just waking up"
    if 9 <= hour < 15: return "school/work hours"
    if 15 <= hour < 18: return "afternoon, after school/work"
    if 18 <= hour < 22: return "evening, free time"
    return "late night, winding down"


def _format_recent_outreach(recent_events: List[Dict[str, Any]]) -> str:
    """Show the user whether the bot has been trying to reach out and if
    the user responded."""
    if not recent_events:
        return "no recent outreach"
    # Events with category 'warm' or 'neutral' that have no follow-up were
    # likely the bot being friendly; we just show the latest few.
    lines = []
    for e in reversed(recent_events[-RECENT_OUTREACH_LOOKBACK:]):
        marker = "→" if e["impact"] > 0 else "·"
        cat = e["category"]
        lines.append(f"  {marker} [{cat}] {e.get('reasoning', '')}")
    return "\n".join(lines)


async def should_reach_out(user_id: str) -> Optional[Dict[str, Any]]:
    """Decide whether to send a proactive reach-out right now.

    Returns a dict with ``should_reach_out``, ``reason``, ``style``,
    ``message`` — or ``None`` if the LLM call failed.
    """
    state = store.get_user_state(user_id)
    last_user_ts = state.get("last_user_msg_time")
    now = store.get_user_current_time(user_id)
    if not last_user_ts:
        return None

    try:
        last_dt = datetime.fromisoformat(last_user_ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=now.tzinfo)
        else:
            last_dt = last_dt.astimezone(now.tzinfo)
        quiet_hours = (now - last_dt).total_seconds() / 3600
    except Exception:
        return None

    # Build context.
    typical_gap = _typical_gap_hours(user_id)
    persona = _format_persona(user_id)
    state_data = store.get_relationship_state(user_id)
    state_desc = (state_data or {}).get("state_description", "neutral, just chatting")
    facts = store.get_all_facts(user_id)
    last_topic = (facts.get("state") or {}).get("current_topic", "general")
    recent_events = store.get_recent_relationship_events(user_id, limit=RECENT_OUTREACH_LOOKBACK)
    typical = _typical_pattern(recent_events, user_id)
    recent_outreach = _format_recent_outreach(recent_events)
    time_of_day = _format_time_of_day(now.hour)

    try:
        provider = build_provider()
        response = await provider.chat(
            [
                {"role": "system", "content": DECISION_PROMPT.format(
                    quiet_hours=quiet_hours,
                    typical_pattern=typical,
                    typical_gap_hours=typical_gap,
                    time_of_day=time_of_day,
                    persona=persona,
                    state_description=state_desc,
                    recent_outreach=recent_outreach,
                    last_topic=last_topic,
                )},
            ],
            temperature=0.85,
            max_tokens=300,
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=__import__("re").M).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        if not data.get("should_reach_out"):
            return None
        return {
            "reason": str(data.get("reason", "")),
            "style": str(data.get("style", "casual")),
            "message": str(data.get("message", "")).strip(),
            "quiet_hours": quiet_hours,
        }
    except Exception as exc:
        logger.debug("Reach-out decision failed: %s", exc)
        return None


# Need re import for the json strip
import re  # noqa: E402