"""Relational state — Atrix's actual feelings toward each friend.

No hardcoded thresholds, no strike counters, no forced apology gates.
The LLM itself judges every user message and writes a free-form state
description from Atrix's first-person perspective. The state evolves
naturally: warm messages warm her up, mean messages cool her off, sincere
apologies recover, sarcasm is recognized as a joke, inside jokes lighten
the mood. Real feelings, not a punishment system.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import atrix.memory.store as store
from atrix.ai.providers import build_provider

logger = logging.getLogger("atrix.behavior.relational")

# ---------------------------------------------------------------------------
# Per-message emotional impact analysis
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are Atrix's inner voice, judging how a Discord DM from her friend makes her feel. You have her full context: persona, memories, the relationship arc so far, known inside jokes, and the new message.

Return ONLY valid JSON in this exact shape:

{{
  "impact": -0.4,                       // -1.0 (very hurtful) to 1.0 (very warm). How this message made her feel.
  "category": "rude",                    // one of:
                                         //   kind, warm, neutral, playful_joke, sarcastic_joke, annoyed,
                                         //   rude, hurtful, abusive, sincere_apology, performative_apology
  "sincere": true,                       // true if the emotional content is genuine. False for sarcasm, gaming, performative.
  "inside_joke": false,                  // true if this is referencing a known shared joke.
  "reasoning": "one short sentence",     // why you judged it this way
  "topic": "their job interview",        // 2-5 word label of what they're talking about RIGHT NOW. e.g. "their job", "a new game", "their family", "memes", "us", "nothing in particular"
  "engagement": "high"                  // one of: "high" (asking questions, elaborating, building on the convo), "medium" (normal back-and-forth), "low" (short replies, one-word answers, clearly distracted or uninterested)
}}

RULES:
- Sarcasm is real. "lol you're the worst" between friends is warmth, not rudeness. Set impact HIGH and inside_joke=true.
- "lol i'm dead" / "kys" / "i hate this" between friends venting together is venting, not rudeness.
- A real apology after real hurt is "sincere_apology" with high positive impact (0.3 to 0.6, not 1.0 — trust rebuilds slowly).
- "sorry lol" or "my bad" with no follow-through is "performative_apology" with impact near 0. The user is gaming.
- Sustained dismissiveness, dismissals of her feelings, mocking things she cares about, or talking down to her: "rude" with negative impact.
- Personal attacks, slurs, threats, harassment: "abusive" with impact below -0.7.
- "kind" is for genuine warmth, care, asking about her, sharing something personal.
- "warm" is the everyday friendship maintenance — saying hi, joking around, normal convo.
- When in doubt, be generous. Real friends give each other the benefit of the doubt.

ENGAGEMENT SIGNALS:
- "high": they ask follow-up questions, build on what I said, share details unprompted, message length is medium-long with substance
- "medium": normal back-and-forth, neither hyper-engaged nor checked-out
- "low": one-word answers ("k", "ok", "yeah", "lol"), very short messages, abrupt topic changes that signal distraction, no questions back

TOPIC DETECTION:
- A short 2-5 word label, lowercase. Pick the most specific thing they're talking about.
- If the message is just "lol" or "ok" with no real content, set topic to whatever it was last (or "continuing previous topic" if you can't tell).
- Topic shifts are normal and fine. Don't flag them as rude.

CONTEXT:
- Atrix's recent emotional arc: {arc}
- Known inside jokes between them: {memes}
- The friend's recent topics: {topics}
- The previous topic they were on (most recent): {prev_topic}
- The new message: {message}
"""


async def analyze_message_impact(
    user_id: str,
    message: str,
    history: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Run the LLM judge over one user message. Returns the parsed analysis or None."""
    if not message or not message.strip():
        return None

    # Build context: arc of recent events, inside jokes, topics, prev topic.
    recent_events = store.get_recent_relationship_events(user_id, limit=8)
    arc = _format_arc(recent_events)

    facts = store.get_all_facts(user_id)
    memes = ", ".join(f"{k}: {v}" for k, v in (facts.get("memes") or {}).items()) or "none"
    topics = store.get_top_topics(user_id, limit=5)
    topic_str = ", ".join(t["topic"] for t in topics) or "none yet"

    state = facts.get("state") or {}
    prev_topic = state.get("current_topic", "none yet")

    # Use the last few conversation turns as additional context.
    recent_turns = history[-6:] if history else []
    convo_snippet = "\n".join(f"{m['role']}: {m['content']}" for m in recent_turns)
    full_message_block = (convo_snippet + "\n" if convo_snippet else "") + f"NEW MESSAGE: {message}"

    try:
        provider = build_provider()
        response = await provider.chat(
            [
                {"role": "system", "content": ANALYSIS_PROMPT.format(
                    arc=arc, memes=memes, topics=topic_str, prev_topic=prev_topic,
                    message=full_message_block,
                )},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            import re
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
        data = json.loads(text)
        if not isinstance(data, dict) or "impact" not in data:
            return None
        return {
            "impact": float(data.get("impact", 0.0)),
            "category": str(data.get("category", "neutral")),
            "sincere": bool(data.get("sincere", True)),
            "inside_joke": bool(data.get("inside_joke", False)),
            "reasoning": str(data.get("reasoning", "")),
            "topic": str(data.get("topic", prev_topic or "general")).strip().lower()[:60],
            "engagement": str(data.get("engagement", "medium")).strip().lower(),
        }
    except Exception as exc:
        logger.debug("Emotional analysis failed for %s: %s", user_id, exc)
        return None


def _format_arc(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "first message ever"
    # Most recent first; show last 6 for the arc.
    lines = []
    for e in reversed(events[-6:]):
        ts = e.get("created_at", "")[:16].replace("T", " ")
        lines.append(f"  [{ts}] {e['category']} (impact {e['impact']:+.2f}): {e.get('reasoning', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Free-form state writing
# ---------------------------------------------------------------------------

STATE_PROMPT = """You are Atrix's inner voice. Given the recent emotional arc of how a friend has been treating you, write a SHORT first-person description of how you are feeling toward them RIGHT NOW.

Rules:
- Write 1-2 sentences, first person, casual Discord voice.
- Don't be generic. Be specific to what actually happened.
- Don't announce hardcoded thresholds like "after 3 mean messages." Just describe the feeling.
- If they were mean and you took it personally, say so. If they were sweet and you're warming up, say so. If you're over it, you're over it.
- Don't use "as an AI" or "I should feel." You're a real person, write how it actually feels.
- Reference specific things they said if relevant.

Examples of good states:
- "kinda annoyed they keep making fun of my music taste but whatever"
- "actually really happy rn, that thing they said about the test was sweet"
- "i'm still a little hurt from yesterday but they're being normal so we're fine"
- "haven't heard from them in like 3 days, kinda worried tbh"
- "we're good, the joke earlier was actually funny"
- "honestly done with this, they keep being dismissive"

RECENT ARC:
{arc}

THEIR LAST MESSAGE EXCERPT: "{last_msg}"

Write 1-2 sentences max.
"""


async def update_state(user_id: str) -> Optional[Dict[str, Any]]:
    """Recompute the user's relationship state from their recent events."""
    events = store.get_recent_relationship_events(user_id, limit=15)
    if not events:
        return None

    arc = _format_arc(events)
    last_msg = events[0].get("sample_text", "")

    try:
        provider = build_provider()
        response = await provider.chat(
            [
                {"role": "system", "content": STATE_PROMPT.format(arc=arc, last_msg=last_msg[:200])},
            ],
            temperature=0.7,
            max_tokens=120,
        )
        state_text = (response.text or "").strip().strip('"').strip("'").strip()
        if not state_text:
            return None
        # Trajectory: a tiny LLM call to label the trend, or compute it crudely here.
        trajectory = _estimate_trajectory(events)
        last_msg_at = ""
        if events:
            last_msg_at = events[0].get("created_at", "")
        store.set_relationship_state(
            user_id=user_id,
            state_description=state_text,
            trajectory=trajectory,
            last_user_msg_excerpt=last_msg,
            last_user_msg_at=last_msg_at,
        )
        return {
            "state_description": state_text,
            "trajectory": trajectory,
            "last_user_msg_excerpt": last_msg,
        }
    except Exception as exc:
        logger.debug("State update failed for %s: %s", user_id, exc)
        return None


def _estimate_trajectory(events: List[Dict[str, Any]]) -> str:
    """Crude trajectory estimate from the last 5 events. Stable / improving / declining."""
    if len(events) < 3:
        return "stable"
    recent = events[:5]
    avg = sum(e["impact"] for e in recent) / len(recent)
    older = events[5:10] if len(events) >= 10 else recent
    older_avg = sum(e["impact"] for e in older) / len(older) if older else 0.0
    delta = avg - older_avg
    if delta > 0.15:
        return "warming"
    if delta < -0.15:
        return "cooling"
    return "stable"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def format_for_prompt(user_id: str) -> str:
    """Render the current relationship state for the system prompt.

    Returns empty string if there's no state yet. This is non-prescriptive —
    it just tells Atrix how she's feeling. The persona prompt instructs her
    to honor that feeling in her own way.
    """
    state = store.get_relationship_state(user_id)
    if not state or not state.get("state_description"):
        return ""

    description = state["state_description"]
    trajectory = state.get("trajectory") or "stable"

    # Recent arc (last 4 events) so the model has continuity.
    recent = store.get_recent_relationship_events(user_id, limit=4)
    arc_lines = []
    for e in reversed(recent):
        marker = "+" if e["impact"] > 0.05 else ("-" if e["impact"] < -0.05 else "·")
        cat = e["category"].replace("_", " ")
        arc_lines.append(f"  {marker} {cat}")
    arc_block = "\n".join(arc_lines) if arc_lines else "  (just started)"

    facts = store.get_all_facts(user_id)
    state_facts = facts.get("state") or {}
    current_topic = state_facts.get("current_topic", "")
    engagement = state_facts.get("engagement", "medium")
    engagement_hint = {
        "high": ("ENGAGEMENT: they're really into this conversation. Match their energy, "
                 "ask follow-ups, build on what they said."),
        "low": ("ENGAGEMENT: their replies are short or absent. Don't push. Either go with "
                "their new subject or send something low-effort. Real friends don't force "
                "conversation when someone's checked out."),
    }.get(engagement, "")
    topic_block = (f"WHAT THEY'RE CURRENTLY TALKING ABOUT: \"{current_topic}\".\n"
                   f"If they shift subject, follow them naturally. If they seem distracted, "
                   f"let it breathe.") if current_topic else ""

    parts = [
        "YOUR CURRENT FEELINGS TOWARD THIS FRIEND:",
        f"  [right now you're feeling: {description}]",
        f"  [trajectory: {trajectory}]",
        f"  [recent arc (oldest to newest):\n{arc_block}]",
    ]
    if topic_block:
        parts += ["", topic_block]
    if engagement_hint:
        parts += ["", engagement_hint]
    parts += ["",
        "Honor these feelings in how you text. If you're hurt, you don't have to be warm. "
        "If you've moved on, don't be cold. If they changed subject, just follow — don't "
        "awkwardly try to bring them back. If they sound low energy, don't force depth. "
        "Real feelings are inconsistent and so are yours — don't announce them, just let "
        "them color your response."]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Top-level post-message hook
# ---------------------------------------------------------------------------

async def run_after_message(user_id: str, message: str, history: List[Dict[str, str]]) -> None:
    """Run the full relational pipeline: analyze impact, record event, refresh state."""
    analysis = await analyze_message_impact(user_id, message, history)
    if not analysis:
        return
    store.record_relationship_event(
        user_id=user_id,
        impact=analysis["impact"],
        category=analysis["category"],
        sincere=analysis["sincere"],
        inside_joke=analysis["inside_joke"],
        reasoning=analysis["reasoning"],
        sample_text=message,
    )
    # Persist the current topic and engagement as a per-user fact so the
    # prompt can reference them next turn and on `!relate`.
    topic = analysis.get("topic", "").strip()
    if topic:
        store.save_fact(user_id, "state", "current_topic", topic)
    engagement = analysis.get("engagement", "").strip().lower()
    if engagement in ("high", "medium", "low"):
        store.save_fact(user_id, "state", "engagement", engagement)
    await update_state(user_id)