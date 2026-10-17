"""Background fact, topic, and persona extraction.

Runs after every bot reply. Pulls long-term facts, nudges topic counts,
updates Atrix's self-disclosure persona based on what's been said, and
picks a pet name for the user if they don't have one yet.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import atrix.memory.store as store
from atrix.ai.persona import _load_template, save_template
from atrix.ai.providers import build_provider
from atrix.ai.semantic import store_embedding

logger = logging.getLogger("atrix.ai.extract")


EXTRACTION_PROMPT = """You are a memory extractor for Atrix, a Discord AI friend.
Current datetime: {now}

Analyze the conversation and identify:
1. Long-term facts about the user (the *human* they are talking to — NOT Atrix, and NOT the user's other friends):
   - "profile": name, age, gender ("boy"/"girl"), location, timezone.
   - "routine": wake-up, school/work schedule, bedtime. ONLY extract if the user EXPLICITLY states a recurring routine. A single "morning" greeting is NOT a routine.
   - "interests": games, music, hobbies.
   - "social": details about the user's friends/family.
   - "preferences": communication-style preferences.
   - "speech_style": the user's natural cadence and energy.
   - "memes": inside jokes they explain or reference.
   - "notes": other personal facts.
2. Upcoming events with target times the friend can be reminded about later.

Respond ONLY with valid JSON:
{{
  "facts": [{{"category": "social", "key": "turkey_friend", "value": "online friend for 6 years from Turkey"}}],
  "events": [{{"description": "math test thursday", "target_time": "2026-10-23T09:00:00"}}]
}}
If nothing new, return {{"facts": [], "events": []}}.
"""


PERSONA_UPDATE_PROMPT = """You keep Atrix's "self" details current. Look at this conversation and update her persona to reflect what's actually been going on in her life.

Current persona:
{current}

Conversation:
{conv}

Update the persona fields that should change based on this conversation. Be subtle — don't flip-flop every line, only update if there's a real signal. Return ONLY valid JSON with any subset of these keys (omit any that shouldn't change):

{{
  "current_obsession": "what she's into right now (rotate every few weeks)",
  "current_annoyance": "what's bugging her this week",
  "current_mood": "general vibe this week",
  "recent_win": "a small recent accomplishment"
}}
"""


async def _extract_facts_and_events(user_id: str, recent_messages: List[Dict[str, str]]) -> None:
    if not recent_messages:
        return
    now_str = store.get_user_current_time(user_id).strftime("%Y-%m-%d %H:%M:%S")
    conv = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages[-16:])

    try:
        provider = build_provider()
        response = await provider.chat(
            [
                {"role": "system", "content": EXTRACTION_PROMPT.format(now=now_str)},
                {"role": "user", "content": f"Conversation:\n{conv}"},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
        res = json.loads(text)

        for fact in res.get("facts", []):
            cat = (fact.get("category") or "notes").strip()
            key = (fact.get("key") or "").strip()
            val = (fact.get("value") or "").strip()
            if key and val:
                store.save_fact(user_id, cat, key, val)

        for event in res.get("events", []):
            desc = (event.get("description") or "").strip()
            target_time = (event.get("target_time") or "").strip()
            rem_type = (event.get("reminder_type") or "followup").strip()
            if desc and target_time:
                store.add_scheduled_event(user_id, desc, target_time, rem_type)

    except Exception as exc:
        logger.debug("Fact extraction failed: %s", exc)


async def _update_persona(user_id: str, recent_messages: List[Dict[str, str]]) -> None:
    """Subtly evolve this specific user's Atrix based on recent conversation.

    Each user gets their own version that evolves from their own chats.
    """
    if not recent_messages:
        return
    conv = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages[-20:])
    current = store.get_user_persona(user_id)
    try:
        provider = build_provider()
        response = await provider.chat(
            [
                {"role": "system", "content": PERSONA_UPDATE_PROMPT.format(
                    current=json.dumps(current, indent=2),
                    conv=conv,
                )},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
        updates = json.loads(text)
        if not isinstance(updates, dict):
            return
        changed = False
        for key, value in updates.items():
            if key in current and isinstance(value, str) and value.strip() and value != current[key]:
                store.update_user_persona_field(user_id, key, value.strip())
                changed = True
        if changed:
            logger.debug("Persona updated for %s: %s", user_id, list(updates.keys()))
    except Exception as exc:
        logger.debug("Persona update failed for %s: %s", user_id, exc)


async def _maybe_pick_pet_name(user_id: str) -> None:
    """Pick a casual nickname for the user if we know enough about them."""
    if store.get_pet_name(user_id):
        return
    facts = store.get_all_facts(user_id)
    profile = facts.get("profile", {})
    if not (profile.get("gender") or profile.get("name")):
        return
    prompt = (
        f"You are Atrix, a teenage girl Discord friend. "
        f"Pick a single short, casual nickname for this user based on: "
        f"name='{profile.get('name', '')}', gender='{profile.get('gender', '')}'. "
        f"Real-friend vibes — could be a name shortening, a vibe, anything. "
        f"Examples: bestie, dude, bro, hun, bubs, their name shortened. "
        f"1 word, lowercase, max 10 chars. Output ONLY the nickname."
    )
    try:
        provider = build_provider()
        response = await provider.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=20,
        )
        words = (response.text or "").strip().split()[:1]
        if words:
            nick = words[0].strip('".,!?').lower()[:12]
            if nick and nick.isalpha():
                store.set_pet_name(user_id, nick)
    except Exception as exc:
        logger.debug("Pet-name pick failed: %s", exc)


async def run_after_reply(user_id: str, recent_messages: List[Dict[str, str]],
                            new_message_id: Optional[int] = None,
                            new_message_text: Optional[str] = None) -> None:
    """Run all post-reply extraction: facts, persona, pet name, embedding, summary."""
    await _extract_facts_and_events(user_id, recent_messages)
    await _update_persona(user_id, recent_messages)
    await _maybe_pick_pet_name(user_id)
    if new_message_id is not None and new_message_text:
        await store_embedding(user_id, new_message_id, new_message_text)
    # Rolling summary (only if we have enough new messages).
    from atrix.ai.summary import needs_summary, refresh_summary
    if needs_summary(user_id):
        try:
            await refresh_summary(user_id)
        except Exception as exc:
            logger.debug("Summary refresh failed: %s", exc)