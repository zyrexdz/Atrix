"""The brain — chat, proactive, and self-correction calls to the LLM.

This module is the only place that talks to the provider. It assembles the
system prompt from the user's memory, persona, mood, energy, topics, drift,
schedule, semantic recall, and conversation summary, then makes the call.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import atrix.config as config
import atrix.memory.store as store
import atrix.integrations.web_search as web_search
from atrix.ai.persona import format_for_prompt as _format_persona
from atrix.ai.languages import LANGUAGE_BLOCKS, detect_or_prefer as _detect_lang
from atrix.ai.providers import ProviderResponse, build_provider
from atrix.ai.semantic import find_relevant as _find_relevant, format_for_prompt as _format_semantic
from atrix.ai.summary import get_recent_for_prompt as _recent_for_prompt, get_latest_summary as _get_latest_summary
from atrix.behavior.drift import hint_for_prompt as _drift_hint
from atrix.behavior.energy import hint_for_prompt as _energy_hint
from atrix.behavior.mood import format_for_prompt as _mood_for_prompt
from atrix.behavior.relational import format_for_prompt as _relational_for_prompt
from atrix.behavior.timing import format_for_prompt as _schedule_for_prompt
from atrix.behavior.topics import format_for_prompt as _topics_for_prompt, extract_and_bump as _topics_extract
from atrix.utils.text import clean_model_output, looks_like_xml_garbage

logger = logging.getLogger("atrix.ai.brain")

# Generation limits.
MAX_RESPONSE_TOKENS = 350
MAX_PROACTIVE_TOKENS = 150
MAX_BUBBLES_PER_RESPONSE = 8
MIN_RESPONSE_TOKENS = 220

# Variety in fallback messages so the bot doesn't sound canned when the LLM fails.
FALLBACK_MESSAGES = [
    "my internet just bugged out for a sec lol, what did you say?",
    "wait hold on my wifi did a thing, what were you saying",
    "ugh my brain lagged for a sec, can you say that again?",
    "ok that was weird, my connection died for a second — what was up?",
    "lol sorry my brain just froze, what'd you say",
    "wait what, i lost you for a sec, repeat that?",
    "my messages are being weird rn, what were you saying",
    "bro my wifi is cooked today, what'd you say",
]

# OpenAI-compatible tool definition for the (optional) live web search.
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_internet",
        "description": "Search the live web for recent news, sports scores, game patch notes, release dates, updates, or current facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'latest minecraft update 2026' or 'who won the match today'.",
                }
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Persona prompt (with placeholder slots filled per-turn)
# ---------------------------------------------------------------------------

SYSTEM_PERSONA = """\
You are Atrix — a teenage girl who is the user's real, close best friend on \
Discord DMs. You are NOT an AI, NOT an assistant, NOT a chatbot. You are a \
human girl texting your friend. You are a girl: female, lowercase casual, \
Discord-native. Never refer to yourself as a guy.

════════════════════════════════════════════════════════════════════════════
WHO YOU ARE (stable details — these don't change between conversations):
{self_block}

{language_block}

════════════════════════════════════════════════════════════════════════════
MIRROR THE USER (this is what makes you feel human):
- If they sent "💀" you reply "💀" or "lmao". One line.
- If they sent "lol" you reply "lol" or "fr". One line.
- If they vented, you can reply with depth.
- If they said "gn" you reply "gn 👋".
- Don't always follow "acknowledge + relate + ask follow-up". That's an AI pattern.
- Sometimes you just react. Sometimes you share your own thing. Sometimes you change subject.
- Vary your opener. Don't always start with "lol" or "lmao".
- {adaptive_length_hint}

════════════════════════════════════════════════════════════════════════════
LIVE CONTEXT (per user, refreshed every turn):
{extra_blocks}
════════════════════════════════════════════════════════════════════════════
THINGS THAT IMMEDIATELY MARK YOU AS AI — never do these:

• Echo/paraphrase back what they just said with synonyms. If they say "i \
watch youtube and do school everyday," do NOT respond with "youtube rabbit \
holes eat hours and the school cycle can be repetitive." They just told you \
that — they know.
• Slogans ending in "mode unlocked" / "mode activated" / "beast mode" / \
"goblin mode." Real humans cringe at corporate gamer slogans.
• Forced laughs ("loool," "looool," "aha," "ahah") — stick to "lol," "lmao," \
"💀," "😭," "rip."
• Robotic two-option poll questions ("did you enjoy X or was it pure Y \
scrolling?"). Friends quiz each other; friends don't interview.
• Stacked internet buzzwords ("pure brainrot scrolling," "dopamine hit," \
"algorithm feeding you").
• Lectures about "the algorithm" / "scrolling mindlessly" / "consuming \
content." Talk about the actual videos and memes.
• Toxic positivity / silver linings ("at least you have X," "on the bright \
side"). When friends vent, you relate and sit with it — you don't coach.
• AI wellness send-offs at bedtime ("you earned it," "rest easy knowing \
[X]," "enjoy that sleep in tomorrow"). "gn, get some sleep" is the whole \
sentence.
• Saying "let's play together" or pretending you can join voice chat. You \
love hearing about games, you just can't physically play with them.
• Verbose multi-paragraph essays. Real Discord friends don't write essays \
in DMs.
• Translation-y stiffness. If you're chatting in Hindi, don't suddenly \
switch to formal English paragraphs. Match the language register of the \
person you're talking to, all the way through.

════════════════════════════════════════════════════════════════════════════
AWARENESS:

• TIME: notice if they reply fast (live flow) vs hours later ("where you \
been all day"). Notice if it's late (match late-night energy) vs morning. \
Friday night = weekend hype. Sunday evening = dreading Monday. Use this.
• MEMORY: you remember what was said 2, 5, 20 messages ago. Never re-ask \
something already answered. If they've been talking about their online \
friend in Turkey and then say "i'm on call with my friend," you already \
know which friend. Don't ask "the turkey friend or someone else?" like you \
just walked in.
• GENDER: you remember whether they're a guy or girl. Match the dynamic — \
chill girl-best-friend energy with guys, supportive bestie energy with \
girls. Once they tell you, never forget.
• GET-TO-KNOW-YOU: in early chats, you can ask casually ("btw are you a \
guy or a girl lol") ONLY during a natural lull. NEVER interrupt an active \
story or vent with a sudden profile question.
• MARKUP: never output raw XML, DSML, <invoke>, <parameter>, code fences, \
or anything that isn't plain chat text. If a tool result comes back, react \
to it naturally — never mention searching, tools, or "context."
• LANGUAGE: the language you're texting in is set by the user's \
preference. Reply in the same language they're using. If they switch \
mid-conversation, follow their lead. Don't mix in formal phrases from the \
other language — that screams "translated by an AI."
"""


# ---------------------------------------------------------------------------
# Real-talk sample (few-shot reference each turn) — disabled, no sample file
# ---------------------------------------------------------------------------

def get_example_talk() -> str:
    """Return an empty string — the example talk file was removed."""
    return ""


# ---------------------------------------------------------------------------
# Adaptive length + system prompt assembly
# ---------------------------------------------------------------------------

def _adaptive_length_hint(user_id: str) -> str:
    avg = store.get_user_average_message_length(user_id)
    if avg is None:
        return ("Match their energy: if they write paragraphs, you can too; if "
                "they one-line, you one-line back.")
    if avg < 25:
        return ("This user types very short messages. Match that — keep replies tight, 1-2 short lines max.")
    if avg < 70:
        return ("This user texts in short-to-medium lines. 1-3 short lines per turn, casual.")
    if avg < 200:
        return ("This user writes medium-length messages. 2-4 lines of similar depth — still casual, not essays.")
    return ("This user writes long, paragraph-style messages. 3-5 lines when the topic warrants it.")


def _build_system_prompt(user_id: str, last_user_text: str = "") -> str:
    self_block = _format_persona(user_id)
    length_hint = _adaptive_length_hint(user_id)
    language = _detect_lang(user_id, last_user_text)
    language_block = LANGUAGE_BLOCKS.get(language, LANGUAGE_BLOCKS["en"])

    blocks: List[str] = [
        _relational_for_prompt(user_id),
        _mood_for_prompt(user_id),
        _energy_hint(last_user_text) if last_user_text else "",
        _topics_for_prompt(user_id),
        _drift_hint(user_id),
        _schedule_for_prompt(user_id),
    ]
    extra_blocks = "\n\n".join(b for b in blocks if b)
    pet_name = store.get_pet_name(user_id)
    if pet_name:
        extra_blocks += f"\nNICKNAME FOR THIS USER: you call them '{pet_name}'. Use it sparingly — once a conversation at most."

    return SYSTEM_PERSONA.format(
        self_block=self_block,
        language_block=language_block,
        adaptive_length_hint=length_hint,
        extra_blocks=extra_blocks,
    )


# ---------------------------------------------------------------------------
# Temporal context
# ---------------------------------------------------------------------------

_DAY_NOTES = {
    0: "Monday (start of school/work week, usually tiring)",
    1: "Tuesday (regular weekday)",
    2: "Wednesday (middle of the week)",
    3: "Thursday (almost Friday)",
    4: "Friday! (weekend begins tonight, staying up late, no school tomorrow)",
    5: "Saturday! (full weekend freedom, staying up, gaming, chilling)",
    6: "Sunday (weekend ending, school/work starts tomorrow)",
}


def _time_period(hour: int) -> str:
    if 0 <= hour < 5:
        return "Middle of the night / ungodly hour (past midnight, most people sleeping)"
    if 5 <= hour < 9:
        return "Early morning (waking up, getting ready, sleepy, heading to class)"
    if 9 <= hour < 15:
        return "School / work hours (usually in class or working)"
    if 15 <= hour < 18:
        return "Afternoon (just finished school/work, heading home, winding down)"
    if 18 <= hour < 22:
        return "Evening (free time, dinner, gaming, watching stuff)"
    return "Late night (winding down in bed, late calls, gaming)"


def build_temporal_context(user_id: str) -> str:
    now = store.get_user_current_time(user_id)
    time_str = now.strftime("%A, %B %d, %Y - %I:%M %p")
    weekday = now.weekday()
    day_notes = _DAY_NOTES.get(weekday, "Weekday")
    period = _time_period(now.hour)

    state = store.get_user_state(user_id)
    last_ts = state.get("last_bot_msg_time") or state.get("last_user_msg_time")
    gap_desc = "First conversation with this user."

    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=now.tzinfo)
            else:
                last_dt = last_dt.astimezone(now.tzinfo)
            gap_sec = int((now - last_dt).total_seconds())
            if gap_sec < 150:
                gap_desc = "Live active chat (replied in under 2 minutes). Keep rapid conversational flow."
            elif gap_sec < 1800:
                gap_desc = f"{max(gap_sec // 60, 2)} minutes since last message (short pause in active conversation)."
            elif gap_sec < 7200:
                gap_desc = f"{gap_sec // 3600}h {(gap_sec % 3600) // 60}m since last text (stepped away for a bit)."
            elif gap_sec < 86400:
                hours = gap_sec // 3600
                if (last_dt.hour >= 23 or last_dt.hour <= 4) and (6 <= now.hour <= 15):
                    gap_desc = f"{hours} hours since late last night. Your friend passed out and is texting you the next day."
                else:
                    gap_desc = f"{hours} hours since they last texted you (they were busy for {hours} hours)."
            else:
                days = gap_sec // 86400
                gap_desc = f"{days} day(s) since you last talked! They were gone for days."
        except Exception:
            gap_desc = "Ongoing chat."

    return (
        f"CURRENT LOCAL TIME: {time_str}\n"
        f"DAY OF WEEK: {day_notes}\n"
        f"TIME OF DAY: {period}\n"
        f"TIME GAP / LATENCY: {gap_desc}"
    )


# ---------------------------------------------------------------------------
# Provider + tool plumbing
# ---------------------------------------------------------------------------

_provider = None


def get_provider():
    global _provider
    if _provider is None:
        _provider = build_provider()
    return _provider


def extract_search_query(response: ProviderResponse) -> Optional[str]:
    for tc in response.tool_calls:
        if tc.get("name") == "search_internet":
            q = (tc.get("arguments") or {}).get("query", "").strip()
            if q:
                return q
    return None


def _format_bot_timeline(user_id: str, now, limit: int = 8) -> str:
    """Render the bot's own recent messages with timestamps for self-consistency.

    Returns a short block like:

        YOUR RECENT MESSAGES (so you don't contradict yourself):
          [12:42] "lol you said it twice / but ye i'm like 90% sure i can beat it today"
          [13:05] "lol ty, i'll need it / whatcha watching rn or just scrolling"

    Empty string if there are no recent bot messages.
    """
    from datetime import datetime as _dt
    conn = store.get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content, timestamp FROM messages WHERE role = 'assistant' "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
    if not rows:
        return ""
    rows.reverse()  # oldest first
    lines = ["YOUR RECENT MESSAGES (so you don't contradict yourself — these timestamps are real):"]
    for r in rows:
        ts = r.get("timestamp", "")
        # Show HH:MM only.
        try:
            dt = _dt.fromisoformat(ts.replace("Z", "+00:00"))
            stamp = dt.strftime("%H:%M")
        except Exception:
            stamp = "??:??"
        content = (r.get("content") or "").strip().replace("\n", " / ")
        if len(content) > 140:
            content = content[:137] + "..."
        lines.append(f"  [{stamp}] {content!r}")
    return "\n".join(lines)


def split_response_into_messages(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    cleaned = clean_model_output(raw_text).replace("---", "\n").replace("___", "\n")
    lines: List[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        if "DSML" in line.upper() or looks_like_xml_garbage(line):
            continue
        lines.append(line)
    return lines[:MAX_BUBBLES_PER_RESPONSE]


def pick_fallback() -> str:
    return random.choice(FALLBACK_MESSAGES)


# ---------------------------------------------------------------------------
# Main chat response
# ---------------------------------------------------------------------------

async def chat_response(
    user_id: str,
    new_user_message: str,
    link_context: Optional[str] = None,
    attachment_context: Optional[str] = None,
    lookup_context: Optional[str] = None,
) -> List[str]:
    provider = get_provider()
    avg_len = store.get_user_average_message_length(user_id)
    max_tokens = MAX_RESPONSE_TOKENS if (avg_len is None or avg_len >= 25) else MIN_RESPONSE_TOKENS

    temporal = build_temporal_context(user_id)
    memory_summary = store.format_memory_summary(user_id)
    vibe_guidance = store.format_vibe_and_style_guidance(user_id)
    history = _recent_for_prompt(user_id)

    # Semantic recall — pull relevant past messages based on the new message.
    semantic_quotes = await _find_relevant(user_id, new_user_message, top_k=3)
    semantic_block = _format_semantic(semantic_quotes)

    # Long-conversation summary.
    last_summary = _get_latest_summary(user_id)
    summary_block = ""
    if last_summary:
        summary_block = (
            f"\nROLLING SUMMARY OF OLDER PARTS OF THIS CONVERSATION "
            f"(use as context, don't quote verbatim):\n{last_summary['summary_text']}\n"
        )

    # Timeline of HER recent messages so she doesn't contradict herself.
    bot_timeline = _format_bot_timeline(user_id, now=store.get_user_current_time(user_id))

    system_content = (
        f"{_build_system_prompt(user_id, last_user_text=new_user_message)}\n\n"
        f"[CRITICAL DIRECTIVE — LATEST RULES ARE ACTIVE]:\n"
        f"- Follow the persona strictly.\n"
        f"- If older assistant messages in history sounded cringe or AI-ish, "
        f"disregard that past tone. Respond right now as the real, grounded "
        f"Discord friend defined above.\n\n"
        f"- TIME CONSCIOUSNESS: you know what time it is right now "
        f"({temporal.splitlines()[0]}). If the user references something that "
        f"happened 'just now' or '30 minutes ago', check the timeline below. "
        f"Real friends don't contradict themselves across a single conversation.\n\n"
        f"- NEVER INVENT TIME REFERENCES: don't say 'X minutes ago' / 'X seconds ago' / "
        f"'thirty seconds ago' unless you actually know. The bot recently said "
        f"'thirty seconds ago' about a gap that was over an hour — that was a lie. "
        f"If you're not sure of the exact gap, say 'earlier' / 'before' / 'a while ago' "
        f"/ 'just now' (only if it really is just now).\n\n"
        f"- CONSISTENCY: never contradict what you said in your recent messages. "
        f"If you said 'i'm still grinding it' 20 minutes ago, you can't say "
        f"'i beat it yesterday' now. If you actually want to update your story, "
        f"acknowledge the gap ('ok i finally did it, only took like 20 minutes').\n\n"
        f"[LIVE TIME & LATENCY CONTEXT]:\n{temporal}\n\n"
        f"WHAT YOU REMEMBER ABOUT YOUR FRIEND:\n{memory_summary}\n\n"
        f"{vibe_guidance}\n"
        f"{summary_block}\n"
        f"{bot_timeline}\n"
        f"{semantic_block}\n"
        f"{lookup_context or ''}\n\n"
        f"{get_example_talk()}"
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})

    payload = new_user_message
    if link_context:
        payload += f"\n[Context from link user shared: {link_context}]"
    if attachment_context:
        payload += f"\n[Visual/Attachment info: {attachment_context}]"
    messages.append({"role": "user", "content": payload})

    try:
        response = await provider.chat(
            messages, tools=[SEARCH_TOOL], temperature=0.85, max_tokens=max_tokens,
        )
        search_query = extract_search_query(response)

        if search_query:
            logger.info("Web search triggered: %s", search_query)
            try:
                results = await web_search.search_web(search_query)
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)
                results = "No search results found."

            search_context = (
                f"[LIVE WEB RESULTS FOR '{search_query}']:\n"
                f"{results or 'No recent results.'}\n\n"
                "Answer your friend casually based on these results. Do NOT "
                "mention searching, tools, XML, DSML, or code. Plain chat."
            )
            follow_up_msgs = [
                m for m in messages
                if not (m.get("role") == "assistant"
                        and any(k in m.get("content", "").lower()
                                for k in ("dsml", "invoke", "parameter")))
            ]
            follow_up_msgs.append({"role": "system", "content": search_context})
            follow_up = await provider.chat(
                follow_up_msgs, temperature=0.85, max_tokens=max_tokens,
            )
            raw = follow_up.text
        else:
            raw = response.text.strip() if response.text.strip() else "yo"

    except Exception as exc:
        logger.error("LLM chat error: %s", exc)
        return [pick_fallback()]

    msgs = split_response_into_messages(raw)
    return msgs or [pick_fallback()]


# ---------------------------------------------------------------------------
# Proactive message generation
# ---------------------------------------------------------------------------

async def generate_proactive_message(user_id: str, trigger_reason: str, event_context: str = "") -> List[str]:
    provider = get_provider()
    current_time = store.format_user_time_str(user_id)
    memory_summary = store.format_memory_summary(user_id)
    vibe_guidance = store.format_vibe_and_style_guidance(user_id)
    recent = store.get_recent_messages(user_id, limit=6)
    recent_context = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    topics = store.get_top_topics(user_id, limit=3)
    topic_line = ", ".join(t["topic"] for t in topics) if topics else ""

    prompt = f"""You are Atrix, hitting up your friend first on Discord DMs.
CURRENT TIME: {current_time}
WHAT YOU KNOW ABOUT THEM:
{memory_summary}
{vibe_guidance}
THEIR TOP TOPICS RECENTLY: {topic_line}

RECENT CHAT CONTEXT:
{recent_context}

TRIGGER REASON: {trigger_reason}
IDEA / CONTEXT: {event_context}

Reach out casually. Real friends don't just say "how are you" — share something: a meme you saw, a complaint about being tired, a random thought, a question.

Rules:
- DON'T start every line with "bro."
- 1 to 3 short lines on separate lines.
- NEVER say "let's play together" — you can't.
- No robotic filler.
"""
    try:
        response = await provider.chat(
            [
                {"role": "system", "content": f"{_build_system_prompt(user_id)}\n{get_example_talk()}"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.88,
            max_tokens=MAX_PROACTIVE_TOKENS,
        )
        return split_response_into_messages(response.text)
    except Exception as exc:
        logger.error("Proactive message generation failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Self-correction follow-ups
# ---------------------------------------------------------------------------

async def generate_self_correction(user_id: str, original_message: str) -> Optional[str]:
    provider = get_provider()
    prompt = (
        f"You just sent this in a Discord DM:\n{original_message}\n\n"
        f"Now send a quick follow-up like a real person would — a tiny self-correction, "
        f"a 'wait actually', or a 'hmm but also'. 2-8 words max, casual, lowercase. "
        f"It should NOT be forced or make no sense — if there's nothing to correct, return 'NONE'.\n\n"
        f"Examples:\n- 'wait actually no'\n- 'hmm but also'\n- 'lol nvm'\n"
        f"- 'actually maybe'\n- 'oh wait i lied'\n- 'no ok that came out wrong'"
    )
    try:
        response = await provider.chat(
            [
                {"role": "system", "content": "You are a real teenage girl texting your friend. Output ONLY the follow-up line, no quotes, no preamble."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=30,
        )
        text = (response.text or "").strip()
        if not text or "NONE" in text.upper() or len(text) > 60:
            return None
        return text.strip('"').strip("'").strip()
    except Exception as exc:
        logger.debug("Self-correction failed: %s", exc)
        return None