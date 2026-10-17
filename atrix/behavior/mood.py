"""Mood tracking with simple keyword-based sentiment.

No LLM call needed for scoring — just a small lexicon of positive and negative
markers. Cheap, fast, and surprisingly accurate for the surface-level "is the
user having a good day or a bad day" question we actually need to answer.
"""
from __future__ import annotations

import re
from typing import Optional

import atrix.memory.store as store

# Tokens weighted by how strongly they tilt the mood reading.
POSITIVE = {
    "love": 0.5, "lit": 0.5, "amazing": 0.6, "awesome": 0.6, "happy": 0.5,
    "good": 0.2, "great": 0.4, "fire": 0.4, "sick": 0.4, "dope": 0.4,
    "vibes": 0.3, "vibe": 0.3, "vibing": 0.3, "best": 0.3, "perfect": 0.5,
    "blessed": 0.3, "grateful": 0.4, "thanks": 0.2, "thx": 0.2, "ty": 0.1,
    "win": 0.3, "won": 0.3, "killed": 0.3, "nailed": 0.3, "succeeded": 0.3,
    "lol": 0.1, "lmao": 0.1, "haha": 0.1, "fun": 0.2, "enjoy": 0.2,
    "excited": 0.4, "hyped": 0.4, "pumped": 0.4, "stoked": 0.4,
    "proud": 0.4, "cute": 0.2, "cool": 0.2, "beautiful": 0.3,
    "chill": 0.2, "relaxed": 0.3, "calm": 0.2, "peaceful": 0.3,
    "yes": 0.1, "yay": 0.4, "yess": 0.3,
}
NEGATIVE = {
    "hate": -0.6, "terrible": -0.6, "awful": -0.6, "horrible": -0.6,
    "worst": -0.5, "sad": -0.5, "depressed": -0.7, "anxious": -0.5,
    "stressed": -0.4, "stress": -0.4, "exhausted": -0.4, "tired": -0.3,
    "drained": -0.4, "burnt": -0.4, "burned": -0.3, "dead": -0.3,
    "cooked": -0.3, "rough": -0.3, "bad": -0.3, "sucks": -0.4, "sucked": -0.4,
    "fucked": -0.4, "fcked": -0.4, "screwed": -0.4, "shit": -0.3,
    "angry": -0.5, "mad": -0.4, "pissed": -0.5, "furious": -0.6,
    "annoyed": -0.3, "annoying": -0.3, "frustrated": -0.4, "frustrating": -0.4,
    "lonely": -0.5, "alone": -0.3, "isolated": -0.4,
    "fail": -0.4, "failed": -0.4, "failure": -0.4, "lost": -0.3,
    "broken": -0.4, "ruined": -0.4, "wasted": -0.3,
    "cry": -0.4, "cried": -0.4, "crying": -0.4, "tears": -0.4,
    "scared": -0.4, "afraid": -0.4, "worried": -0.3, "worry": -0.3,
    "empty": -0.4, "ugh": -0.3, "meh": -0.2, "bleh": -0.2, "lame": -0.3,
    "boring": -0.3, "bored": -0.2,
    "pain": -0.3, "hurt": -0.3, "hurts": -0.3,
    "overwhelmed": -0.5, "panic": -0.5, "panicking": -0.5,
    "ffs": -0.2, "smh": -0.2, "rip": -0.2,
}
PHRASES = [
    (r"\bno\s+way\b", -0.2), (r"\bcan'?t\s+even\b", -0.4),
    (r"\bso\s+tired\b", -0.3), (r"\bwtf\b", -0.1),
    (r"\bfeels?\s+good\b", 0.4),
    (r"\blow\s+key\s+(happy|excited|proud)\b", 0.3),
    (r"\blow\s+key\s+(sad|tired|stressed|down)\b", -0.3),
    (r"\bgonna\s+(cry|scream)\b", -0.4),
    (r"\bmiss\s+(you|her|him|them)\b", -0.2),
    (r"\bgood\s+(day|time|night|morning)\b", 0.3),
    (r"\bbad\s+(day|time|night|morning)\b", -0.3),
    (r"\bschool\s+sucks\b", -0.4), (r"\bwork\s+sucks\b", -0.4),
]
_INTENSIFIERS = {"really": 1.3, "so": 1.2, "very": 1.3, "kinda": 0.7, "lowkey": 0.8, "highkey": 1.2}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def score_text(text: str) -> float:
    """Return a sentiment score in [-1.0, 1.0] for *text*."""
    if not text:
        return 0.0
    lower = text.lower()
    total, n_hits = 0.0, 0

    for pattern, weight in PHRASES:
        if re.search(pattern, lower):
            total += weight
            n_hits += 1

    multiplier = 1.0
    for tok in _tokenize(lower):
        if tok in _INTENSIFIERS:
            multiplier = _INTENSIFIERS[tok]
            continue
        if tok in POSITIVE:
            total += POSITIVE[tok] * multiplier
            n_hits += 1
            multiplier = 1.0
        elif tok in NEGATIVE:
            total += NEGATIVE[tok] * multiplier
            n_hits += 1
            multiplier = 1.0
        else:
            multiplier = 1.0

    if n_hits == 0:
        return 0.0
    return max(-1.0, min(1.0, total / n_hits * 2.5))


def classify(score: float) -> str:
    if score is None:
        return "unknown"
    if score < -0.45: return "rough"
    if score < -0.15: return "low"
    if score < 0.15:  return "neutral"
    if score < 0.45:  return "good"
    return "great"


def record_message_mood(user_id: str, text: str) -> float:
    s = score_text(text)
    if s != 0.0:
        store.record_mood(user_id, s, text)
    return s


def format_for_prompt(user_id: str) -> str:
    recent = store.get_recent_mood(user_id, days=7)
    if not recent:
        return ""
    avg = sum(r["score"] for r in recent) / len(recent)
    label = classify(avg)
    today_avg = store.get_today_mood_average(user_id)
    today_label = classify(today_avg) if today_avg is not None else None

    if today_label and today_label != label:
        return (
            f"MOOD CONTEXT: this user has felt generally '{label}' over the past week "
            f"(avg {avg:+.2f}), but TODAY specifically is reading '{today_label}'. "
            f"Match today's energy, not the week's."
        )
    return (
        f"MOOD CONTEXT: this user has been feeling '{label}' lately (avg {avg:+.2f} over the past week). "
        f"Sit with the mood — don't force cheerfulness if they're down, and don't be heavy if they're up."
    )