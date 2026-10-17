"""Energy mirroring — detect how wired/tired the user is from message shape."""
from __future__ import annotations

import re
from typing import Literal

EnergyLevel = Literal["dead", "tired", "neutral", "warmed", "hyped"]


def detect_energy(text: str) -> EnergyLevel:
    if not text or not text.strip():
        return "neutral"

    stripped = text.strip()
    length = len(stripped)
    n_caps = sum(1 for c in stripped if c.isupper())
    n_excl = stripped.count("!")
    caps_ratio = n_caps / max(length, 1)
    has_shouting = bool(re.search(r"[A-Z]{4,}", stripped))
    has_repeat = bool(re.search(r"!{2,}|\?{2,}|!+\?+", stripped))
    n_emoji = sum(1 for c in stripped if ord(c) > 0x2700)

    if (has_shouting and caps_ratio > 0.4) or has_repeat or n_excl >= 3:
        return "hyped"
    if length < 12 and not has_shouting and n_emoji <= 1:
        return "dead"
    if length < 30 and n_excl <= 1 and n_emoji <= 1:
        return "tired"
    if length > 120 or n_excl >= 2 or n_emoji >= 3:
        return "warmed"
    return "neutral"


_ENERGY_HINT = {
    "dead":    "User is running on zero. Keep your reply to 1-3 short words max. Match the energy: 'rip', '💀', 'gn', 'fair', 'same'.",
    "tired":   "User is low-energy. Keep it brief and soft. 1-2 short lines, lowercase, no exclamation marks, no hype.",
    "neutral": "User is at baseline. Match their typical length and energy.",
    "warmed":  "User is engaged. You can be a bit more talkative — 2-3 lines, some emoji, reactive.",
    "hyped":   "User is hyped. Match the energy — quick bursts, exclamation marks ok, emoji ok, mirror the volume.",
}


def hint_for_prompt(text: str) -> str:
    return f"ENERGY: {detect_energy(text)} → {_ENERGY_HINT[detect_energy(text)]}"