"""Spontaneous emoji + sticker reactions on incoming messages."""
from __future__ import annotations

import logging
import random
import re

import discord

logger = logging.getLogger("atrix.discord_ui.reactions")

# Discord server sticker IDs that work on any server. These are the public
# stickers Discord ships in every server's "Wumpus" and friends packs.
# Real IDs from Discord's public sticker packs.
PUBLIC_STICKERS = [
    749054690769252494,   # :wave:
    749054669317199120,   # :hugyou:
    749054673784582195,   # :slap:
    821035583962337280,   # :yikes:
    821034842003415051,   # :thisisfine:
    821034832169779241,   # :notlike:
]

_SLEEP_WORDS = re.compile(
    r"\b(gn|goodnight|gonna sleep|heading to bed|sleep now|hop off|going to sleep)\b", re.I
)
_LAUGH_WORDS = re.compile(r"\b(lmaoo|lmfao|💀|xd|ahah)\b", re.I)


async def maybe_react(message: discord.Message) -> None:
    """Add a small spontaneous emoji or sticker reaction. Best-effort."""
    try:
        text = (message.content or "").lower().strip()
        if _SLEEP_WORDS.search(text):
            await message.add_reaction("💤")
        elif _LAUGH_WORDS.search(text) and random.random() < 0.45:
            await message.add_reaction("💀")
        elif message.attachments and random.random() < 0.35:
            await message.add_reaction("👀")
        # Occasional sticker (~1% of messages, on anything with emotion).
        elif random.random() < 0.01 and message.content:
            try:
                sticker_id = random.choice(PUBLIC_STICKERS)
                await message.channel.send(stickers=[discord.Sticker(id=sticker_id)])
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Reaction failed: %s", exc)