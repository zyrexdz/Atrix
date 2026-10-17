"""Discord presence (status) — dynamic, context-aware."""
from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Optional

import discord
from discord.ext import tasks

logger = logging.getLogger("atrix.discord_ui.presence")

# Status messages pool. We pick a few that fit the current context.
_GAME_STATUSES = [
    ("playing", "Minecraft"),
    ("playing", "roblox"),
    ("playing", "stardew"),
    ("playing", "hollow knight"),
]
_WATCHING_STATUSES = [
    ("watching", "YouTube"),
    ("watching", "TikTok edits"),
    ("watching", "a tutorial i'll never finish"),
]
_LISTENING_STATUSES = [
    ("listening", "Spotify"),
    ("listening", "lo-fi on repeat"),
    ("listening", "the same song for 4 hours"),
]
_CUSTOM_STATUSES_LATE = [
    ("custom", "half asleep 😴", "half asleep"),
    ("custom", "ghost mode 👻", "ghost mode"),
    ("custom", "do not disturb 🛏️", "do not disturb"),
]


def _pick_status(now: datetime) -> tuple:
    """Pick a (type, name, state) status tuple based on time of day."""
    hour = now.hour
    if 0 <= hour < 5:
        return ("watching", "YouTube at 2am", "2am")
    if 5 <= hour < 9:
        return ("custom", "half asleep 😴", "half asleep")
    if 9 <= hour < 15:
        return ("listening", "Spotify", "online")
    if 15 <= hour < 20:
        kind, name = random.choice(_GAME_STATUSES)
        return (kind, name, "online")
    if 20 <= hour < 23:
        kind, name = random.choice(_WATCHING_STATUSES)
        return (kind, name, "online")
    return ("custom", "do not disturb 🛏️", "do not disturb")


async def set_presence(client: discord.Client) -> None:
    now = datetime.now()
    kind, name, state = _pick_status(now)
    try:
        if kind == "custom":
            activity = discord.Activity(type=discord.ActivityType.custom, name=name, state=state)
            status = discord.Status.idle if "asleep" in state or "disturb" in state else discord.Status.online
        elif kind == "playing":
            activity = discord.Activity(type=discord.ActivityType.playing, name=name)
            status = discord.Status.online
        elif kind == "watching":
            activity = discord.Activity(type=discord.ActivityType.watching, name=name)
            status = discord.Status.idle if "2am" in name else discord.Status.online
        elif kind == "listening":
            activity = discord.Activity(type=discord.ActivityType.listening, name=name)
            status = discord.Status.online
        else:
            activity = None
            status = discord.Status.online
        await client.change_presence(activity=activity, status=status)
    except Exception as exc:
        logger.debug("Could not update presence: %s", exc)


@tasks.loop(minutes=30)
async def refresh_presence() -> None:
    """Refresh Discord presence every 30 minutes. Bound at startup via ``.start()``."""
    # Imported here to avoid a circular import at module load.
    from atrix.bot import client
    await set_presence(client)