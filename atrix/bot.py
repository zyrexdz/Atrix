"""Atrix — Discord entry point.

The bot is DM-only: it ignores every message outside a 1-on-1 direct message
channel. The flow is:

    user message ──► debouncer (waits for multi-message bursts to settle)
                   ──► voice transcription (if a voice message)
                   ──► brain.chat_response
                   ──► typing indicator + bubble-by-bubble send
                   ──► background memory + persona + summary extraction

Run with ``python -m atrix`` (or via ``run.bat`` on Windows).
"""
from __future__ import annotations

import asyncio
import logging

import discord

import atrix.config as config
import atrix.memory.store as store
from atrix.ai import brain
from atrix.discord_ui.commands import handle_command
from atrix.discord_ui.debouncer import UserDebouncer
from atrix.discord_ui.presence import refresh_presence, set_presence
from atrix.proactive.engine import ProactiveEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for noisy in ("httpx", "httpcore", "discord.http"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("atrix.bot")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.dm_messages = True
intents.typing = True

client = discord.Client(intents=intents)
proactive_engine = ProactiveEngine(client)
debouncer = UserDebouncer(client)


@client.event
async def on_ready() -> None:
    logger.info("Bot logged in as %s (%s)", client.user.name, client.user.id)
    logger.info("Default fallback timezone: %s", config.USER_TIMEZONE)
    # Run schema migrations on every existing user DB so the latest columns
    # (weight, last_referenced, persona_state, relationship_events, etc.)
    # are guaranteed to exist before any query runs.
    migrated = store.migrate_all_user_dbs()
    if migrated:
        logger.info("Schema migration applied to %d user DB(s).", migrated)
    proactive_engine.start()
    refresh_presence.start()
    await set_presence(client)
    logger.info("Friend AI Bot is ONLINE — listening for DMs.")


@client.event
async def on_typing(channel, user, when) -> None:
    if user == client.user or user.bot:
        return
    if isinstance(channel, discord.DMChannel):
        debouncer.on_user_typing(str(user.id))


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user or message.author.bot:
        return

    # DM-only.
    if message.guild is not None:
        return
    if not isinstance(message.channel, discord.DMChannel):
        return

    user_id = str(message.author.id)
    if config.ALLOWED_USER_ID and user_id != config.ALLOWED_USER_ID:
        return

    user_text = (message.content or "").strip()
    has_attachments = bool(message.attachments)
    if not user_text and not has_attachments:
        return

    author_name = message.author.name
    preview = user_text or f"[Attachment: {message.attachments[0].filename}]"
    logger.info("[DM from @%s (%s)]: %s", author_name, user_id, preview)

    # Engagement: if user is replying to a bot message, mark it as having a follow-up.
    state = store.get_user_state(user_id)
    last_bot = state.get("last_bot_msg_time") or ""
    last_user = state.get("last_user_msg_time") or ""
    if last_bot and (not last_user or last_bot > last_user):
        store.mark_latest_engagement_followup(user_id)

    if await handle_command(message, user_id, user_text, proactive_engine=proactive_engine):
        return

    await debouncer.add_message(message)


def main() -> None:
    if not config.DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN is missing. Set it in .env and try again.")
        return
    if not config.DEEPSEEK_API_KEY and config.LLM_PROVIDER in ("deepseek", ""):
        logger.error(
            "DEEPSEEK_API_KEY is missing and LLM_PROVIDER is unset or 'deepseek'. "
            "Either set DEEPSEEK_API_KEY, or set LLM_PROVIDER to a provider you do have a key for."
        )
        return
    client.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    main()