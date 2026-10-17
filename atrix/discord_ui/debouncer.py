"""Multi-message debouncer — coalesces a rapid burst of DMs into one reply.

Real people send 2-5 quick messages in a row ("yo" + "wait" + "i was gonna
say..."). A bot that replies to each one looks frantic. So we wait for the
burst to settle, then send one cohesive reply.

Also responsible for: voice transcription, link/image extraction, mood
recording, and triggering the post-reply background work.
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from typing import List, Optional

import discord

import atrix.config as config
import atrix.memory.store as store
import atrix.integrations.link_preview as link_preview
import atrix.integrations.reels as reels
import atrix.integrations.vision as vision
import atrix.integrations.voice as voice
from atrix.ai import brain, extract
from atrix.behavior import mood
from atrix.behavior.relational import run_after_message as run_relational
from atrix.discord_ui.reactions import maybe_react

logger = logging.getLogger("atrix.discord_ui.debouncer")

# Discord's hard limit is 2000 chars; we leave a small safety margin.
DISCORD_MESSAGE_LIMIT = 1900

# Debounce tuning.
WAIT_DEFAULT = 2.8
WAIT_SHORT_FRAGMENT = 3.4
WAIT_COMPLETE_QUESTION = 2.0
WAIT_COMPLETE_SENTENCE = 2.2
WAIT_WHEN_USER_TYPING = 3.2

DANGLING_WORDS = {
    "and", "but", "so", "or", "because", "wait", "like",
    "then", "if", "also", "bro", "yo", "and-", "plus",
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".ogg", ".ogv")


class UserDebouncer:
    def __init__(self, bot: discord.Client) -> None:
        self.bot = bot
        self.buffers: dict[str, list[discord.Message]] = {}
        self.timer_tasks: dict[str, asyncio.Task] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        if user_id not in self.locks:
            self.locks[user_id] = asyncio.Lock()
        return self.locks[user_id]

    async def add_message(self, message: discord.Message) -> None:
        user_id = str(message.author.id)
        self.buffers.setdefault(user_id, []).append(message)

        if user_id in self.timer_tasks and not self.timer_tasks[user_id].done():
            self.timer_tasks[user_id].cancel()

        wait_seconds = self._calculate_wait_time(message.content)
        self.timer_tasks[user_id] = asyncio.create_task(
            self._wait_and_process(user_id, wait_seconds),
        )

    def on_user_typing(self, user_id: str) -> None:
        if user_id not in self.buffers or not self.buffers[user_id]:
            return
        if user_id in self.timer_tasks and not self.timer_tasks[user_id].done():
            self.timer_tasks[user_id].cancel()
        self.timer_tasks[user_id] = asyncio.create_task(
            self._wait_and_process(user_id, WAIT_WHEN_USER_TYPING),
        )

    @staticmethod
    def _calculate_wait_time(text: str) -> float:
        clean = text.strip().lower()
        if not clean:
            return WAIT_DEFAULT
        words = clean.split()
        last_word = words[-1] if words else ""
        if len(clean) < 22 and not clean.endswith("?"):
            return WAIT_SHORT_FRAGMENT
        if clean.endswith(("...", ",", "-")):
            return WAIT_SHORT_FRAGMENT
        if last_word in DANGLING_WORDS:
            return WAIT_SHORT_FRAGMENT
        if clean.endswith("?"):
            return WAIT_COMPLETE_QUESTION
        if clean.endswith((".", "!")):
            return WAIT_COMPLETE_SENTENCE
        return WAIT_DEFAULT

    async def _wait_and_process(self, user_id: str, wait_seconds: float) -> None:
        try:
            await asyncio.sleep(wait_seconds)
        except asyncio.CancelledError:
            return
        async with self._lock_for(user_id):
            messages = self.buffers.pop(user_id, [])
            if not messages:
                return
            try:
                await self._process_batch(user_id, messages)
            except Exception as exc:
                logger.error("Batch processing failed for %s: %s", user_id, exc, exc_info=True)

    async def _process_batch(self, user_id: str, messages: list[discord.Message]) -> None:
        last_msg = messages[-1]
        channel = last_msg.channel

        # ----- Build combined text --------------------------------------
        text_lines = [m.content.strip() for m in messages if m.content.strip()]
        combined_text = "\n".join(text_lines)

        # ----- Voice transcription ---------------------------------------
        voice_msgs = [m for m in messages if voice.is_voice_message(m)]
        if voice_msgs:
            for vm in voice_msgs:
                transcript = await voice.transcribe_message(vm)
                if transcript:
                    combined_text = (combined_text + "\n" + transcript).strip()
                    logger.info("Voice transcript (%d chars): %s", len(transcript), transcript[:120])

        # ----- Link metadata ---------------------------------------------
        link_info = None
        for m in messages:
            if m.content.strip():
                # Instagram Reel URL takes priority — it's video content,
                # not just metadata.
                reel_url = reels.extract_reel_url(m.content)
                if reel_url:
                    link_info = await reels.extract_reel_context(reel_url)
                    if link_info:
                        break
                info = await link_preview.extract_link_info(m.content.strip())
                if info:
                    link_info = info
                    break

        # ----- Attachment metadata ---------------------------------------
        attachment_info = None
        for m in messages:
            if m.attachments and not voice.is_voice_message(m):
                first_att = m.attachments[0]
                ext = first_att.filename.lower()
                mime = first_att.content_type or "application/octet-stream"
                if any(ext.endswith(suffix) for suffix in VIDEO_SUFFIXES) or mime.startswith("video/"):
                    attachment_info = await vision.understand_video(
                        first_att.url, first_att.filename, mime=mime,
                    )
                elif any(ext.endswith(suffix) for suffix in IMAGE_SUFFIXES) or mime.startswith("image/"):
                    attachment_info = await vision.understand_image(
                        first_att.url, first_att.filename, mime=mime,
                    )
                else:
                    filenames = [a.filename for a in m.attachments]
                    attachment_info = f"Attached media: {', '.join(filenames)}"
                break

        # ----- Persist messages ------------------------------------------
        now_iso = store.get_user_current_time(user_id).isoformat()
        for m in messages:
            if voice.is_voice_message(m):
                if combined_text:
                    store.add_message(user_id, "user", "[Voice Message]\n" + combined_text)
                    break
                store.add_message(user_id, "user", f"[Voice Message: {m.attachments[0].filename}]")
                break
            if m.content.strip():
                store.add_message(user_id, "user", m.content.strip())
            elif m.attachments:
                store.add_message(user_id, "user", f"[Sent media: {m.attachments[0].filename}]")

        store.update_user_state(user_id, last_user_msg_time=now_iso, pending_reply=0)
        store.track_wake_up_consistency(user_id, combined_text or "[voice]")

        # ----- Mood + reading pause --------------------------------------
        if combined_text and not combined_text.startswith("["):
            mood.record_message_mood(user_id, combined_text)
        read_text = combined_text or "[image/voice]"
        read_delay = min(max(len(read_text) * 0.02, 1.2), 3.2)
        if read_text and not read_text.startswith("[") and random.random() < 0.15:
            read_delay += random.uniform(1.0, 2.2)
        await asyncio.sleep(read_delay)

        # ----- Live lookup for named entities the user just mentioned -----
        # If the message mentions people/things the LLM might not know
        # (e.g. niche actors, regional figures, new releases), proactively
        # web-search them so Atrix doesn't ask "who are they?".
        lookup_context = ""
        if combined_text:
            try:
                from atrix.integrations.web_search import (
                    extract_entities, lookup_entities, format_for_context,
                )
                entities = extract_entities(combined_text)
                if entities:
                    lookups = await lookup_entities(entities)
                    lookup_context = format_for_context(lookups)
                    if lookups:
                        logger.info("[Atrix] entity lookups for %s: %s",
                                    user_id, [e for e, _ in lookups])
            except Exception as exc:
                logger.debug("Entity lookup failed: %s", exc)

        # ----- Generate ---------------------------------------------------
        async with channel.typing():
            responses = await brain.chat_response(
                user_id,
                combined_text or "[Sent image/voice]",
                link_context=link_info,
                attachment_context=attachment_info,
                lookup_context=lookup_context,
            )

        # ----- Spontaneous reactions --------------------------------------
        await maybe_react(last_msg)

        # ----- Deliver each bubble with its own typing indicator ----------
        author_name = messages[0].author.name if messages else user_id
        for idx, part in enumerate(responses):
            if idx > 0:
                await asyncio.sleep(random.uniform(0.6, 1.3))
            typing_delay = min(max(len(part) * 0.035, 0.8), 2.5)
            async with channel.typing():
                await asyncio.sleep(typing_delay)
            await _send_safe(channel, part)
            store.add_message(user_id, "assistant", part)
            logger.info("[Atrix -> @%s]: %s", author_name, part)

        bot_now_iso = store.get_user_current_time(user_id).isoformat()
        store.update_user_state(user_id, last_bot_msg_time=bot_now_iso, pending_reply=0)
        store.record_bot_message(user_id, bot_message_id=0, style_tags="")  # placeholder

        # ----- Background work --------------------------------------------
        recent = store.get_recent_messages(user_id, limit=20)
        asyncio.create_task(extract.run_after_reply(user_id, recent))
        # Relational state: judge how this message made Atrix feel, log it,
        # then update the free-form "current feelings" description.
        asyncio.create_task(run_relational(user_id, combined_text or "", recent))


async def aiohttp_session_get(url: str):
    """Placeholder kept for back-compat."""
    raise NotImplementedError


async def _send_safe(channel: discord.abc.Messageable, text: str, max_len: int = DISCORD_MESSAGE_LIMIT) -> None:
    if len(text) <= max_len:
        await channel.send(text)
        return
    lines = text.splitlines(keepends=True)
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) > max_len:
            if chunk.strip():
                await channel.send(chunk.strip())
            chunk = ""
        chunk += line
    if chunk.strip():
        await channel.send(chunk.strip())