"""Proactive DM engine — the part that messages the user first.

Every 2 minutes we sweep every user and decide whether to drop a casual
line: scheduled event follow-up, bedtime recap, or spontaneous ping.
Hardcoded time windows are gone; the timing engine decides when.
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from typing import Optional

import discord
from discord.ext import tasks

import atrix.ai.brain as brain
import atrix.config as config
import atrix.memory.store as store
from atrix.behavior import recap, timing
from atrix.behavior.reach_out import should_reach_out

logger = logging.getLogger("atrix.proactive")

# Probabilistic caps so we don't ping every user every 2 minutes.
SPONTANEOUS_PING_PROBABILITY = 0.25

# Recap pings only fire within this window around the user's typical bedtime.
RECAP_WINDOW_START = 22
RECAP_WINDOW_END = 2

# Proactive-message pacing: real humans pause between burst messages.
PROACTIVE_TYPING_MIN = 1.4
PROACTIVE_TYPING_MAX = 3.8
PROACTIVE_PAUSE_MIN = 1.0
PROACTIVE_PAUSE_MAX = 2.0

# Self-correction follow-up probability after a proactive message.
SELF_CORRECTION_PROBABILITY = 0.08
SELF_CORRECTION_DELAY_MIN = 3.0
SELF_CORRECTION_DELAY_MAX = 6.0

# (reason, context) tuples for spontaneous pings.
_SPONTANEOUS_IDEAS = [
    ("sharing_random_thought", "share a random funny realization or observation"),
    ("relatable_complaint", "talk about being exhausted or having a cooked sleep schedule"),
    ("casual_vibe", "spontaneous casual check-in or wondering if they're alive"),
    ("evening_chill", "asking what they're up to or sharing a late-night random thought"),
    ("topic_match", "bring up something from their favorite topics that just popped into your head"),
]


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


class ProactiveEngine:
    def __init__(self, bot: discord.Client):
        self.bot = bot
        self.loop = tasks.loop(minutes=2)(self._check_all_users)

    def start(self) -> None:
        if not self.loop.is_running():
            self.loop.start()
            logger.info("Proactive engine started (sweep every 2 minutes).")

    def stop(self) -> None:
        if self.loop.is_running():
            self.loop.stop()

    async def _check_all_users(self) -> None:
        user_ids = store.get_all_active_users()
        if config.ALLOWED_USER_ID and config.ALLOWED_USER_ID not in user_ids:
            user_ids.append(config.ALLOWED_USER_ID)
        for uid in user_ids:
            # Defensive: skip anything that isn't a valid Discord snowflake.
            if not uid.isdigit():
                logger.debug("Skipping non-snowflake user id: %s", uid)
                continue
            try:
                user_now = store.get_user_current_time(uid)
                await self._evaluate_user(uid, user_now)
            except Exception as exc:
                logger.error("Proactive sweep failed for %s: %s", uid, exc, exc_info=True)

    async def _evaluate_user(self, user_id: str, now: datetime) -> None:
        state = store.get_user_state(user_id)
        if state.get("pending_reply", 0) == 1:
            return

        if timing.should_skip_for_recent_activity(user_id):
            return
        if not timing.is_within_active_window(user_id, now):
            return

        # 1. Scheduled events (always fire if due — these are time-sensitive).
        now_iso = now.isoformat()
        due_events = store.get_due_events(user_id, now_iso)
        if due_events:
            event = due_events[0]
            store.mark_event_triggered(user_id, event["id"])
            await self._send_proactive_dm(user_id, "event_followup", event["description"])
            return

        # 2. Bedtime recap (once per day, in the recap window).
        today_key = now.strftime("%Y-%m-%d")
        recap_sent_flag = state.get(f"recap_sent_{today_key}")
        recap_hour_ok = now.hour >= RECAP_WINDOW_START or now.hour < RECAP_WINDOW_END
        if recap_hour_ok and not recap_sent_flag:
            recap_text = await recap.generate_recap(user_id)
            if recap_text:
                store.update_user_state(user_id, **{f"recap_sent_{today_key}": 1})
                await self._send_proactive_dm(
                    user_id, "bedtime_recap", recap_text, raw_text=recap_text,
                )
                return

        # 3. Spontaneous check-in.
        last_ping_time = _parse_iso(state.get("last_proactive_ping_at"))
        hours_since_ping = (
            (now - last_ping_time).total_seconds() / 3600
            if last_ping_time else 168.0
        )
        if hours_since_ping >= 4.0 and random.random() < SPONTANEOUS_PING_PROBABILITY:
            reason, context = random.choice(_SPONTANEOUS_IDEAS)
            await self._send_proactive_dm(user_id, reason, context)
            return

        # 4. LLM-driven reach-out when the user has been unusually quiet.
        # The LLM decides whether to reach out, what style, and what to say.
        # No hardcoded "if 7 days silent then ping" rules — it reasons from
        # the user's typical pattern, relationship state, and context.
        last_user_time = _parse_iso(state.get("last_user_msg_time"))
        if last_user_time:
            hours_since_user = (now - last_user_time).total_seconds() / 3600
            if hours_since_user >= 6.0 and hours_since_ping >= 12.0:
                decision = await should_reach_out(user_id)
                if decision and decision.get("message"):
                    logger.info("LLM reach-out to %s (%.1fh quiet, style=%s): %s",
                                user_id, decision["quiet_hours"], decision["style"],
                                decision["message"][:80])
                    await self._send_proactive_dm(
                        user_id,
                        f"reach_out:{decision['style']}",
                        decision["reason"] or decision["style"],
                        raw_text=decision["message"],
                    )
                    return

    async def _send_proactive_dm(
        self, user_id: str, reason: str, context: str = "", *,
        raw_text: Optional[str] = None,
    ) -> None:
        try:
            user = await self.bot.fetch_user(int(user_id))
            if not user:
                return
            dm_channel = user.dm_channel or await user.create_dm()

            if raw_text:
                messages = [raw_text]
            else:
                messages = await brain.generate_proactive_message(user_id, reason, context)
            if not messages:
                return

            for idx, msg in enumerate(messages):
                typing_seconds = min(
                    max(len(msg) * 0.05, PROACTIVE_TYPING_MIN),
                    PROACTIVE_TYPING_MAX,
                )
                async with dm_channel.typing():
                    await asyncio.sleep(typing_seconds)
                await dm_channel.send(msg)
                store.add_message(user_id, "assistant", msg)
                logger.info("Proactive DM -> @%s [%s]: %s", user.name, reason, msg)
                if idx < len(messages) - 1:
                    await asyncio.sleep(random.uniform(PROACTIVE_PAUSE_MIN, PROACTIVE_PAUSE_MAX))

                # Self-correction follow-up.
                if idx == len(messages) - 1 and random.random() < SELF_CORRECTION_PROBABILITY:
                    await asyncio.sleep(random.uniform(
                        SELF_CORRECTION_DELAY_MIN, SELF_CORRECTION_DELAY_MAX,
                    ))
                    correction = await brain.generate_self_correction(user_id, msg)
                    if correction:
                        async with dm_channel.typing():
                            await asyncio.sleep(min(max(len(correction) * 0.05, 0.8), 2.0))
                        await dm_channel.send(correction)
                        store.add_message(user_id, "assistant", correction)
                        logger.info("Self-correction follow-up -> @%s: %s", user.name, correction)

            now_iso = store.get_user_current_time(user_id).isoformat()
            store.update_user_state(
                user_id,
                last_bot_msg_time=now_iso,
                last_proactive_ping_at=now_iso,
                pending_reply=1,
            )
            logger.info("Proactive message sent to %s (reason: %s)", user_id, reason)
        except Exception as exc:
            logger.error("Failed to send proactive DM to %s: %s", user_id, exc)