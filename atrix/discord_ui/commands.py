"""DM command router — !-prefixed commands the user can send in DMs."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord

import atrix.config as config
import atrix.memory.store as store
from atrix.ai import brain
from atrix.ai.providers import list_providers
from atrix.behavior import recap as recap_mod
from atrix.behavior import timing
from atrix.behavior import mood
from atrix.integrations import voice

logger = logging.getLogger("atrix.discord_ui.commands")


async def _send(channel, text: str):
    if len(text) <= 1900:
        await channel.send(text)
        return
    for i in range(0, len(text), 1900):
        await channel.send(text[i:i + 1900])


async def handle_command(
    message: discord.Message,
    user_id: str,
    user_text: str,
    *,
    proactive_engine=None,
) -> bool:
    """Handle ``!command``-style messages. Returns True if a command ran."""
    if not user_text.startswith("!"):
        return False

    parts = user_text.split()
    cmd = parts[0].lower()
    logger.info("[Command from @%s (%s)]: %s", message.author.name, user_id, user_text)

    if cmd in ("!rewind", "!undo"):
        count = 1
        if len(parts) > 1 and parts[1].isdigit():
            count = max(1, min(int(parts[1]), 30))
        me_id = message.channel.me.id if hasattr(message.channel, "me") else None
        deleted_discord = 0
        try:
            bot_msgs = []
            async for m in message.channel.history(limit=80):
                if me_id is not None and m.author.id == me_id:
                    bot_msgs.append(m)
                if len(bot_msgs) >= count:
                    break
            for bm in bot_msgs:
                try:
                    await bm.delete()
                    deleted_discord += 1
                    await asyncio.sleep(0.2)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Discord rewind delete failed: %s", exc)
        db_deleted = store.rewind_messages(user_id, count)
        confirm = await message.channel.send(
            f"↺ Rewound last {count} bot response(s) — {db_deleted} messages forgotten from memory."
        )
        await asyncio.sleep(4)
        try:
            await confirm.delete()
        except Exception:
            pass
        return True

    if cmd == "!memory":
        facts = store.format_memory_summary(user_id)
        user_tz = store.get_user_timezone(user_id)
        full = (
            f"**[What I remember about you]:**\n"
            f"- Your timezone: `{user_tz}`\n"
            f"- Your local time: {store.format_user_time_str(user_id)}\n\n"
            f"{facts}"
        )
        await _send(message.channel, full)
        return True

    if cmd in ("!clear", "!wipe", "!reset"):
        deleted = store.clear_chat_history(user_id)
        confirm = await message.channel.send(
            f"✓ Chat history cleared ({deleted} messages forgotten). Profile facts preserved."
        )
        await asyncio.sleep(4)
        try:
            await confirm.delete()
        except Exception:
            pass
        return True

    if cmd == "!forget":
        term = " ".join(parts[1:]).strip()
        if not term:
            await message.channel.send("Usage: `!forget <topic/fact>` (e.g. `!forget gaming` or `!forget age`)")
            return True
        deleted = store.delete_fact(user_id, term)
        if deleted:
            confirm = await message.channel.send(f"✓ Forgot: {', '.join(deleted)}")
        else:
            confirm = await message.channel.send(f"Couldn't find any facts matching '{term}'.")
        await asyncio.sleep(4)
        try:
            await confirm.delete()
        except Exception:
            pass
        return True

    if cmd in ("!tz", "!timezone"):
        if len(parts) < 2:
            current_tz = store.get_user_timezone(user_id)
            await message.channel.send(
                f"Current timezone: `{current_tz}`\n"
                "Usage: `!tz <Timezone>` (e.g. `!tz Europe/Bucharest` or `!tz US/Eastern`)"
            )
            return True
        new_tz = parts[1].strip()
        if store.set_user_timezone(user_id, new_tz):
            await message.channel.send(
                f"✓ Timezone set to `{new_tz}`. Local time: {store.format_user_time_str(user_id)}"
            )
        else:
            await message.channel.send(
                f"Invalid timezone `{new_tz}`. Examples: `Europe/Bucharest`, `Europe/London`, "
                "`America/New_York`, `Asia/Tokyo`."
            )
        return True

    if cmd == "!status":
        user_tz = store.get_user_timezone(user_id)
        local_time = store.format_user_time_str(user_id)
        msg_count = len(store.get_recent_messages(user_id, limit=500))
        facts = store.get_all_facts(user_id)
        pref_count = len(facts.get("preferences", {}))
        avg_len = store.get_user_average_message_length(user_id)
        avg_line = f"\n- Avg message length: `{avg_len:.0f} chars`" if avg_len else ""
        provider_line = f"\n- LLM provider: `{config.LLM_PROVIDER}`"
        from atrix.ai import languages as lang_mod
        lang_line = f"\n- {lang_mod.status_line(user_id)}"
        await message.channel.send(
            "**[Atrix Status]**\n"
            f"- Timezone: `{user_tz}`\n"
            f"- Local Time: `{local_time}`\n"
            f"- Messages in Context: `{msg_count}`\n"
            f"- Learned Preferences: `{pref_count}`"
            f"{avg_line}{provider_line}{lang_line}\n"
            "- Proactive Checks: `Active`\n"
            f"- Voice: {'enabled' if config.OPENAI_API_KEY else 'disabled (set OPENAI_API_KEY)'}"
        )
        return True

    if cmd == "!voice":
        await message.channel.send(voice.status_line())
        return True

    if cmd in ("!provider", "!providers", "!llm"):
        try:
            current = brain.get_provider()
            current_line = f"Active: **{current.name}**"
        except Exception as exc:
            current_line = f"Active: **error** ({exc})"
        available = "\n".join(
            f"• `{p['id']}` — {p['label']} (default model: `{p['default_model']}`, env: `{p['key_env']}`)"
            for p in list_providers()
        )
        await message.channel.send(
            f"**[LLM Provider]**\n{current_line}\n\n"
            f"**Available providers** (set `LLM_PROVIDER` in `.env`):\n{available}\n\n"
            f"Free-tier picks: `gemini`, `groq`, `openrouter` (some models)."
        )
        return True

    if cmd == "!mood":
        recent = store.get_recent_mood(user_id, days=7)
        if not recent:
            await message.channel.send("not enough data yet — chat more and I'll pick up on it")
            return True
        avg = sum(r["score"] for r in recent) / len(recent)
        label = mood.classify(avg)
        today_avg = store.get_today_mood_average(user_id)
        today_line = f"\n- Today so far: {mood.classify(today_avg)} ({today_avg:+.2f})" if today_avg is not None else ""
        await message.channel.send(
            f"**[Mood — last 7 days]**\n- Overall vibe: {label} ({avg:+.2f})\n- Samples: {len(recent)}{today_line}"
        )
        return True

    if cmd == "!topics":
        top = store.get_top_topics(user_id, limit=10)
        if not top:
            await message.channel.send("no topics tracked yet — chat more")
            return True
        lines = "\n".join(f"• {r['topic']} (×{r['count']})" for r in top)
        await message.channel.send(f"**[Topics you talk about most]**\n{lines}")
        return True

    if cmd in ("!petname", "!nickname"):
        nick = store.get_pet_name(user_id)
        if nick:
            await message.channel.send(f"i call you **{nick}** rn. want me to change it? reply with the new one.")
        else:
            await message.channel.send("i haven't picked a nickname for you yet — give me a few more convos")
        return True

    if cmd == "!schedule":
        hours = timing.get_learned_active_hours(user_id)
        await message.channel.send(timing.format_for_prompt(user_id) or "no learned schedule yet")
        return True

    if cmd == "!recap":
        text = await recap_mod.generate_recap(user_id)
        if text:
            await message.channel.send(text)
        else:
            await message.channel.send("nothing to recap rn — chat more and check back tonight")
        return True

    if cmd == "!persona":
        from atrix.ai import persona as persona_mod
        data = persona_mod.get_user_persona(user_id)
        lines = "\n".join(
            f"• **{k.replace('_', ' ')}**: {v}" for k, v in data.items() if v
        )
        await message.channel.send(
            f"**[My current vibe, just for you]**\n{lines}\n\n"
            f"_this is your personal Atrix — every friend who DMs me gets their own version_"
        )
        return True

    if cmd in ("!relate", "!feelings", "!vibe"):
        from atrix.behavior import relational as relational_mod
        state = store.get_relationship_state(user_id)
        if not state:
            await message.channel.send("we haven't talked enough yet for me to know how i feel about you, chat more")
            return True
        trajectory = state.get("trajectory", "stable")
        last_msg = state.get("last_user_msg_excerpt", "")
        recent = store.get_recent_relationship_events(user_id, limit=6)
        arc = "\n".join(
            f"  {('+' if e['impact'] > 0.05 else '-' if e['impact'] < -0.05 else '·')} "
            f"{e['category'].replace('_', ' ')} (impact {e['impact']:+.2f})"
            for e in reversed(recent)
        )
        last_msg_block = f"\n- Your last message: \"{last_msg[:120]}\"" if last_msg else ""
        await message.channel.send(
            f"**[How I'm feeling toward you]**\n"
            f"> {state.get('state_description', '')}\n"
            f"\n- Trajectory: `{trajectory}`"
            f"{last_msg_block}\n\n"
            f"**Recent arc (oldest to newest):**\n{arc}"
        )
        return True

    if cmd in ("!snapshot", "!view"):
        from atrix.ai import persona as persona_mod
        persona = persona_mod.get_user_persona(user_id)
        facts = store.get_all_facts(user_id)
        state = store.get_relationship_state(user_id) or {}
        profile = facts.get("profile", {})
        state_facts = facts.get("state", {})
        mood_history = store.get_recent_mood(user_id, days=7)
        topics = store.get_top_topics(user_id, limit=5)
        pet_name = store.get_pet_name(user_id)
        active_hours = timing.get_learned_active_hours(user_id)
        avg_len = store.get_user_average_message_length(user_id)
        msg_count = len(store.get_recent_messages(user_id, limit=1000))

        # --- Her current vibe (per-user persona) ---
        g = persona
        persona_lines = [
            f"• name: **{g.get('name', 'Atrix')}**",
            f"• current obsession: {g.get('current_obsession', '—')}",
            f"• current mood: {g.get('current_mood', '—')}",
            f"• current annoyance: {g.get('current_annoyance', '—')}",
            f"• recent win: {g.get('recent_win', '—')}",
        ]

        # --- What she's learned about you ---
        about_lines = []
        if profile.get("name"):    about_lines.append(f"• name: {profile['name']}")
        if profile.get("gender"):  about_lines.append(f"• gender: {profile['gender']}")
        if profile.get("timezone"): about_lines.append(f"• timezone: {profile['timezone']}")
        if profile.get("location"): about_lines.append(f"• location: {profile['location']}")
        if profile.get("age"):     about_lines.append(f"• age: {profile['age']}")
        if pet_name:                 about_lines.append(f"• pet name i picked: **{pet_name}**")
        if avg_len:                  about_lines.append(f"• your avg message length: {avg_len:.0f} chars")
        if active_hours:
            about_lines.append(f"• your active hours: {', '.join(f'{h:02d}h' for h in active_hours)}")
        if topics:
            about_lines.append(f"• topics you talk about most: {', '.join(t['topic'] for t in topics[:5])}")

        # Count facts by category.
        category_counts = {k: len(v) for k, v in facts.items() if v}
        if category_counts:
            about_lines.append(
                f"• things I remember: {sum(category_counts.values())} facts across "
                f"{len(category_counts)} categories ({', '.join(f'{k} ({v})' for k, v in category_counts.items())})"
            )
        if msg_count:
            about_lines.append(f"• messages we've exchanged: {msg_count}+")

        # --- How she's feeling right now ---
        feeling_block = state.get("state_description", "neutral, just chatting") if state else "neutral, just chatting"
        trajectory = state.get("trajectory", "stable")
        last_topic = state_facts.get("current_topic", "general")
        engagement = state_facts.get("engagement", "medium")

        # Mood trend.
        if mood_history:
            avg_mood = sum(m["score"] for m in mood_history) / len(mood_history)
            mood_label = (
                "rough" if avg_mood < -0.45 else
                "low"  if avg_mood < -0.15 else
                "neutral" if avg_mood < 0.15 else
                "good"  if avg_mood < 0.45 else "great"
            )
            mood_line = f"your mood lately: {mood_label} ({avg_mood:+.2f}, {len(mood_history)} samples)"
        else:
            mood_line = "your mood: not enough data yet"

        text = (
            "**Atrix's view of you**\n\n"
            "**Her current vibe (just for you):**\n"
            + "\n".join(persona_lines) +
            "\n\n**What she's learned about you:**\n"
            + ("\n".join(about_lines) if about_lines else "• nothing yet — chat more") +
            f"\n\n**How she's feeling toward you right now:**\n"
            f"> {feeling_block}\n"
            f"\n- trajectory: `{trajectory}`"
            f"\n- what you're talking about: `{last_topic}`"
            f"\n- your engagement: `{engagement}`"
            f"\n- {mood_line}"
        )
        await _send(message.channel, text)
        return True

    if cmd in ("!lang", "!language"):
        from atrix.ai import languages as lang_mod
        if len(parts) < 2:
            await message.channel.send(
                f"{lang_mod.status_line(user_id)}\n\n{lang_mod.list_for_user()}"
            )
            return True
        target = parts[1].strip().lower()
        if lang_mod.set_user_language(user_id, target):
            if target == "auto":
                await message.channel.send("✓ Language reset to auto-detect.")
            else:
                await message.channel.send(f"✓ Language set to **{lang_mod.LANGUAGES.get(target, target)}**.")
        else:
            await message.channel.send(
                f"Unknown language `{target}`. {lang_mod.list_for_user()}"
            )
        return True

    if cmd == "!pingme":
        await message.channel.send("testing proactive ping in 5 seconds...")
        await asyncio.sleep(5)
        if proactive_engine is not None:
            await proactive_engine._send_proactive_dm(user_id, "test_checkin", "testing proactive ping")
        else:
            await message.channel.send("(proactive engine not running)")
        return True

    if cmd == "!help":
        await message.channel.send(
            "**[Atrix Commands]**\n"
            "• `!rewind [N]` — rewind last N bot responses\n"
            "• `!memory` — view what I remember about you\n"
            "• `!forget <topic>` — remove a specific fact\n"
            "• `!clear` — wipe chat history (keeps profile facts)\n"
            "• `!tz <timezone>` — set your timezone\n"
            "• `!status` — bot status, local time, voice availability\n"
            "• `!voice` — show whether voice transcription is enabled\n"
            "• `!provider` — show current LLM provider\n"
            "• `!mood` — show your mood trend over the last week\n"
            "• `!topics` — show topics you talk about most\n"
            "• `!petname` — show the nickname I picked for you\n"
            "• `!schedule` — show your learned active hours\n"
            "• `!recap` — generate a one-line recap of today\n"
            "• `!persona` — see the current Atrix vibe (your personalized version)\n"
            "• `!relate` — see how I'm currently feeling toward you + recent arc\n"
            "• `!snapshot` — one-page view of everything I know about you\n"
            "• `!lang [code]` — show or set your language (`en`/`hi`/`ne`/`tr`/`ro`/`auto`)\n"
            "• `!pingme` — test proactive messaging with an immediate ping"
        )
        return True

    return False