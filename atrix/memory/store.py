"""Per-user SQLite memory store.

Each Discord user gets their own ``.db`` file under ``data/users/`` so memories
physically cannot leak across users. This module owns:

* Schema setup and migration.
* Message history, facts, scheduled events, user state.
* Mood, topic, engagement, and summary tracking.
* Pruning helpers that decay stale data (see ``pruning.py``).
"""
from __future__ import annotations

import glob
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz

import atrix.config as config
from atrix.utils.text import clean_model_output

logger = logging.getLogger("atrix.memory")

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
USERS_DIR = os.path.abspath(os.path.join(DB_DIR, "users"))
os.makedirs(USERS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        summary TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        last_referenced DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(category, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scheduled_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        target_time TEXT NOT NULL,
        reminder_type TEXT DEFAULT 'followup',
        triggered INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        last_user_msg_time TEXT,
        last_bot_msg_time TEXT,
        last_proactive_ping_at TEXT,
        last_routine_day TEXT,
        pending_reply INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        date_str TEXT NOT NULL,
        hour INTEGER NOT NULL,
        minute INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(event_type, date_str)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mood_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        score REAL NOT NULL,
        sample_text TEXT,
        date_str TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS topic_frequency (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL UNIQUE,
        count INTEGER DEFAULT 1,
        last_mentioned DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS engagement_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bot_message_id INTEGER,
        style_tags TEXT,
        had_followup INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_str TEXT NOT NULL UNIQUE,
        summary_text TEXT NOT NULL,
        message_count INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS message_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id_marker TEXT,
        summary_text TEXT NOT NULL,
        message_count INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # Per-user persona state. Each user sees their own version of Atrix —
    # her current obsession, mood, etc. evolve independently based on
    # the conversations she has with that specific friend.
    """
    CREATE TABLE IF NOT EXISTS persona_state (
        field TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # Append-only log of every user message's emotional impact, judged by
    # an LLM. Used to build the relationship state over time.
    """
    CREATE TABLE IF NOT EXISTS relationship_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        impact REAL NOT NULL,
        category TEXT NOT NULL,
        sincere INTEGER DEFAULT 1,
        inside_joke INTEGER DEFAULT 0,
        reasoning TEXT,
        sample_text TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # The current relationship state, written by the LLM in free-form text
    # from Atrix's first-person perspective. Single row per user.
    """
    CREATE TABLE IF NOT EXISTS relationship_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        state_description TEXT NOT NULL,
        trajectory TEXT,
        last_user_msg_excerpt TEXT,
        last_user_msg_at TEXT,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def init_user_db(conn: sqlite3.Connection) -> None:
    """Create all tables for a brand-new user, then migrate an existing one.

    Safe to call on a fresh DB or on one that predates the current schema.
    """
    with conn:
        for stmt in _SCHEMA:
            conn.execute(stmt)
    _migrate_schema(conn)


def _column_names(conn: sqlite3.Connection, table: str) -> set:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}
    except Exception:
        return set()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Idempotent migration for pre-existing user databases.

    Older user DBs may be missing entire tables (added in later schema
    revisions) or new columns on existing tables. This runs every
    connection and only adds what's missing.
    """
    # 1. Add any missing tables (CREATE TABLE IF NOT EXISTS is a no-op if
    #    the table is already there, so this is safe to run on every
    #    open).
    with conn:
        for stmt in _SCHEMA:
            conn.execute(stmt)

    # 2. Add any missing columns to existing tables.
    column_migrations: list[tuple[str, str, str]] = [
        # (table, column, definition)
        ("messages", "summary", "TEXT"),
        ("facts", "weight", "REAL DEFAULT 1.0"),
        ("facts", "last_referenced", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ]
    with conn:
        for table, column, definition in column_migrations:
            existing = _column_names(conn, table)
            if column not in existing:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                    logger.info("Migrated %s: added column %s", table, column)
                except Exception as exc:
                    logger.debug("Migration skipped for %s.%s: %s", table, column, exc)


def get_db(user_id: str) -> sqlite3.Connection:
    """Open (and if necessary initialize + migrate) the user's database."""
    safe_uid = re.sub(r"[^a-zA-Z0-9_\-]", "", str(user_id))
    db_file = os.path.join(USERS_DIR, f"{safe_uid}.db")
    is_new = not os.path.exists(db_file)
    conn = sqlite3.connect(db_file, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if is_new:
        init_user_db(conn)
    else:
        # Pre-existing DB: ensure the latest schema and any missing columns.
        # CREATE TABLE IF NOT EXISTS is a no-op for tables that already
        # exist, so this is safe to run unconditionally.
        _migrate_schema(conn)
    return conn


def migrate_all_user_dbs() -> int:
    """Run schema migration on every existing user DB at startup.

    Returns the number of DBs that were processed.
    """
    db_files = glob.glob(os.path.join(USERS_DIR, "*.db"))
    migrated = 0
    for path in db_files:
        uid = os.path.splitext(os.path.basename(path))[0]
        if not uid or not uid.isdigit():
            continue
        try:
            conn = sqlite3.connect(path)
            _migrate_schema(conn)
            conn.close()
            migrated += 1
        except Exception as exc:
            logger.warning("Migration failed for %s: %s", uid, exc)
    return migrated


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def add_message(user_id: str, role: str, content: str, summary: Optional[str] = None) -> None:
    cleaned = clean_model_output(content)
    if not cleaned:
        return
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO messages (role, content, timestamp, summary) VALUES (?, ?, ?, ?)",
                (role, cleaned, datetime.utcnow().isoformat(), summary),
            )
    finally:
        conn.close()


def get_recent_messages(user_id: str, limit: int = 100) -> List[Dict[str, str]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    cleaned: List[Dict[str, str]] = []
    for row in reversed(rows):
        content = row["content"]
        if not content or looks_like_residue(content):
            continue
        cleaned.append({"role": row["role"], "content": content})
    return cleaned


def looks_like_residue(text: str) -> bool:
    upper = text.upper()
    if "DSML" in upper or "<invoke" in text or "<parameter" in text:
        return True
    return False


def rewind_messages(user_id: str, bot_count: int = 1) -> int:
    bot_count = max(1, min(bot_count, 50))
    conn = get_db(user_id)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM messages
                WHERE role = 'assistant'
                ORDER BY id DESC
                LIMIT 1 OFFSET ?
                """,
                (bot_count,),
            )
            row = cursor.fetchone()
            if row:
                keep_id = row["id"]
                cursor.execute("SELECT COUNT(*) AS c FROM messages WHERE id > ?", (keep_id,))
                del_count = cursor.fetchone()["c"]
                cursor.execute("DELETE FROM messages WHERE id > ?", (keep_id,))
                return del_count
            cursor.execute("SELECT COUNT(*) AS c FROM messages")
            del_count = cursor.fetchone()["c"]
            cursor.execute("DELETE FROM messages")
            return del_count
    finally:
        conn.close()


def clear_chat_history(user_id: str) -> int:
    conn = get_db(user_id)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS c FROM messages")
            del_count = cursor.fetchone()["c"]
            cursor.execute("DELETE FROM messages")
            return del_count
    finally:
        conn.close()


def get_user_average_message_length(user_id: str, lookback: int = 50) -> Optional[float]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content FROM messages WHERE role = 'user' ORDER BY id DESC LIMIT ?",
            (lookback,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    lengths: List[int] = []
    for row in rows:
        content = (row["content"] or "").strip()
        if not content or content.startswith("["):
            continue
        lengths.append(len(content))
    if not lengths:
        return None
    return sum(lengths) / len(lengths)


# ---------------------------------------------------------------------------
# Facts (long-term memory)
# ---------------------------------------------------------------------------

def delete_fact(user_id: str, pattern: str) -> List[str]:
    conn = get_db(user_id)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT category, key FROM facts WHERE key LIKE ? OR category LIKE ? OR value LIKE ?",
                (f"%{pattern}%", f"%{pattern}%", f"%{pattern}%"),
            )
            matches = cursor.fetchall()
            deleted: List[str] = []
            for row in matches:
                cat, key = row["category"], row["key"]
                cursor.execute("DELETE FROM facts WHERE category = ? AND key = ?", (cat, key))
                deleted.append(f"{cat}.{key}")
            return deleted
    finally:
        conn.close()


def save_fact(user_id: str, category: str, key: str, value: str) -> None:
    """Insert or update a fact.

    Defensive against schema drift: tries the full schema (with weight and
    last_referenced) first, then falls back to the basic 4-column schema if
    the new columns are missing.
    """
    conn = get_db(user_id)
    try:
        with conn:
            try:
                conn.execute(
                    """
                    INSERT INTO facts (category, key, value, weight, last_referenced, updated_at)
                    VALUES (?, ?, ?, 1.0, ?, ?)
                    ON CONFLICT(category, key) DO UPDATE SET
                        value = excluded.value,
                        weight = MIN(2.0, weight + 0.2),
                        last_referenced = excluded.last_referenced,
                        updated_at = excluded.updated_at
                    """,
                    (
                        category.strip().lower(),
                        key.strip().lower(),
                        value.strip(),
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat(),
                    ),
                )
                return
            except Exception:
                pass
            conn.execute(
                """
                INSERT INTO facts (category, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (
                    category.strip().lower(),
                    key.strip().lower(),
                    value.strip(),
                    datetime.utcnow().isoformat(),
                ),
            )
    finally:
        conn.close()


def touch_fact(user_id: str, category: str, key: str) -> None:
    """Mark a fact as recently referenced (used to prevent pruning)."""
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                """
                UPDATE facts
                SET last_referenced = ?, weight = MIN(2.0, weight + 0.05)
                WHERE category = ? AND key = ?
                """,
                (datetime.utcnow().isoformat(), category, key),
            )
    finally:
        conn.close()


def get_all_facts(user_id: str) -> Dict[str, Dict[str, str]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT category, key, value FROM facts ORDER BY category, key")
        rows = cursor.fetchall()
    finally:
        conn.close()

    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        cat = row["category"]
        result.setdefault(cat, {})[row["key"]] = row["value"]
    return result


def get_all_facts_with_weight(user_id: str) -> List[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT category, key, value, weight, last_referenced, updated_at "
            "FROM facts ORDER BY weight DESC, updated_at DESC"
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

def parse_custom_timezone(tz_val: str) -> Optional[pytz.BaseTzInfo]:
    if not tz_val:
        return None
    val = tz_val.strip()
    try:
        return pytz.timezone(val)
    except Exception:
        pass

    abbr_map = {
        "EST": "America/New_York", "EDT": "America/New_York",
        "CST": "America/Chicago", "CDT": "America/Chicago",
        "MST": "America/Denver", "MDT": "America/Denver",
        "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
        "CET": "Europe/Berlin", "CEST": "Europe/Berlin",
        "EET": "Europe/Bucharest", "EEST": "Europe/Bucharest",
        "BST": "Europe/London",
        "GMT": "UTC", "UTC": "UTC",
        "ROMANIA": "Europe/Bucharest",
    }
    if val.upper() in abbr_map:
        try:
            return pytz.timezone(abbr_map[val.upper()])
        except Exception:
            pass

    offset_match = re.search(r'(?:UTC|GMT)?\s*([+-]\d{1,2})(?::?(\d{2}))?', val, re.I)
    if offset_match:
        hours = int(offset_match.group(1))
        minutes = int(offset_match.group(2) or 0)
        total_minutes = -(hours * 60 + minutes)
        return pytz.FixedOffset(total_minutes)
    return None


def set_user_timezone(user_id: str, tz_name: str) -> bool:
    parsed = parse_custom_timezone(tz_name)
    if not parsed:
        return False
    save_fact(user_id, "profile", "timezone", tz_name.strip())
    return True


def get_user_timezone(user_id: str) -> pytz.BaseTzInfo:
    facts = get_all_facts(user_id)
    profile = facts.get("profile", {})
    tz_val = profile.get("timezone") or profile.get("location")
    if tz_val:
        parsed = parse_custom_timezone(tz_val)
        if parsed:
            return parsed
    return config.USER_TIMEZONE


def get_user_current_time(user_id: str) -> datetime:
    return datetime.now(get_user_timezone(user_id))


def format_user_time_str(user_id: str) -> str:
    return get_user_current_time(user_id).strftime("%A, %B %d, %Y - %I:%M %p")


# ---------------------------------------------------------------------------
# Memory summary (used in system prompt)
# ---------------------------------------------------------------------------

_CATEGORY_LABELS = {
    "profile": "Basic Info & Timezone",
    "routine": "Daily Routine & Schedule",
    "interests": "Hobbies, Games & Interests",
    "social": "People, Friends & Relationships",
    "preferences": "Communication & People Preferences",
    "speech_style": "Natural Speaking Style & Slang",
    "memes": "Inside Jokes & Memes We Share",
    "notes": "Personal Facts & Habits",
}


def format_memory_summary(user_id: str) -> str:
    facts = get_all_facts(user_id)
    profile = facts.get("profile", {})
    gender = profile.get("gender")
    name = profile.get("name")

    lines: List[str] = []
    meta: List[str] = []
    if gender:
        meta.append(f"Gender: {gender}")
    else:
        meta.append(
            "Gender: UNKNOWN (subtly ask during a natural lull — "
            "'btw are you a guy or a girl lol' — NEVER hijack an active topic)",
        )
    if name:
        meta.append(f"Name/Nickname: {name}")
    if meta:
        lines.append("[Friend Profile Overview]\n" + "\n".join(f"- {m}" for m in meta))

    for cat, items in facts.items():
        label = _CATEGORY_LABELS.get(cat, cat.capitalize())
        item_strs = [
            f"- {k.replace('_', ' ')}: {v}"
            for k, v in items.items()
            if k not in ("gender", "name")
        ]
        if item_strs:
            lines.append(f"[{label}]\n" + "\n".join(item_strs))

    if not lines:
        return "No specific long-term notes remembered yet. Gender is currently UNKNOWN."
    return "\n\n".join(lines)


def format_vibe_and_style_guidance(user_id: str) -> str:
    facts = get_all_facts(user_id)
    prefs = facts.get("preferences", {})
    style = facts.get("speech_style", {})

    lines: List[str] = []
    if prefs:
        items = [f"- {k.replace('_', ' ')}: {v}" for k, v in prefs.items()]
        lines.append("WHAT THIS USER PREFERS IN PEOPLE & CONVERSATION:\n" + "\n".join(items))
    if style:
        items = [f"- {k.replace('_', ' ')}: {v}" for k, v in style.items()]
        lines.append("HOW THIS USER TEXTS (MIRROR THEIR CADENCE & ENERGY):\n" + "\n".join(items))

    if not lines:
        return (
            "[DYNAMIC PER-USER VIBE GUIDANCE]:\n"
            "- Baseline: grounded, authentic, low-key Discord friend.\n"
            "- Observe how this user texts and match their energy.\n"
            "- If they share preferences, validate them naturally and adapt silently."
        )
    return (
        "[DYNAMIC PER-USER VIBE & STYLE GUIDANCE]:\n"
        + "\n\n".join(lines)
        + "\n- INSTRUCTION: naturally align your cadence with what this user likes. "
        + "Acknowledge their preferences as a peer and reflect them silently."
    )


# ---------------------------------------------------------------------------
# Scheduled events
# ---------------------------------------------------------------------------

def add_scheduled_event(user_id: str, description: str, target_time: str, reminder_type: str = "followup") -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO scheduled_events (description, target_time, reminder_type, triggered) "
                "VALUES (?, ?, ?, 0)",
                (description.strip(), target_time, reminder_type),
            )
    finally:
        conn.close()


def get_due_events(user_id: str, current_iso: str) -> List[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, description, target_time, reminder_type "
            "FROM scheduled_events WHERE triggered = 0 AND target_time <= ? "
            "ORDER BY target_time ASC",
            (current_iso,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def mark_event_triggered(user_id: str, event_id: int) -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute("UPDATE scheduled_events SET triggered = 1 WHERE id = ?", (event_id,))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Activity / wake-up tracking
# ---------------------------------------------------------------------------

def track_wake_up_consistency(user_id: str, message_text: str) -> Optional[str]:
    now = get_user_current_time(user_id)
    text_lower = message_text.lower().strip()
    is_morning_phrase = any(
        p in text_lower
        for p in ("morning", "gm", "just woke up", "woke up", "awake now", "good morning")
    )

    conn = get_db(user_id)
    try:
        with conn:
            cursor = conn.cursor()
            date_str = now.strftime("%Y-%m-%d")
            cursor.execute(
                "SELECT id FROM activity_logs WHERE date_str = ? AND event_type = 'wake_up'",
                (date_str,),
            )
            already = cursor.fetchone()
            if not already and (is_morning_phrase or (5 <= now.hour <= 14)):
                cursor.execute(
                    "INSERT INTO activity_logs (event_type, date_str, hour, minute) "
                    "VALUES ('wake_up', ?, ?, ?) "
                    "ON CONFLICT(event_type, date_str) DO NOTHING",
                    (date_str, now.hour, now.minute),
                )
            cursor.execute(
                "SELECT date_str, hour, minute FROM activity_logs "
                "WHERE event_type = 'wake_up' ORDER BY date_str DESC LIMIT 10"
            )
            logs = cursor.fetchall()
            if len(logs) >= 5:
                recent_5 = logs[:5]
                hours = [r["hour"] + r["minute"] / 60.0 for r in recent_5]
                min_h, max_h = min(hours), max(hours)
                if (max_h - min_h) <= 2.0:
                    avg_h = sum(hours) / len(hours)
                    h_int = int(avg_h)
                    m_int = int((avg_h - h_int) * 60)
                    ampm = "AM" if h_int < 12 else "PM"
                    display_h = h_int if 1 <= h_int <= 12 else (h_int - 12 if h_int > 12 else 12)
                    val = f"consistently wakes up around {display_h}:{m_int:02d} {ampm} (verified across 5+ days)"
                    cursor.execute(
                        "INSERT INTO facts (category, key, value, updated_at) "
                        "VALUES ('routine', 'verified_wake_up', ?, datetime('now')) "
                        "ON CONFLICT(category, key) DO UPDATE SET "
                        "value = excluded.value, updated_at = excluded.updated_at",
                        (val,),
                    )
                    return val
    finally:
        conn.close()
    return None


# ---------------------------------------------------------------------------
# User state (single-row, used by proactive engine)
# ---------------------------------------------------------------------------

_STATE_DEFAULTS = {
    "last_user_msg_time": None,
    "last_bot_msg_time": None,
    "last_proactive_ping_at": None,
    "last_routine_day": None,
    "pending_reply": 0,
}


def get_user_state(user_id: str) -> Dict[str, Any]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_state WHERE id = 1")
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {"id": 1, **_STATE_DEFAULTS}
    finally:
        conn.close()


def update_user_state(user_id: str, **kwargs: Any) -> None:
    allowed = {k: v for k, v in kwargs.items() if k in _STATE_DEFAULTS}
    conn = get_db(user_id)
    try:
        current = get_user_state(user_id)
        current.update(allowed)
        with conn:
            conn.execute(
                "INSERT INTO user_state "
                "(id, last_user_msg_time, last_bot_msg_time, last_proactive_ping_at, last_routine_day, pending_reply) "
                "VALUES (1, :last_user_msg_time, :last_bot_msg_time, :last_proactive_ping_at, :last_routine_day, :pending_reply) "
                "ON CONFLICT(id) DO UPDATE SET "
                "last_user_msg_time = excluded.last_user_msg_time, "
                "last_bot_msg_time = excluded.last_bot_msg_time, "
                "last_proactive_ping_at = excluded.last_proactive_ping_at, "
                "last_routine_day = excluded.last_routine_day, "
                "pending_reply = excluded.pending_reply",
                current,
            )
    finally:
        conn.close()


def get_all_active_users() -> List[str]:
    """List every user that has actually sent at least one message.

    Skips DB files that are missing the messages table, are empty, or
    don't look like real Discord snowflakes. Test fixtures (e.g.
    ``test_user_b`` or ``petname_test_*``) and brand-new inits are
    filtered out so the proactive engine never tries to DM them.
    """
    db_files = glob.glob(os.path.join(USERS_DIR, "*.db"))
    out: List[str] = []
    for p in db_files:
        uid = os.path.splitext(os.path.basename(p))[0]
        if not uid or not uid.isdigit():
            continue
        try:
            conn = sqlite3.connect(p)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            conn.close()
            if count == 0:
                continue
        except Exception:
            continue
        out.append(uid)
    return out


# ---------------------------------------------------------------------------
# Mood, topics, engagement, summaries, pet name
# ---------------------------------------------------------------------------

def record_mood(user_id: str, score: float, sample_text: str = "") -> None:
    score = max(-1.0, min(1.0, float(score)))
    now = get_user_current_time(user_id)
    date_str = now.strftime("%Y-%m-%d")
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO mood_entries (score, sample_text, date_str) VALUES (?, ?, ?)",
                (score, sample_text[:200], date_str),
            )
    finally:
        conn.close()


def get_recent_mood(user_id: str, days: int = 7) -> List[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT score, sample_text, date_str, created_at "
            "FROM mood_entries ORDER BY created_at DESC LIMIT 200"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    cutoff = (get_user_current_time(user_id) - timedelta(days=days)).isoformat()
    return [dict(r) for r in reversed(rows) if r["created_at"] >= cutoff]


def get_today_mood_average(user_id: str) -> Optional[float]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT AVG(score) AS avg_score FROM mood_entries WHERE date_str = ?",
            (get_user_current_time(user_id).strftime("%Y-%m-%d"),),
        )
        row = cursor.fetchone()
        return row["avg_score"] if row and row["avg_score"] is not None else None
    finally:
        conn.close()


def bump_topic(user_id: str, topic: str) -> None:
    topic = (topic or "").strip().lower()
    if not topic or len(topic) > 60:
        return
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO topic_frequency (topic, count, last_mentioned) VALUES (?, 1, ?) "
                "ON CONFLICT(topic) DO UPDATE SET count = count + 1, last_mentioned = excluded.last_mentioned",
                (topic, datetime.utcnow().isoformat()),
            )
    finally:
        conn.close()


def get_top_topics(user_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT topic, count, last_mentioned FROM topic_frequency "
            "ORDER BY count DESC, last_mentioned DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def record_bot_message(user_id: str, bot_message_id: int, style_tags: str = "") -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO engagement_scores (bot_message_id, style_tags) VALUES (?, ?)",
                (bot_message_id, style_tags),
            )
    finally:
        conn.close()


def mark_engagement_followup(user_id: str, bot_message_id: int) -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "UPDATE engagement_scores SET had_followup = 1 WHERE id = "
                "(SELECT MAX(id) FROM engagement_scores WHERE bot_message_id = ?)",
                (bot_message_id,),
            )
    finally:
        conn.close()


def mark_latest_engagement_followup(user_id: str) -> None:
    """Mark the most recent bot message as having a follow-up."""
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "UPDATE engagement_scores SET had_followup = 1 "
                "WHERE id = (SELECT MAX(id) FROM engagement_scores)"
            )
    finally:
        conn.close()


def get_engagement_rate(user_id: str, lookback: int = 50) -> Optional[float]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT had_followup FROM engagement_scores ORDER BY id DESC LIMIT ?",
            (lookback,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    return sum(1 for r in rows if r["had_followup"]) / len(rows)


def get_active_hours(user_id: str, min_messages: int = 5) -> Optional[List[int]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp FROM messages WHERE role = 'user' ORDER BY id DESC LIMIT 500"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if len(rows) < min_messages:
        return None

    hours: Dict[int, int] = {h: 0 for h in range(24)}
    for r in rows:
        try:
            hours[datetime.fromisoformat(r["timestamp"]).hour] += 1
        except Exception:
            continue
    return [hours[h] for h in range(24)]


def is_user_currently_active(user_id: str, window_minutes: int = 30) -> bool:
    state = get_user_state(user_id)
    last = state.get("last_user_msg_time")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
        now = get_user_current_time(user_id)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=now.tzinfo)
        else:
            last_dt = last_dt.astimezone(now.tzinfo)
        return (now - last_dt).total_seconds() < window_minutes * 60
    except Exception:
        return False


def get_pet_name(user_id: str) -> Optional[str]:
    return get_all_facts(user_id).get("profile", {}).get("pet_name")


def set_pet_name(user_id: str, name: str) -> None:
    save_fact(user_id, "profile", "pet_name", name.strip())


def save_daily_summary(user_id: str, date_str: str, summary: str, message_count: int) -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO daily_summaries (date_str, summary_text, message_count) VALUES (?, ?, ?) "
                "ON CONFLICT(date_str) DO UPDATE SET summary_text = excluded.summary_text, "
                "message_count = excluded.message_count",
                (date_str, summary, message_count),
            )
    finally:
        conn.close()


def get_daily_summary(user_id: str, date_str: str) -> Optional[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT summary_text, message_count FROM daily_summaries WHERE date_str = ?",
            (date_str,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Embeddings (semantic memory)
# ---------------------------------------------------------------------------

def store_message_embedding(user_id: str, message_id: int, embedding: bytes) -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO message_embeddings (message_id, embedding) VALUES (?, ?)",
                (message_id, embedding),
            )
    finally:
        conn.close()


def get_recent_message_embeddings(user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT me.message_id, me.embedding, m.role, m.content "
            "FROM message_embeddings me "
            "JOIN messages m ON m.id = me.message_id "
            "ORDER BY me.id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Conversation summaries
# ---------------------------------------------------------------------------

def save_conversation_summary(user_id: str, summary_text: str, message_count: int) -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO conversation_summaries (user_id_marker, summary_text, message_count) "
                "VALUES (?, ?, ?)",
                (user_id, summary_text, message_count),
            )
    finally:
        conn.close()


def get_latest_conversation_summary(user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT summary_text, message_count, created_at FROM conversation_summaries "
            "WHERE user_id_marker = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Per-user persona state
# ---------------------------------------------------------------------------

def get_user_persona(user_id: str) -> Dict[str, str]:
    """Return the user's per-user persona state. Seeds from defaults on first access.

    This is what makes each user see a different "version" of Atrix — her
    current obsession, mood, etc. evolve independently per user based on
    their conversations.
    """
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT field, value FROM persona_state")
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        seed_user_persona(user_id)
        return get_user_persona(user_id)
    return {row["field"]: row["value"] for row in rows}


def seed_user_persona(user_id: str) -> None:
    """Seed a new user with the default persona template from persona.py."""
    from atrix.ai.persona import _load_template
    template = _load_template().get("global", {})
    conn = get_db(user_id)
    try:
        with conn:
            for field, value in template.items():
                conn.execute(
                    "INSERT OR IGNORE INTO persona_state (field, value) VALUES (?, ?)",
                    (field, value),
                )
    finally:
        conn.close()


def update_user_persona_field(user_id: str, field: str, value: str) -> None:
    """Update one field of the user's per-user persona. Inserts if missing."""
    if not field or value is None:
        return
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO persona_state (field, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(field) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (field, value, datetime.utcnow().isoformat()),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Relationship events (emotional log) and relationship state
# ---------------------------------------------------------------------------

def record_relationship_event(
    user_id: str,
    impact: float,
    category: str,
    sincere: bool,
    inside_joke: bool,
    reasoning: str,
    sample_text: str,
) -> int:
    """Append a single emotional event to the user's log. Returns the new row id."""
    impact = max(-1.0, min(1.0, float(impact)))
    conn = get_db(user_id)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO relationship_events "
                "(impact, category, sincere, inside_joke, reasoning, sample_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    impact, category, int(bool(sincere)), int(bool(inside_joke)),
                    reasoning, sample_text[:300],
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_recent_relationship_events(user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, impact, category, sincere, inside_joke, reasoning, sample_text, created_at "
            "FROM relationship_events ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_relationship_state(user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db(user_id)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM relationship_state WHERE id = 1")
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def set_relationship_state(
    user_id: str,
    state_description: str,
    trajectory: str = "",
    last_user_msg_excerpt: str = "",
    last_user_msg_at: str = "",
) -> None:
    conn = get_db(user_id)
    try:
        with conn:
            conn.execute(
                "INSERT INTO relationship_state "
                "(id, state_description, trajectory, last_user_msg_excerpt, last_user_msg_at, last_updated) "
                "VALUES (1, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "state_description = excluded.state_description, "
                "trajectory = excluded.trajectory, "
                "last_user_msg_excerpt = excluded.last_user_msg_excerpt, "
                "last_user_msg_at = excluded.last_user_msg_at, "
                "last_updated = excluded.last_updated",
                (
                    state_description, trajectory,
                    last_user_msg_excerpt[:200], last_user_msg_at,
                    datetime.utcnow().isoformat(),
                ),
            )
    finally:
        conn.close()