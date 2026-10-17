"""Memory pruning — decay stale facts and clean up.

Facts that haven't been referenced in a long time lose weight. Below a
threshold they get dropped entirely. This keeps the system prompt focused
on what actually matters and prevents the user's fact table from bloating
indefinitely.

Run via the periodic sweep or call ``prune_user`` directly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import atrix.memory.store as store

logger = logging.getLogger("atrix.memory.pruning")

# A fact must not have been referenced in this many days to start decaying.
DECAY_THRESHOLD_DAYS = 30

# How much weight to subtract per weekly check.
DECAY_AMOUNT = 0.1

# Below this weight, the fact is dropped entirely.
DROP_THRESHOLD = 0.2

# Initial weight assigned to a brand-new fact (matches store.save_fact).
INITIAL_WEIGHT = 1.0


def decay_user(user_id: str, *, now: datetime | None = None) -> int:
    """Apply one round of decay to a user's facts. Returns rows dropped."""
    if now is None:
        now = store.get_user_current_time(user_id)
    cutoff = (now - timedelta(days=DECAY_THRESHOLD_DAYS)).isoformat()
    conn = store.get_db(user_id)
    try:
        with conn:
            cursor = conn.cursor()
            # First drop any fact that is way past the threshold AND has very low weight.
            cursor.execute(
                "DELETE FROM facts WHERE last_referenced < ? AND weight < ?",
                (cutoff, DROP_THRESHOLD),
            )
            dropped = cursor.rowcount
            # Then decay facts that are past the threshold but not yet dropped.
            cursor.execute(
                "UPDATE facts SET weight = MAX(0.0, weight - ?) "
                "WHERE last_referenced < ? AND weight >= ?",
                (DECAY_AMOUNT, cutoff, DROP_THRESHOLD),
            )
        return dropped
    finally:
        conn.close()


def prune_all_users() -> None:
    """Run decay across every known user. Call this from a weekly task."""
    total_dropped = 0
    for uid in store.get_all_active_users():
        try:
            total_dropped += decay_user(uid)
        except Exception as exc:
            logger.warning("Decay failed for %s: %s", uid, exc)
    if total_dropped:
        logger.info("Pruning pass dropped %d stale facts", total_dropped)