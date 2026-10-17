"""Behavior modules: mood, energy, topics, drift, timing, recap, quote recall, relational, reach_out."""

from atrix.behavior.mood import (
    score_text as mood_score,
    classify as mood_classify,
    record_message_mood,
    format_for_prompt as mood_for_prompt,
)
from atrix.behavior.energy import detect_energy, hint_for_prompt as energy_hint
from atrix.behavior.topics import extract_and_bump, format_for_prompt as topics_for_prompt
from atrix.behavior.drift import hint_for_prompt as drift_hint
from atrix.behavior.timing import (
    get_learned_active_hours,
    is_within_active_window,
    should_skip_for_recent_activity,
    format_for_prompt as schedule_for_prompt,
)
from atrix.behavior.recap import generate_recap
from atrix.behavior.quote_recall import find_relevant_past_quotes, format_for_prompt as quotes_for_prompt
from atrix.behavior.relational import (
    analyze_message_impact,
    update_state as update_relationship_state,
    format_for_prompt as relational_for_prompt,
    run_after_message as run_relational_after_message,
)
from atrix.behavior.reach_out import should_reach_out

__all__ = [
    "mood_score", "mood_classify", "record_message_mood", "mood_for_prompt",
    "detect_energy", "energy_hint",
    "extract_and_bump", "topics_for_prompt",
    "drift_hint",
    "get_learned_active_hours", "is_within_active_window",
    "should_skip_for_recent_activity", "schedule_for_prompt",
    "generate_recap",
    "find_relevant_past_quotes", "quotes_for_prompt",
    "analyze_message_impact", "update_relationship_state",
    "relational_for_prompt", "run_relational_after_message",
    "should_reach_out",
]