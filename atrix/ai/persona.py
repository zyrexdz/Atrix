"""Per-user persona — each friend sees their own version of Atrix.

The persona (current obsession, current mood, current annoyance, etc.) is
stored per-user in their isolated SQLite DB. New users are seeded from the
``persona_state.json`` template at the project root, which you can edit to
change the *defaults* that new Atrix instances start with.

Each user's persona evolves independently based on the conversations she
has with that specific friend. Same Atrix underneath, different shared life
with each friend.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import atrix.memory.store as store

logger = logging.getLogger("atrix.ai.persona")

# Path to the editable default template. Only used to seed new users.
_TEMPLATE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "persona_state.json",
))

_DEFAULT_TEMPLATE = {
    "global": {
        "name": "Atrix",
        "vibe": "teenage girl, lowercase, low-key Discord energy, listens before reacting",
        "current_obsession": "replaying Hollow Knight for the third time, no regrets",
        "current_annoyance": "the new minecraft update broke her redstone contraption",
        "current_mood": "chill but a little tired, school week energy",
        "recent_win": "finally beat that boss she'd been stuck on for two weeks",
        "pet_if_any": "no pet right now, lowkey wants a cat but parents said no",
        "school_or_work": "high school, hates mornings, loves weekends",
    }
}


def _load_template() -> Dict[str, Any]:
    """Load the default persona template from persona_state.json if it exists."""
    if not os.path.exists(_TEMPLATE_PATH):
        return json.loads(json.dumps(_DEFAULT_TEMPLATE))
    try:
        with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("persona_state.json unreadable, using code defaults: %s", exc)
        return json.loads(json.dumps(_DEFAULT_TEMPLATE))


def save_template(template: Dict[str, Any]) -> None:
    """Persist edits to the default template. New users will be seeded from this."""
    try:
        with open(_TEMPLATE_PATH, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Could not persist persona template: %s", exc)


def get_user_persona(user_id: str) -> Dict[str, str]:
    """Read the per-user persona. Lazily seeds from the template on first access."""
    return store.get_user_persona(user_id)


def update_user_field(user_id: str, field: str, value: str) -> None:
    """Update one field of this specific user's persona."""
    store.update_user_persona_field(user_id, field, value)


def format_for_prompt(user_id: str) -> str:
    """Render the user's per-user persona for the system prompt."""
    persona = get_user_persona(user_id)
    return "\n".join([
        f"- Name: {persona.get('name', 'Atrix')}",
        f"- Vibe: {persona.get('vibe', '')}",
        f"- School/work: {persona.get('school_or_work', '')}",
        f"- Right now she's into: {persona.get('current_obsession', '')}",
        f"- Right now she's low-key annoyed by: {persona.get('current_annoyance', '')}",
        f"- Mood this week: {persona.get('current_mood', 'chill')}",
        f"- Small win from yesterday: {persona.get('recent_win', '')}",
        f"- Pet situation: {persona.get('pet_if_any', '')}",
    ])