"""Atrix — a Discord AI friend that tries really hard to feel like a real person.

The package is organized into focused submodules:

    atrix.config          — env loading & validation
    atrix.bot             — Discord entry point & orchestration
    atrix.ai              — LLM providers, prompts, memory extraction
    atrix.memory          — per-user SQLite store & pruning
    atrix.behavior        — mood, energy, topics, drift, timing, recap
    atrix.proactive       — the "should I DM them first?" engine
    atrix.integrations    — voice, vision, image gen, link preview, web search
    atrix.discord_ui      — presence, commands, debouncer, reactions
    atrix.utils           — shared text helpers
"""

__version__ = "1.0.0"
__all__ = ["config", "bot"]