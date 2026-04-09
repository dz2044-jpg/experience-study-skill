"""Centralized model configuration for the unified copilot."""

from __future__ import annotations

import os


DEFAULT_COPILOT_MODEL = "gpt-5-mini"


def resolve_copilot_model(explicit_model: str | None = None) -> str:
    """Resolve the active model using the new env var with legacy fallbacks."""
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    env_candidates = (
        "OPENAI_COPILOT_MODEL",
        "OPENAI_ACTUARY_MODEL",
        "OPENAI_STEWARD_MODEL",
        "OPENAI_ROUTER_MODEL",
    )
    for env_var in env_candidates:
        value = os.getenv(env_var, "").strip()
        if value:
            return value

    return DEFAULT_COPILOT_MODEL

