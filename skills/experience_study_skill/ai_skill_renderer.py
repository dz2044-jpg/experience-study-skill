"""Render simple internal AI action prompts from sanitized packets."""

from __future__ import annotations

import json
from typing import Any

from skills.experience_study_skill.ai_models import AIActionName, AISweepPacket
from skills.experience_study_skill.ai_skill_loader import load_ai_action


def render_action_prompt(
    *,
    action_name: AIActionName,
    packet: AISweepPacket,
    action_context: dict[str, Any] | None = None,
) -> str:
    """Render a prompt with action Markdown, packet JSON, and optional context."""

    action = load_ai_action(action_name)
    packet_json = packet.model_dump_json(indent=2)
    context_json = json.dumps(action_context or {}, indent=2, sort_keys=True)
    return "\n\n".join(
        [
            action.body,
            "Use only this sanitized packet JSON:",
            packet_json,
            "Action context JSON:",
            context_json,
        ]
    )
