"""Internal orchestration for AI interpretation actions."""

from __future__ import annotations

from typing import Any

from skills.experience_study_skill.ai_fallbacks import (
    build_fallback_response,
    select_action_rows,
)
from skills.experience_study_skill.ai_models import (
    AIActionName,
    AIActionResponse,
    AISweepPacket,
)
from skills.experience_study_skill.ai_skill_loader import AI_ACTION_NAMES
from skills.experience_study_skill.ai_skill_renderer import render_action_prompt
from skills.experience_study_skill.ai_validation import validate_ai_response


def _packet_caution_flags(packet: AISweepPacket) -> list[str]:
    flags: list[str] = []
    for row in packet.rows:
        flags.extend(row.caution_flags)
        if row.low_credibility and row.masking_reason:
            flags.append(row.masking_reason)
    flags.extend(warning.code for warning in packet.warnings)
    return list(dict.fromkeys(flags))


def run_ai_action(
    *,
    action_name: AIActionName,
    packet: AISweepPacket,
    client: Any | None = None,
    model: str | None = None,
    action_context: dict[str, Any] | None = None,
) -> AIActionResponse:
    """Run an internal AI action, falling back deterministically when needed."""

    if action_name not in AI_ACTION_NAMES:
        raise ValueError(f"Unknown AI action: {action_name}")

    if client is None:
        return build_fallback_response(
            action_name=action_name,
            packet=packet,
            action_context=action_context,
        )

    prompt = render_action_prompt(
        action_name=action_name,
        packet=packet,
        action_context=action_context,
    )
    try:
        completion = client.chat.completions.create(
            model=model or "gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize deterministic actuarial packet data. "
                        "Do not make causal claims, pricing recommendations, "
                        "underwriting recommendations, assumption changes, or final conclusions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = completion.choices[0].message.content or ""
    except Exception:
        return build_fallback_response(
            action_name=action_name,
            packet=packet,
            action_context=action_context,
        )

    validation = validate_ai_response(text)
    if validation.blocked_issues:
        return build_fallback_response(
            action_name=action_name,
            packet=packet,
            validation=validation,
            action_context=action_context,
        )

    selected_rows = select_action_rows(action_name, packet, action_context)
    evidence_refs = [row.evidence_ref for row in selected_rows]
    return AIActionResponse(
        action_name=action_name,
        source_mode="llm",
        response_text=text,
        evidence_refs=evidence_refs,
        caution_flags=_packet_caution_flags(packet),
        next_review_steps=[
            "Review evidence refs against the deterministic sweep artifact.",
            "Confirm cautions and credibility before report use.",
        ],
        state_fingerprint=packet.state_fingerprint,
        packet_fingerprint=packet.packet_fingerprint,
        validation=validation,
    )
