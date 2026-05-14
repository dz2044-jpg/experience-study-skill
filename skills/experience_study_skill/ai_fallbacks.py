"""Deterministic fallback responses for internal AI interpretation actions."""

from __future__ import annotations

import math
from typing import Any

from skills.experience_study_skill.ai_baselines import (
    ACTION_TITLES,
    DEFAULT_NEXT_REVIEW_STEPS,
)
from skills.experience_study_skill.ai_cautions import collect_packet_caution_flags
from skills.experience_study_skill.ai_models import (
    AIActionName,
    AIActionResponse,
    AICohortRow,
    AISweepPacket,
    AIValidationResult,
)


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def _rank_metric_value(row: AICohortRow, metric: str) -> float:
    value = getattr(row, metric)
    if value is None or not math.isfinite(float(value)):
        return float("-inf")
    return float(value)


def _divergence_rank_value(row: AICohortRow) -> float:
    amount_value = _rank_metric_value(row, "AE_Ratio_Amount")
    count_value = _rank_metric_value(row, "AE_Ratio_Count")
    if amount_value == float("-inf") or count_value == float("-inf"):
        return float("-inf")
    return abs(amount_value - count_value)


def _rank_rows(packet: AISweepPacket, metric: str = "AE_Ratio_Amount") -> list[AICohortRow]:
    return sorted(packet.rows, key=lambda row: _rank_metric_value(row, metric), reverse=True)


def _cohort_line(row: AICohortRow) -> str:
    if row.low_credibility:
        return (
            f"- {row.evidence_ref}: masked or low-credibility cohort; "
            "review deterministic output before interpretation."
        )
    return (
        f"- {row.evidence_ref}: {row.Dimensions}; "
        f"MAC={_format_number(row.Sum_MAC)}, MEC={_format_number(row.Sum_MEC)}, "
        f"A/E count={_format_number(row.AE_Ratio_Count)}, "
        f"A/E amount={_format_number(row.AE_Ratio_Amount)}."
    )


def select_action_rows(
    action_name: AIActionName,
    packet: AISweepPacket,
    action_context: dict[str, Any] | None,
) -> list[AICohortRow]:
    if not packet.rows:
        return []
    if action_context and action_context.get("evidence_ref"):
        requested_ref = str(action_context["evidence_ref"])
        selected = [row for row in packet.rows if row.evidence_ref == requested_ref]
        if selected:
            return selected
        if action_name == "explain_cohort":
            return []
    if action_name == "compare_cohorts":
        return _rank_rows(packet)[:2]
    if action_name == "analyze_count_amount_divergence":
        return sorted(
            packet.rows,
            key=_divergence_rank_value,
            reverse=True,
        )[:3]
    return _rank_rows(packet)[:3]


def requested_evidence_ref_not_found(
    action_name: AIActionName,
    packet: AISweepPacket,
    action_context: dict[str, Any] | None,
) -> bool:
    if action_name != "explain_cohort":
        return False
    if not action_context or not action_context.get("evidence_ref"):
        return False
    requested_ref = str(action_context["evidence_ref"])
    return not any(row.evidence_ref == requested_ref for row in packet.rows)


def build_fallback_response(
    *,
    action_name: AIActionName,
    packet: AISweepPacket,
    validation: AIValidationResult | None = None,
    action_context: dict[str, Any] | None = None,
) -> AIActionResponse:
    """Return a safe deterministic response when LLM output is unavailable or blocked."""

    selected_rows = select_action_rows(action_name, packet, action_context)
    evidence_refs = [row.evidence_ref for row in selected_rows]
    caution_flags = collect_packet_caution_flags(packet)
    if requested_evidence_ref_not_found(action_name, packet, action_context):
        caution_flags.append("requested_evidence_ref_not_found")
    if validation and validation.blocked_issues:
        caution_flags.extend(issue.code for issue in validation.blocked_issues)
    caution_flags = list(dict.fromkeys(caution_flags))

    title = ACTION_TITLES[action_name]
    lines = [
        f"{title}",
        "",
        "Source mode: fallback.",
        f"State fingerprint: {packet.state_fingerprint or 'unavailable'}.",
        f"Packet fingerprint: {packet.packet_fingerprint or 'unavailable'}.",
        "",
        "Evidence refs: " + (", ".join(evidence_refs) if evidence_refs else "none"),
        "Caution flags: " + (", ".join(caution_flags) if caution_flags else "none"),
        "",
    ]
    if selected_rows:
        lines.append("Deterministic packet observations:")
        lines.extend(_cohort_line(row) for row in selected_rows)
    else:
        lines.append("No cohort rows are available in the deterministic packet.")
    lines.extend(["", "Next review steps:"])
    lines.extend(f"- {step}" for step in DEFAULT_NEXT_REVIEW_STEPS)

    return AIActionResponse(
        action_name=action_name,
        source_mode="fallback",
        response_text="\n".join(lines),
        evidence_refs=evidence_refs,
        caution_flags=caution_flags,
        next_review_steps=DEFAULT_NEXT_REVIEW_STEPS,
        state_fingerprint=packet.state_fingerprint,
        packet_fingerprint=packet.packet_fingerprint,
        validation=validation or AIValidationResult(),
    )
