"""Shared caution flag helpers for AI interpretation packets."""

from __future__ import annotations

from skills.experience_study_skill.ai_models import AISweepPacket


def collect_packet_caution_flags(packet: AISweepPacket) -> list[str]:
    """Collect ordered, de-duplicated caution flags from a sanitized AI packet."""
    flags: list[str] = []
    for row in packet.rows:
        flags.extend(row.caution_flags)
        if row.low_credibility and row.masking_reason:
            flags.append(row.masking_reason)
    flags.extend(warning.code for warning in packet.warnings)
    return list(dict.fromkeys(flags))
