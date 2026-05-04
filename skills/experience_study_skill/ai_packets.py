"""Build sanitized AI packets from deterministic sweep artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.artifact_manifest import file_sha256, normalize_json_value, read_artifact_manifest
from skills.experience_study_skill.ai_models import (
    AI_DEFAULT_MASKING_MIN_MAC,
    AI_SWEEP_PACKET_SCHEMA_VERSION,
    AICohortRow,
    AISweepPacket,
    AIValidationIssue,
    ThresholdOperator,
)
from skills.experience_study_skill.ai_sanitization import (
    MASKED_COHORT_LABEL,
    MASKED_DIMENSION_COLUMN,
    dimension_column_is_sensitive,
    parse_dimension_label,
    sanitize_filters,
    sanitize_selected_columns,
)


REQUIRED_AI_SWEEP_COLUMNS = [
    "Dimensions",
    "Sum_MAC",
    "Sum_MOC",
    "Sum_MEC",
    "Sum_MAF",
    "Sum_MEF",
    "AE_Ratio_Count",
    "AE_Ratio_Amount",
    "AE_Count_CI_Lower",
    "AE_Count_CI_Upper",
    "AE_Amount_CI_Lower",
    "AE_Amount_CI_Upper",
]


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, coerced)


def _threshold_from_operator(
    value: int | None,
    operator: ThresholdOperator,
) -> int | None:
    if value is None:
        return None
    threshold = max(0, int(value))
    return threshold + 1 if operator == ">" else threshold


def resolve_ai_masking_min_mac(
    *,
    masking_min_mac: int | None = None,
    masking_operator: ThresholdOperator = ">=",
    parsed_min_mac: int | None = None,
    parsed_operator: ThresholdOperator = ">=",
    manifest_min_mac: int | None = None,
    explicit_no_volume_masking: bool = False,
) -> int:
    """Resolve the AI-specific masking floor for cohort interpretation."""

    if explicit_no_volume_masking:
        return 0

    explicit_threshold = _threshold_from_operator(masking_min_mac, masking_operator)
    if explicit_threshold is not None:
        return explicit_threshold

    parsed_threshold = _threshold_from_operator(parsed_min_mac, parsed_operator)
    if parsed_threshold is not None:
        return parsed_threshold

    manifest_threshold = _coerce_optional_int(manifest_min_mac) or 0
    return max(manifest_threshold, AI_DEFAULT_MASKING_MIN_MAC)


def _read_manifest_metadata(
    artifact_manifest_path: str | Path | None,
) -> tuple[dict[str, Any], int | None]:
    if artifact_manifest_path is None:
        return {}, None
    manifest_path = Path(artifact_manifest_path)
    if not manifest_path.exists():
        return {}, None
    manifest = read_artifact_manifest(manifest_path)
    fingerprint_inputs = manifest.get("fingerprint_inputs") or {}
    if not isinstance(fingerprint_inputs, dict):
        fingerprint_inputs = {}
    return manifest, _coerce_optional_int(fingerprint_inputs.get("min_mac"))


def _numeric_or_none(value: Any, *, column: str, evidence_ref: str) -> tuple[float | None, AIValidationIssue | None]:
    if pd.isna(value):
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError):
        return (
            None,
            AIValidationIssue(
                code="invalid_numeric_value",
                message=f"Invalid numeric value in required sweep column `{column}`.",
                evidence_refs=[evidence_ref],
            ),
        )


def _packet_fingerprint(packet: AISweepPacket) -> str:
    payload = packet.model_dump(mode="json", exclude={"packet_fingerprint"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_latest_sweep_packet(
    *,
    sweep_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    masking_min_mac: int | None = None,
    masking_operator: ThresholdOperator = ">=",
    parsed_min_mac: int | None = None,
    parsed_operator: ThresholdOperator = ">=",
    explicit_no_volume_masking: bool = False,
) -> AISweepPacket:
    """Build an allowlisted, cohort-level packet from the latest sweep artifact."""

    sweep_artifact_path = Path(sweep_path)
    sweep_df = pd.read_csv(sweep_artifact_path)
    missing_columns = [
        column for column in REQUIRED_AI_SWEEP_COLUMNS if column not in sweep_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Sweep artifact is missing required AI packet columns: {missing_columns}."
        )
    sweep_df = sweep_df.loc[:, REQUIRED_AI_SWEEP_COLUMNS]

    manifest, manifest_min_mac = _read_manifest_metadata(artifact_manifest_path)
    fingerprint_inputs = manifest.get("fingerprint_inputs") or {}
    if not isinstance(fingerprint_inputs, dict):
        fingerprint_inputs = {}

    ai_masking_min_mac = resolve_ai_masking_min_mac(
        masking_min_mac=masking_min_mac,
        masking_operator=masking_operator,
        parsed_min_mac=parsed_min_mac,
        parsed_operator=parsed_operator,
        manifest_min_mac=manifest_min_mac,
        explicit_no_volume_masking=explicit_no_volume_masking,
    )

    sanitized_filters, filter_warnings = sanitize_filters(
        fingerprint_inputs.get("filters") if isinstance(fingerprint_inputs, dict) else []
    )
    sanitized_selected_columns, selected_column_warnings = sanitize_selected_columns(
        fingerprint_inputs.get("selected_columns")
        if isinstance(fingerprint_inputs.get("selected_columns"), list)
        else None
    )

    packet_warnings: list[AIValidationIssue] = [
        *filter_warnings,
        *selected_column_warnings,
    ]
    packet_rows: list[AICohortRow] = []

    for row_index, row in sweep_df.iterrows():
        evidence_ref = f"row_{row_index + 1:04d}"
        parsed_dimensions, parse_warnings = parse_dimension_label(
            str(row["Dimensions"]),
            evidence_ref=evidence_ref,
        )
        packet_warnings.extend(parse_warnings)
        caution_flags = [warning.code for warning in parse_warnings]

        dimension_columns = [part.column for part in parsed_dimensions]
        has_sensitive_dimension = any(
            dimension_column_is_sensitive(column) for column in dimension_columns
        )

        sum_mac, numeric_issue = _numeric_or_none(
            row["Sum_MAC"],
            column="Sum_MAC",
            evidence_ref=evidence_ref,
        )
        if numeric_issue:
            raise ValueError(numeric_issue.message)
        assert sum_mac is not None

        low_credibility = False
        masking_reason: str | None = None
        dimensions = str(row["Dimensions"])
        sanitized_dimension_columns = list(dimension_columns)

        if has_sensitive_dimension:
            dimensions = MASKED_COHORT_LABEL
            sanitized_dimension_columns = [MASKED_DIMENSION_COLUMN]
            low_credibility = True
            masking_reason = "sensitive_or_disallowed_dimension"
            caution_flags.append("sensitive_or_disallowed_dimension")
            packet_warnings.append(
                AIValidationIssue(
                    code="sensitive_or_disallowed_dimension",
                    message="A sensitive or disallowed dimension was masked.",
                    evidence_refs=[evidence_ref],
                )
            )
        elif sum_mac < ai_masking_min_mac:
            dimensions = MASKED_COHORT_LABEL
            low_credibility = True
            masking_reason = "low_volume"
            caution_flags.append("low_volume")

        numeric_values: dict[str, float | None] = {}
        for column in REQUIRED_AI_SWEEP_COLUMNS:
            if column == "Dimensions":
                continue
            numeric_value, issue = _numeric_or_none(
                row[column],
                column=column,
                evidence_ref=evidence_ref,
            )
            if issue:
                raise ValueError(issue.message)
            numeric_values[column] = numeric_value

        packet_rows.append(
            AICohortRow(
                evidence_ref=evidence_ref,
                Dimensions=dimensions,
                Dimension_Columns=sanitized_dimension_columns,
                Sum_MAC=float(numeric_values["Sum_MAC"] or 0.0),
                Sum_MOC=float(numeric_values["Sum_MOC"] or 0.0),
                Sum_MEC=float(numeric_values["Sum_MEC"] or 0.0),
                Sum_MAF=float(numeric_values["Sum_MAF"] or 0.0),
                Sum_MEF=float(numeric_values["Sum_MEF"] or 0.0),
                AE_Ratio_Count=float(numeric_values["AE_Ratio_Count"] or 0.0),
                AE_Ratio_Amount=float(numeric_values["AE_Ratio_Amount"] or 0.0),
                AE_Count_CI_Lower=numeric_values["AE_Count_CI_Lower"],
                AE_Count_CI_Upper=numeric_values["AE_Count_CI_Upper"],
                AE_Amount_CI_Lower=numeric_values["AE_Amount_CI_Lower"],
                AE_Amount_CI_Upper=numeric_values["AE_Amount_CI_Upper"],
                low_credibility=low_credibility,
                masking_reason=masking_reason,
                caution_flags=list(dict.fromkeys(caution_flags)),
            )
        )

    packet = AISweepPacket(
        schema_version=AI_SWEEP_PACKET_SCHEMA_VERSION,
        source_artifact_path=str(sweep_artifact_path),
        source_content_hash=file_sha256(sweep_artifact_path),
        state_fingerprint=manifest.get("state_fingerprint"),
        depth=_coerce_optional_int(fingerprint_inputs.get("depth")),
        filters=normalize_json_value(sanitized_filters),
        selected_columns=normalize_json_value(sanitized_selected_columns),
        sort_by=(
            str(fingerprint_inputs.get("sort_by"))
            if fingerprint_inputs.get("sort_by") is not None
            else None
        ),
        deterministic_min_mac=manifest_min_mac,
        ai_masking_min_mac=ai_masking_min_mac,
        rows=packet_rows,
        warnings=list({warning.model_dump_json(): warning for warning in packet_warnings}.values()),
    )
    packet.packet_fingerprint = _packet_fingerprint(packet)
    return packet
