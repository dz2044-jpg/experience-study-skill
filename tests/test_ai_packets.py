import json
from pathlib import Path

import pandas as pd
import pytest

from core.artifact_manifest import (
    build_state_fingerprint,
    build_state_fingerprint_payload,
    update_manifest_fingerprint,
)
from skills.experience_study_skill.ai_models import AI_SWEEP_PACKET_SCHEMA_VERSION
from skills.experience_study_skill.native_tools import get_tool_handlers
from skills.experience_study_skill.ai_packets import (
    REQUIRED_AI_SWEEP_COLUMNS,
    build_latest_sweep_packet,
)
from skills.experience_study_skill.schemas import get_tool_specs


EXPECTED_PUBLIC_TOOL_NAMES = {
    "profile_dataset",
    "inspect_dataset_schema",
    "run_actuarial_data_checks",
    "create_categorical_bands",
    "regroup_categorical_features",
    "run_dimensional_sweep",
    "generate_combined_report",
}


def _sweep_row(dimensions: str, mac: int = 1) -> dict[str, object]:
    return {
        "Dimensions": dimensions,
        "Sum_MAC": mac,
        "Sum_MOC": 10.0,
        "Sum_MEC": 2.0,
        "Sum_MAF": 100000.0,
        "Sum_MEF": 90000.0,
        "AE_Ratio_Count": 0.5,
        "AE_Ratio_Amount": 1.11,
        "AE_Count_CI_Lower": 0.05,
        "AE_Count_CI_Upper": 1.50,
        "AE_Amount_CI_Lower": 0.10,
        "AE_Amount_CI_Upper": 2.25,
    }


def _write_sweep(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_manifest(
    path: Path,
    *,
    min_mac: int,
    selected_columns: list[str] | None = None,
    filters: list[dict[str, object]] | None = None,
) -> Path:
    fingerprint_inputs = build_state_fingerprint_payload(
        source_hashes={"latest_sweep_path": "abc"},
        selected_columns=selected_columns,
        filters=filters or [],
        depth=1,
        sort_by="AE_Ratio_Amount",
        min_mac=min_mac,
        packet_schema_version=AI_SWEEP_PACKET_SCHEMA_VERSION,
        skill_name="experience-study-skill",
        skill_version="1.0.0",
    )
    fingerprint = build_state_fingerprint(
        source_hashes={"latest_sweep_path": "abc"},
        selected_columns=selected_columns,
        filters=filters or [],
        depth=1,
        sort_by="AE_Ratio_Amount",
        min_mac=min_mac,
        packet_schema_version=AI_SWEEP_PACKET_SCHEMA_VERSION,
        skill_name="experience-study-skill",
        skill_version="1.0.0",
    )
    update_manifest_fingerprint(
        path,
        fingerprint=fingerprint,
        fingerprint_inputs=fingerprint_inputs,
    )
    return path


def test_packet_builds_from_sweep_and_manifest_metadata(tmp_path: Path):
    sweep_path = _write_sweep(tmp_path / "sweep_summary.csv", [_sweep_row("Gender=M")])
    manifest_path = _write_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        min_mac=3,
        selected_columns=["Gender"],
        filters=[{"column": "Smoker", "operator": "=", "value": "Yes"}],
    )

    packet = build_latest_sweep_packet(
        sweep_path=sweep_path,
        artifact_manifest_path=manifest_path,
    )

    assert packet.schema_version == AI_SWEEP_PACKET_SCHEMA_VERSION
    assert packet.state_fingerprint is not None
    assert packet.packet_fingerprint is not None
    assert packet.depth == 1
    assert packet.sort_by == "AE_Ratio_Amount"
    assert packet.deterministic_min_mac == 3
    assert packet.ai_masking_min_mac == 3
    assert packet.selected_columns == ["Gender"]
    assert packet.filters == [{"column": "Smoker", "operator": "=", "value": "Yes"}]
    assert [row.evidence_ref for row in packet.rows] == ["row_0001"]


def test_packet_requires_known_columns_and_drops_unknown_columns(tmp_path: Path):
    sweep_path = tmp_path / "sweep_summary.csv"
    row = _sweep_row("Gender=M")
    row["Unexpected_Name"] = "Alice Example"
    _write_sweep(sweep_path, [row])

    packet = build_latest_sweep_packet(sweep_path=sweep_path)

    packet_json = packet.model_dump_json()
    assert "Unexpected_Name" not in packet_json
    assert "Alice Example" not in packet_json

    missing_path = tmp_path / "missing.csv"
    missing_row = dict(row)
    missing_row.pop("AE_Ratio_Amount")
    _write_sweep(missing_path, [missing_row])

    with pytest.raises(ValueError, match="missing required AI packet columns"):
        build_latest_sweep_packet(sweep_path=missing_path)


def test_packet_json_excludes_sensitive_field_names_and_raw_values(tmp_path: Path):
    sweep_path = _write_sweep(
        tmp_path / "sweep_summary.csv",
        [_sweep_row("Policy_Number=P001 | Gender=M", mac=2)],
    )
    manifest_path = _write_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        min_mac=0,
        selected_columns=["Policy_Number", "Gender"],
        filters=[
            {"column": "Email", "operator": "=", "value": "person@example.com"},
            {"column": "Gender", "operator": "=", "value": "M"},
        ],
    )

    packet = build_latest_sweep_packet(
        sweep_path=sweep_path,
        artifact_manifest_path=manifest_path,
    )

    packet_json = packet.model_dump_json()
    for forbidden in ("Policy_Number", "P001", "person@example.com", "Email"):
        assert forbidden not in packet_json
    assert packet.rows[0].Dimensions == "[masked cohort label]"
    assert packet.rows[0].Dimension_Columns == ["[masked dimension]"]
    assert packet.rows[0].masking_reason == "sensitive_or_disallowed_dimension"
    assert packet.filters[0] == {
        "column": "[masked column]",
        "operator": "=",
        "value": "[masked value]",
    }


def test_sensitive_dimension_matching_is_token_aware(tmp_path: Path):
    sweep_path = _write_sweep(
        tmp_path / "sweep_summary.csv",
        [
            _sweep_row("Certificate_Number=C123", mac=2),
            _sweep_row("Duration=1", mac=2),
            _sweep_row("Grid=A", mac=2),
        ],
    )

    packet = build_latest_sweep_packet(sweep_path=sweep_path)

    assert packet.rows[0].Dimensions == "[masked cohort label]"
    assert packet.rows[0].masking_reason == "sensitive_or_disallowed_dimension"
    assert packet.rows[1].Dimensions == "Duration=1"
    assert packet.rows[1].masking_reason is None
    assert packet.rows[2].Dimensions == "Grid=A"
    assert packet.rows[2].masking_reason is None


def test_rare_cohort_masking_threshold_precedence_and_absent_rows(tmp_path: Path):
    default_path = _write_sweep(
        tmp_path / "default.csv",
        [_sweep_row("Gender=F", mac=0), _sweep_row("Gender=M", mac=1)],
    )
    default_packet = build_latest_sweep_packet(sweep_path=default_path)
    assert default_packet.ai_masking_min_mac == 1
    assert default_packet.rows[0].masking_reason == "low_volume"
    assert default_packet.rows[1].masking_reason is None

    explicit_gte = build_latest_sweep_packet(
        sweep_path=default_path,
        masking_min_mac=1,
        masking_operator=">=",
    )
    assert explicit_gte.ai_masking_min_mac == 1
    assert explicit_gte.rows[0].masking_reason == "low_volume"
    assert explicit_gte.rows[1].masking_reason is None

    explicit_gt = build_latest_sweep_packet(
        sweep_path=default_path,
        masking_min_mac=1,
        masking_operator=">",
    )
    assert explicit_gt.ai_masking_min_mac == 2
    assert explicit_gt.rows[0].masking_reason == "low_volume"
    assert explicit_gt.rows[1].masking_reason == "low_volume"

    parsed_intent = build_latest_sweep_packet(
        sweep_path=default_path,
        parsed_min_mac=1,
        parsed_operator=">=",
    )
    assert parsed_intent.ai_masking_min_mac == 1
    assert parsed_intent.rows[0].masking_reason == "low_volume"
    assert parsed_intent.rows[1].masking_reason is None

    manifest_path = _write_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        min_mac=0,
    )
    manifest_packet = build_latest_sweep_packet(
        sweep_path=default_path,
        artifact_manifest_path=manifest_path,
    )
    assert manifest_packet.ai_masking_min_mac == 1

    filtered_path = _write_sweep(
        tmp_path / "filtered_min_mac_5.csv",
        [_sweep_row("Gender=M", mac=5), _sweep_row("Gender=F", mac=6)],
    )
    filtered_manifest_path = _write_manifest(
        tmp_path / "audit_filtered" / "artifact_manifest.json",
        min_mac=5,
    )
    filtered_packet = build_latest_sweep_packet(
        sweep_path=filtered_path,
        artifact_manifest_path=filtered_manifest_path,
    )
    assert len(filtered_packet.rows) == 2
    assert filtered_packet.ai_masking_min_mac == 5
    assert all(row.masking_reason is None for row in filtered_packet.rows)


def test_explicit_no_volume_masking_can_override_ai_default(tmp_path: Path):
    sweep_path = _write_sweep(tmp_path / "sweep_summary.csv", [_sweep_row("Gender=F", mac=0)])

    packet = build_latest_sweep_packet(
        sweep_path=sweep_path,
        explicit_no_volume_masking=True,
    )

    assert packet.ai_masking_min_mac == 0
    assert packet.rows[0].masking_reason is None


def test_dimension_parser_is_conservative_without_failing_packet(tmp_path: Path):
    sweep_path = _write_sweep(
        tmp_path / "sweep_summary.csv",
        [
            _sweep_row("Plan=Base=Plus | Region=West", mac=2),
            _sweep_row("Plan=Base | Plus | Region=West", mac=2),
            _sweep_row("MalformedLabel", mac=2),
        ],
    )

    packet = build_latest_sweep_packet(sweep_path=sweep_path)

    assert packet.rows[0].Dimension_Columns == ["Plan", "Region"]
    assert packet.rows[0].Dimensions == "Plan=Base=Plus | Region=West"
    assert packet.rows[1].Dimension_Columns == ["Plan", "Region"]
    assert "dimension_parse_warning" in packet.rows[1].caution_flags
    assert packet.rows[2].Dimension_Columns == []
    assert "dimension_parse_warning" in packet.rows[2].caution_flags
    assert any(warning.code == "dimension_parse_warning" for warning in packet.warnings)


def test_packet_serializes_with_only_allowlisted_sweep_columns(tmp_path: Path):
    sweep_path = _write_sweep(tmp_path / "sweep_summary.csv", [_sweep_row("Gender=M")])

    packet = build_latest_sweep_packet(sweep_path=sweep_path)
    row_payload = packet.rows[0].model_dump()

    for column in REQUIRED_AI_SWEEP_COLUMNS:
        if column == "Dimensions":
            assert column in row_payload
        else:
            assert column in row_payload
    assert "Policy_Number" not in json.dumps(packet.model_dump(), sort_keys=True)


def test_pr6_does_not_change_public_deterministic_tools():
    exposed_schema_tool_names = {spec["function"]["name"] for spec in get_tool_specs()}

    assert exposed_schema_tool_names == EXPECTED_PUBLIC_TOOL_NAMES
    assert set(get_tool_handlers()) == EXPECTED_PUBLIC_TOOL_NAMES
