import json
from pathlib import Path

from core.artifact_manifest import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    build_state_fingerprint,
    file_sha256,
    read_artifact_manifest,
    upsert_artifact_entry,
)
from core.methodology_log import (
    METHODOLOGY_LOG_SCHEMA_VERSION,
    MethodologyEvent,
    append_methodology_event,
    read_methodology_log,
)
from skills.experience_study_skill.ai_models import AI_SWEEP_PACKET_SCHEMA_VERSION


def test_methodology_log_appends_events_in_order(tmp_path: Path):
    log_path = tmp_path / "audit" / "methodology_log.json"

    append_methodology_event(
        log_path,
        MethodologyEvent(
            step_name="Schema inspected",
            tool_name="inspect_dataset_schema",
            input_path="analysis_inforce.parquet",
            output_path=None,
            parameters={},
            timestamp="2026-05-04T00:00:00Z",
        ),
    )
    append_methodology_event(
        log_path,
        MethodologyEvent(
            step_name="Validation checks run",
            tool_name="run_actuarial_data_checks",
            input_path="analysis_inforce.parquet",
            output_path=None,
            parameters={"status": "PASS"},
            timestamp="2026-05-04T00:00:01Z",
        ),
    )

    payload = read_methodology_log(log_path)

    assert payload["schema_version"] == METHODOLOGY_LOG_SCHEMA_VERSION
    assert [event["step_name"] for event in payload["events"]] == [
        "Schema inspected",
        "Validation checks run",
    ]
    assert payload["events"][0]["output_path"] is None


def test_artifact_manifest_hash_upsert_and_source_relationship(tmp_path: Path):
    manifest_path = tmp_path / "audit" / "artifact_manifest.json"
    source_path = tmp_path / "source.csv"
    artifact_path = tmp_path / "analysis_inforce.parquet"
    source_path.write_text("id,value\n1,10\n", encoding="utf-8")
    artifact_path.write_text("first version", encoding="utf-8")
    first_hash = file_sha256(artifact_path)
    source_hash = file_sha256(source_path)

    first_entry = upsert_artifact_entry(
        manifest_path,
        artifact_type="prepared_dataset",
        path=artifact_path,
        generating_tool="profile_dataset",
        parameters={"data_path": str(source_path)},
        source_artifacts=[
            {
                "artifact_type": "source_dataset",
                "path": str(source_path),
                "content_hash": source_hash,
            }
        ],
    )
    artifact_path.write_text("second version", encoding="utf-8")
    second_entry = upsert_artifact_entry(
        manifest_path,
        artifact_type="prepared_dataset",
        path=artifact_path,
        generating_tool="create_categorical_bands",
        parameters={"source_column": "Issue_Age"},
        source_artifacts=[
            {
                "artifact_type": "prepared_dataset",
                "path": str(artifact_path),
                "content_hash": first_hash,
            }
        ],
    )

    payload = read_artifact_manifest(manifest_path)

    assert payload["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert first_entry is not None
    assert second_entry is not None
    assert first_entry["content_hash"] == first_hash
    assert second_entry["content_hash"] != first_hash
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["source_artifacts"][0]["content_hash"] == first_hash


def test_artifact_manifest_does_not_manifest_audit_files(tmp_path: Path):
    manifest_path = tmp_path / "audit" / "artifact_manifest.json"
    audit_log_path = tmp_path / "audit" / "methodology_log.json"
    audit_log_path.parent.mkdir(parents=True)
    audit_log_path.write_text("{}", encoding="utf-8")

    result = upsert_artifact_entry(
        manifest_path,
        artifact_type="methodology_log",
        path=audit_log_path,
        generating_tool="audit",
        parameters={},
    )

    assert result is None
    assert not manifest_path.exists()


def test_state_fingerprint_is_stable_for_key_order_and_changes_for_source_hash():
    base_args = {
        "selected_columns": ["Gender", "Risk_Class"],
        "filters": [{"value": "Yes", "operator": "=", "column": "Smoker"}],
        "depth": 2,
        "sort_by": "AE_Ratio_Amount",
        "min_mac": 1,
        "packet_schema_version": AI_SWEEP_PACKET_SCHEMA_VERSION,
        "skill_name": "experience-study-skill",
        "skill_version": "1.0.0",
    }

    first = build_state_fingerprint(
        source_hashes={"prepared_dataset_path": "abc", "raw_input_path": "def"},
        **base_args,
    )
    reordered = build_state_fingerprint(
        source_hashes={"raw_input_path": "def", "prepared_dataset_path": "abc"},
        **base_args,
    )
    changed = build_state_fingerprint(
        source_hashes={"prepared_dataset_path": "changed", "raw_input_path": "def"},
        **base_args,
    )

    assert first == reordered
    assert first != changed


def test_malformed_methodology_log_raises_clear_error(tmp_path: Path):
    log_path = tmp_path / "audit" / "methodology_log.json"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps({"schema_version": "unknown", "events": []}), encoding="utf-8")

    try:
        read_methodology_log(log_path)
    except ValueError as exc:
        assert "Unsupported methodology log schema" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected malformed methodology log to raise ValueError.")
