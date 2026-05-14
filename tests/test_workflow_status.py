from __future__ import annotations

from pathlib import Path

import pytest

from core.artifact_manifest import upsert_artifact_entry
from core.methodology_log import MethodologyEvent, append_methodology_event
from core.session_state import SessionArtifactState
from core.workflow_status import AIWorkflowSnapshot, WorkflowStep, derive_workflow_steps


def _state(tmp_path: Path) -> SessionArtifactState:
    return SessionArtifactState(session_id="session-a", output_base_dir=tmp_path)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact", encoding="utf-8")
    return path


def _append_event(
    state: SessionArtifactState,
    *,
    tool_name: str,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    append_methodology_event(
        state.default_methodology_log_path,
        MethodologyEvent(
            step_name=tool_name,
            tool_name=tool_name,
            input_path=str(input_path) if input_path else None,
            output_path=str(output_path) if output_path else None,
            parameters={},
        ),
    )
    state.methodology_log_path = state.default_methodology_log_path


def _steps_by_id(
    state: SessionArtifactState,
    ai_snapshot: AIWorkflowSnapshot | None = None,
) -> dict[str, WorkflowStep]:
    return {step.id: step for step in derive_workflow_steps(state, ai_snapshot)}


def test_empty_session_blocks_downstream_workflow_steps(tmp_path: Path) -> None:
    steps = _steps_by_id(_state(tmp_path))

    assert steps["dataset"].status == "not_started"
    assert steps["schema"].status == "blocked"
    assert steps["validation"].status == "blocked"
    assert steps["feature_engineering"].status == "blocked"
    assert steps["sweep"].status == "blocked"
    assert steps["visualization"].status == "blocked"
    assert steps["ai_interpretation"].status == "blocked"
    assert "report" not in steps


def test_prepared_dataset_makes_analysis_steps_ready(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.prepared_dataset_path = _touch(state.output_dir / "analysis_inforce.parquet")

    steps = _steps_by_id(state)

    assert steps["dataset"].status == "completed"
    assert steps["schema"].status == "ready"
    assert steps["validation"].status == "ready"
    assert steps["feature_engineering"].status == "ready"
    assert steps["sweep"].status == "ready"
    assert steps["visualization"].status == "blocked"


def test_missing_recorded_artifacts_are_stale(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.prepared_dataset_path = _touch(state.output_dir / "analysis_inforce.parquet")
    state.latest_sweep_path = state.output_dir / "missing_sweep.csv"
    state.latest_visualization_path = state.output_dir / "missing_report.html"

    steps = _steps_by_id(state)

    assert steps["dataset"].status == "completed"
    assert steps["sweep"].status == "stale"
    assert steps["visualization"].status == "stale"


def test_schema_and_validation_are_stale_after_prepared_dataset_change(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    prepared_path = _touch(state.output_dir / "analysis_inforce.parquet")
    state.prepared_dataset_path = prepared_path
    _append_event(state, tool_name="inspect_dataset_schema", input_path=prepared_path)
    _append_event(state, tool_name="run_actuarial_data_checks", input_path=prepared_path)

    completed_steps = _steps_by_id(state)

    assert completed_steps["schema"].status == "completed"
    assert completed_steps["validation"].status == "completed"

    _append_event(
        state,
        tool_name="create_categorical_bands",
        input_path=prepared_path,
        output_path=prepared_path,
    )

    stale_steps = _steps_by_id(state)

    assert stale_steps["schema"].status == "stale"
    assert stale_steps["validation"].status == "stale"
    assert stale_steps["feature_engineering"].status == "completed"


def test_sweep_and_visualization_follow_artifact_existence_and_event_order(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    prepared_path = _touch(state.output_dir / "analysis_inforce.parquet")
    sweep_path = _touch(state.output_dir / "sweep_summary.csv")
    visualization_path = _touch(state.output_dir / "combined_ae_report.html")
    state.prepared_dataset_path = prepared_path
    state.latest_sweep_path = sweep_path
    state.latest_visualization_path = visualization_path
    _append_event(
        state,
        tool_name="run_dimensional_sweep",
        input_path=prepared_path,
        output_path=sweep_path,
    )
    _append_event(
        state,
        tool_name="generate_combined_report",
        input_path=sweep_path,
        output_path=visualization_path,
    )

    completed_steps = _steps_by_id(state)

    assert completed_steps["sweep"].status == "completed"
    assert completed_steps["visualization"].status == "completed"

    newer_sweep_path = _touch(state.output_dir / "sweep_summary_new.csv")
    state.latest_sweep_path = newer_sweep_path
    _append_event(
        state,
        tool_name="run_dimensional_sweep",
        input_path=prepared_path,
        output_path=newer_sweep_path,
    )

    stale_steps = _steps_by_id(state)

    assert stale_steps["sweep"].status == "completed"
    assert stale_steps["visualization"].status == "stale"


def test_sweep_is_stale_when_manifest_source_hash_is_not_current_prepared_dataset(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    current_prepared_path = _touch(state.output_dir / "analysis_inforce.parquet")
    sweep_path = _touch(state.output_dir / "sweep_summary.csv")
    state.prepared_dataset_path = current_prepared_path
    state.latest_sweep_path = sweep_path
    state.artifact_manifest_path = state.default_artifact_manifest_path
    upsert_artifact_entry(
        state.artifact_manifest_path,
        artifact_type="prepared_dataset",
        path=current_prepared_path,
        generating_tool="profile_dataset",
        parameters={},
        source_artifacts=[],
    )
    upsert_artifact_entry(
        state.artifact_manifest_path,
        artifact_type="sweep_summary",
        path=sweep_path,
        generating_tool="run_dimensional_sweep",
        parameters={},
        source_artifacts=[
            {
                "artifact_type": "prepared_dataset",
                "path": str(current_prepared_path),
                "content_hash": "prior-hash",
            }
        ],
    )

    steps = _steps_by_id(state)

    assert steps["sweep"].status == "stale"
    assert steps["sweep"].basis == "artifact manifest source artifact"


@pytest.mark.parametrize(
    ("snapshot", "expected_status"),
    [
        (AIWorkflowSnapshot(ready=False), "blocked"),
        (AIWorkflowSnapshot(ready=True), "ready"),
        (
            AIWorkflowSnapshot(
                ready=True,
                has_response=True,
                response_is_fresh=True,
            ),
            "completed",
        ),
        (
            AIWorkflowSnapshot(
                ready=True,
                has_response=True,
                response_is_fresh=False,
                freshness_mismatches=("state fingerprint",),
            ),
            "stale",
        ),
        (
            AIWorkflowSnapshot(
                ready=False,
                has_response=True,
                response_is_fresh=False,
            ),
            "stale",
        ),
    ],
)
def test_ai_interpretation_status_mirrors_ai_snapshot(
    tmp_path: Path,
    snapshot: AIWorkflowSnapshot,
    expected_status: str,
) -> None:
    steps = _steps_by_id(_state(tmp_path), snapshot)

    assert steps["ai_interpretation"].status == expected_status
