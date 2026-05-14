from pathlib import Path

import pytest

from core.copilot_agent import UnifiedCopilot
from core.session_state import SUPPORTED_SWEEP_DEPTHS, SessionArtifactState
from tests.conftest import final_message


def test_session_isolation_uses_separate_output_directories(
    tmp_path: Path,
    sample_csv_path: Path,
):
    output_base_dir = tmp_path / "sessions"
    copilot_a = UnifiedCopilot(session_id="session-a", output_base_dir=output_base_dir)
    copilot_b = UnifiedCopilot(session_id="session-b", output_base_dir=output_base_dir)

    list(copilot_a.process_message(f"Profile {sample_csv_path}"))
    list(copilot_b.process_message(f"Profile {sample_csv_path}"))

    assert copilot_a.state.prepared_dataset_path is not None
    assert copilot_b.state.prepared_dataset_path is not None
    assert copilot_a.state.prepared_dataset_path != copilot_b.state.prepared_dataset_path
    assert copilot_a.state.prepared_dataset_path.exists()
    assert copilot_b.state.prepared_dataset_path.exists()


def test_reset_session_removes_only_old_session_directory(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))
    old_output_dir = copilot.state.output_dir
    old_output_dir.joinpath("scratch.txt").write_text("artifact", encoding="utf-8")

    new_session_id = copilot.reset_session()

    assert new_session_id != "session-a"
    assert not old_output_dir.exists()
    assert copilot.state.output_dir.exists()
    assert copilot.history == []


def test_missing_visualization_prerequisite_returns_guidance(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(copilot.process_message("Generate the combined report for the latest sweep."))

    assert "Run a dimensional sweep first" in final_message(events)


def test_prompt_state_contract_exposes_readiness_flags_and_artifact_paths(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    list(copilot.process_message(f"Profile {sample_csv_path}"))
    list(copilot.process_message("Run a 1-way sweep on Gender."))

    prompt_state = copilot.state.to_prompt()

    assert "Current Session State:" in prompt_state
    assert f"- output_dir: {copilot.state.output_dir}" in prompt_state
    assert "- prepared_dataset_ready: True" in prompt_state
    assert f"- prepared_dataset_path: {copilot.state.prepared_dataset_path}" in prompt_state
    assert "- latest_sweep_ready: True" in prompt_state
    assert f"- latest_sweep_path: {copilot.state.latest_sweep_path}" in prompt_state
    assert "- latest_sweep_paths_by_depth: {1:" in prompt_state
    assert "- existing_sweep_artifact_depths: [1]" in prompt_state
    assert f"- supported_sweep_depths: {SUPPORTED_SWEEP_DEPTHS}" in prompt_state
    assert "available_sweep_depths" not in prompt_state
    assert "- latest_visualization_ready: False" in prompt_state


def test_llm_prompt_omits_absolute_paths_but_keeps_artifact_refs(tmp_path: Path):
    state = SessionArtifactState(session_id="session-a", output_base_dir=tmp_path)
    raw_path = tmp_path / "private" / "synthetic_inforce.csv"
    prepared_path = state.output_dir / "analysis_inforce.parquet"
    sweep_path = state.output_dir / "sweep_summary.csv"
    depth_path = state.output_dir / "sweep_summary_latest_1.csv"
    visualization_path = state.output_dir / "combined_ae_report.html"
    for path in (raw_path, prepared_path, sweep_path, depth_path, visualization_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
    state.raw_input_path = raw_path
    state.prepared_dataset_path = prepared_path
    state.latest_sweep_path = sweep_path
    state.latest_sweep_paths_by_depth = {1: depth_path}
    state.latest_visualization_path = visualization_path
    state.methodology_log_path = state.default_methodology_log_path
    state.artifact_manifest_path = state.default_artifact_manifest_path
    state.latest_state_fingerprint = "state-a"

    prompt_state = state.to_llm_prompt()

    assert str(tmp_path) not in prompt_state
    assert "synthetic_inforce.csv" not in prompt_state
    assert "- raw_input_ref: current source dataset (path omitted)" in prompt_state
    assert "- prepared_dataset_ref: analysis_inforce.parquet" in prompt_state
    assert "- latest_sweep_ref: sweep_summary.csv" in prompt_state
    assert "- latest_sweep_refs_by_depth: {1: 'sweep_summary_latest_1.csv'}" in prompt_state
    assert "- existing_sweep_artifact_depths: [1]" in prompt_state
    assert f"- supported_sweep_depths: {SUPPORTED_SWEEP_DEPTHS}" in prompt_state
    assert "available_sweep_depths" not in prompt_state
    assert "- latest_visualization_ref: combined_ae_report.html" in prompt_state
    assert "- latest_state_fingerprint: state-a" in prompt_state
    assert "omit" in prompt_state


def test_llm_prompt_does_not_treat_existing_sweep_artifacts_as_capabilities(
    tmp_path: Path,
) -> None:
    state = SessionArtifactState(session_id="session-a", output_base_dir=tmp_path)
    prepared_path = state.output_dir / "analysis_inforce.parquet"
    sweep_path = state.output_dir / "sweep_summary.csv"
    depth_path = state.output_dir / "sweep_summary_latest_1.csv"
    for path in (prepared_path, sweep_path, depth_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
    state.prepared_dataset_path = prepared_path
    state.latest_sweep_path = sweep_path
    state.latest_sweep_paths_by_depth = {1: depth_path}

    prompt_state = state.to_llm_prompt()

    assert "- prepared_dataset_ready: True" in prompt_state
    assert "- existing_sweep_artifact_depths: [1]" in prompt_state
    assert "- supported_sweep_depths: [1, 2, 3]" in prompt_state
    assert "available_sweep_depths" not in prompt_state


def test_llm_messages_use_safe_session_prompt(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path)
    raw_path = tmp_path / "sensitive" / "synthetic_inforce.csv"
    prepared_path = copilot.state.output_dir / "analysis_inforce.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("raw", encoding="utf-8")
    prepared_path.write_text("prepared", encoding="utf-8")
    copilot.state.raw_input_path = raw_path
    copilot.state.prepared_dataset_path = prepared_path

    messages = copilot._llm_messages("Inspect the schema.")
    prompt_state = messages[1]["content"]

    assert str(tmp_path) not in prompt_state
    assert str(raw_path) not in prompt_state
    assert "synthetic_inforce.csv" not in prompt_state
    assert "analysis_inforce.parquet" in prompt_state


@pytest.mark.parametrize("result_kind", ["profile", "feature_engineering"])
def test_prepared_dataset_writes_invalidate_downstream_artifacts_but_preserve_audit_paths(
    tmp_path: Path,
    result_kind: str,
):
    state = SessionArtifactState(session_id="session-a", output_base_dir=tmp_path)
    state.output_dir.mkdir(parents=True)
    prepared_path = state.output_dir / "analysis_inforce.parquet"
    sweep_path = state.output_dir / "sweep_summary.csv"
    depth_path = state.output_dir / "sweep_summary_latest_1.csv"
    visualization_path = state.output_dir / "combined_ae_report.html"
    methodology_log_path = state.default_methodology_log_path
    artifact_manifest_path = state.default_artifact_manifest_path
    for path in (
        prepared_path,
        sweep_path,
        depth_path,
        visualization_path,
        methodology_log_path,
        artifact_manifest_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    state.prepared_dataset_path = prepared_path
    state.prepared_dataset_ready = True
    state.latest_sweep_path = sweep_path
    state.latest_sweep_ready = True
    state.latest_sweep_paths_by_depth = {1: depth_path}
    state.latest_visualization_path = visualization_path
    state.latest_visualization_ready = True
    state.methodology_log_path = methodology_log_path
    state.artifact_manifest_path = artifact_manifest_path
    state.latest_state_fingerprint = "state-a"

    changed = state.apply_tool_result(
        {
            "ok": True,
            "kind": result_kind,
            "message": "prepared dataset updated",
            "artifacts": {"prepared_dataset_path": str(prepared_path)},
            "data": {},
        }
    )

    assert changed is True
    assert state.prepared_dataset_path == prepared_path
    assert state.latest_sweep_path is None
    assert state.latest_sweep_ready is False
    assert state.latest_sweep_paths_by_depth == {}
    assert state.latest_visualization_path is None
    assert state.latest_visualization_ready is False
    assert state.latest_state_fingerprint is None
    assert state.methodology_log_path == methodology_log_path
    assert state.artifact_manifest_path == artifact_manifest_path
