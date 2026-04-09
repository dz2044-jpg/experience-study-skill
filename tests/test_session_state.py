from pathlib import Path

from core.copilot_agent import UnifiedCopilot
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
    assert "- latest_visualization_ready: False" in prompt_state
