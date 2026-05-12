import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.copilot_agent import UnifiedCopilot
from skills.experience_study_skill.ai_models import AI_SWEEP_PACKET_SCHEMA_VERSION
from skills.experience_study_skill.native_tools import get_tool_handlers
from skills.experience_study_skill.schemas import get_tool_specs
from tests.conftest import final_message


EXPECTED_PUBLIC_TOOL_NAMES = {
    "profile_dataset",
    "inspect_dataset_schema",
    "run_actuarial_data_checks",
    "create_categorical_bands",
    "regroup_categorical_features",
    "run_dimensional_sweep",
    "generate_combined_report",
}

GOLDEN_SWEEP_COLUMNS = [
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


class _FakeToolCall:
    def __init__(
        self,
        *,
        tool_call_id: str,
        name: str,
        arguments: dict[str, object] | str,
    ) -> None:
        self.id = tool_call_id
        serialized_arguments = (
            arguments if isinstance(arguments, str) else json.dumps(arguments)
        )
        self.function = SimpleNamespace(name=name, arguments=serialized_arguments)

    def model_dump(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _FakeClient:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self._messages = list(messages)
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = self._messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_tool_gating_hides_visualization_before_sweep(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    intent = copilot._summarize_intent("Generate the combined report.")

    assert "generate_combined_report" not in copilot._enabled_tool_names(intent)


def test_analysis_requires_prepared_dataset(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(copilot.process_message("Run a 1-way sweep on Gender."))

    assert "Profile a dataset first" in final_message(events)


def test_extract_top_n_caps_large_requests(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    assert copilot._extract_top_n("Run a sweep and show the top 500 cohorts.") == 20


def test_public_tool_names_remain_stable():
    exposed_schema_tool_names = {
        spec["function"]["name"] for spec in get_tool_specs()
    }

    assert exposed_schema_tool_names == EXPECTED_PUBLIC_TOOL_NAMES
    assert set(get_tool_handlers()) == EXPECTED_PUBLIC_TOOL_NAMES


def test_profile_and_columns_request_profiles_then_inspects_prepared_dataset(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(copilot.process_message(f"Profile {sample_csv_path} and tell me the columns."))

    tool_starts = [event.message for event in events if event.type == "tool_start"]
    message = final_message(events)

    assert tool_starts == [
        "Executing `profile_dataset`.",
        "Executing `inspect_dataset_schema`.",
    ]
    assert "Profiled the source dataset and saved" in message
    assert "Columns in" in message
    assert message.index("`Policy_Number`") < message.index("`Duration`") < message.index("`MAC`")
    assert copilot.state.prepared_dataset_ready is True
    assert copilot.state.prepared_dataset_path is not None
    assert copilot.state.prepared_dataset_path.exists()


def test_schema_request_for_bare_filename_uses_session_artifact_without_reprofiling(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    events = list(copilot.process_message("What are columns in analysis_inforce.parquet?"))

    tool_starts = [event.message for event in events if event.type == "tool_start"]
    message = final_message(events)

    assert tool_starts == ["Executing `inspect_dataset_schema`."]
    assert copilot.state.prepared_dataset_path is not None
    assert f"Columns in `{copilot.state.prepared_dataset_path.resolve()}`" in message
    assert "`Policy_Number`" in message
    assert "`Risk_Class`" in message


def test_pure_schema_request_does_not_mutate_prepared_artifact_state(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    prepared_path = copilot.state.prepared_dataset_path
    events = list(copilot.process_message("Show me the schema for the current dataset."))

    assert prepared_path is not None
    assert copilot.state.prepared_dataset_path == prepared_path
    assert copilot.state.prepared_dataset_ready is True
    assert any(event.type == "artifact_update" for event in events)
    assert copilot.state.methodology_log_path is not None
    assert copilot.state.methodology_log_path.exists()
    assert "Columns in" in final_message(events)


def test_analysis_only_request_runs_after_profile(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    events = list(copilot.process_message("Run a 1-way sweep on Gender."))
    message = final_message(events)

    assert "Completed a 1-way dimensional sweep" in message
    assert "Summary of Sweep Results" in message
    assert "| Cohort Dimension | Actual Deaths (MAC) | Expected (MEC) | A/E Ratio (Count) | A/E Ratio (Amount) |" in message
    assert "| Gender=M | 2.00 | 0.79 | 2.53 | 0.85 |" in message
    assert "| Gender=F | 1.00 | 0.63 | 1.59 | 0.39 |" in message
    assert "Credibility interval detail is available in the chat explorer and generated report." in message
    assert copilot.state.latest_sweep_ready is True
    assert copilot.state.latest_sweep_path is not None
    assert copilot.state.latest_sweep_path.exists()


def test_visualize_only_request_runs_after_sweep(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))
    list(copilot.process_message("Run a 1-way sweep on Gender."))

    events = list(copilot.process_message("Generate the combined report."))

    assert "visualization report" in final_message(events).lower()
    assert copilot.state.latest_visualization_ready is True
    assert copilot.state.latest_visualization_path is not None
    assert copilot.state.latest_visualization_path.exists()


def test_full_pipeline_runs_in_order_with_session_local_artifacts(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(
        copilot.process_message(
            f"Profile {sample_csv_path}, inspect the schema columns, validate the data, "
            "group Issue_Age into 3 equal-width bands, "
            "run a 1-way sweep on Gender, and generate the combined report."
        )
    )

    tool_starts = [event.message for event in events if event.type == "tool_start"]
    assert tool_starts == [
        "Executing `profile_dataset`.",
        "Executing `inspect_dataset_schema`.",
        "Executing `run_actuarial_data_checks`.",
        "Executing `create_categorical_bands`.",
        "Executing `run_dimensional_sweep`.",
        "Executing `generate_combined_report`.",
    ]

    tool_results = [event.data["result"] for event in events if event.type == "tool_result"]
    validation_result = next(result for result in tool_results if result["kind"] == "validation")
    analysis_result = next(result for result in tool_results if result["kind"] == "analysis")

    assert validation_result["data"]["status"] == "PASS"
    assert analysis_result["artifacts"]["sweep_output_path"].endswith(
        "sweep_summary_1_gender.csv"
    )

    message = final_message(events).lower()
    assert "prepared dataset" in message
    assert "inspected the schema" in message
    assert "actuarial validation completed" in message
    assert "dimensional sweep" in message
    assert "visualization report" in message

    assert copilot.state.prepared_dataset_ready is True
    assert copilot.state.latest_sweep_ready is True
    assert copilot.state.latest_visualization_ready is True

    prepared_path = copilot.state.prepared_dataset_path
    sweep_path = copilot.state.latest_sweep_path
    visualization_path = copilot.state.latest_visualization_path
    sweep_depth_path = copilot.state.latest_sweep_paths_by_depth[1]

    assert prepared_path is not None
    assert prepared_path.name == "analysis_inforce.parquet"
    assert prepared_path.exists()
    prepared_df = pd.read_parquet(prepared_path)
    assert len(prepared_df) == 8
    assert "Issue_Age_band" in prepared_df.columns

    assert sweep_path is not None
    assert sweep_path.name == "sweep_summary.csv"
    assert sweep_path.exists()
    assert sweep_depth_path.name == "sweep_summary_latest_1.csv"
    assert sweep_depth_path.exists()

    sweep_df = pd.read_csv(sweep_path)
    assert list(sweep_df.columns) == GOLDEN_SWEEP_COLUMNS
    assert len(sweep_df) == 2

    sweep_by_dimension = sweep_df.set_index("Dimensions")
    assert sweep_by_dimension.loc["Gender=M", "Sum_MAC"] == 2
    assert sweep_by_dimension.loc["Gender=F", "Sum_MAC"] == 1
    assert sweep_by_dimension.loc["Gender=M", "AE_Ratio_Count"] == pytest.approx(
        2.5316455696202533
    )
    assert sweep_by_dimension.loc["Gender=M", "AE_Ratio_Amount"] == pytest.approx(
        0.8517350157728707
    )
    assert sweep_by_dimension.loc["Gender=F", "AE_Ratio_Count"] == pytest.approx(
        1.5873015873015872
    )
    assert sweep_by_dimension.loc["Gender=F", "AE_Ratio_Amount"] == pytest.approx(
        0.39215686274509803
    )
    assert sweep_df[
        [
            "AE_Count_CI_Lower",
            "AE_Count_CI_Upper",
            "AE_Amount_CI_Lower",
            "AE_Amount_CI_Upper",
        ]
    ].notna().all().all()

    assert visualization_path is not None
    assert visualization_path.name.startswith("combined_ae_report_")
    assert visualization_path.suffix == ".html"
    assert visualization_path.exists()

    methodology_log_path = copilot.state.methodology_log_path
    artifact_manifest_path = copilot.state.artifact_manifest_path
    assert copilot.state.audit_ready is True
    assert methodology_log_path is not None
    assert methodology_log_path.name == "methodology_log.json"
    assert methodology_log_path.exists()
    assert artifact_manifest_path is not None
    assert artifact_manifest_path.name == "artifact_manifest.json"
    assert artifact_manifest_path.exists()
    assert copilot.state.latest_state_fingerprint is not None

    methodology_log = json.loads(methodology_log_path.read_text(encoding="utf-8"))
    step_names = [event["step_name"] for event in methodology_log["events"]]
    assert step_names == [
        "Source dataset profiled",
        "Schema inspected",
        "Validation checks run",
        "Age bands created",
        "Dimensional sweep run",
        "Visualization generated",
    ]
    sweep_event = next(
        event for event in methodology_log["events"] if event["tool_name"] == "run_dimensional_sweep"
    )
    assert sweep_event["parameters"]["depth"] == 1
    assert sweep_event["parameters"]["selected_columns"] == ["Gender"]
    assert sweep_event["parameters"]["filters"] == []
    assert sweep_event["parameters"]["sort_by"] == "AE_Ratio_Amount"
    assert sweep_event["parameters"]["min_mac"] == 0

    manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    fingerprint_inputs = manifest["fingerprint_inputs"]
    assert fingerprint_inputs["packet_schema_version"] == AI_SWEEP_PACKET_SCHEMA_VERSION
    assert fingerprint_inputs["depth"] == 1
    assert fingerprint_inputs["selected_columns"] == ["Gender"]
    assert fingerprint_inputs["sort_by"] == "AE_Ratio_Amount"
    assert fingerprint_inputs["min_mac"] == 0
    source_hashes = fingerprint_inputs["source_hashes"]
    assert "latest_visualization_path" not in source_hashes
    assert not any(key.startswith("pre_call_") for key in source_hashes)

    manifest_paths = {Path(entry["path"]).name for entry in manifest["entries"]}
    assert "analysis_inforce.parquet" in manifest_paths
    assert "sweep_summary.csv" in manifest_paths
    assert "sweep_summary_latest_1.csv" in manifest_paths
    assert visualization_path.name in manifest_paths
    assert "methodology_log.json" not in manifest_paths
    assert "artifact_manifest.json" not in manifest_paths

    prepared_entry = next(
        entry
        for entry in manifest["entries"]
        if entry["artifact_type"] == "prepared_dataset" and entry["path"] == str(prepared_path)
    )
    assert prepared_entry["source_artifacts"][0]["artifact_type"] == "prepared_dataset"
    assert (
        prepared_entry["source_artifacts"][0]["content_hash"]
        != prepared_entry["content_hash"]
    )


def test_report_generation_preserves_latest_sweep_fingerprint(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))
    list(copilot.process_message("Run a 1-way sweep on Gender."))

    sweep_fingerprint = copilot.state.latest_state_fingerprint
    assert sweep_fingerprint is not None
    artifact_manifest_path = copilot.state.artifact_manifest_path
    assert artifact_manifest_path is not None
    manifest_after_sweep = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    sweep_fingerprint_inputs = manifest_after_sweep["fingerprint_inputs"]

    list(copilot.process_message("Generate the combined report."))

    manifest_after_report = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    assert copilot.state.latest_state_fingerprint == sweep_fingerprint
    assert manifest_after_report["state_fingerprint"] == sweep_fingerprint
    assert manifest_after_report["fingerprint_inputs"] == sweep_fingerprint_inputs


def test_validation_methodology_event_records_fail_status(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
):
    invalid_path = tmp_path / "invalid_inforce.csv"
    invalid_df = sample_dataframe.copy()
    invalid_df.loc[0, "MAC"] = 2
    invalid_df.to_csv(invalid_path, index=False)

    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    events = list(copilot.process_message(f"Profile {invalid_path} and validate the data."))

    tool_results = [event.data["result"] for event in events if event.type == "tool_result"]
    validation_result = next(result for result in tool_results if result["kind"] == "validation")
    issue_count = len(validation_result["data"]["issues"])
    assert validation_result["data"]["status"] == "FAIL"
    assert issue_count > 0

    methodology_log_path = copilot.state.methodology_log_path
    assert methodology_log_path is not None
    methodology_log = json.loads(methodology_log_path.read_text(encoding="utf-8"))
    validation_event = next(
        event
        for event in methodology_log["events"]
        if event["tool_name"] == "run_actuarial_data_checks"
    )
    assert validation_event["parameters"]["status"] == "FAIL"
    assert validation_event["parameters"]["issue_count"] == issue_count


def test_filtered_analysis_request_respects_filter_clause(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    list(copilot.process_message("Run a 1-way sweep on Gender where Smoker = Yes."))

    sweep_path = copilot.state.latest_sweep_path
    assert sweep_path is not None
    result_df = pd.read_csv(sweep_path)
    assert not result_df.empty
    assert "Dimensions" in result_df.columns


@pytest.mark.parametrize(
    ("prompt", "expected_depth", "expected_columns"),
    [
        (
            "Run a 1-way sweep on Gender with at least 1 deaths.",
            1,
            ["Gender"],
        ),
        (
            "Run a 2-way sweep on Gender and Risk_Class with at least 1 deaths.",
            2,
            ["Gender", "Risk_Class"],
        ),
    ],
)
def test_min_mac_phrase_request_runs_without_absorbing_threshold_into_dimension(
    tmp_path: Path,
    sample_csv_path: Path,
    prompt: str,
    expected_depth: int,
    expected_columns: list[str],
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    events = list(copilot.process_message(prompt))

    assert f"Completed a {expected_depth}-way dimensional sweep" in final_message(events)
    sweep_path = copilot.state.latest_sweep_path
    assert sweep_path is not None
    result_df = pd.read_csv(sweep_path)
    assert not result_df.empty
    assert result_df["Sum_MAC"].ge(1).all()

    methodology_log_path = copilot.state.methodology_log_path
    assert methodology_log_path is not None
    methodology_log = json.loads(methodology_log_path.read_text(encoding="utf-8"))
    sweep_event = [
        event
        for event in methodology_log["events"]
        if event["tool_name"] == "run_dimensional_sweep"
    ][-1]
    assert sweep_event["parameters"]["depth"] == expected_depth
    assert sweep_event["parameters"]["selected_columns"] == expected_columns
    assert sweep_event["parameters"]["min_mac"] == 1


def test_runtime_preserves_tool_result_when_audit_log_is_malformed(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))
    assert copilot.state.methodology_log_path is not None
    copilot.state.methodology_log_path.write_text("{not-json", encoding="utf-8")
    copilot.state.refresh()
    assert copilot.state.audit_ready is False

    events = list(copilot.process_message("Show me the schema for the current dataset."))

    assert "Columns in" in final_message(events)
    assert any(
        "Audit recording skipped; deterministic tool result was preserved" in event.message
        for event in events
        if event.type == "status"
    )
    assert copilot.state.prepared_dataset_ready is True
    assert copilot.state.audit_ready is False


def test_failed_tool_call_surfaces_exact_error_without_retrying(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    events = list(copilot.process_message("Group Unknown Column into 3 equal-width bands."))

    tool_starts = [event.message for event in events if event.type == "tool_start"]
    message = final_message(events)

    assert tool_starts == ["Executing `create_categorical_bands`."]
    assert copilot.state.prepared_dataset_path is not None
    assert (
        f"Column `Unknown_Column` was not found in `{copilot.state.prepared_dataset_path}`."
        in message
    )


def test_final_llm_response_strips_thinking_from_user_output_and_history(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    copilot.client = _FakeClient(
        [
            SimpleNamespace(
                content="<thinking>Check state and prerequisites.</thinking>\nSchema inspection is ready.",
                tool_calls=[],
            )
        ]
    )

    events = list(copilot.process_message("Show me the schema for /tmp/example.csv."))

    assert final_message(events) == "Schema inspection is ready."
    assert "<thinking>" not in final_message(events)
    assert [event.message for event in events if event.type == "text_delta"] == [
        "Schema ",
        "inspection ",
        "is ",
        "ready.",
    ]
    assert copilot.history[-1]["content"] == "Schema inspection is ready."


def test_tool_call_keeps_internal_thinking_for_model_but_not_user_output(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    fake_client = _FakeClient(
        [
            SimpleNamespace(
                content="<thinking>Check state, then inspect exact schema path.</thinking>",
                tool_calls=[
                    _FakeToolCall(
                        tool_call_id="call_1",
                        name="inspect_dataset_schema",
                        arguments={"data_path": str(sample_csv_path)},
                    )
                ],
            ),
            SimpleNamespace(
                content="<thinking>Summarize the tool output only.</thinking>\nSchema inspection completed.",
                tool_calls=[],
            ),
        ]
    )
    copilot.client = fake_client

    events = list(copilot.process_message(f"Show me the schema for {sample_csv_path}."))

    assert [event.message for event in events if event.type == "tool_start"] == [
        "Executing `inspect_dataset_schema`."
    ]
    assert final_message(events) == "Schema inspection completed."
    assert "<thinking>" not in final_message(events)
    assert copilot.history[-1]["content"] == "Schema inspection completed."
    second_call_messages = fake_client.calls[1]["messages"]
    assistant_messages = [
        message
        for message in second_call_messages
        if message["role"] == "assistant"
    ]
    assert any("<thinking>" in message["content"] for message in assistant_messages)


def _latest_tool_result(events) -> dict[str, object]:
    tool_results = [event.data["result"] for event in events if event.type == "tool_result"]
    assert tool_results
    return tool_results[-1]


def _assert_error_result_shape(result: dict[str, object]) -> None:
    assert set(result) == {"ok", "kind", "message", "artifacts", "data"}
    assert result["ok"] is False
    assert result["artifacts"] == {}
    assert isinstance(result["message"], str)
    assert "Traceback" not in result["message"]


def test_unknown_llm_tool_call_returns_safe_validation_error(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    copilot.client = _FakeClient(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        tool_call_id="call_1",
                        name="unknown_tool",
                        arguments={},
                    )
                ],
            )
        ]
    )

    events = list(copilot.process_message(f"Show me the schema for {sample_csv_path}."))

    result = _latest_tool_result(events)
    _assert_error_result_shape(result)
    assert result["kind"] == "validation_error"
    assert result["data"] == {"error_type": "unknown_tool"}
    assert final_message(events) == "Tool `unknown_tool` is not available."


def test_malformed_llm_tool_arguments_return_safe_validation_error(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    copilot.client = _FakeClient(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        tool_call_id="call_1",
                        name="inspect_dataset_schema",
                        arguments='{"data_path": ',
                    )
                ],
            )
        ]
    )

    events = list(copilot.process_message(f"Show me the schema for {sample_csv_path}."))

    result = _latest_tool_result(events)
    _assert_error_result_shape(result)
    assert result["kind"] == "validation_error"
    assert result["data"] == {"error_type": "malformed_json"}
    assert "malformed JSON" in final_message(events)


def test_runtime_validation_forbids_unknown_tool_fields(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    copilot.client = _FakeClient(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        tool_call_id="call_1",
                        name="inspect_dataset_schema",
                        arguments={
                            "data_path": str(sample_csv_path),
                            "unexpected": "value",
                        },
                    )
                ],
            )
        ]
    )

    events = list(copilot.process_message(f"Show me the schema for {sample_csv_path}."))

    result = _latest_tool_result(events)
    _assert_error_result_shape(result)
    assert result["kind"] == "validation_error"
    assert result["data"]["error_type"] == "schema_validation"
    assert "unexpected" in final_message(events)


def test_runtime_validation_catches_missing_required_tool_fields(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    copilot.client = _FakeClient(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    _FakeToolCall(
                        tool_call_id="call_1",
                        name="profile_dataset",
                        arguments={},
                    )
                ],
            )
        ]
    )

    events = list(copilot.process_message(f"Profile {sample_csv_path}."))

    result = _latest_tool_result(events)
    _assert_error_result_shape(result)
    assert result["kind"] == "validation_error"
    assert result["data"]["error_type"] == "schema_validation"
    assert "data_path" in final_message(events)


def test_handler_exception_returns_safe_runtime_error_without_crashing(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    def failing_handler(args, context):
        raise RuntimeError("internal secret stack detail")

    copilot.active_skill.tool_handlers["inspect_dataset_schema"] = failing_handler

    events = list(copilot.process_message(f"Show me the schema for {sample_csv_path}."))

    result = _latest_tool_result(events)
    _assert_error_result_shape(result)
    assert result["kind"] == "runtime_error"
    assert result["data"]["error_type"] == "handler_exception"
    assert result["data"]["exception_type"] == "RuntimeError"
    assert "internal secret stack detail" not in final_message(events)
    assert "failed during deterministic execution" in final_message(events)
