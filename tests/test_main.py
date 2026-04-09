from __future__ import annotations

import pandas as pd

from core.copilot_agent import CopilotEvent
import main


class _FakeStatusPanel:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.updates: list[dict[str, object]] = []

    def write(self, message: str) -> None:
        self.writes.append(message)

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class _FakeResponsePlaceholder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def markdown(self, message: str) -> None:
        self.messages.append(message)


def test_consume_copilot_events_keeps_fallback_status_only(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()
    final_message = "Columns in `/tmp/analysis_inforce.parquet` (2):\n- `Policy_Number`: `string`"
    success_tool_message = "Inspected the schema for `/tmp/analysis_inforce.parquet`."

    response, visualization_path, sweep_results = main._consume_copilot_events(
        [
            CopilotEvent("status", message="Copilot received a new request."),
            CopilotEvent(
                "status",
                message="OpenAI is unavailable. Using deterministic local planning.",
            ),
            CopilotEvent("tool_start", message="Executing `inspect_dataset_schema`."),
            CopilotEvent(
                "tool_result",
                message=success_tool_message,
                data={"result": {"ok": True, "kind": "schema"}},
            ),
            CopilotEvent("text_delta", message=final_message),
            CopilotEvent("final", message=final_message, data={"artifact_state": {}}),
        ],
        status_panel=status_panel,
        response_placeholder=response_placeholder,
    )

    assert "OpenAI is unavailable. Using deterministic local planning." in status_panel.writes
    assert success_tool_message not in status_panel.writes
    assert response == final_message
    assert visualization_path is None
    assert sweep_results is None


def test_consume_copilot_events_captures_analysis_rows(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()
    rows = [
        {
            "Dimensions": "Gender=M",
            "Sum_MAC": 2,
            "Sum_MOC": 3.7,
            "Sum_MEC": 0.789,
            "Sum_MAF": 270000.0,
            "Sum_MEF": 317000.0,
            "AE_Ratio_Count": 2.5316,
            "AE_Ratio_Amount": 0.8517,
            "AE_Count_CI_Lower": 0.8,
            "AE_Count_CI_Upper": 4.2,
            "AE_Amount_CI_Lower": 0.2,
            "AE_Amount_CI_Upper": 1.5,
        }
    ]

    _, _, sweep_results = main._consume_copilot_events(
        [
            CopilotEvent("status", message="Copilot received a new request."),
            CopilotEvent(
                "tool_result",
                message="Completed a 1-way dimensional sweep.",
                data={"result": {"ok": True, "kind": "analysis", "data": {"results": rows}}},
            ),
            CopilotEvent("final", message="done", data={"artifact_state": {}}),
        ],
        status_panel=status_panel,
        response_placeholder=response_placeholder,
    )

    assert sweep_results == rows


def test_build_sweep_display_frame_renames_and_rounds() -> None:
    display_df = main._build_sweep_display_frame(
        [
            {
                "Dimensions": "Risk_Class=Standard",
                "Sum_MAC": 14,
                "Sum_MOC": 10.135,
                "Sum_MEC": 1.384,
                "Sum_MAF": 120000.123,
                "Sum_MEF": 95000.567,
                "AE_Ratio_Count": 10.115,
                "AE_Ratio_Amount": 1.789,
                "AE_Count_CI_Lower": 0.904,
                "AE_Count_CI_Upper": 2.226,
                "AE_Amount_CI_Lower": 0.377,
                "AE_Amount_CI_Upper": 2.444,
            }
        ]
    )

    assert list(display_df.columns) == [
        "Cohort Dimension",
        "Actual Deaths (MAC)",
        "Exposure (MOC)",
        "Expected (MEC)",
        "Actual Amount (MAF)",
        "Expected Amount (MEF)",
        "A/E Ratio (Count)",
        "A/E Ratio (Amount)",
        "Count CI Lower",
        "Count CI Upper",
        "Amount CI Lower",
        "Amount CI Upper",
    ]
    row = display_df.iloc[0].to_dict()
    assert row["Cohort Dimension"] == "Risk_Class=Standard"
    assert row["Actual Deaths (MAC)"] == 14.0
    assert row["Expected (MEC)"] == 1.38
    assert row["A/E Ratio (Count)"] == 10.12
    assert row["Amount CI Upper"] == 2.44
    assert isinstance(display_df, pd.DataFrame)
