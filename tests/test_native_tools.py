from pathlib import Path

import pandas as pd

from skills.experience_study_skill.native_tools import (
    ToolExecutionContext,
    compute_ae_ci,
    compute_ae_ci_amount,
    generate_combined_report,
    run_dimensional_sweep,
)


def test_zero_claim_ci_amount_still_returns_upper_bound():
    lower, upper = compute_ae_ci_amount(
        mac=0,
        moc=1000,
        mec=4,
        actual_amount=0,
        expected_amount=400000,
    )

    assert lower is not None
    assert upper is not None
    assert upper > 0


def test_count_ci_returns_values_for_standard_case():
    lower, upper = compute_ae_ci(mac=5, moc=1000, mec=4)

    assert lower is not None
    assert upper is not None
    assert lower < upper


def test_visualization_tool_enforces_prerequisite(tmp_path: Path):
    context = ToolExecutionContext(
        session_id="session-a",
        output_dir=tmp_path / "sessions" / "session-a",
    )

    result = generate_combined_report(context=context)

    assert result["ok"] is False
    assert result["kind"] == "missing_prerequisite"
    assert "Run dimensional sweep first" in result["message"]


def test_dimensional_sweep_caps_top_n_payload(tmp_path: Path):
    rows = [
        {
            "Policy_Number": f"P{index:03d}",
            "MAC": 1,
            "MOC": 1.0,
            "MEC": 0.25,
            "MAF": 100000.0 + index,
            "MEF": 80000.0,
            "Segment": f"segment_{index:02d}",
        }
        for index in range(25)
    ]
    data_path = tmp_path / "prepared.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    context = ToolExecutionContext(
        session_id="session-a",
        output_dir=tmp_path / "sessions" / "session-a",
    )

    result = run_dimensional_sweep(
        context=context,
        data_path=str(data_path),
        selected_columns=["Segment"],
        top_n=500,
    )

    assert result["ok"] is True
    assert len(result["data"]["results"]) == 20
